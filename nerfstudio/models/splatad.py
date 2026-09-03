# Copyright 2025 the authors of NeuRAD and contributors.
# ruff: noqa: E741
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Per-Gaussian Atmospheric Scattering Model + LiDAR Residual Offsets (FINAL).
===========================================================================
Innovation 1 — per-Gaussian transmission -> t_map ASM :
    1. Surface rasterization -> CNN decoder -> clean RGB.
    2. Per-Gaussian t_i = exp(-beta * d_i),  d_i detach -> protect geometry.
    3. t_i -> 2D t_map (second lightweight raster).
    4. Foggy output = clean RGB * t_map + A * (1 - t_map).
    5. Dehazing : skip steps 2-4 -> zero-shot clean output.

Innovation 2 — LiDAR residual offsets :
    LiDAR renders use mu + Delta_mu_offsets, camera uses mu (shared backbone).

Conservative refinements (all verified in ablation):
    * d_gauss.detach()          — geometry protection (no blur).
    * fog_t_min = 0.20          — full fog expression range.
    * fog_beta_reg = 0.005      — allows beta to freely match data.
    * sqrt depth weighting      — balanced near/far supervision.
    * A learnable + weak L2     — adapts across fog densities.
    * t_map_reg lambda=0.05 (alpha>0.7) — anchors transmission to physics.
    * ssim_lambda = 0.30        — slightly stronger structure prior.

Ablation : use_dual_stream / use_lidar_offset independent toggles.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_msssim import SSIM
from torch.nn import BCEWithLogitsLoss, Parameter
from typing_extensions import Literal

from nerfstudio.cameras.camera_optimizers import (
    CameraOptimizer,
    CameraOptimizerConfig,
    CameraVelocityOptimizer,
    CameraVelocityOptimizerConfig,
)
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.lidars import Lidars, transform_points, transform_points_pairwise
from nerfstudio.data.datamanagers.full_images_lidar_datamanager import (
    AZIM_CHANNELS_PER_TILE,
    ELEV_CHANNELS_PER_TILE,
)
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.data.utils.data_utils import points_in_box
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.field_components.mlp import MLP
from nerfstudio.model_components.cnns import BasicBlock
from nerfstudio.model_components.losses import L1Loss, MSELoss
from nerfstudio.model_components.strategy import ADDefaultStrategy, ADMCMCStrategy
from nerfstudio.models.ad_model import ADModel, ADModelConfig
from nerfstudio.models.splatfacto import get_viewmat, resize_image
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.math import chamfer_distance
from nerfstudio.utils.poses import inverse as pose_inverse, to4x4
from nerfstudio.viewer.viewer_elements import ViewerSlider

try:
    from gsplat.rendering import lidar_rasterization, rasterization
except ImportError:
    print("Please install gsplat>=1.0.0")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def random_quat_tensor(N):
    u, v, w = torch.rand(N), torch.rand(N), torch.rand(N)
    return torch.stack([
        torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
        torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
        torch.sqrt(u) * torch.sin(2 * math.pi * w),
        torch.sqrt(u) * torch.cos(2 * math.pi * w),
    ], dim=-1)


def get_ray_dirs_pinhole(cameras, width, height, c2w):
    ys = (torch.arange(height, device=cameras.device, dtype=torch.float32)
          + (0.5 - cameras.cy[0, 0])) / cameras.fy[0, 0]
    xs = (torch.arange(width, device=cameras.device, dtype=torch.float32)
          + (0.5 - cameras.cx[0, 0])) / cameras.fx[0, 0]
    grid = torch.meshgrid(ys, xs, indexing="ij")
    dirs = torch.stack([grid[1], -grid[0], -torch.ones_like(grid[0])], dim=-1)
    dirs = dirs.view(-1, 3)
    dirs = torch.matmul(dirs, c2w[0, :3, :3].transpose(0, 1))
    return (dirs / dirs.norm(dim=-1, keepdim=True)).view(height, width, 3)


# ---------------------------------------------------------------------------
# CNN decoder
# ---------------------------------------------------------------------------

class RGBDecoderCNN(torch.nn.Module):
    def __init__(
        self,
        in_dim=6,
        out_dim=6,
        skip_dim=3,
        weight_init_scale=1e-2,
        hidden_dim=32,
        kernel_size=3,
        num_hidden_blocks=1,
    ):
        super().__init__()
        last = torch.nn.Conv2d(hidden_dim, out_dim, 1)
        last.weight.data *= weight_init_scale

        layers = [BasicBlock(in_dim, hidden_dim, kernel_size,
                             padding=kernel_size // 2, use_bn=False)]
        for _ in range(num_hidden_blocks):
            layers.append(BasicBlock(hidden_dim, hidden_dim, kernel_size,
                                     padding=kernel_size // 2, use_bn=False))
        layers.append(last)
        self.net = torch.nn.Sequential(*layers)
        self.skip_dim = skip_dim
        self.out_dim = out_dim

    def forward(self, features, ray_dirs):
        features = features.view(1, *features.shape[-3:])
        albedo, spec = features.split([self.skip_dim, features.shape[-1] - self.skip_dim], dim=-1)
        spec = torch.cat([spec, ray_dirs], dim=-1).permute(0, 3, 1, 2)
        spec = self.net(spec).permute(0, 2, 3, 1)
        return albedo * (1 + spec[..., :3]) + spec[..., 3:]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SplatADModelConfig(ADModelConfig):
    _target: Type = field(default_factory=lambda: SplatADModel)

    # ---- original splat-ad ----
    warmup_length: int = 500
    refine_every: int = 100
    resolution_schedule: int = 3000
    background_color: Literal["random", "black", "white"] = "random"
    num_downscales: int = 2
    strategy: Literal["default", "mcmc"] = "mcmc"
    cull_alpha_thresh: float = 0.1
    cull_scale_thresh: float = 500.0
    continue_cull_post_densification: bool = True
    reset_alpha_every: int = 30
    densify_grad_thresh: float = 0.0006
    densify_size_thresh: float = 0.5
    n_split_samples: int = 2
    cull_screen_size: float = 0.15
    split_screen_size: float = 0.05
    stop_screen_size_at: int = 4000
    use_absgrad: bool = True
    mcmc_cap_max: int = 5_000_000
    mcmc_noise_lr: float = 5e5
    mcmc_min_opacity: float = 0.005
    verbose: bool = True
    max_steps: int = 30_000
    init_opacities: float = 0.1
    init_scale: float = 1.0
    max_num_seed_points: int = 2_000_000
    ssim_lambda: float = 0.30
    stop_split_at: int = 15000
    mcmc_scale_reg_lambda: float = 0.001
    mcmc_opacity_reg_lambda: float = 0.005
    output_depth_during_training: bool = False
    rasterize_mode: Literal["classic", "antialiased"] = "antialiased"
    camera_optimizer: CameraOptimizerConfig = field(
        default_factory=lambda: CameraOptimizerConfig(mode="off"))
    camera_velocity_optimizer: CameraVelocityOptimizerConfig = field(
        default_factory=lambda: CameraVelocityOptimizerConfig(enabled=True))
    feature_dim: int = 13
    appearance_dim: int = 8
    implementation: Literal["tcnn", "torch"] = "tcnn"
    actor_flip_probability: float = 0.5
    flip_actors_at_init: bool = True
    n_far_points: float = 300_000
    depth_lambda: float = 0.1
    depth_loss_quantile_threshold: float = 0.95
    intensity_lambda: float = 1.0
    ray_drop_lambda: float = 0.1
    compensate_rs_camera: bool = True
    compensate_rs_lidar: bool = True
    radius_clip_pix: float = 0.0
    radius_clip_lidar: float = 0.0
    line_of_sight_lambda: float = 0.1
    line_of_sight_dist: float = 0.8
    use_camopt_in_eval: bool = False
    min_points_per_actor: int = 500
    rgb_decoder_hidden_dim: int = 32
    rgb_decoder_kernel_size: int = 3
    rgb_decoder_num_hidden_blocks: int = 1

    # ======== Innovation 1 – per-Gaussian t -> t_map ASM (FINAL) ========
    use_dual_stream: bool = True
    """Master switch for the dual-stream architecture."""

    fog_beta_init: float = 0.03
    """Initial fog density beta (before softplus + beta_min)."""

    fog_beta_min: float = 0.02
    """Hard lower bound — ONLY prevents collapse to zero."""

    fog_beta_reg: float = 0.005
    """L2 regularisation on beta — minimal, lets recon loss drive beta freely."""

    fog_t_min: float = 0.20
    """Minimum per-Gaussian transmission — full fog expression range."""

    fog_atmospheric_light_reg: float = 0.05
    """L2 weight pulling atmospheric light A toward target grey-white."""

    fog_atmospheric_light_target: Tuple[float, float, float] = (0.95, 0.95, 0.95)
    """Target RGB value for A regularisation — matches synthetic fog generator."""

    render_weather: bool = True
    """If True, apply ASM during eval.  Set False for zero-shot dehazing."""

    t_map_reg_lambda: float = 0.05
    """Weak L1 supervision of t_map from depth-derived transmission (alpha>0.7 only)."""

    depth_weighted_loss_lambda: float = 0.30
    """Weight for sqrt depth-weighted reconstruction loss."""

    # ======== Innovation 2 – LiDAR residual offsets ========
    use_lidar_offset: bool = True
    """Learn per-Gaussian LiDAR position offsets."""

    lidar_offset_reg: float = 0.01
    """L2 regularisation keeping offsets small."""

    def __post_init__(self):
        if self.strategy == "mcmc":
            self.init_opacities = 0.5
            self.init_scale = 0.2


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SplatADModel(ADModel):
    config: SplatADModelConfig

    def __init__(self, *args, seed_points: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], **kwargs):
        self.seed_points = seed_points
        self.last_size = (1, 1)
        super().__init__(*args, **kwargs)

    @property
    def has_environment_stream(self):
        return self.config.use_dual_stream and self.raw_fog_beta is not None

    # ── populate ──────────────────────────────────────────────────

    def populate_modules(self):
        super().populate_modules()
        self.collider = None

        sp, dp = self.split_seed_points(self.seed_points)
        sw = self.scene_box.aabb.diff(dim=0)[..., 0].item()
        sl = self.scene_box.aabb.diff(dim=0)[..., 1].item()

        # Far points sampled along rays, then close random points inside the box.
        rand_d = torch.rand(int(self.config.n_far_points), 3) - 0.5
        rand_d[:, -1] = rand_d[:, -1].abs()
        rand_d = rand_d / rand_d.norm(dim=-1, keepdim=True)
        rand_dist = torch.rand(int(self.config.n_far_points), 1)
        n, f = min(sw, sl) / 2, 1e4
        rand_dist = 1 / (1 / n * (1 - rand_dist) + 1 / f * rand_dist)
        far_pts = rand_d * rand_dist
        far_pts = torch.cat([far_pts, torch.randint_like(far_pts, 0, 255)], -1)

        close_pts = torch.rand(int(self.config.n_far_points), 3) - 0.5
        close_pts = close_pts * torch.tensor([sw, sl, 50])
        close_pts = torch.cat([close_pts, torch.randint_like(close_pts, 0, 255)], -1)

        sp = torch.cat([sp, far_pts, close_pts], 0)
        self.gauss_params = self.create_gauss_param_dict(dp, [sp], self.config.flip_actors_at_init)

        # Innovation 2: per-Gaussian LiDAR position offsets.
        if self.config.use_lidar_offset:
            self.gauss_params["lidar_offsets"] = torch.nn.Parameter(
                torch.zeros(self.gauss_params["means"].shape[0], 3))

        # Innovation 1: learnable ASM parameters.
        if self.config.use_dual_stream:
            ba = max(self.config.fog_beta_init - self.config.fog_beta_min, 1e-6)
            self.raw_fog_beta = torch.nn.Parameter(
                torch.tensor(math.log(math.exp(ba) - 1.0)))
            target = self.config.fog_atmospheric_light_target
            self.fog_atmospheric_light = torch.nn.Parameter(
                torch.logit(torch.tensor(target, dtype=torch.float32)))
        else:
            self.raw_fog_beta = None
            self.fog_atmospheric_light = None

        ds = self.kwargs["metadata"]
        ns = len(ds["sensor_idx_to_name"])

        self.camera_optimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu")
        self.camera_velocity_optimizer = self.config.camera_velocity_optimizer.setup(
            num_cameras=self.num_train_data, num_unique_cameras=ns, device="cpu")

        vd = 3
        self.rgb_decoder = torch.compile(RGBDecoderCNN(
            self.config.feature_dim + self.config.appearance_dim + vd,
            hidden_dim=self.config.rgb_decoder_hidden_dim,
            kernel_size=self.config.rgb_decoder_kernel_size,
            num_hidden_blocks=self.config.rgb_decoder_num_hidden_blocks,
        ), disable=True)

        self.appearance_embedding = torch.nn.Embedding(ns, self.config.appearance_dim)
        self.fallback_sensor_idx = ViewerSlider("fallback sensor idx", 0, 0, ns - 1, step=1)

        self.setup_rs_editing()

        self.lidar_decoder = MLP(
            in_dim=self.config.feature_dim + self.config.appearance_dim + vd,
            layer_width=32,
            out_dim=2,
            num_layers=3,
            implementation=self.config.implementation,
            out_activation=None,
        )

        self.render_weather_slider = ViewerSlider(
            "Render Weather", 1.0, 0.0, 1.0, step=1.0,
            cb_hook=lambda o: setattr(self.config, 'render_weather', o.value > 0.5))

        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

        self.step = 0
        self.median_l2 = lambda p, g: torch.median((p - g) ** 2)
        self.mean_rel_l2 = lambda p, g: torch.mean(((p - g) / g) ** 2)
        self.rmse = lambda p, g: torch.sqrt(torch.mean((p - g) ** 2))
        self.chamfer_distance = lambda p, g: chamfer_distance(p, g, 1000, True)

        self.depth_loss = L1Loss(reduction="none")
        self.intensity_loss = MSELoss()
        self.ray_drop_loss = BCEWithLogitsLoss()

        self.background_color = (
            torch.tensor([0.1490, 0.1647, 0.2157])
            if self.config.background_color == "random"
            else get_color(self.config.background_color)
        )

        if self.config.strategy == "mcmc":
            self.strategy = ADMCMCStrategy(
                cap_max=self.config.mcmc_cap_max,
                noise_lr=self.config.mcmc_noise_lr,
                refine_start_iter=self.config.warmup_length,
                refine_stop_iter=self.config.stop_split_at,
                refine_every=self.config.refine_every,
                min_opacity=self.config.mcmc_min_opacity,
                verbose=self.config.verbose,
            )
            self.strategy_state = self.strategy.initialize_state()
            self.config.init_opacities = self.config.mcmc_min_opacity

        elif self.config.strategy == "default":
            self.strategy = ADDefaultStrategy(
                prune_opa=self.config.cull_alpha_thresh,
                grow_grad2d=self.config.densify_grad_thresh,
                grow_scale3d=self.config.densify_size_thresh,
                grow_scale2d=self.config.split_screen_size,
                prune_scale3d=self.config.cull_scale_thresh,
                prune_scale2d=self.config.cull_screen_size,
                refine_scale2d_stop_iter=self.config.stop_screen_size_at,
                refine_start_iter=self.config.warmup_length,
                refine_stop_iter=self.config.stop_split_at,
                reset_every=self.config.reset_alpha_every * self.config.refine_every,
                refine_every=self.config.refine_every,
                pause_refine_after_reset=self.num_train_data + self.config.refine_every,
                absgrad=self.config.use_absgrad,
                revised_opacity=False,
                verbose=self.config.verbose,
            )
            self.strategy_state = self.strategy.initialize_state(scene_scale=1.0)

        else:
            raise NotImplementedError

        self.optimizers = {}

    # ── properties ──

    @property
    def num_points(self):
        return self.means.shape[0]

    @property
    def means(self):
        return self.gauss_params["means"]

    @property
    def scales(self):
        return self.gauss_params["scales"]

    @property
    def quats(self):
        return self.gauss_params["quats"]

    @property
    def features_dc(self):
        return self.gauss_params["features_dc"]

    @property
    def features_rest(self):
        return self.gauss_params["features_rest"]

    @property
    def opacities(self):
        return self.gauss_params["opacities"]

    @property
    def id(self):
        return self.gauss_params["id"]

    # ── rolling-shutter editing sliders ──

    def setup_rs_editing(self):
        self.rs_editing = {
            "rs_time": 0.0,
            "lin_vel_x": 0.0, "lin_vel_y": 0.0, "lin_vel_z": 0.0,
            "ang_vel_x": 0.0, "ang_vel_y": 0.0, "ang_vel_z": 0.0,
        }
        self.rs_time_slider = ViewerSlider("rs time", 0.0, 0.0, 0.2, 0.001,
            cb_hook=lambda o: self.rs_editing.update({"rs_time": o.value}))
        self.rs_lin_vel_x_slider = ViewerSlider("rs lin vel x", 0.0, -30, 30, 0.01,
            cb_hook=lambda o: self.rs_editing.update({"lin_vel_x": o.value}))
        self.rs_lin_vel_y_slider = ViewerSlider("rs lin vel y", 0.0, -30, 30, 0.01,
            cb_hook=lambda o: self.rs_editing.update({"lin_vel_y": o.value}))
        self.rs_lin_vel_z_slider = ViewerSlider("rs lin vel z", 0.0, -30, 30, 0.01,
            cb_hook=lambda o: self.rs_editing.update({"lin_vel_z": o.value}))
        self.rs_ang_vel_x_slider = ViewerSlider("rs ang vel x", 0.0, -1, 1, 0.01,
            cb_hook=lambda o: self.rs_editing.update({"ang_vel_x": o.value}))
        self.rs_ang_vel_y_slider = ViewerSlider("rs ang vel y", 0.0, -1, 1, 0.01,
            cb_hook=lambda o: self.rs_editing.update({"ang_vel_y": o.value}))
        self.rs_ang_vel_z_slider = ViewerSlider("rs ang vel z", 0.0, -1, 1, 0.01,
            cb_hook=lambda o: self.rs_editing.update({"ang_vel_z": o.value}))

    # ── state dict ──

    def load_state_dict(self, d, **kwargs):
        self.step = 30000
        newp = d["gauss_params.means"].shape[0]

        for n, p in self.gauss_params.items():
            self.gauss_params[n] = torch.nn.Parameter(
                torch.zeros(newp, *p.shape[1:], device=self.device))

        if self.config.use_lidar_offset and "gauss_params.lidar_offsets" not in d:
            d["gauss_params.lidar_offsets"] = torch.zeros(newp, 3)
        if self.raw_fog_beta is not None and "raw_fog_beta" not in d:
            d["raw_fog_beta"] = self.raw_fog_beta.data
        if self.fog_atmospheric_light is not None and "fog_atmospheric_light" not in d:
            d["fog_atmospheric_light"] = self.fog_atmospheric_light.data

        super().load_state_dict(d, **kwargs)

    # ── gaussian param construction ──

    def create_gauss_param_dict(self, dyn, st, flip_actors=True):
        pds = []
        self.xys_grad_norm = None
        self.max_2Dsize = None

        for i, sp in enumerate(dyn + st):
            flip = flip_actors and i < len(dyn)
            m = torch.nn.Parameter(sp[:, :3])
            n = m.shape[0]

            if n < 4:
                warnings.warn(f"Actor {i}<4")
                dist = torch.ones(n, 3)
            else:
                dist, _ = self.k_nearest_sklearn(m.data, 3)
                dist = torch.from_numpy(dist)

            sc = torch.nn.Parameter(torch.log(dist.mean(-1, keepdim=True).repeat(1, 3) * self.config.init_scale))
            q = torch.nn.Parameter(random_quat_tensor(n))
            fdc = torch.nn.Parameter(sp[:, 3:] / 255)
            fr = torch.nn.Parameter(torch.randn(n, max(self.config.feature_dim, 0), dtype=sp.dtype, device=sp.device))
            op = torch.nn.Parameter(torch.logit(self.config.init_opacities * torch.ones(n, 1)))
            ids = torch.nn.Parameter(torch.full((n, 1), min(float(i), len(dyn))), requires_grad=False)

            if flip:
                mm = m.clone()
                mm[:, 0] *= -1
                mq = q.clone()
                mq[:, 1] *= -1
                m = torch.nn.Parameter(torch.cat([m, mm], 0))
                sc = torch.nn.Parameter(torch.cat([sc, sc.clone()], 0))
                q = torch.nn.Parameter(torch.cat([q, mq], 0))
                fdc = torch.nn.Parameter(torch.cat([fdc, fdc.clone()], 0))
                fr = torch.nn.Parameter(torch.cat([fr, fr.clone()], 0))
                op = torch.nn.Parameter(torch.cat([op, op.clone()], 0))
                ids = torch.nn.Parameter(torch.cat([ids, ids.clone()], 0))

            pds.append({
                "means": m, "scales": sc, "quats": q,
                "features_dc": fdc, "features_rest": fr,
                "opacities": op, "id": ids,
            })

        return torch.nn.ParameterDict({k: torch.cat([p[k] for p in pds], 0) for k in pds[0]})

    @torch.no_grad()
    def split_seed_points(self, seed_points):
        na = self.dynamic_actors.n_actors
        sp = []
        dp = [[] for _ in range(na)]

        for a in range(na):
            n = self.config.min_points_per_actor
            rp = (torch.rand(n, 3, device=seed_points[0].device) - 0.5) * self.dynamic_actors.actor_sizes[a]
            dp[a].append(torch.cat([rp, torch.rand(n, 3, device=seed_points[0].device) * 255], -1))

        for ct in seed_points[2].unique():
            p = seed_points[0][seed_points[2] == ct]
            c = seed_points[1][seed_points[2] == ct]
            mask = torch.ones(p.shape[0], dtype=torch.bool)
            b2w, e = self.dynamic_actors.get_boxes2world(ct.unsqueeze(-1), flatten=False)
            b2w = b2w.squeeze(0)
            e = e.squeeze(0)

            for a in range(na):
                if e[a]:
                    am = points_in_box(p, b2w[a], self.dynamic_actors.actor_sizes[a] + self.dynamic_actors.actor_padding)
                    if am.any():
                        w2b = pose_inverse(b2w[a]).reshape(-1, 3, 4)
                        pl = transform_points(p[am].reshape(-1, 3), w2b)
                        mi = pl.clone()
                        mi[:, 0] *= -1
                        pl = torch.cat([pl, mi], 0)
                        ac = torch.cat([c[am], c[am]], 0)
                        dp[a].append(torch.cat([pl, ac], -1))
                        mask = mask & ~am
                    else:
                        dp[a].append(torch.empty(0, 6, device=p.device))
            sp.append(torch.cat([p[mask], c[mask]], -1))

        dp = [self.prune_seed_points(torch.cat(x), True) for x in dp]
        return self.prune_seed_points(torch.cat(sp)), dp

    def prune_seed_points(self, pts, is_dynamic=False):
        if pts is None or pts.shape[0] == 0:
            return pts
        if 0 < self.config.max_num_seed_points < pts.shape[0]:
            return pts[torch.randperm(pts.shape[0])[:self.config.max_num_seed_points]]
        return pts

    def k_nearest_sklearn(self, x, k):
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean")
        nn.fit(x.cpu().numpy())
        d, _ = nn.kneighbors(x.cpu().numpy())
        return d[:, 1:].astype(np.float32), None

    def set_background(self, bg):
        self.background_color = bg

    # ── training callbacks / strategy ──

    def step_post_backward(self, step):
        assert step == self.step
        if isinstance(self.strategy, ADDefaultStrategy):
            self.strategy.step_post_backward(
                params=self.gauss_params, optimizers=self.optimizers,
                state=self.strategy_state, step=self.step, info=self.info,
                packed=False, dynamic_actors=self.dynamic_actors)
        elif isinstance(self.strategy, ADMCMCStrategy):
            self.strategy.step_post_backward(
                params=self.gauss_params, optimizers=self.optimizers,
                state=self.strategy_state, step=self.step, info=self.info,
                lr=self.optimizers["means"].param_groups[0]["lr"])

    def get_training_callbacks(self, attrs):
        return [
            TrainingCallback(
                [TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                self.step_cb, args=[attrs.optimizers]),
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.step_post_backward),
        ]

    def step_cb(self, optimizers, step):
        self.step = step
        self.optimizers = optimizers.optimizers

    # ── param groups ──

    def get_gaussian_param_groups(self):
        g = {n: [self.gauss_params[n]] for n in
             ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]}
        if self.config.use_lidar_offset:
            g["lidar_offsets"] = [self.gauss_params["lidar_offsets"]]
        return g

    def get_param_groups(self):
        pg = super().get_param_groups()
        pg.update(self.get_gaussian_param_groups())
        self.camera_optimizer.get_param_groups(pg)
        self.camera_velocity_optimizer.get_param_groups(pg)
        pg["fields"] = (
            list(self.rgb_decoder.parameters())
            + list(self.appearance_embedding.parameters())
            + list(self.lidar_decoder.parameters())
        )
        if self.config.use_dual_stream and self.raw_fog_beta is not None:
            pg["fog_params"] = [self.raw_fog_beta, self.fog_atmospheric_light]
        return pg

    # ── resolution / background helpers ──

    def _get_downscale_factor(self):
        if self.training:
            return 2 ** max((self.config.num_downscales - self.step // self.config.resolution_schedule), 0)
        return 1

    def _downscale_if_required(self, im):
        d = self._get_downscale_factor()
        return resize_image(im, d) if d > 1 else im

    def _get_background_color(self):
        if self.config.background_color == "random":
            return torch.rand(3, device=self.device) if self.training else self.background_color.to(self.device)
        return torch.tensor(get_color(self.config.background_color), device=self.device)

    def _get_actor_adjusted_means(self, means, times, ids, calc_vels=True):
        miw = means.clone()
        b2w, _ = self.dynamic_actors.get_boxes2world(times, flatten=False)
        b2w = b2w.squeeze(0)

        if self.training:
            fm = torch.eye(4, device=b2w.device).unsqueeze(0).repeat(b2w.shape[0], 1, 1)
            fm[:, 0, 0] += (torch.rand(b2w.shape[0], device=b2w.device)
                             < self.config.actor_flip_probability) * -2
            b2w = b2w @ fm

        b2w = b2w[..., :3, :]
        na = self.dynamic_actors.actor_sizes.shape[0]
        actor_mask = (ids < na).squeeze()

        # Default velocity.
        vels = None
        if calc_vels:
            vels = torch.zeros_like(means)

        if actor_mask.any():
            ai = actor_mask.nonzero().squeeze(-1)               # [N_actor]
            ci = ids.index_select(0, ai).squeeze().long()
            bp = b2w.index_select(0, ci)

            if calc_vels and len(b2w) > 0:
                lv, av = self.dynamic_actors.get_velocities(times).split([3, 3], -1)
                lv = lv.squeeze(0).squeeze(0)
                av = av.squeeze(0).squeeze(0)
                vels[ai] += lv.index_select(0, ci) + transform_points_pairwise(
                    torch.cross(av.index_select(0, ci), means.index_select(0, ai)),
                    bp, with_translation=False)

            miw[ai] = transform_points_pairwise(miw.index_select(0, ai), bp)

        return miw, vels

    # ─────────────────── Camera rendering (per-Gaussian ASM, FINAL) ────────

    def get_camera_outputs(self, camera: Cameras) -> Dict:
        if not isinstance(camera, Cameras):
            return {}

        if self.training or self.config.use_camopt_in_eval:
            assert camera.shape[0] == 1
            oc2w = self.camera_optimizer.apply_to_camera(camera)
        else:
            oc2w = camera.camera_to_worlds

        BLK = 16
        csf = self._get_downscale_factor()
        if csf != 1:
            camera.rescale_output_resolution(1 / csf)

        K = camera.get_intrinsics_matrices()
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)
        rd = get_ray_dirs_pinhole(camera, W, H, oc2w)

        if csf != 1:
            camera.rescale_output_resolution(csf)

        render_mode = "RGB+ED"
        clv = cav = None
        rst = None
        ctimes = camera.times

        if camera.metadata is not None and self.config.compensate_rs_camera:
            rst = camera.metadata.get("rolling_shutter_time",
                                       torch.zeros((1, 1), device=self.device))[0]
            tcp = camera.metadata.get("time_to_center_pixel",
                                      torch.zeros((1, 1), device=self.device))
            vels = self.camera_velocity_optimizer.apply_to_camera_velocity(
                camera, return_init_only=(not self.training) and (not self.config.use_camopt_in_eval))
            clv, cav = torch.split(vels, 3, -1)
            tcp = tcp + self.camera_velocity_optimizer.get_time_to_center_pixel_adjustment(camera)
            oc2w = torch.cat([
                oc2w[:, :3, :3],
                oc2w[:, :3, 3:4] + (torch.matmul(clv, oc2w[0, :3, :3].transpose(0, 1)) * tcp)[..., None],
            ], -1)
            ctimes = camera.times + tcp
            ft = torch.ones(3, device=clv.device, dtype=clv.dtype)
            ft[1:] = -1
            clv = clv * ft
            cav = cav * ft
        else:
            clv = torch.tensor([[
                self.rs_editing["lin_vel_x"],
                self.rs_editing["lin_vel_y"],
                self.rs_editing["lin_vel_z"],
            ]], device=self.device)
            cav = torch.tensor([[
                self.rs_editing["ang_vel_x"],
                self.rs_editing["ang_vel_y"],
                self.rs_editing["ang_vel_z"],
            ]], device=self.device)
            rst = torch.tensor([self.rs_editing["rs_time"]], device=self.device)

        vm = get_viewmat(oc2w)
        means_w, vels = self._get_actor_adjusted_means(self.means, ctimes, self.id)

        surf_op = torch.sigmoid(self.opacities).squeeze(-1)
        surf_colors = torch.cat((self.features_dc, self.features_rest), -1)

        # ── Surface rasterization -> features + depth ──
        render_s, alpha, self.info = rasterization(
            means=means_w,
            quats=self.quats,
            scales=torch.exp(self.scales),
            opacities=surf_op,
            colors=surf_colors,
            velocities=vels,
            viewmats=vm,
            Ks=K,
            width=W,
            height=H,
            linear_velocity=clv,
            angular_velocity=cav,
            rolling_shutter_time=rst,
            tile_size=BLK,
            packed=False,
            near_plane=0.5,
            far_plane=1e10,
            radius_clip=self.config.radius_clip_pix,
            render_mode=render_mode,
            sh_degree=None,
            sparse_grad=False,
            absgrad=self.config.use_absgrad,
            rasterize_mode=self.config.rasterize_mode,
            channel_chunk=128,
            eps2d=0.3,
        )

        if self.training:
            self.strategy.step_pre_backward(
                self.gauss_params, self.optimizers, self.strategy_state, self.step, self.info)

        bg = self._get_background_color()
        r_feat = render_s[..., :-1]                       # [1, H, W, 3]
        raw_d = render_s[..., -1:]                        # [1, H, W, 1]

        # ── CNN decoder -> clean RGB ──
        af = self._get_appearance_embedding(camera, r_feat)
        rgb_clean = self.rgb_decoder(torch.cat((r_feat, af), -1), rd.unsqueeze(0))
        rgb_clean = rgb_clean + (1 - alpha) * bg
        rgb_clean = torch.clamp(rgb_clean, 0.0, 1.0)

        # ── Environment stream: per-Gaussian t -> t_map -> ASM composition ──
        should_fog = (
            self.config.use_dual_stream
            and self.raw_fog_beta is not None
            and (self.training or (self.config.render_weather and self.render_weather_slider.value > 0.5))
        )
        t_map = None

        if should_fog:
            beta = F.softplus(self.raw_fog_beta) + self.config.fog_beta_min

            # Per-Gaussian camera-space depth — detach protects geometry.
            means_h = torch.cat([means_w, torch.ones_like(means_w[..., :1])], -1)
            means_cam = (vm[0] @ means_h.T).T
            d_gauss = means_cam[..., 2].detach().clamp(min=0.5)
            t_per_gauss = torch.clamp(torch.exp(-beta * d_gauss), min=self.config.fog_t_min)

            # ── Second raster: t_map ──
            t_render, _, _ = rasterization(
                means=means_w,
                quats=self.quats,
                scales=torch.exp(self.scales),
                opacities=surf_op,
                colors=t_per_gauss.unsqueeze(-1),
                velocities=vels,
                viewmats=vm,
                Ks=K,
                width=W,
                height=H,
                linear_velocity=clv,
                angular_velocity=cav,
                rolling_shutter_time=rst,
                tile_size=BLK,
                packed=False,
                near_plane=0.5,
                far_plane=1e10,
                radius_clip=self.config.radius_clip_pix,
                render_mode="RGB",
                sh_degree=None,
                sparse_grad=False,
                absgrad=False,
                rasterize_mode=self.config.rasterize_mode,
                channel_chunk=128,
                eps2d=0.3,
            )
            t_map = t_render.squeeze(-1).unsqueeze(-1)         # [1, H, W, 1]

            fog_A = torch.sigmoid(self.fog_atmospheric_light)   # [3]
            rgb = rgb_clean * t_map + fog_A.view(1, 1, 1, 3) * (1.0 - t_map)
            rgb = torch.clamp(rgb, 0.0, 1.0)
        else:
            rgb = rgb_clean

        # ── Depth output ──
        if self.config.output_depth_during_training or not self.training:
            depth_im = torch.where(alpha > 0, raw_d, raw_d.detach().max()).squeeze(0)
        else:
            depth_im = None

        if bg.shape[0] == 3 and not self.training:
            bg = bg.expand(H, W, 3)

        out = {
            "rgb": rgb.squeeze(0),
            "depth": depth_im,
            "accumulation": alpha.squeeze(0),
            "background": bg,
        }
        if t_map is not None:
            out["t_map"] = t_map
        return out

    # ─────────────────── LiDAR rendering ──────────────────────────

    def get_lidar_outputs(self, lidar: Lidars) -> Dict:
        if not isinstance(lidar, Lidars):
            return {}
        assert (lidar.azimuths is not None and lidar.elevations is not None) \
            or (lidar.metadata and "raster_pts" in lidar.metadata)

        if self.training or self.config.use_camopt_in_eval:
            assert lidar.shape[0] == 1
            ol2w = self.camera_optimizer.apply_to_camera(lidar)
        else:
            ol2w = lidar.lidar_to_worlds

        if lidar.metadata and "raster_pts" in lidar.metadata:
            raster_pts = lidar.metadata["raster_pts"][..., :-1]
            tb = lidar.metadata["elevation_boundaries"]
            ma = -180
            Ma = 180
            me = tb.min()
            Me = tb.max()
            ar = lidar.metadata["azimuth_resolution"]
        else:
            ev, az = torch.meshgrid(
                torch.rad2deg(lidar.elevations.flatten()),
                torch.rad2deg(lidar.azimuths.flatten()),
                indexing="ij",
            )
            raster_pts = torch.stack([az, ev, torch.ones_like(az), torch.zeros_like(az)], -1).to(self.device)[None]
            tb = torch.rad2deg(lidar.elevations[0, ::ELEV_CHANNELS_PER_TILE]).to(self.device).flatten()
            tb = torch.cat([tb, torch.tensor([tb[-1].item() + 1], device=self.device)])
            ar = float(torch.rad2deg(lidar.azimuths[0, 1] - lidar.azimuths[0, 0]))
            ma = -180
            Ma = 180
            me = tb.min().item()
            Me = tb.max().item() + 1e-6

        llv = torch.zeros(1, 3, device=self.device)
        lav = torch.zeros(1, 3, device=self.device)
        rst_l = torch.zeros(1, device=self.device)
        lt = lidar.times

        if lidar.metadata is not None and self.config.compensate_rs_lidar:
            mxo, mno = raster_pts[..., 3].max(), raster_pts[..., 3].min()
            rst_l = (mxo - mno).unsqueeze(0)
            vels = self.camera_velocity_optimizer.apply_to_camera_velocity(
                lidar, return_init_only=(not self.training) and (not self.config.use_camopt_in_eval))
            llv, lav = torch.split(vels, 3, -1)
            tca = (mxo + mno) / 2
            ol2w = torch.cat([
                ol2w[:, :3, :3],
                ol2w[:, :3, 3:4] + (torch.einsum("bij,bj->bi", ol2w[..., :3, :3], llv) * tca)[..., None],
            ], -1)
            lt = lidar.times + tca
            raster_pts[..., 3] = raster_pts[..., 3] - tca

        lf = self.features_rest.unsqueeze(0)
        bs = raster_pts.shape[0]
        vm_l = to4x4(pose_inverse(ol2w))

        if bs > 1:
            vm_l = vm_l.repeat(bs, 1, 1)
            lf = lf.repeat(bs, 1, 1)
            llv = llv.repeat(bs, 1)
            lav = lav.repeat(bs, 1)
            rst_l = rst_l.repeat(bs)

        lidar_means = self.means + (self.gauss_params["lidar_offsets"] if self.config.use_lidar_offset else 0)
        means, vels = self._get_actor_adjusted_means(lidar_means, lt, self.id)

        render, alpha, asup, self.info = lidar_rasterization(
            means=means,
            quats=self.quats,
            scales=torch.exp(self.scales),
            opacities=torch.sigmoid(self.opacities).squeeze(-1),
            lidar_features=lf,
            velocities=vels,
            viewmats=vm_l,
            min_azimuth=ma,
            max_azimuth=Ma,
            min_elevation=me,
            max_elevation=Me,
            n_elevation_channels=raster_pts.shape[1],
            azimuth_resolution=ar,
            raster_pts=raster_pts,
            tile_width=AZIM_CHANNELS_PER_TILE,
            tile_height=ELEV_CHANNELS_PER_TILE,
            tile_elevation_boundaries=tb,
            linear_velocity=llv,
            angular_velocity=lav,
            rolling_shutter_time=rst_l,
            near_plane=0.2,
            far_plane=300,
            radius_clip=self.config.radius_clip_lidar,
            compute_alpha_sum_until_points=(self.config.line_of_sight_lambda > 0) and self.training,
            compute_alpha_sum_until_points_threshold=self.config.line_of_sight_dist,
            sparse_grad=False,
            absgrad=self.config.use_absgrad,
            rasterize_mode=self.config.rasterize_mode,
            channel_chunk=128,
            eps2d=0.01718873385,
        )

        self.info["width"] = -1
        self.info["height"] = -1
        self.last_size = (self.last_size[0], self.last_size[1], -1)

        if self.training:
            self.strategy.step_pre_backward(
                self.gauss_params, self.optimizers, self.strategy_state, self.step, self.info)

        di = render[:,...,-1:]
        rf = render[..., :-1]
        af_l = self._get_appearance_embedding(lidar, rf)
        rf = torch.cat((rf, af_l), -1)

        rd_l = torch.deg2rad(raster_pts[..., :2])
        lrd = torch.cat([
            torch.cos(rd_l[..., 0:1]) * torch.cos(rd_l[..., 1:2]),
            torch.sin(rd_l[..., 0:1]) * torch.cos(rd_l[..., 1:2]),
            torch.sin(rd_l[..., 1:2]),
        ], -1)
        lrdw = (ol2w[:, :3, :3].reshape(1, 1, 1, 3, 3) @ lrd.unsqueeze(-1)).squeeze(-1)

        intensity, rdl = (
            self.lidar_decoder(torch.cat([
                rf.reshape(-1, rf.shape[-1]),
                lrdw.reshape(-1, lrdw.shape[-1]),
            ], -1))
            .reshape((*lrdw.shape[:-1], self.lidar_decoder.out_dim))
            .split([1, 1], -1)
        )

        out = {
            "depth": di,
            "accumulation": alpha,
            "median_depth": self.info["median_depths"] + (alpha <= 0.5) * (di / alpha.clamp_min(1e-10)),
        }
        if intensity is not None:
            out["intensity"] = intensity.sigmoid().float()
        if rdl is not None:
            out["ray_drop_logits"] = rdl.float()
            out["ray_drop_prob"] = rdl.sigmoid().float()
        if asup is not None:
            out["alpha_sum_until_points"] = asup
        return out

    def get_outputs(self, sensor):
        return self.get_camera_outputs(sensor) if isinstance(sensor, Cameras) else self.get_lidar_outputs(sensor)

    # ─────────────────── Metrics / Loss ───────────────────────────

    def get_gt_img(self, image):
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        return self._downscale_if_required(image).to(self.device)

    def composite_with_background(self, image, bg):
        if image.shape[2] == 4:
            a = image[..., -1].unsqueeze(-1).repeat((1, 1, 3))
            return a * image[..., :3] + (1 - a) * bg
        return image

    def filter_lidar_pred_and_gt(self, outputs, batch, output_point_cloud=False):
        gl = batch["raster_pts"]
        v = batch["raster_pts_valid_depth_and_did_return"]
        dr = batch["raster_pts_did_return"].flatten()
        vn = batch["raster_pts_valid_depth_and_did_not_return"]

        gt = {
            "depth": gl[..., 2].flatten()[v],
            "intensity": gl[..., 4].flatten()[v],
            "ray_drop": ~dr,
            "valid": gl[..., 2].flatten() > 0,
        }
        pred = {
            "depth": outputs["depth"].flatten()[v],
            "depth_dropped": outputs["depth"].flatten()[vn],
            "intensity": outputs["intensity"].flatten()[v],
            "intensity_dropped": outputs["intensity"].flatten()[vn],
            "ray_drop": outputs["ray_drop_logits"].flatten() * gt["valid"] - (~gt["valid"]) * 10000,
            "accumulation": outputs["accumulation"].flatten()[v],
            "accumulation_dropped": outputs["accumulation"].flatten()[vn],
            "median_depth": outputs["median_depth"].flatten()[v],
        }
        if "alpha_sum_until_points" in outputs:
            pred["alpha_sum_until_points"] = outputs["alpha_sum_until_points"].flatten()[v]
            pred["alpha_sum_until_points_dropped"] = outputs["alpha_sum_until_points"].flatten()[vn]

        if output_point_cloud:
            ad = torch.deg2rad(gl[..., 0].flatten())
            ed = torch.deg2rad(gl[..., 1].flatten())
            dirs = torch.stack([
                torch.cos(ed) * torch.cos(ad),
                torch.cos(ed) * torch.sin(ad),
                torch.sin(ed),
            ], -1)
            gt["point_cloud"] = batch["lidar"][batch["lidar_pts_did_return"].squeeze(), :3]
            pred["point_cloud"] = (
                outputs["depth"].view(-1, 1) * dirs
                + batch["linear_velocities_local"] * gl[..., 3].view(-1, 1)
            )[(pred["ray_drop"].sigmoid() <= 0.5) * gt["valid"]]
            pred["median_point_cloud"] = (
                outputs["median_depth"].view(-1, 1) * dirs
                + batch["linear_velocities_local"] * gl[..., 3].view(-1, 1)
            )[(pred["ray_drop"].sigmoid() <= 0.5) * gt["valid"]]

        return pred, gt

    def get_metrics_dict(self, outputs, batch):
        md = {}

        if "image" in batch:
            gt = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
            pr = outputs["rgb"]
            if gt.shape[:2] != pr.shape[:2]:
                gt = gt[:pr.shape[0], :pr.shape[1]]
            md["psnr"] = self.psnr(pr, gt)
            md["gaussian_count"] = self.num_points
            if self.raw_fog_beta is not None:
                md["fog_beta"] = float(F.softplus(self.raw_fog_beta) + self.config.fog_beta_min)

        if "raster_pts" in batch:
            p, g = self.filter_lidar_pred_and_gt(outputs, batch)
            vn = g["valid"].sum()
            ra = float((((p["ray_drop"].sigmoid() > 0.5) == g["ray_drop"]) * g["valid"]).sum() / vn) if vn > 0 else 0.0
            md.update(
                depth_median_l2=float(self.median_l2(p["depth"], g["depth"])),
                depth_mean_rel_l2=float(self.mean_rel_l2(p["depth"], g["depth"])),
                median_depth_median_l2=float(self.median_l2(p["median_depth"], g["depth"])),
                median_depth_mean_rel_l2=float(self.mean_rel_l2(p["median_depth"], g["depth"])),
                intensity_rmse=float(self.rmse(p["intensity"], g["intensity"])),
                ray_drop_accuracy=ra,
            )

        self.camera_optimizer.get_metrics_dict(md)
        self.camera_velocity_optimizer.get_metrics_dict(md)
        return md

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        ld = {}

        if "image" in batch:
            gt_img = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
            pred_img = outputs["rgb"]
            if gt_img.shape[:2] != pred_img.shape[:2]:
                gt_img = gt_img[:pred_img.shape[0], :pred_img.shape[1]]

            if "mask" in batch:
                mk = self._downscale_if_required(batch["mask"]).to(self.device)
                gt_img = gt_img * mk
                pred_img = pred_img * mk

            Ll1 = torch.abs(gt_img - pred_img).mean()
            simloss = (
                (1 - self.ssim(gt_img.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...]))
                if self.config.ssim_lambda > 0 else 0
            )
            ld["main_loss"] = (1 - self.config.ssim_lambda) * Ll1 + self.config.ssim_lambda * simloss

            # ── sqrt depth-weighted reconstruction loss (balanced near/far) ──
            if (self.training and self.config.depth_weighted_loss_lambda > 0
                    and "depth" in outputs and outputs["depth"] is not None):
                rd_dw = outputs["depth"]
                if rd_dw.dim() == 3:
                    rd_dw = rd_dw.squeeze(-1)
                if rd_dw.shape[:2] != gt_img.shape[:2]:
                    rd_dw = rd_dw[:gt_img.shape[0], :gt_img.shape[1]]
                with torch.no_grad():
                    dw = rd_dw.clamp(min=0.5).sqrt()
                    dw = dw / (dw.mean() + 1e-8)
                    dw = dw.unsqueeze(-1)
                ld["depth_weighted_loss"] = self.config.depth_weighted_loss_lambda * (
                    torch.abs(gt_img - pred_img) * dw).mean()

        ld["mcmc_scale_reg"] = (
            torch.abs(torch.exp(self.scales).mean()) * self.config.mcmc_scale_reg_lambda
            if self.config.mcmc_scale_reg_lambda and isinstance(self.strategy, ADMCMCStrategy)
            else torch.zeros(1, device=self.device)
        )
        ld["mcmc_opacity_reg"] = (
            torch.abs(torch.sigmoid(self.opacities).mean()) * self.config.mcmc_opacity_reg_lambda
            if self.config.mcmc_opacity_reg_lambda and isinstance(self.strategy, ADMCMCStrategy)
            else torch.zeros(1, device=self.device)
        )

        if self.training:
            self.camera_optimizer.get_loss_dict(ld)
            self.camera_velocity_optimizer.get_loss_dict(ld)

        if "raster_pts" in batch:
            p, g = self.filter_lidar_pred_and_gt(outputs, batch)
            ul = self.depth_loss(p["depth"], g["depth"])
            q = torch.quantile(ul, self.config.depth_loss_quantile_threshold)
            qm = ul < q
            ld["depth_loss"] = self.config.depth_lambda * (ul * qm).mean()
            ld["intensity_loss"] = self.config.intensity_lambda * self.intensity_loss(p["intensity"] * qm, g["intensity"] * qm)
            ld["ray_drop_loss"] = self.config.ray_drop_lambda * self.ray_drop_loss(
                p["ray_drop"], g["ray_drop"].to(p["ray_drop"]))
            if "alpha_sum_until_points" in p and self.config.line_of_sight_lambda > 0:
                ld["alpha_sum_until_points_loss"] = self.config.line_of_sight_lambda * (p["alpha_sum_until_points"] * qm).mean()

        # ── Innovation 1: fog regularisation ──
        if self.config.use_dual_stream and self.raw_fog_beta is not None:
            eb = F.softplus(self.raw_fog_beta) + self.config.fog_beta_min
            ld["fog_beta_reg"] = (eb ** 2) * self.config.fog_beta_reg

            fog_A = torch.sigmoid(self.fog_atmospheric_light)
            target = torch.tensor(self.config.fog_atmospheric_light_target, device=self.device)
            ld["fog_A_reg"] = ((fog_A - target) ** 2).mean() * self.config.fog_atmospheric_light_reg

            # ── Weak t_map supervision (solid surfaces only, alpha>0.7) ──
            if (self.training and self.config.t_map_reg_lambda > 0
                    and "t_map" in outputs and "depth" in outputs and outputs["depth"] is not None):
                t_map = outputs["t_map"]                           # [1,H,W,1] or [H,W,1]
                rd_ts = outputs["depth"].detach()
                if rd_ts.dim() == 3:
                    rd_ts = rd_ts.squeeze(-1)                      # [H,W]
                accum = outputs.get("accumulation")                # [H,W]
                if accum is not None and accum.dim() == 2:
                    mask = (accum > 0.7).float().unsqueeze(-1)     # [H,W,1]
                    # Crop to common size if needed.
                    if t_map.shape[:2] != mask.shape[:2]:
                        t_map = t_map[:mask.shape[0], :mask.shape[1]]
                        rd_ts = rd_ts[:mask.shape[0], :mask.shape[1]]
                    if t_map.dim() == 3 and t_map.shape[-1] == 1:
                        t_map = t_map.squeeze(-1)                  # [H,W]
                    t_depth = torch.exp(-eb.detach() * rd_ts.clamp(min=0.5))
                    ld["t_map_reg"] = self.config.t_map_reg_lambda * (
                        (torch.abs(t_map - t_depth) * mask.squeeze(-1)).sum()
                        / mask.sum().clamp(min=1.0))

        # ── Innovation 2: LiDAR offset regularisation ──
        if self.config.use_lidar_offset:
            ld["lidar_offset_reg"] = (self.gauss_params["lidar_offsets"] ** 2).mean() * self.config.lidar_offset_reg

        return ld

    @torch.no_grad()
    def get_outputs_for_camera(self, camera, obb_box=None):
        return self.get_outputs(camera.to(self.device))

    def get_image_metrics_and_images(self, outputs, batch):
        imd, med = {}, {}

        if "image" in batch:
            gt = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
            pr = outputs["rgb"]
            if gt.shape[:2] != pr.shape[:2]:
                gt = gt[:pr.shape[0], :pr.shape[1]]
            imd["img"] = torch.cat([gt, pr], 1)
            gt4, pr4 = torch.moveaxis(gt, -1, 0)[None, ...], torch.moveaxis(pr, -1, 0)[None, ...]
            med.update(
                psnr=float(self.psnr(gt4, pr4)),
                ssim=float(self.ssim(gt4, pr4)),
                lpips=float(self.lpips(gt4, pr4)),
            )

        if "raster_pts" in batch:
            p, g = self.filter_lidar_pred_and_gt(outputs, batch, output_point_cloud=True)
            vn = g["valid"].sum()
            ra = float((((p["ray_drop"].sigmoid() > 0.5) == g["ray_drop"]) * g["valid"]).sum() / vn) if vn > 0 else 0.0
            med.update(
                depth_median_l2=float(self.median_l2(p["depth"], g["depth"])),
                depth_mean_rel_l2=float(self.mean_rel_l2(p["depth"], g["depth"])),
                median_depth_median_l2=float(self.median_l2(p["median_depth"], g["depth"])),
                median_depth_mean_rel_l2=float(self.mean_rel_l2(p["median_depth"], g["depth"])),
                intensity_rmse=float(self.rmse(p["intensity"], g["intensity"])),
                ray_drop_accuracy=ra,
            )

        return med, imd

    def _get_appearance_embedding(self, sensor, features):
        md = sensor.metadata if sensor.metadata is not None else {}
        si = md.get("sensor_idxs", None)
        if si is None:
            assert not self.training
            si = torch.full((1,), self.fallback_sensor_idx.value, device=features.device, dtype=torch.long)
        return self.appearance_embedding(si).expand(*features.shape[:-1], -1)

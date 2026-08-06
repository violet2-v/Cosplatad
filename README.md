<p align="center">
    <!-- project badges -->
    <a href="https://research.zenseact.com/publications/neurad/"><img src="https://img.shields.io/badge/NeuRAD-Project-ffa"/></a>
    <a href="https://research.zenseact.com/publications/splatad/"><img src="https://img.shields.io/badge/SplatAD-Project-ffa"/></a>
    <!-- paper badges -->
    <a href="https://arxiv.org/abs/2311.15260">
        <img src='https://img.shields.io/badge/NeuRAD-Arxiv-aff'>
    </a>
    <a href="https://arxiv.org/abs/2411.16816">
        <img src='https://img.shields.io/badge/SplatAD-Arxiv-aff'>
    </a>
</p>

<div align="center">
<picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/_static/imgs/neurad_logo_with_text_dark.png" />
    <img alt="tyro logo" src="docs/_static/imgs/neurad_logo_with_text.png" width="80%"/>
</picture>
</div>

<div align="center">
<h3 style="font-size:2.0em;">Neural Rendering for Autonomous Driving</h3>
<h4>CVPR 2024 highlight + CVPR 2025</h4>
</div>

<div align="center">

[Quickstart](#quickstart) ·
[Learn more](#learn-more) ·
[Planned Features](#planned-featurestodos) ·
[Project page](https://research.zenseact.com/publications/neurad/)

</div>

# About

This repository is built on top of [neurad-studio](https://github.com/georghess/neurad-studio), the official code release of the following papers:

- CVPR 2024 [paper](https://arxiv.org/abs/2311.15260) _NeuRAD: Neural Rendering for Autonomous Driving_
- CVPR 2025 [paper](https://arxiv.org/abs/2411.16816) _SplatAD: Real-Time Lidar and Camera Rendering with 3D Gaussian Splatting for Autonomous Driving_

**CoSplat** — our extension in this fork — adds zero‑shot dehazing to SplatAD by introducing a surface‑environment dual‑stream architecture and hierarchical LiDAR residual offsets. The method is based on a per‑Gaussian atmospheric scattering model (ASM) that decomposes foggy scenes into a clean surface stream and a volumetric environment stream, enabling dehazed rendering at inference time without any clean‑image supervision.

Besides releasing the code for NeuRAD and SplatAD, we hope that this can lay the ground‑work for research on applying neural rendering methods in autonomous driving. In line with Nerfstudio's mission, this is a contributor‑friendly repo with the goal of building a community where users can more easily build upon each other's contributions.

Do you have feature requests or want to add **your** new AD‑NeRF model? Or maybe provide structures for a new dataset? **We welcome contributions!**

<div align="center">
<a href="https://zenseact.com/">
<picture style="padding-left: 10px; padding-right: 10px;">
    <source media="(prefers-color-scheme: dark)" srcset="docs/_static/imgs/ZEN_Vertical_logo_white.svg" />
    <img alt="zenseact logo" src="docs/_static/imgs/ZEN_Vertical_logo_black.svg" height="100px" />
</picture>
</a>
<a href="https://www.chalmers.se/en/">
<picture style="padding-left: 10px; padding-right: 10px; padding-bottom: 10px;">
    <source media="(prefers-color-scheme: dark)" srcset="docs/_static/imgs/EN_Avancez_CH_white.png" />
    <img alt="chalmers logo" src="docs/_static/imgs/EN_Avancez_CH_black.png" height="90px" />
</picture>
</a>
<a href="https://www.lunduniversity.lu.se/">
<picture style="padding-left: 10px; padding-right: 10px;">
    <source media="(prefers-color-scheme: dark)" srcset="docs/_static/imgs/LundUniversity_C2line_NEG.png" />
    <img alt="lund logo" src="docs/_static/imgs/LundUniversity_C2line_BLACK.png" height="100px" />
</picture>
</a>
<a href="https://liu.se/en">
<picture style="padding-left: 10px; padding-right: 10px;">
    <source media="(prefers-color-scheme: dark)" srcset="docs/_static/imgs/LiU_secondary_1_white-PNG.png" />
    <img alt="liu logo" src="docs/_static/imgs/LiU_secondary_1_black-PNG.png" height="100px" />
</picture>
</a>
<a href="https://wasp-sweden.org/">
<picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/_static/imgs/WASP-logotype-white.png" />
    <img alt="wasp logo" src="docs/_static/imgs/WASP_logotyp_grey_180116.png" height="80px" />
</picture>
</a>
</div>

# News

[June 2025] Code for [SplatAD](https://research.zenseact.com/publications/splatad/) is now released in neurad‑studio. The code uses our custom fork of gsplat, found [here](https://github.com/carlinds/splatad), for handling rolling shutter and lidar rendering. For Apptainer users we provide [a recipe](apptainer_recipe) that lets you build an image with all the needed dependencies.

[2025] **CoSplat** is added — zero‑shot fog removal for autonomous driving using a bias‑driven per‑Gaussian atmospheric scattering model with LiDAR residual offsets.

# Quickstart

The quickstart will help you get started with the NeuRAD model on a PandaSet sequence. For more complex changes (e.g., running with your own data/setting up a new NeRF graph), please refer to our [references](#learn-more).

## 1. Installation: Setup the environment

### Prerequisites

Our installation steps largely follow Nerfstudio, with some added dataset-specific dependencies. You must have an NVIDIA video card with CUDA installed on the system. This library has been tested with version 11.8 of CUDA. You can find more information about installing CUDA [here](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html).

### Create environment

The models require `python >= 3.10`. We recommend using conda to manage dependencies. Make sure to install [Conda](https://docs.conda.io/miniconda.html) before proceeding.

```bash
conda create --name neurad -y python=3.10
conda activate neurad
pip install --upgrade pip
```

### Dependencies

Install PyTorch with CUDA (this repo has been tested with CUDA 11.7 and CUDA 11.8) and tiny-cuda-nn. `cuda-toolkit` is required for building `tiny-cuda-nn`.

For CUDA 11.8:

```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit

# Some need to upgrade dill prior to tiny-cuda-nn install
pip install dill --upgrade
pip install --upgrade pip "setuptools<70.0"

pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

For support of Waymo-Open-Dataset v2 (requires python3.10, also dependencies from this package are very strict so cannot add it to pyproject.toml and need install first):

```bash
pip install waymo-open-dataset-tf-2-11-0==1.6.1
```

We refer to Nerfstudio for more installation support.

### Installing neurad-studio

```bash
git clone https://github.com/georghess/neurad-studio.git
cd neurad-studio
pip install -e .
```

### Installing kernels for SplatAD / CoSplat

If you want to use SplatAD, our 3DGS-based method for camera and lidar rendering, or CoSplat which builds upon SplatAD for zero‑shot dehazing, you will need to also install our custom fork of gsplat.

**OR** if you want to skip all installation steps and directly start using NeuRAD, use the provided docker image or apptainer recipe:

[Dockerfile](Dockerfile) or [Apptainer recipe](apptainer_recipe).

## 2. Training your first model!

The following will train a NeuRAD model. However, training SplatAD or CoSplat instead is as easy as calling a different method name.

### Data preparation

Begin by downloading PandaSet and unzip it under `data/pandaset`. The dataset is no longer hosted by Scale but can be downloaded from the provided huggingface link.

For foggy scene training with CoSplat, add synthetic fog to the camera images using the Koschmieder atmospheric scattering model. We provide fog generation scripts in the repository (see `scripts/add_fog_pandaset.py` for PandaSet and `scripts/add_fog_nuscenes.py` for nuScenes).

Example fog generation command:

```bash
python scripts/add_fog_pandaset.py \
    --data data/pandaset \
    --seq 001 \
    --beta 0.01 \
    --output data/pandaset_fog
```

### Training

Training models is done the same way as in nerfstudio, i.e.,

```bash
# Train NeuRAD
python nerfstudio/scripts/train.py neurad pandaset-data

# Train SplatAD
python nerfstudio/scripts/train.py splatad pandaset-data

# Train CoSplat (zero‑shot dehazing, β=0.01 light fog)
python nerfstudio/scripts/train.py cosplat pandaset-data \
    --data data/pandaset_fog \
    --sequence 001 \
    --pipeline.model.fog-beta-init 0.04 \
    --pipeline.model.fog-beta-min 0.03

# Train CoSplat for moderate fog (β=0.02)
python nerfstudio/scripts/train.py cosplat pandaset-data \
    --data data/pandaset_fog \
    --sequence 001 \
    --pipeline.model.fog-beta-init 0.05 \
    --pipeline.model.fog-beta-min 0.04
```

If everything works, you should see training progress like the following:

Navigating to the link at the end of the terminal will load the webviewer. If you are running on a remote machine, you will need to port forward the websocket port (defaults to 7007).

### Troubleshooting

If you run into issues, it could be due to the training taking up too much memory. You can try to adjust the model parameters according to the neurad-tiny vscode launch config.

### Resume from checkpoint / visualize existing run

It is possible to load a pretrained model by running

```bash
python nerfstudio/scripts/train.py neurad pandaset-data --load-dir {outputs/.../nerfstudio_models}
```

### Visualize existing run

Given a pretrained model checkpoint, you can start the viewer by running

```bash
python nerfstudio/scripts/viewer/run_viewer.py --load-config {outputs/.../config.yml}
```

## 3. Exporting Results

Once you have a model you can render its output. There are multiple different renders, more info available using

```bash
python nerfstudio/scripts/render.py --help
```

For CoSplat, you can render dehazed images (zero‑shot) by disabling the environment stream:

```bash
python nerfstudio/scripts/render.py cosplat \
    --load-config outputs/cosplat/{experiment}/config.yml \
    --output-path ./dehazed_output/ \
    --pipeline.model.render-weather False
```

## 4. Advanced Options

### Available models

We currently provide implementations for the following models:

- `splatad`: 3DGS-based. Official implementation for SplatAD. This is currently our fastest and best performing model on AD scenes.
- `cosplat`: Zero‑shot dehazing variant of SplatAD. Adds:
  - Per‑Gaussian atmospheric scattering model (ASM) to separate fog from clear appearance. Each Gaussian primitive is assigned a transmission $t_i = \exp(-\beta \cdot d_i)$, and a second lightweight rasterization pass renders the transmission map $t_{\text{map}}$.
  - Bias‑driven $\beta_{\text{min}}$ training strategy to prevent optimization collapse. The scattering coefficient lower bound is set above the true fog density ($\beta_{\text{min}} \approx 2\text{--}3\times \beta_{\text{true}}$), forcing the CNN decoder to output cleaner colors.
  - Hierarchical LiDAR residual offsets for multi‑modal geometry consistency. LiDAR renders use $\mu + \Delta\mu_{\text{offset}}$ while camera shares the backbone $\mu$, absorbing calibration errors and penetration noise.

  Train as usual, then set `render_weather: false` at inference to obtain dehazed outputs.

- `neurad`: NeRF‑based. Official implementation for NeuRAD.
- `unisim`: NeRF‑based. This is an unofficial implementation of UniSim. Available as plugin, see https://github.com/carlinds/unisim for more info.

Any of these models can be trained as described above

```bash
python nerfstudio/scripts/train.py <model name> pandaset-data
```

Further, as we build on top of nerfstudio, models such as `nerfacto` or `splatfacto` are available as well, see nerfstudio for details. However, note that these are made for static scenes.

For a full list of included models run `python nerfstudio/scripts/train.py --help`.

### Modify Configuration

Each model contains many parameters that can be changed, too many to list here. Use the `--help` command to see the full list of configuration options.

```bash
python nerfstudio/scripts/train.py cosplat --help
```

### CoSplat Fog Parameters

The following table lists the key fog‑related configuration parameters for CoSplat, defined in `SplatADModelConfig`:

| Parameter | Default (β=0.01) | Description |
| --- | --- | --- |
| `fog_beta_init` | 0.04 | Initial scattering coefficient β (before softplus + β_min) |
| `fog_beta_min` | 0.03 | Hard lower bound on β; set to ~2‑3× true β to prevent collapse |
| `fog_beta_reg` | 0.005 | L2 regularization strength on β; kept minimal to let reconstruction loss drive β freely |
| `fog_t_min` | 0.20 | Minimum per‑Gaussian transmission value |
| `fog_atmospheric_light_reg` | 0.05 | L2 regularization pulling atmospheric light A toward target (0.95, 0.95, 0.95) |
| `render_weather` | True | If True, apply ASM fog synthesis at eval; set False for zero‑shot dehazing |
| `t_map_reg_lambda` | 0.05 | Weak L1 supervision strength for the transmission map $t_{\text{map}}$ |
| `depth_weighted_loss_lambda` | 0.30 | Weight for sqrt‑depth‑weighted reconstruction loss |
| `ssim_lambda` | 0.30 | SSIM loss weight in the reconstruction objective |
| `use_lidar_offset` | True | Enable per‑Gaussian LiDAR residual offsets (Innovation 2) |
| `lidar_offset_reg` | 0.01 | L2 regularization on LiDAR offsets, keeping them numerically sparse |

For moderate fog (β=0.02), we recommend:

```bash
--pipeline.model.fog-beta-init 0.05 --pipeline.model.fog-beta-min 0.04
```

All other parameters remain identical across fog densities, demonstrating the robustness of the bias‑driven strategy.

### CoSplat Ablation Studies

CoSplat supports two independent boolean toggles for systematic ablation experiments, allowing you to quantify the individual contributions of each innovation:

```bash
# Ablation 1: Baseline (neither innovation — equivalent to vanilla SplatAD)
ns-train cosplat pandaset-data --data data/pandaset_fog --sequence 001 \
    --pipeline.model.use-dual-stream False \
    --pipeline.model.use-lidar-offset False

# Ablation 2: +ASM only (Innovation 1 — per‑Gaussian transmission map)
ns-train cosplat pandaset-data --data data/pandaset_fog --sequence 001 \
    --pipeline.model.use-dual-stream True \
    --pipeline.model.use-lidar-offset False

# Ablation 3: +Offset only (Innovation 2 — LiDAR residual offsets)
ns-train cosplat pandaset-data --data data/pandaset_fog --sequence 001 \
    --pipeline.model.use-dual-stream False \
    --pipeline.model.use-lidar-offset True

# Ablation 4: Full model (both innovations)
ns-train cosplat pandaset-data --data data/pandaset_fog --sequence 001
```

Expected outcomes from the ablation study (PandaSet, β=0.01):

| Configuration | Dehaze PSNR↑ | Dehaze SSIM↑ | Dehaze LPIPS↓ | Notes |
| --- | --- | --- | --- | --- |
| Baseline (vanilla SplatAD) | 13.68 | 0.696 | 0.286 | No dehazing capability |
| +ASM only | 19.29 | 0.749 | 0.237 | Innovation 1 alone provides +5.61 dB |
| +Offset only | 13.71 | 0.696 | 0.285 | No dehazing (expected), geometry improved |
| Full model | 19.54 | 0.752 | 0.236 | 1+1>2 synergy: +0.25 dB over ASM‑only |

The $0.25$ dB improvement from ASM‑only to Full is consistent and repeatable, arising from the feedback loop: LiDAR offsets improve geometric accuracy → better per‑Gaussian depth estimates → more accurate transmission map → cleaner ASM decomposition.

### CoSplat Evaluation

We provide dedicated evaluation scripts for CoSplat that compare the dehazed output against clean ground‑truth images. These compute PSNR, SSIM, LPIPS, and save both per‑image metrics and 4‑panel visual comparisons.

For PandaSet:

```bash
python eval_dehaze_full.py \
    --load-config outputs/cosplat/ablation_full/config.yml \
    --clean-gt-dir data/pandaset/001 \
    --output-dir ./eval_results/ \
    --max-vis 10
```

For nuScenes:

```bash
python eval_dehaze_nuscenes.py \
    --load-config outputs/cosplat/nufog_full/config.yml \
    --clean-data-root data/nuscenes \
    --output-dir ./eval_nuscenes/ \
    --max-vis 10
```

The scripts output `dehaze_metrics.json` containing summary statistics and per‑image breakdowns, a `comparisons/` directory with side‑by‑side visualizations, and a `summary_grid.png` overview.

### Tensorboard / WandB / Comet / Viewer

There are four different methods to track training progress, using the viewer, tensorboard, Weights and Biases, and Comet. You can specify which visualizer to use by appending `--vis {viewer, tensorboard, wandb, comet viewer+wandb, viewer+tensorboard, viewer+comet}` to the training command. Simultaneously utilizing the viewer alongside wandb or tensorboard may cause stuttering issues during evaluation steps.

# Learn More

And that's it for getting started with the basics of NeuRAD, SplatAD, and CoSplat. If you are missing some features, have a look at Planned Features to see if we have plans on implementing this. Otherwise, feel free to open an issue, or even better implement it yourself and open a PR!

If you want to add a dataset, look [here](#adding-datasets). If you want to add a method, have a look [here](#adding-methods).

## Adding Datasets

We have provided dataparsers for multiple autonomous driving datasets, see below for a complete list. However, your favorite AD dataset might still be missing.

To add a dataset, create `nerfstudio/data/dataparsers/mydataset.py` containing one dataparser config class `MyADDataParserConfig` and one dataparser class `MyADData`. Preferably, these inherit from `ADDataParserConfig` and `ADDataParser`, as these provide common functionality and streamline the expected format of AD data. For most datasets, it should then be sufficient to overwrite `_get_cameras`, `_get_lidars`, `_read_lidars`, `_get_actor_trajectories`, and `_generate_dataparser_outputs`.

| Data | Cameras | Lidars |
| --- | --- | --- |
| 🚗 nuScenes | 6 cameras | 32-beam lidar |
| 🚗 ZOD (Annotations) | 1 camera | 128-beam + 2 × 16-beam lidars |
| 🚗 Argoverse 2 | 7 ring cameras + 2 stereo cameras | 2 × 32-beam lidars |
| 🚗 PandaSet (huggingface download) | 6 cameras | 64-beam lidar |
| 🚗 KITTIMOT (Timestamps) | 2 stereo cameras | 64-beam lidar |
| 🚗 Waymo v2 | 5 cameras | 64-beam lidar |

A brief introduction about Waymo dataparser for NeuRAD can be found in `waymo_dataparser.md`.

## Adding Methods

Nerfstudio has made it easy to add new methods, see [here](https://docs.nerf.studio/developer_guides/new_methods.html) for details. We have added our UniSim reimplementation as a plugin, which can be run as any other method using the `ns-train` command:

```bash
ns-train unisim pandaset-data --data data/pandaset
```

and follow the instructions in the terminal.

See our UniSim repo for reference on how to add a new method as a plugin.

# Key features

- Dataparser for multiple autonomous driving datasets including:
    - Dataparsing of lidar data (3D + intensity + time)
    - Dataparsing of annotations
- Datamanager for lidar + image data
- Rolling shutter handling for ray generation
- Viewer improvements:
    - Lidar rendering
    - Dynamic actor modifications
- NeuRAD — SOTA NeRF‑based rendering method for dynamic AD scenes
- SplatAD — SOTA splatting‑based rendering method for dynamic AD scenes
- CoSplat — Zero‑shot dehazing for foggy driving scenes:
    - Surface‑environment dual‑stream decoupling with per‑Gaussian atmospheric scattering model
    - Hierarchical LiDAR residual offsets for multi‑modal geometry consistency
    - Bias‑driven training strategy to prevent β collapse and enable stable decomposition
    - Independent ablation toggles for systematic evaluation

# Planned Features/TODOs

- [x] 3DGS implementation supporting dynamic objects
- [x] UniSim plug-in
- [x] Release code
- [x] Zero‑shot dehazing with surface‑environment decoupling (CoSplat)
- [ ] Height‑adaptive fog density (β as a function of elevation)
- [ ] View‑dependent atmospheric light for improved sky/ground consistency
- [ ] LiDAR true‑depth supervision for transmission map refinement

# Built On

Collaboration friendly studio for NeRFs

# Citation

You can find our papers for NeuRAD and SplatAD on arXiv. You can also head over to our research blog for project pages and more papers on AD.

If you use this code or find our papers useful, please consider citing:

```bibtex
@inproceedings{tonderski2024neurad,
  title={{NeuRAD}: Neural rendering for autonomous driving},
  author={Tonderski, Adam and Lindstr{\"o}m, Carl and Hess, Georg and Ljungbergh, William and Svensson, Lennart and Petersson, Christoffer},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={14895--14904},
  year={2024}
}

@inproceedings{hess2024splatad,
  title={{SplatAD}: Real-Time Lidar and Camera Rendering with 3D Gaussian Splatting for Autonomous Driving},
  author={Hess, Georg and Lindstr{\"o}m, Carl and Fatemi, Maryam and Petersson, Christoffer and Svensson, Lennart},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```

If you use the CoSplat extension, please cite:

```bibtex
@article{cosplat2025,
  title={{CoSplat}: Consistent Multi-Modal Simulation with Surface-Environment Decoupled 3D Gaussian Splatting},
  author={},
  journal={arXiv preprint},
  year={2025}
}
```

# Contributors

<a href="https://github.com/georghess">
    <img src="https://github.com/georghess.png" width="60px;" style="border-radius: 50%;"/>
</a>
<a href="https://github.com/carlinds">
    <img src="https://github.com/carlinds.png" width="60px;" style="border-radius: 50%;"/>
</a>
<a href="https://github.com/atonderski">
    <img src="https://github.com/atonderski.png" width="60px;" style="border-radius: 50%;"/>
</a>
<a href="https://github.com/wljungbergh">
    <img src="https://github.com/wljungbergh.png" width="60px;" style="border-radius: 50%;"/>
</a>
<a href="https://github.com/MartinEthier">
    <img src="https://github.com/MartinEthier.png" width="60px;" style="border-radius: 50%;"/>
</a>
<a href="https://github.com/JulienStanguennec-Leddartech">
    <img src="https://github.com/JulienStanguennec-Leddartech.png" width="60px;" style="border-radius: 50%;"/>
</a>

\+ [nerfstudio contributors](https://github.com/nerfstudio-project/nerfstudio/graphs/contributors)
# TimeBridge
Official Pytorch Implementation of ["TimeBridge: Better Diffusion Prior Design with Bridge Models for Time Series Generation"](https://arxiv.org/abs/2408.06672) (KDD 2026)

## Code Implementation 

For code implementation, we use the official code of **Diffusion-TS (ICLR 24)**, including dataset and running code. 

### Dataset

All the four real-world datasets (Stocks, ETTh1, Energy and fMRI) can be obtained from [Google Drive](https://drive.google.com/file/d/11DI22zKWtHjXMnNGPWNUbyGz-JiEtZy6/view?usp=sharing). Please download **dataset.zip**, then unzip and copy it to the folder `./Data` in our repository. EEG dataset can be downloaded from [here](https://drive.google.com/file/d/1IqwE0wbCT1orVdZpul2xFiNkGnYs4t89/view?usp=sharing) and should also be placed in the aforementioned `./Data/dataset` folder.

### Environment & Libraries

The full libraries list is provided as a `requirements.txt` in this repo. Please create a virtual environment with `conda` or `venv` and run

~~~bash
(myenv) $ pip install -r requirements.txt
~~~

### Training & Sampling


**Note:** We provide `.yaml` config files (stock, sines, mujoco, etth, energy, fmri) under `./Config` folder.

---

#### Common Parameters

| Parameter | Description | Options/Values |
|-----------|-------------|----------------|
| `--data` | Dataset name | energy, etth, fmri, mujoco, sines, stock |
| `--name` | Experiment name | Custom string |
| `--output` | Output directory | Default: `OUTPUT` |
| `--sample` | Task type | `0`: unconditional, `1`: conditional |
| `--mode` | Task mode | `generation`, `infill`|

#### TimeBridge Parameters

| Parameter | Description |  Notes |
|-----------|-------------|-------|
| `--prior` | Prior distribution | `normal`, `uniform`, `trend`, `trend-poly`, `trend-linear`, `gp` |
| `--pred_mode` | Prediction mode | `vp` or `ve`. **Important:** `vp` requires `sigma_max=1.0` |
| `--sampler` | Sampling method | `sde` or `ode` |
| `--sigma_min` | Min noise level | Diffusion noise parameter |
| `--sigma_max` | Max noise level | **Must be 1.0 for `vp` mode** |
| `--sigma_data` | Data noise scale  | Diffusion noise parameter |
| `--beta_min` | Beta schedule min | Beta schedule parameter |
| `--beta_d` | Beta schedule max | Beta schedule parameter |
| `--kernel_type` | GP kernel type |For GP prior |
| `--bw` | Bandwidth type | For GP prior |
| `--var` | Prior variance | Variance parameter |
| `--model_matching` | Matching type | Model matching strategy |

---

### Usage Examples

Refer to **run** folder for example notebooks including best settings (`01_Unconditional_Generation.ipynb`, etc.).


#### 1. Unconditional Generation 
01_Unconditional_Generation.ipynb


#### 2. Trend Priors (Trend-Conditional Generation)
02_Conditional_Generation.ipynb

Using different trend-based priors:
--prior [trend-poly, trend-linear, trend]

#### 3. Imputation (Fixed-data Conditional)
03_Imputation_energy.ipynb

For missing value imputation:
--prior [impt-linear, impt-quadratic]


## Acknowledgement

Our code is based on the Diffusion-TS (ICLR 24) and DDBM (ICLR 24) below.

1. Diffusion-TS: https://github.com/Y-debug-sys/Diffusion-TS
2. DDBM: https://github.com/alexzhou907/DDBM

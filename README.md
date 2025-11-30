# TimeBridge
Official Pytorch Implementation of "TimeBridge: Better Diffusion Prior Design with Bridge Models for Time Series Generation" (KDD 2026)

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

Refer to **run** folder.

For training, you can reproduce the experimental results of all benchmarks by runing

~~~bash
(myenv) $ python main.py --name {name} --config_file {config.yaml} --gpu 0 --train
~~~

**Note:** We also provided the corresponding `.yml` files (only stocks, sines, mujoco, etth, energy and fmri) under the folder `./Config` where all possible option can be altered. 

#### Unconstrained
```bash
(myenv) $ python main.py --name {name} --config_file {config.yaml} --gpu 0 --sample 0 --milestone {checkpoint_number}
```

#### Imputation
```bash
(myenv) $ python main.py --name {name} --config_file {config.yaml} --gpu 0 --sample 1 --milestone {checkpoint_number} --mode infill --missing_ratio {missing_ratio}
```



## Acknowledgement

Our code is based on the Diffusion-TS (ICLR 24) and DDBM (ICLR 24) below.

1. Diffusion-TS: https://github.com/Y-debug-sys/Diffusion-TS
2. DDBM: https://github.com/alexzhou907/DDBM
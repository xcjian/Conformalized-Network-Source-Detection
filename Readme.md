# Conformalized Network Source Detection

## About the Project

This repository contains the source code and data for reproducing the paper *Conformal Prediction for Multi-Source Detection on a Network*.

## Getting Started

### Environmental Setup

Initiate the environment:

```conda create -n [env_name] python=3.8.20```

(Optional) install `cudatoolkit`, if the cuda version is not 10.2:

```conda install -c conda-forge cudatoolkit=10.1.243```

Install required packages:

```pip install -r requirements.txt```

### Data Preprocessing

Run the command

```
cd SD-STGCN/output/test_res
python data_prepare.py --combine 1 --split 0
cd ../../../
nohup ./batch_CP_process_loaddata.sh > /dev/null 2>&1 &
```
For the first time of executing the `.sh` file, run the following command to make it executable:
```
chmod +x batch_CP_process.sh
```


## Usage

Example: SIR simulations with 10 sources on the highSchool network
```
python main.py --graph highSchool --train_exp_name SIR_nsrc10_Rzero43.44_beta0.25_gamma0.15_T30_ls21200_nf16 --test_exp_name SIR_nsrc10_Rzero43.44_beta0.25_gamma0.15_T30_ls8000_nf16 --pow_expected 0.5 --prop_model SIR --set_recall 1 --set_prec 1 --ArbiTree_CQC 1 > logfiles/runCPSIR10pow5.log 2>&1
```
See `batch_CP_process.sh` and `batch_CP_process_SI1.sh` for all commands to reproduce the experiments.

## Acknowledgements

The `DSI` module is derived from the [source code](https://github.com/lab-sigma/Diffusion-Source-Identification) of the existing work

Dawkins, Q. E., Li, T., & Xu, H. (2021, July). Diffusion source identification on networks with statistical confidence. In International Conference on Machine Learning


The `SD-STGCN` module is derived from the [source code](https://github.com/anonymous-anuthor/SD-STGCN) of the existing work

H. Sha, M. Al Hasan and G. Mohler, "Source detection on networks using spatial temporal graph convolutional networks," 2021 IEEE 8th International Conference on Data Science and Advanced Analytics (DSAA), Porto, Portugal.


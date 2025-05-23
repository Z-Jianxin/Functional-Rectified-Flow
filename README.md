# Functional Diffusion Process

---
Our code is based on the code release of [Franzese et al., 2023](https://github.com/giulio98/functional-diffusion-processes). 
---

## Quickstart

### Setup the Development Environment

```bash
conda env create -f env.yaml
conda activate fdp
pip install -e .[dev]
pre-commit install
```
---

## Setup the Project
Before you begin with any experiments, ensure to create a `.env` file with the following content:
```plaintext
export WANDB_API_KEY=<your wandb api key>
export HOME=<your_home_directory>
export CUDA_HOME=/usr/local/cuda
export PROJECT_ROOT=<your_project_directory>
export DATA_ROOT=${PROJECT_ROOT}/data
export LOGS_ROOT=${PROJECT_ROOT}/logs
export TFDS_DATA_DIR=${DATA_ROOT}/tensorflow_datasets
export PYTHONPATH=${PROJECT_ROOT}
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export WANDB_DISABLE_SERVICE=true
export CUDA_VISIBLE_DEVICES=<your cuda devices>
```
All experiments utilize wandb for logging. However, you can opt out of using wandb by setting `trainer_logging.use_wandb=False` in the yaml files in `conf/trainers/trainer_maml` and `conf/trainers/trainer_vit`. We implement the functional rectified flow by adding samplers/ode_sampler.py and sdetools/ode.py.

# Experiments
### Download Dataset
```bash
pip install gdown
```
```bash
cd ~/functional-rectified-flow/data/tensorflow_datasets/
gdown --folder https://drive.google.com/drive/folders/1eHdU3N4Tiv6BAezAAI7LAvJTItIF8GD2?usp=share_link
```
The download scripts of MNIST are integrated into the code.

### Training
Run the default training script:

```bash
python3 src/functional_diffusion_processes/run.py --multirun +experiments_maml=exp_mnist_frf
python3 src/functional_diffusion_processes/run.py --multirun +experiments_vit=exp_celeba_frf
```
### Generation
```bash
python3 src/functional_diffusion_processes/run.py --multirun +experiments_maml=eval_mnist_frf
python3 src/functional_diffusion_processes/run.py --multirun +experiments_vit=eval_celeba_frf
```
### Super Resolution
```bash
python3 src/functional_diffusion_processes/run.py --multirun +experiments_maml=eval_mnist_ss
```


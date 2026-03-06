# TAPPI

## Installation

First clone the repository from GitHub:

```bash
git clone git@github.com:NENUBioCompute/TAPPI.git
cd TAPPI
```

Then install the required dependencies:

```bash
pip install -r requirements.txt
```

It is recommended to use a Python environment with **Python ≥ 3.8** and **PyTorch ≥ 2.0**.

After installing the dependencies, the environment will be ready for running the TAPPI model and related experiments.


## introduction
 a deep learning framework utilizes LLMs and improved backbones to predict protein-protein interaction variation trends 
 <div align="center">
    <img src="docs/base_back.png", width="800">
</div>

* train&benchmark -> **source code** of training and benchmark evaluation
* external -> Apply TAPPI to **external datasets** and ***pathogenic analyse**
* ## Pre-trained Models and Test Dataset
The pre-trained models and test datasets can be loaded from [Figshare](https://figshare.com/articles/dataset/x_test_fold_mirror_multi_csv/29637008).
## Model Evaluation
You can easily understand the model's performance by executing the notebook available at [mippi_simply_variation.ipynb](https://github.com/NENUBioCompute/TAPPI/blob/main/mippi_simply_variation.ipynb).
# Requirement
* python 3.8+
* pytorch 1.12.0+
* numpy
* pandas
* fair-esm

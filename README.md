# Training and testing a model
The relevant scripts for training a model are in the IVF folder - specifically `conmodel.py`,  `btrain.py` and `test.py`. `conmodel.py` is the model that is used for training and testing by both the other scripts. Running the code requires the use of a conda environment so let us first set that up
Make sure that you have conda installed, installation instructions can be found [here](https://docs.anaconda.com/miniconda/) Upon successful installation of conda, we setup the environment:
```
cd IVF/
conda env create -f env.yml
conda activate BTV_train
```
Test if the environment works by running
```
python btrain.py -h
python test.py -h
```

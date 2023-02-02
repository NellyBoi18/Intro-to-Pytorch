# Intro to Pytorch
 Implementing and training a basic neural network using Pytorch

**You will need to use Python 3 and a Python Virtual Environment with torch v1.12.1+cpu, torchvision v0.13.1+cpu and torchaudio v0.12.1+cpu**

The following steps sets up a Python Virtual Environment using the venv module but you can use other virtual envs such as Conda.

**Step 1:** Set up a Python Virtual Environment named Pytorch:
```
python3 -m venv /path/to/new/virtual/environment
```
For example, if you want to put the virtual environment in your working directoy:
```
python3 -m venv Pytorch
```

**Step 2:** Active the environment
```
source Pytorch/bin/activate
```

**Step 3:** Upgrade pip and install the CPU version of Pytorch. (Note: you may be using pip3 instead of pip so just add 3 if needed)
```
pip install --upgrade pip
pip install torch==1.12.1+cpu torchvision==0.13.1+cpu torchaudio==0.12.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
```


You can check the version of the packages installed using:
```
pip freeze
```
To deactive the virtual environment:
```
deactivate
```
# Installation

These installation instructions are created for Ubuntu 20.04. If you are using a different OS you made need to make some changes to the installing instructions. 


## Ubuntu 20.04

```
sudo apt-get install libosmesa6-dev
sudo apt-get install patchelf
```


## Install Python Environment


There are two options:

A. (Recommended) Install with conda:

1. Install conda, if you don't already have it, by following the instructions at [this link](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)

This install will modify the `PATH` variable in your bashrc.
You need to open a new terminal for that path change to take place (to be able to find 'conda' in the next step).

2. Create a conda environment that will contain python 3:
```
conda create -n grid python=3.7
```
4. (Optional) If you have any problem with the pygam library do :
```
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```
3. Activate python environment
```
source activate grid
```

4. Run this command
```
./mingrid/main_agent.py
```
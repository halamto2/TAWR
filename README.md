### Transformer Architecture for Watermark Removal (TAWR)
This project is a part of my thesis on the topic of Detection and removal of watermarks from image data. The project is built using PyTorch and various components from other libraries and sources. This repository in its current state is equivalent to the *TAWR Segformer* variant presented in the thesis. 

## Installation
The following steps will guide you to set up the environment for running this project. Please make sure to execute these commands in the terminal:

1. Create a new conda environment:
```
conda create --name TAWR python=3.8 -y
```

2. Activate the newly created conda environment:
```
conda activate TAWR
```

3. Install PyTorch, torchvision, torchaudio, and cudatoolkit:
```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
```

4. Install OpenMIM:
```
pip install -U openmim
```

5. Install MMsegmentation:
```
mim install mmcv-full
pip install mmsegmentation
```

6. Install other required libraries:
```
conda install scipy matplotlib progress scikit-image==0.17.2 tensorboardX scikit-learn conda-build
pip install -U albumentations --no-binary qudida,albumentations
pip install ninja
conda install -c conda-forge timm
```

7. Clone required external repositories:
```
git clone https://github.com/bcmi/SLBR-Visible-Watermark-Removal.git external_repos/SLBR
git clone https://github.com/fenglinglwb/MAT.git external_repos/MAT
git clone https://github.com/whai362/PVT external_repos/PVT
```

8. Finally, install the project and its dependencies:
```
conda-develop . && conda-develop ./external_repos/ && conda-develop ./external_repos/MAT && conda-develop ./external_repos/SLBR
```

Additionally, a file named `conda_packages_env.txt` is provided which lists the exact packages and dependencies used for this project, which can be used to recreate the environment if needed.

## Usage
To use the project, first make sure to activate the conda environment:
```
conda activate TAWR
```
Then, navigate to the project directory and run the desired script.

## Results
The results of the project are documented in the thesis text, which will be linked here once it becomes available.

## Acknowledgments
Parts of the codebase for this project are based on the previous work [SLBR](https://github.com/bcmi/SLBR-Visible-Watermark-Removal)

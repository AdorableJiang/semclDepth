# semclDepth

## Prerequisites

- Linux or macOS (Windows is in experimental support)
- Python 3.8
- PyTorch 1.8
- CUDA 11.1+ (If you build PyTorch from source, CUDA 9.0 is also compatible)
- GCC 5+
- [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)

## Installation

I ran experiments with PyTorch 1.8.2, CUDA 11.1, Python 3.8, and Ubuntu 20.04. Other settings that satisfy the requirement would work.

Use Anaconda to create a conda environment:

```bash
conda create -n dep python=3.8
conda activate dep
```

### Pytorch

```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c conda-forge
```

### mmcv-full

Then, install MMCV and install our toolbox:

```bash
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html

# clone and enter this repo
pip install -e . # probably need root privilege
```

Recently, I have tested our codes based on mmcv==1.5.0, which also works.

More information about installation can be found in docs of MMSegmentation (see [get_started.md](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/get_started.md#installation)).

If training, you should install the tensorboard:

```shell
pip install future tensorboard
```

### pytorch3d

When reproducing Adabins, [Pytorch3d](https://github.com/facebookresearch/pytorch3d) is needed to calculate the chamfer loss. 

```bash
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install cuda -c nvidia
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

Please follow [pytorch3d/INSTALL.md](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md#installation) if you have problem installing `Pytorch3d`.

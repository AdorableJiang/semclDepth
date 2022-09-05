## Prerequisites
- Linux or macOS (Windows is in experimental support)
- Python 3.6+
- PyTorch 1.3+
- CUDA 9.2+ (If you build PyTorch from source, CUDA 9.0 is also compatible)
- GCC 5+
- [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)

## Installation
I ran experiments with PyTorch 1.8.0, CUDA 11.1, Python 3.7, and Ubuntu 20.04. Other settings that satisfact the requirement would work.

### **If you have a similar environment**
You can simply follow our settings:

Use Anaconda to create a conda environment:

```bash
conda create -n dep python=3.8
conda activate dep
```

#### Pytorch

```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c conda-forge
```

#### mmcv-full

Then, install MMCV and install our toolbox:

```bash
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html

# clone and enter this repo
pip install -e . # probably need root privilege
```

If training, you should install the tensorboard:

```shell
pip install future tensorboard
```

#### pytorch3d



Recently, I have tested our codes based on mmcv==1.5.0, which also works.

More information about installation can be found in docs of MMSegmentation (see [get_started.md](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/get_started.md#installation)).

### Other Dependence
When reproducing Adabins, [Pytorch3d](https://github.com/facebookresearch/pytorch3d) is needed to calculate the chamfer loss. You can de-annotate the import in depth/models/losses/chamferloss.py if you wanna train Adabins, or you should install Pytorch3d:
```
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
```
In the future, I will try to remove this dependence.


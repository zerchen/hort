<div align="center">

# HORT: Monocular Hand-held Objects Reconstruction with Transformers

[Zerui Chen](https://zerchen.github.io/)<sup>1</sup> &emsp; [Rolandos Alexandros Potamias](https://rolpotamias.github.io)<sup>2</sup> &emsp; [Shizhe Chen](https://cshizhe.github.io/)<sup>1</sup> &emsp; [Cordelia Schmid](https://cordeliaschmid.github.io/)<sup>1</sup>

<sup>1</sup>WILLOW, INRIA Paris, France <br>
<sup>2</sup>Imperial College London, UK

<a href='https://zerchen.github.io/projects/hort.html'><img src='https://img.shields.io/badge/Project-Page-blue'></a>
<a href='https://arxiv.org/abs/2503.21313'><img src='https://img.shields.io/badge/Paper-arXiv-red'></a>
<a href='https://huggingface.co/spaces/zerchen/HORT'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-green'></a>
</div>

This is the implementation of **[HORT](https://zerchen.github.io/projects/hort.html)**, an state-of-the-art hand-held object reconstruction algorithm:

![teaser](assets/teaser.png)

## Installation 👷
```
git clone https://github.com/zerchen/hort.git
cd hort
```

The code has been tested with PyTorch 2.4.1 and CUDA 12.1. It is suggested to use an anaconda encironment to install the the required dependencies:
```bash
conda create --name hort python=3.12
conda activate hort

conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
# Install requirements
pip install -r requirements.txt
pip install -U git+https://github.com/luca-medeiros/lang-segment-anything.git
conda install pytorch3d-0.7.8-py312_cu121_pyt241.tar.bz2 # https://anaconda.org/pytorch3d/pytorch3d/files?page=2
cd /home/zerchen/workspace/code/hort_init/hort/models/tgs/models/snowflake/pointnet2_ops_lib && python setup.py install
```
Download the pretrained models using: 
```bash
wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt -P ./pretrained_models/
wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/wilor_final.ckpt -P ./pretrained_models/
wget https://huggingface.co/zerchen/hort_models/resolve/main/hort_final.pth.tar -P ./pretrained_models/
```
It is also required to download MANO model from [MANO website](https://mano.is.tue.mpg.de). 
Create an account by clicking Sign Up and download the models (mano_v*_*.zip). Unzip and place the right hand model `MANO_RIGHT.pkl` under the `mano_data/mano/` folder. 
Note that MANO model falls under the [MANO license](https://mano.is.tue.mpg.de/license.html).
## Demo 🎮
```bash
python demo.py --img_folder demo_img 
python vis_ho.py -e out_demo/test1.json # visualize the result in open3d
```
## Start a local gradio demo 🤗
You can start a local demo for inference by running:
```bash
python gradio_demo.py
```
## Acknowledgements
Parts of the code are based on [WiLoR](https://github.com/rolpotamias/WiLoR), [SnowflakeNet](https://github.com/AllenXiangX/SnowflakeNet) and [Lang-SAM](https://github.com/luca-medeiros/lang-segment-anything).

## License 📚
HORT is licensed under MIT License. This repository also depends on [WiLoR](https://github.com/rolpotamias/WiLoR), [Ultralytics library](https://github.com/ultralytics/ultralytics) and [MANO Model](https://mano.is.tue.mpg.de/license.html), which are fall under their own licenses.
## Citation  📝
If you find HORT useful for your research, please consider citing our paper:

```bibtex
@article{chen2025hort,
  title={{HORT}: Monocular Hand-held Objects Reconstruction with Transformers},
  author={Chen, Zerui and Potamias, Rolandos Alexandros and Chen, Shizhe and Schmid, Cordelia},
  journal={arXiv preprint arXiv:2503.21313},
  year={2025}
}
```

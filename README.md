# GIQ:  Benchmarking 3D Geometric Reasoning of Vision Foundation Models with Simulated and Real Polyhedra

<a href="https://arxiv.org/pdf/2412.19920"><img src="https://img.shields.io/badge/Arxiv-2408.00653-B31B1B.svg"></a>


<br>

This repository contains the official codebase for **G-IQ**, a comprehensive benchmark designed to assess the geometric reasoning capabilities of vision and vision-language foundation models on a diverse collection of polyhedra exhibiting rich geometric properties—convexity, symmetry, and varying levels of complexity.

<br>
<p align="center">
    <img width="450" src="demo_files/sample2.png"/>
</p>



<p align="center">
    <img width="450" src="demo_files/comp.gif"/>
</p>


Stable Fast 3D is based on [TripoSR](https://github.com/VAST-AI-Research/TripoSR) but introduces several new key techniques. For one, we explicitly optimize our model to produce good meshes without artifacts alongside textures with UV unwrapping. We also delight the color and predict material parameters so the assets can be easily integrated into a game. We achieve all of this while still maintaining the fast inference speeds of TripoSR.

## Getting Started

### Installation

Ensure your environment is:
- Python >= 3.8
- Optional: CUDA or MPS has to be available
- For Windows **(experimental)**: Visual Studio 2022
- Has PyTorch installed according to your platform: https://pytorch.org/get-started/locally/ [Make sure the Pytorch CUDA version matches your system's.]
- Update setuptools by `pip install -U setuptools==69.5.1`
- Install wheel by `pip install wheel`

Then, install the remaining requirements with `pip install -r requirements.txt`.
For the gradio demo, an additional `pip install -r requirements-demo.txt` is required.



## Citation
```BibTeX
@article{michalkiewicz2024not,
  title={Not all Views are Created Equal: Analyzing Viewpoint Instabilities in Vision Foundation Models},
  author={Michalkiewicz, Mateusz and Bai, Sheena and Baktashmotlagh, Mahsa and Jampani, Varun and Balakrishnan, Guha},
  journal={arXiv preprint arXiv:2412.19920},
  year={2024}
}
```

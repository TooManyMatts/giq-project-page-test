# GIQ:  Benchmarking 3D Geometric Reasoning of Vision Foundation Models with Simulated and Real Polyhedra

<a href="https://arxiv.org/pdf/2412.19920"><img src="https://img.shields.io/badge/Arxiv-2408.00653-B31B1B.svg"></a>


<br>

This repository contains the official codebase for **G-IQ**, a comprehensive benchmark designed to assess the geometric reasoning capabilities of vision and vision-language foundation models on a diverse collection of polyhedra exhibiting rich geometric properties—convexity, symmetry, and varying levels of complexity.

<br>
<p align="center">
    <img width="450" src="demo_files/sample_2.png"/>
</p>


## Table of Contents

1. [Data Files (`json`, `txt`)](#data-files-json-txt)
2. [Download Links to Data](#download-links-to-data)
3. [Results: Monocular 3D Reconstruction](#results-monocular-3d-reconstruction)
4. [Citation](#citation)
5. [License](#license)

---

## Data Files (`json`, `txt`)

### `shapes.json`

A JSON file describing each polyhedron in G-IQ:

```json
{
  "<shape_id>": {
    "group": "<group_name>",
    "sym": "<symmetry_type>",
    "name": "<polyhedron_name>"
  },
  …
}
```

- **Shape IDs**  
  - `cid_x` – Catalan solid number *x*  
  - `jid_x` – Johnson solid number *x*  
  - `wid_x` – Wenninger model ID number *x*  

- **Groups** (one of):  
  `platonic`, `archimedean`, `catalan`, `johnson`, `stellations`,  
  `kepler-poinsot`, `compounds`, `uniform non-convex`

- **Symmetries** (one of):  
  `central point reflection`, `5-fold rotation`, `4-fold rotation`

- **Name**  
  The canonical polyhedron name (e.g., `"cube"`, `"hexahedron"`).

**Example entry**:

```json
{
  "wid_3": {
    "group": "platonic",
    "sym": "4-fold rotation",
    "name": "cube, hexahedron"
  }
}
```

### Hard Split (`hard_split.txt`)

Defines challenging “hard” pairs for the Mental Rotation Test. Each line is a comma-separated pair. Sections:

```
# negative pairs
cid 7, wid 5
cid 13, wid 9
cid 7, cid 13

# positive pairs
cid 7, cid 7
…
```

---

## Download Links to Data

- **Rendered images** (256×256 PNGs, 20 views per shape):  
  `<dummy link to renderings archive>`

- **3D meshes** (OBJ/PLY format):  
  `<dummy link to mesh archive>`

---

## Results: Monocular 3D Reconstruction

Below are representative results of monocular 3D reconstruction using state-of-the-art methods (Shap-E, Stable Fast 3D, OpenLRM):

<p align="center">
  <img width="600" src="demo_files/m3dr.gif" alt="Monocular 3D Reconstruction Results"/>
  <!-- Replace with actual GIF showcasing reconstruction results -->
</p>

---



## Citation
```BibTeX
@article{michalkiewicz2024not,
  title={Not all Views are Created Equal: Analyzing Viewpoint Instabilities in Vision Foundation Models},
  author={Michalkiewicz, Mateusz and Bai, Sheena and Baktashmotlagh, Mahsa and Jampani, Varun and Balakrishnan, Guha},
  journal={arXiv preprint arXiv:2412.19920},
  year={2024}
}
```

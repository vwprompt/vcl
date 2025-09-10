# Visual Coherence Loss (VCL) for Coherent & Visually Grounded Story Generation

> Code and resources for **“Visual Coherence Loss for Coherent and Visually Grounded Story Generation” (Findings of ACL 2023)** by Xudong Hong, Vera Demberg, Asad Sayeed, Qiankun Zheng, and Bernt Schiele.

---

## 🌟 TL;DR

- **VCL**: a self-supervised **visual coherence ranking loss** that encourages models to represent character recurrence across an image sequence.
- **Character-centric features**: plug in **object** and **face** features to build stronger character representations.
- **CM (Character Matching)**: a reference-free **image–text relevance** metric that checks whether generated stories refer to the **right characters** in the input images.

---

## Repository Structure
```
.
├── configs/ # YAML configs for training/eval experiments
├── data/ # (created by you) VWP dataset, splits, annotations
├── models/ # Encoders/decoders, loss (VCL), training loops
├── features/ # Scripts to extract object/face features
│ ├── objects/ # e.g., Faster R-CNN feature extraction
│ └── faces/ # e.g., face detection & embeddings
├── metrics/ # Character Matching (CM) implementation
├── scripts/ # Useful scripts
├── notebooks/ # Exploration & visualization (optional)
```

---

## Installation

```bash
# 1) Create environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) Install dependencies (PyTorch per your CUDA/CPU)
pip install -r requirements.txt
# If you need a starting point, requirements.txt typically includes:
# torch, torchvision, transformers, scipy, numpy, pandas, pyyaml, pillow, opencv-python, tqdm
```
> 💡 For GPU training, install the matching `torch`/`torchvision` wheels from https://pytorch.org/get-started/locally/

---

## Data: Visual Writing Prompts (VWP)

This project evaluates on **VWP** (Visual Writing Prompts), a curated dataset of image sequences with grounded characters and aligned crowd-sourced stories.

- Website: https://vwprompt.github.io/  
- Repo: https://github.com/vwprompt/vwp  
- Paper: https://arxiv.org/abs/2301.08571

---

## Feature Extraction (Objects & Faces)

VCL benefits from **character-aware** visual features:

- **Object features**: extracted with a general detector (e.g., Faster R-CNN) over each image; pooled per frame.
- **Face features**: extracted with a face detector + embedding model; aggregated per character instance.

---

## Tips & Notes

- **Coreference**: For extracting referring expressions in generated stories, you can plug in any off-the-shelf English **coreference resolver** (configure in `metrics/eval_cm.py`).
- **Speed**: precompute and cache all visual features; enable mixed precision for LM decoding.

---

## Citing

If you use this repository, please cite the paper:

```bibtex
@inproceedings{hong-etal-2023-visual,
  title     = {Visual Coherence Loss for Coherent and Visually Grounded Story Generation},
  author    = {Hong, Xudong and Demberg, Vera and Sayeed, Asad and Zheng, Qiankun and Schiele, Bernt},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2023},
  month     = jul,
  year      = {2023},
  address   = {Toronto, Canada},
  publisher = {Association for Computational Linguistics},
  url       = {https://aclanthology.org/2023.findings-acl.603/},
  doi       = {10.18653/v1/2023.findings-acl.603},
  pages     = {9456--9470}
}
```

---

## Acknowledgements

This work builds on prior efforts in **visual storytelling**, **transformer LMs**, and reference-free **image–text relevance** evaluation. We thank the authors and dataset maintainers for making their resources available.

---

## Contact

- For issues/bugs: open a GitHub issue.
- Research questions: see the paper authors’ pages or reach out via email listed in the publication.



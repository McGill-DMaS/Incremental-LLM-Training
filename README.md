# On the Effectiveness of Incremental Training of Large Language Models

This repository contains the source code, experimental setup scripts used in the paper:

**"On the Effectiveness of Incremental Training of Large Language Models"**

> Miles Q. Li, Benjamin Fung, and Shih-Chia Huang. *arXiv preprint arXiv:2411.18700 (2024)*

---

## Paper Abstract
Incremental layer-wise training has been proposed as a strategy to optimize large language model (LLM) training by progressively introducing layers. This paper investigates its effectiveness, revealing that incremental training initially shows computational efficiency but ultimately incurs higher overall costs to match full-scale training. The results highlight limitations of incremental layer-wise training for initial model training but suggest a potential for scaling pretrained checkpoints.

---

## Code Structure

### Training Code
The training code is built on **nanoGPT** by [Andrej Karpathy](https://github.com/karpathy/build-nanogpt). nanoGPT provides a minimalistic, clean implementation of GPT-like transformer models.

- **Training**: Our experiments apply incremental layer-wise training to progressively add layers during optimization.
- **Baselines**: We compare incremental training with full-scale model training.

### Evaluation Code
The evaluation scripts for HellaSwag are implemented based on [hellaswag](https://github.com/rowanz/hellaswag).

- **HellaSwag**: A commonsense reasoning benchmark used to evaluate model generalization.


---

## Results
Our findings demonstrate that while incremental training initially shows computational gains, it requires substantially more resources to achieve similar performance as full-scale training. For details, refer to the [paper](https://arxiv.org/abs/2411.18700).

---

## Citation
If you find this work useful, please cite:
```bibtex
@article{li2024effectiveness,
  title={On the Effectiveness of Incremental Training of Large Language Models},
  author={Li, Miles Q and Fung, Benjamin and Huang, Shih-Chia},
  journal={arXiv preprint arXiv:2411.18700},
  year={2024}
}
```

---

## Acknowledgments
- The training and model implementation is built upon [nanoGPT](https://github.com/karpathy/build-nanogpt) by Andrej Karpathy.
- HellaSwag evaluation code is based on [nanoGPT](https://github.com/karpathy/build-nanogpt)  and [hellaswag](https://github.com/rowanz/hellaswag) by Rowan Zellers.

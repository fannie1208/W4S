# Weak-for-Strong: Training Weak Meta-Agent to Harness Strong Executors

[Fan Nie](https://scholar.google.com/citations?user=o2lsU8YAAAAJ&hl=en), [Lan Feng](https://alan-lanfeng.github.io/), [Haotian Ye](https://haotianye.com/), [Weixin Liang](https://ai.stanford.edu/~wxliang/), [Pan Lu](https://lupantech.github.io/), [Huaxiu Yao](https://www.huaxiuyao.io/), [Alexandre Alahi](https://people.epfl.ch/alexandre.alahi?lang=en), [James Zou](https://www.james-zou.com/)

[**Paper**](https://arxiv.org/abs/2504.04785)

---

## Install
Store your API keys in `key.ev`:
```
OPENAI_API_KEY=""
ANTHROPIC_API_KEY=""
```
Then install the environment:
```
conda create -n w4s python=3.11
conda activate w4s
pip install -r requirements.txt
```

---

## 📖 Citation

If you find this work useful, please cite our paper:

```bibtex
@misc{nie2025weakforstrong,
      title={Weak-for-Strong: Training Weak Meta-Agent to Harness Strong Executors}, 
      author={Fan Nie and Lan Feng and Haotian Ye and Weixin Liang and Pan Lu and Huaxiu Yao and Alexandre Alahi and James Zou},
      year={2025},
      eprint={2504.04785},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2504.04785}, 
}
```

## Acknowledgment

We thank ADAS and AFlow for their codebase and prompts.

# *Any*Body: A Benchmark Suite for Cross-Embodiment Manipulation

This repository contains code for the [AnyBody benchmark suite](https://arxiv.org/abs/2505.14986). It includes a configurable robot set, standardized benchmark scenarios, and training pipelines for RL algorithms. The benchmark is based on [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim). 

This repository includes:

* [**Installation**](docs/installation.md)
* [**Benchmark**](docs/benchmark.md)
* [**RL training**](docs/train.md)
* [**Cluster Guide**](docs/cluster.md)

## Code Attribution

Parts of this codebase are adapted from the following repositories:

- [Isaac Lab](https://github.com/isaac-sim/IsaacLab): Much of the utilities (VSCode, Docker, Cluster, SKRL wrapper) are based on the Isaac Lab repository. The Isaac Lab framework is released under BSD-3 License.
- [SKRL](https://github.com/Toni-SM/skrl): Multi-PPO and buffer implementations. SKRL is released under the MIT License.
- [MetaMorph](https://github.com/agrimgupta92/metamorph): Morphology-aware transformer architecture and config system.  
  Gupta, Agrim, Linxi (Jim) Fan, Surya Ganguli, and Li Fei-Fei. "MetaMorph: Learning Universal Controllers with Transformers." arXiv preprint arXiv:2203.11931 (2022).

We thank the authors of these repositories, which have been instrumental in building this benchmark suite.

## Ongoing Development
This project is under active development. Specifically:
- The training and evaluation configs (and commands) in this repo are provided as examples for running policies. The exact hyperparameters used in the paper’s experiments will be released soon.

## Citation

If you use this repository in your work, please consider citing:

```
@misc{parakh2025anybodybenchmarksuitecrossembodiment,
      title={AnyBody: A Benchmark Suite for Cross-Embodiment Manipulation}, 
      author={Meenal Parakh and Alexandre Kirchmeyer and Beining Han and Jia Deng},
      year={2025},
      eprint={2505.14986},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2505.14986}, 
}
```

## Contact

For issues, open an Issue or contact [meenalp@princeton.edu](mailto:meenalp@princeton.edu).

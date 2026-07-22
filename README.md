<h1 align="center">Nikita Sviridov</h1>
<p align="center"><b>ML/NLP Engineer · Research-to-Production ML</b></p>

<p align="center">
  <a href="mailto:nikswirlo@gmail.com"><img src="https://img.shields.io/badge/Email-nikswirlo@gmail.com-15192b?style=flat-square&logo=gmail&logoColor=white"></a>
  <a href="https://github.com/nikswir"><img src="https://img.shields.io/badge/GitHub-nikswir-15192b?style=flat-square&logo=github&logoColor=white"></a>
  <a href="https://t.me/n1ksw1r"><img src="https://img.shields.io/badge/Telegram-@n1ksw1r-2CA5E0?style=flat-square&logo=telegram&logoColor=white"></a>
  <a href="https://www.linkedin.com/in/n1ksw1r/"><img src="https://img.shields.io/badge/LinkedIn-n1ksw1r-0A66C2?style=flat-square&logo=linkedin&logoColor=white"></a>
</p>

<p align="center">
  <a href="cv/Nikita_CV_EN.pdf"><img src="https://img.shields.io/badge/Download%20CV-EN-4338ca?style=for-the-badge"></a>
  &nbsp;
  <a href="cv/Nikita_CV_RU.pdf"><img src="https://img.shields.io/badge/Скачать%20CV-RU-0d9488?style=for-the-badge"></a>
</p>

---

## Repositories

### [model-trainer](https://github.com/nikswir/model-trainer)

Configurable training harness for BERT-like and GPT-like models built with Hydra and PyTorch Lightning.
The project focuses on reusable experiment assembly: datasets, models, losses, optimizers, schedulers,
callbacks, loggers and task adapters are wired through YAML configs.

### [Myohuman](https://github.com/nikswir/Myohuman) · [MyoTrainer](https://github.com/nikswir/MyoTrainer)

Full-body musculoskeletal human model in MuJoCo and a training pipeline for biomechanical control.
The environment contains a detailed body model with muscles and degrees of freedom, while the trainer
contains PPO/Transformer-based experiments for motion-control policies.

### [multiagent-navigation](https://github.com/nikswir/multiagent-navigation) · [mobile-robot-navigation](https://github.com/nikswir/mobile-robot-navigation)

Reinforcement-learning navigation environments for mobile robots with randomized obstacles, goals and
LiDAR observations. The repositories cover both single-robot and multi-robot setups using DDPG/TD3-style
control policies.

### [pinn-implementation](https://github.com/nikswir/pinn-implementation)

Automatic differentiation engine implemented from scratch in NumPy and used to train a Physics-Informed
Neural Network on the heat equation. The project includes low-level memory management and Numba/CUDA
experiments for accelerating computations.

### [neuron-style-transfer](https://github.com/nikswir/neuron-style-transfer)

Neural style transfer implementation in PyTorch using image optimization. The project compares frozen
CNN feature extractors and combines content, style and total-variation losses.

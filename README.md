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

I am an **ML/NLP Engineer** with 3+ years of experience building NLP and LLM systems, including
LLM safety models, agentic workflows, model fine-tuning, hallucination detection, and
production-oriented ML pipelines. I have a strong **physics and mathematical background**
from Lomonosov Moscow State University, with hands-on experience moving research ideas
from experiments to deployed ML systems.

## Core Stack

`ML/NLP` PyTorch · PyTorch Lightning · Transformers · BERT · GPT · LLM Finetuning · PEFT (LoRA) · DDP/FSDP · Mixed Precision · W&B · LangChain · LangGraph · ReAct · vLLM · Ollama · scikit-learn · Gradient Boosting · Logistic Regression · RL (PPO/TD3/DDPG) · Linear Algebra · Mathematical Analysis · Probability & Statistics  
`Engineering Tooling` Git · uv · Hydra · Ruff · mypy · pytest · Hypothesis · pre-commit hooks · GitHub Actions CI · Claude Code CLI · Async

## Projects

### 2026 · Thesis — Control of a Human Biomechanical Model

[Myohuman](https://github.com/nikswir/Myohuman) · [MyoTrainer](https://github.com/nikswir/MyoTrainer)

Developed the MyoHuman MuJoCo environment: built the first open full-body musculoskeletal human model
with `45 bodies`, `86 DoF` and `338 muscles`, prepared an AMASS/KIT dataset with `2,822 trajectories`
(`~7.5 h`) and wrote a PPO training harness. Reimplemented key SOTA solutions for musculoskeletal
control, proposed a 3-stage MLP-policy training scheme and distilled it into a Transformer policy.

**Result:** `0.45 success rate` · `0.68 frame coverage` · `6.1 cm MPJPE`  
**Stack:** `MyoSuite` `MuJoCo` `PPO` `Transformer` `Mixture of Experts` `Hard Negative Mining` `Curriculum Learning`

### 2026–present · Training Harness for BERT-like & GPT-like Models

[model-trainer](https://github.com/nikswir/model-trainer)

Training harness built on Hydra + PyTorch Lightning. The core idea is configurable ML-pipeline assembly:
datasets, model, loss, optimizer, scheduler, callbacks, logger, metrics and task adapters are described
in YAML configs and created via Hydra instantiate. This makes it easy to assemble new experiments, add
new components and reuse existing ones without rewriting pipeline code.

**Result:** `battle-tested for fine-tuning BERT-like classifiers` · `full SFT of Gemma 4B`  
**Stack:** `PyTorch Lightning` `Hydra` `DDP/FSDP` `Mixed Precision` `W&B` `Torchmetrics` `Transformers`

### 2025 · Multi-robot Navigation with a Shared TD3 Policy

[multiagent-navigation](https://github.com/nikswir/multiagent-navigation)

Implemented a multi-robot navigation environment: each episode randomly samples positions for 1 to 8
robots in an obstacle field, and every robot receives its own goal. Each robot observes its state through
LiDAR readings. A shared TD3 control policy is trained and applied independently to every robot without
inter-agent communication; task difficulty is increased stage by stage by adding more robots to the environment.

**Result:** `96.7% goal-reaching rate across robots at n=3` · `92.4% at n=8`  
**Stack:** `TD3` `Gym` `PyTorch` `RL` `MLP` `Curriculum learning`

### 2024 · Mobile Robot Navigation in Randomized Environments

[mobile-robot-navigation](https://github.com/nikswir/mobile-robot-navigation)

Implemented a Gym environment for a mobile robot: each episode randomly generates obstacles, the start
position and the goal, while the agent observes the scene through 8 LiDAR sensors. Trained an MLP-based
DDPG policy for robot control and added a point-of-interest heuristic that selects an intermediate sub-goal
from LiDAR observations.

**Result:** `74.7% successful-episode rate on a held-out evaluation set`  
**Stack:** `DDPG` `PyTorch` `Gym` `RL` `MLP` `POI heuristic`

### 2023 · Autograd Engine & Physics-Informed Neural Network

[pinn-implementation](https://github.com/nikswir/pinn-implementation)

Built an automatic differentiation system from scratch with all functionality required to train a PINN on
the heat equation — a second-order PDE. The backend was implemented in pure NumPy: memory is allocated in
a single 1D buffer and distributed by a memory manager; Numba/CUDA kernels were written to move computations
from CPU to GPU.

**Result:** `PINN converged successfully` · `max deviation up to 5% vs. analytic solution`  
**Stack:** `NumPy` `Numba/CUDA` `Autograd` `PINN`

### 2022 · Neural Style Transfer

[neuron-style-transfer](https://github.com/nikswir/neuron-style-transfer)

Implemented the neural style transfer method through image optimization. The objective combined content loss,
style loss based on Gram matrices and TV regularization for smoother outputs. Compared VGG16, VGG19 and
ResNet50 as frozen feature extractors on the same content/style pairs.

**Result:** `VGG19 produced the best transfer quality` · `visual results are in the repository`  
**Stack:** `PyTorch` `VGG` `ResNet` `CNN feature extraction`

# TAAN-GRPO: Type-Aware Advantage Normalization for Multi-Task RL Training

> **One-line summary**: TAAN adds a second-level normalization on top of GRPO's within-group normalization, balancing gradient contributions across task types (math/code/writing/chat) to prevent high-SNR tasks from dominating multi-task RL fine-tuning.

---

## Algorithm

### GRPO Baseline

For each prompt with G sampled responses and rewards $r_1, \ldots, r_G$:

$$A_i^{\text{GRPO}} = \frac{r_i - \bar{r}}{\text{std}(r, \text{ddof}=1) + \varepsilon}$$

When all rewards are equal ($\text{std} = 0$), $A_i^{\text{GRPO}} = 0$.

### TAAN Extension

After GRPO normalization, group advantages by task type $\tau$ and apply a second normalization:

$$A_i^{\text{TAAN}} = \frac{A_i^{\text{GRPO}} - \hat{\mu}_\tau}{\hat{\sigma}_\tau + \varepsilon}$$

Where $\hat{\mu}_\tau$ and $\hat{\sigma}_\tau$ are **EMA-tracked** (bias-corrected) location/scale estimates per type:

$$\hat{\mu}_\tau^{(t)} = \frac{\alpha \hat{\mu}_\tau^{(t-1)} + (1-\alpha)\mu_\tau^{\text{batch}}}{1 - \alpha^t}$$

$$\hat{\sigma}_\tau^{(t)} = \frac{\sqrt{\alpha \hat{v}_\tau^{(t-1)} + (1-\alpha)(\sigma_\tau^{\text{batch}})^2}}{\sqrt{1 - \alpha^t}}$$

### Three Robustness Enhancements

1. **Robust statistics**: When a type has ≥ 20 samples in a batch, use median + IQR × 0.7413 instead of mean + std to resist outliers.
2. **EMA with bias correction**: Adam-style bias correction prevents under-estimation in early training steps.
3. **Advantage clipping**: $A_i^{\text{final}} = \text{clip}(A_i^{\text{TAAN}}, -c, c)$ with $c = 5.0$ by default.

---

## Quick Start

```bash
pip install -e ".[train]"

# Single GPU
python scripts/train.py --config configs/qwen3_4b_multitask.yaml

# Multi-GPU (4 GPUs)
torchrun --nproc_per_node=4 scripts/train.py --config configs/qwen3_4b_multitask.yaml
```

---

## Repository Structure

```
taan-grpo/
├── README.md
├── pyproject.toml
├── configs/
│   ├── base_config.yaml               # Base training config
│   ├── qwen3_4b_multitask.yaml        # Qwen3-4B multi-task config
│   └── task_types.yaml                # Task type definitions & reward mappings
├── src/
│   ├── taan/
│   │   ├── advantage.py               # Core: TAAN advantage normalization
│   │   ├── ema_tracker.py             # EMA statistics tracker
│   │   ├── robust_stats.py            # Robust statistics (median, IQR, MAD)
│   │   └── type_registry.py           # Type registry
│   ├── grpo/
│   │   ├── trainer.py                 # GRPO+TAAN training loop
│   │   ├── policy_loss.py             # Policy loss (clip + KL penalty)
│   │   └── rollout.py                 # vLLM sampling rollout
│   ├── rewards/
│   │   ├── base.py                    # Reward function base class
│   │   ├── math_reward.py             # Math rule reward (0/1)
│   │   ├── code_reward.py             # Code execution reward (pass/fail)
│   │   └── model_reward.py            # Reward model scoring (continuous)
│   ├── data/
│   │   ├── multitask_dataset.py       # Multi-task mixed dataset
│   │   └── task_sampler.py            # Type-aware sampler
│   └── utils/
│       ├── distributed.py             # Distributed AllReduce utilities
│       └── logging.py                 # Training logging & metrics
├── scripts/
│   ├── train.py                       # Training entry point
│   ├── eval.py                        # Evaluation script
│   └── prepare_data.py                # Data preparation
└── tests/
    ├── test_taan_advantage.py         # TAAN unit tests
    ├── test_robust_stats.py           # Robust statistics tests
    └── test_ema_tracker.py            # EMA tracker tests
```

---

## Configuration

Key parameters in `configs/qwen3_4b_multitask.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `G` | 8 | Samples per prompt |
| `batch_size` | 128 | Prompts per step |
| `taan_alpha` | 0.99 | EMA decay coefficient |
| `taan_clip` | 5.0 | Advantage clip threshold |
| `taan_min_samples` | 20 | Min samples for robust statistics |
| `clip_eps` | 0.2 | PPO clip range |
| `beta` | 0.01 | KL penalty coefficient |
| `lr` | 1e-6 | Learning rate |

---

## Experimental Results

> TODO: Add benchmark results comparing GRPO vs TAAN-GRPO on math, code, and writing tasks.

---

## Citation

If you use this code, please cite:

```bibtex
@article{taan2024,
  title     = {Type-Aware Advantage Normalization for Multi-Task Reinforcement Learning from Human Feedback},
  author    = {TBD},
  year      = {2024},
}

@article{deepseekmath2024,
  title     = {DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models},
  author    = {Shao, Zhihong and Wang, Peiyi and Zhu, Qihao and Xu, Runxin and Song, Junxian and Bi, Xiao and Zhang, Haowei and Zhang, Mingchuan and Li, Y.K. and Wu, Y. and Guo, Daya},
  journal   = {arXiv preprint arXiv:2402.03300},
  year      = {2024},
}
```

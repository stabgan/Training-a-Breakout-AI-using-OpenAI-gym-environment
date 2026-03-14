# Breakout AI — A3C with LSTM

Training an AI agent to play Atari Breakout using Asynchronous Advantage Actor-Critic (A3C) and LSTM, built with PyTorch and OpenAI Gym.

## Overview

This project implements the A3C reinforcement learning algorithm from [DeepMind's seminal paper](https://arxiv.org/abs/1602.01783) to train an agent that learns to play Breakout. Instead of a single agent training for weeks, A3C runs multiple agents in parallel — each exploring independently and sharing gradients with a central model.

## Architecture

The model (`model.py`) combines:

- **4 convolutional layers** (32 filters each, 3×3 kernels, stride 2) with ELU activations — processes 42×42 grayscale frames
- **LSTM cell** (256 hidden units) — captures temporal dependencies across frames
- **Actor head** — outputs action probabilities (policy π)
- **Critic head** — outputs state value V(s)

Training uses Generalized Advantage Estimation (GAE) with entropy regularization to encourage exploration.

### Hyperparameters

| Parameter | Value |
|---|---|
| Learning rate | 0.0001 |
| Discount (γ) | 0.99 |
| GAE (τ) | 1.0 |
| Parallel workers | 16 |
| Steps per update | 20 |
| Max episode length | 10,000 |
| Gradient clipping | 40 |

## Project Structure

```
Breakout/Code_With_Comments/
├── main.py        # Entry point — spawns training and test processes
├── model.py       # ActorCritic network (CNN + LSTM + actor/critic heads)
├── train.py       # A3C training loop with GAE
├── test.py        # Evaluation agent (greedy policy, records video)
├── envs.py        # Gym environment wrappers (frame preprocessing)
├── my_optim.py    # SharedAdam optimizer for cross-process gradient sharing
└── test/          # Recorded gameplay videos (mp4)
```

## Dependencies

- Python 2.7+ or 3.x
- PyTorch (0.3.x or 0.4.x era — see deprecation notes below)
- OpenAI Gym with Atari environments
- OpenCV (`cv2`)
- NumPy

```bash
pip install torch gym gym[atari] opencv-python numpy
```

## Usage

```bash
cd Breakout/Code_With_Comments
python main.py
```

This spawns 16 training workers + 1 test worker. The test agent periodically evaluates the shared model and saves gameplay videos to `test/`.

## Known Issues and Deprecations

This code was written circa 2018 against older versions of PyTorch and Gym. Running it on modern versions will require fixes:

**PyTorch deprecations:**

- `Variable` is deprecated since PyTorch 0.4. Tensors now track gradients natively — all `Variable(...)` wrapping can be removed.
- `volatile=True` (used in `test.py`) is removed. Use `torch.no_grad()` context manager instead.
- `F.softmax(action_values)` and `F.log_softmax(action_values)` require an explicit `dim` argument in modern PyTorch (e.g., `dim=1`).
- `prob.multinomial()` should be `prob.multinomial(num_samples=1)`.
- `torch.nn.utils.clip_grad_norm` is renamed to `clip_grad_norm_` (with trailing underscore).
- `exp_avg.mul_(beta1).add_(1 - beta1, grad)` — the alpha/value two-arg form of `add_`, `addcmul_`, and `addcdiv_` is removed. Use `exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)` style instead.

**OpenAI Gym deprecations:**

- `Breakout-v0` is removed in recent Gym/Gymnasium versions. Use `BreakoutNoFrameskip-v4` or `ALE/Breakout-v5`.
- `env.seed()` is deprecated. Pass `seed` to `gym.make()` instead.
- `gym.wrappers.Monitor` is removed. Use `gym.wrappers.RecordVideo`.
- The `_observation` method in custom wrappers should be renamed to `observation`.
- `Box(0.0, 1.0, [1, 42, 42])` — the shape argument should use `shape=` keyword and be a tuple.

**Functional bugs:**

- In `test.py`, `action[0, 0]` will fail if `action` is 1D. The `.data.numpy()` call on the multinomial result may need reshaping depending on PyTorch version.
- `ensure_shared_grads` in `train.py` returns early if *any* shared param already has a gradient, which may skip gradient sharing for remaining parameters.

## References

- [Asynchronous Methods for Deep Reinforcement Learning (Mnih et al., 2016)](https://arxiv.org/abs/1602.01783)
- [Playing Atari with Deep Reinforcement Learning (Mnih et al., 2013)](https://arxiv.org/abs/1312.5602)

## License

MIT — Kaustabh Ganguly, 2018

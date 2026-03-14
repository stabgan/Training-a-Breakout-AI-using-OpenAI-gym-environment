# 🎮 Breakout AI — A3C with LSTM

Train an AI agent to play Atari Breakout using Asynchronous Advantage Actor-Critic (A3C) with PyTorch and OpenAI Gym.

## What It Does

Implements the [A3C algorithm](https://arxiv.org/abs/1602.01783) to train a reinforcement learning agent on Breakout. Multiple agents explore in parallel across CPU cores, sharing gradients with a central model for fast, stable learning.

## Architecture

```
Input (42×42 grayscale frames)
  → 4× Conv2d (32 filters, 3×3, stride 2, ELU)
  → LSTMCell (288 → 256 hidden)
  ├→ Critic head → V(s)       (state value)
  └→ Actor head  → π(a|s)     (action probabilities)
```

Training uses Generalized Advantage Estimation (GAE) with entropy regularization. A custom `SharedAdam` optimizer enables cross-process gradient sharing.

| Hyperparameter | Value |
|---|---|
| Learning rate | 0.0001 |
| Discount (γ) | 0.99 |
| GAE (τ) | 1.0 |
| Workers | 16 |
| Steps/update | 20 |
| Gradient clip | 40 |

## Project Structure

```
Breakout/Code_With_Comments/
├── main.py        # Entry point — spawns 16 training + 1 test process
├── model.py       # ActorCritic network (CNN + LSTM)
├── train.py       # A3C training loop with GAE
├── test.py        # Greedy evaluation agent, records video
├── envs.py        # Gym wrappers (frame resize, normalization)
├── my_optim.py    # SharedAdam optimizer
└── test/          # Recorded gameplay videos
```

## Getting Started

### Install dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install torch gymnasium gymnasium[atari] gymnasium[accept-rom-license] opencv-python numpy
```

### Run training

```bash
cd Breakout/Code_With_Comments
python main.py
```

The test agent evaluates periodically and saves gameplay videos to `test/`.

## 🛠 Tech Stack

| | Technology |
|---|---|
| 🧠 | **PyTorch** — neural network and autograd |
| 🕹️ | **Gymnasium** — Atari environment |
| 👁️ | **OpenCV** — frame preprocessing |
| 🔢 | **NumPy** — numerical operations |
| ⚡ | **Python multiprocessing** — parallel A3C workers |

## ⚠️ Known Issues

- **CPU-only**: A3C uses shared memory multiprocessing, which doesn't map cleanly to GPU. Training is CPU-bound and can be slow.
- **Atari ROM license**: You need to accept the Atari ROM license (`gymnasium[accept-rom-license]`) for the environment to work.
- **Long training time**: Expect several hours to see meaningful improvement. The test agent sleeps 60s between evaluations.

## References

- [Asynchronous Methods for Deep Reinforcement Learning (Mnih et al., 2016)](https://arxiv.org/abs/1602.01783)
- [Playing Atari with Deep Reinforcement Learning (Mnih et al., 2013)](https://arxiv.org/abs/1312.5602)

## License

MIT

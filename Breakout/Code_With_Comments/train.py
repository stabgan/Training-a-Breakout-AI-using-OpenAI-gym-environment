# Training the AI

import torch
import torch.nn.functional as F
from envs import create_atari_env
from model import ActorCritic


# Implementing a function to make sure the models share the same gradient
def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def train(rank, params, shared_model, optimizer):
    torch.manual_seed(params.seed + rank)
    env = create_atari_env(params.env_name)
    model = ActorCritic(env.observation_space.shape[0], env.action_space)
    state, _ = env.reset()
    state = torch.from_numpy(state)
    done = True
    episode_length = 0
    while True:
        episode_length += 1
        model.load_state_dict(shared_model.state_dict())
        if done:
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
        else:
            cx = cx.detach()
            hx = hx.detach()
        values = []
        log_probs = []
        rewards = []
        entropies = []
        for step in range(params.num_steps):
            value, action_values, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))
            prob = F.softmax(action_values, dim=1)
            log_prob = F.log_softmax(action_values, dim=1)
            entropy = -(log_prob * prob).sum(1)
            entropies.append(entropy)
            action = prob.multinomial(num_samples=1).data
            log_prob = log_prob.gather(1, action)
            values.append(value)
            log_probs.append(log_prob)
            state, reward, terminated, truncated, _ = env.step(action.numpy()[0, 0])
            done = terminated or truncated or episode_length >= params.max_episode_length
            reward = max(min(reward, 1), -1)
            if done:
                episode_length = 0
                state, _ = env.reset()
            state = torch.from_numpy(state)
            rewards.append(reward)
            if done:
                break
        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model((state.unsqueeze(0), (hx, cx)))
            R = value.data
        values.append(R)
        policy_loss = 0
        value_loss = 0
        R = R
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = params.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)
            TD = rewards[i] + params.gamma * values[i + 1].data - values[i].data
            gae = gae * params.gamma * params.tau + TD
            policy_loss = policy_loss - log_probs[i] * gae.detach() - 0.01 * entropies[i]
        optimizer.zero_grad()
        (policy_loss + 0.5 * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 40)
        ensure_shared_grads(model, shared_model)
        optimizer.step()

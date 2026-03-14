# Test Agent

import torch
import torch.nn.functional as F
from envs import create_atari_env
from model import ActorCritic
import time
from collections import deque


# Making the test agent (won't update the model but will just use the shared model to explore)
def test(rank, params, shared_model):
    torch.manual_seed(params.seed + rank)
    env = create_atari_env(params.env_name, video=True)
    model = ActorCritic(env.observation_space.shape[0], env.action_space)
    model.eval()
    state, _ = env.reset()
    state = torch.from_numpy(state)
    reward_sum = 0
    done = True
    start_time = time.time()
    actions = deque(maxlen=100)
    episode_length = 0
    while True:
        episode_length += 1
        if done:
            model.load_state_dict(shared_model.state_dict())
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
        else:
            cx = cx.detach()
            hx = hx.detach()
        with torch.no_grad():
            value, action_value, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))
        prob = F.softmax(action_value, dim=1)
        action = prob.max(1)[1].numpy()  # greedy action
        state, reward, terminated, truncated, _ = env.step(action[0])
        done = terminated or truncated
        reward_sum += reward
        if done:
            print("Time {}, episode reward {}, episode length {}".format(
                time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)),
                reward_sum, episode_length))
            reward_sum = 0
            episode_length = 0
            actions.clear()
            state, _ = env.reset()
            time.sleep(60)
        state = torch.from_numpy(state)

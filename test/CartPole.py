import torch.nn as nn
import gym
import torch
import tqdm
import numpy as np
from torch.distributions import Categorical
from collections import deque

class Predictor(nn.Module):
    def __init__(self, state_dim, nacts):
        super(Predictor, self).__init__()

        # feature network
        self.featnet = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        # policy network
        self.pnet = nn.Linear(256, nacts)

    def forward(self, x):
        feat = self.featnet(x)
        logits = self.pnet(feat)
        return logits

    # 动作决策
    def act(self, x):
        with torch.no_grad():
            # print(x)
            logits = self(x)
            # print(logits)
            dist = Categorical(logits=logits)
            return dist.sample().cpu().item()

class ActionBuffer(object):

    def __init__(self, buffer_size):
        super().__init__()
        self.buffer = deque(maxlen=buffer_size)

    def reset(self):
        self.buffer.clear()

    def push(self, state, action, reward, done):
        self.buffer.append((state, action, reward, done))

    def sample(self):
        # 轨迹采样
        state, action, reward, done = \
            zip(*self.buffer)
        reward = np.stack(reward, 0)
        # 计算回报函数
        for i in reversed(range(len(reward)-1)):
            reward[i] = reward[i] + GAMMA*reward[i+1]
        # 减去平均值，提高稳定性
        reward = reward - np.mean(reward)
        return np.stack(state, 0), np.stack(action, 0), reward, np.stack(done, 0)

    def __len__(self):
        return len(self.buffer)

class Trainer(object):

    def __init__(self, model):
        self.model = model


    def train(self, dataset):

        pass


class Tester(object):

    def __init__(self, model):
        self.model = model


    def test(self, dataset):

        pass

def train(buffer, model, optimizer):
    # 获取训练数据
    state, action, reward, _ = buffer.sample()
    state = torch.tensor(state, dtype=torch.float32)
    action = torch.tensor(action, dtype=torch.long)
    reward = torch.tensor(reward, dtype=torch.float32)

    print(state)

    # 计算损失函数
    logits = model(state)
    dist = Categorical(logits=logits)
    lossp = -(reward*dist.log_prob(action)).mean()
    
    optimizer.zero_grad()
    lossp.backward()
    optimizer.step()
    return lossp.item()

BATCH_SIZE = 16
GAMMA = 0.99


buffer = ActionBuffer(BATCH_SIZE)

if __name__ == "__main__":

    env = gym.make("CartPole-v1")

    # print(env.observation_space.shape[0], env.action_space.n)

    model = Predictor(env.observation_space.shape[0], env.action_space.n)

    # print(model.parameters())

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    all_rewards = []
    all_losses = []
    episode_reward = 0
    loss = 0.0
    NSTEPS = 1000000

    state = env.reset()
    state = state[0]
    print(state)
    # state_array = np.array(state)
    # print(state_array)
    for nstep in tqdm.tqdm(range(NSTEPS)):
        state_t = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0)
        print(state_t)
        action = model.act(state_t)
        next_state, reward, done, _, _ = env.step(action)
        buffer.push(state, action, reward, done)
        state = next_state
        episode_reward += reward

        if done:
            print("Episode Reward: ", episode_reward)
            state = env.reset()
            state = state[0]
            all_rewards.append(episode_reward)
            episode_reward = 0

        if done or len(buffer) == BATCH_SIZE:
            loss = train(buffer, model, optimizer)
            buffer.reset()






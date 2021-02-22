import argparse
import gym
import torch

import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

import matplotlib.pyplot as plt


class ObservationBuffer:

    def __init__(self, obssize, size):
        self.observation = np.zeros((*obssize, 3 * size))
        self.size = size
    
    def push(self, obs):
        self.observation = np.concatenate([self.observation[:, :, 3:], obs], axis=2)
        return self.make_input(self.observation)

    def push(self, obs):
        self.observation = np.concatenate([self.observation[:, :, 3:], obs], axis=2)
        return self.make_input(self.observation)

    def make_input(self, obs):
        # return obs / 255
        # img=obs[:,:,-3:]/255
        # print(img.shape)
        # print(np.max(img), np.mean(img), np.min(img))
        # plt.imshow(img)
        # plt.show()
        return obs
        # obs = obs / 255
        # return np.concatenate([modify_obs(i) for i in np.split(obs, self.size, axis=2)], axis=2)
    
    def current_obs(self):
        return self.observation

def modify_obs(img):
    """
    modify the observations, making it easier to handle
    convert img representation to 1 digit / pixel
    """
    # print(img.shape)
    # print(np.max(img), np.mean(img), np.min(img))
    # plt.imshow(img)
    # plt.show()
    input_img = np.expand_dims(np.sum(img, axis=2) / 3, axis=2)

    # plt.imshow(np.sum(img, axis=2) / 3, cmap='gray')
    # plt.show()
    return input_img

class Policy(nn.Module):
    def __init__(self, obssize, num_consec_obs, num_actions):
        super(Policy, self).__init__()

        self.num_actions = num_actions

        w, h = obssize
        self.convnet = nn.Sequential(
            nn.Conv2d(in_channels=num_consec_obs * 3, out_channels=32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten()
            # nn.Linear(22816, 512),
            # nn.ReLU(),
            # nn.Linear(512, 256),
            # nn.ReLU(),
            # nn.Linear(256, 64),
            # nn.ReLU(),
            # nn.Linear(64, num_actions)
        )

        self.saved_log_probs = []
        self.rewards = []
        # self.entropy = []

        self.epsilon = 0.99
        self.exploration = 1

    def forward(self, observation):
        """
        Sample action from agents output distribution over actions.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Unsqueeze to give a batch size of 1.
        state = torch.from_numpy(observation).float().unsqueeze(0).permute(0, 3, 1, 2).to(device) # (1, 12, 210, 160)
        # print(state.shape)

        # forward pass
        logits = self.convnet(state)
        # print(logits.shape); input()
        action_probs = F.softmax(logits, dim=-1)

        # sample action
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        # print("action:", (action))

        # epsilon greedy
        if np.random.rand() < self.exploration:
            action[0] = np.random.randint(action_probs.shape[1])
            # print("exp")
        # else:
            # print('   no exp')
        self.exploration *= self.epsilon
        

        # save log prob
        self.saved_log_probs.append(dist.log_prob(action))

        # save entropy
        # self.entropy.append(dist.entropy())

        # take action
        return action.item()


def finish_episode(optimizer, policy, config):
    """Updates model using REINFORCE."""
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + config.gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns).to('cuda')
    # returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).mean()
    print("GPU Memory Allocated: ", torch.cuda.memory_allocated() >> 20, "MB")
    policy_loss.backward()
    optimizer.step()
    policy.rewards.clear()
    policy.saved_log_probs.clear()
    # policy.entropy.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=10_000)
    parser.add_argument("--num_repeat_action", type=int, default=4)
    parser.add_argument("--reward_threshold", type=int, default=1_000)
    parser.add_argument("--max_steps", type=int, default=10_000)
    parser.add_argument("--num_consec_obs", type=int, default=4)
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        metavar="G",
        help="discount factor (default: 0.99)",
    )
    parser.add_argument(
        "--seed", type=int, default=543, metavar="N", help="random seed (default: 543)"
    )
    parser.add_argument("--render", action="store_true", help="render the environment")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        metavar="N",
        help="interval between training status logs (default: 10)",
    )
    parser.add_argument(
        "--save-model-interval",
        type=int,
        default=250,
        help="interval between saving models.",
    )
    config = parser.parse_args()

    # setup env
    # env = gym.make("CartPole-v1")
    env = gym.make("Seaquest-v0")

    def reset(env):
        env.reset()
        return env.render(mode='rgb_array')
    
    def step(env, action):
        obs, rwd, done, _ = env.step(action)
        obs = env.render(mode='rgb_array')
        return obs, rwd, done, _

    torch.manual_seed(config.seed)
    env.seed(config.seed)

    # setup policy
    num_actions = env.action_space.n
    obs_shape = reset(env).shape[:2]
    policy = Policy(obs_shape, config.num_consec_obs, num_actions)
    if torch.cuda.is_available():
        policy.cuda()

    # optimizer
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)
    eps = np.finfo(np.float32).eps.item()

    # save criteria
    running_reward = 10.0

    # Vanilla Policy Gradient
    for i_episode in range(config.num_episodes):
        observation = reset(env)
        obsbuffer = ObservationBuffer(obs_shape, config.num_consec_obs)
        ep_reward = 0

        for t in range(config.max_steps):
            action = policy(obsbuffer.push(observation))
            reward = 0.0
            # for _ in range(config.num_repeat_action):
            #     if config.render:
            #         env.render()
            #     observation, _reward, done, _ = step(env, action)
            #     print(action)
            #     reward += _reward
            #     if done:
            #         break
            if config.render:
                env.render()
            observation, _reward, done, _ = step(env, action)
            reward += _reward
            policy.rewards.append(reward)
            ep_reward += reward
            if done:
                running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
                finish_episode(optimizer, policy, config)
                if i_episode % config.log_interval == 0:
                    print(
                        "Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}".format(
                            i_episode, ep_reward, running_reward
                        )
                    )
                if running_reward > config.reward_threshold:
                    print(
                        "Solved! Running reward is now {} and "
                        "the last episode runs to {} time steps!".format(
                            running_reward, t
                        )
                    )
                break
    
    torch.save(policy.state_dict(), f"./models/policy-final.pt")
    env.close()
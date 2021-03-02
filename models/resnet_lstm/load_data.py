import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from collections import defaultdict

def load_reward_data(dir):
    data = defaultdict(list)
    for file in os.listdir(dir):
        try:
            with open(f"{dir}/{file}", "rb") as f:
                d = pickle.load(f)
                episode = file.split('.')[0].split('-')[1]
                if episode == "final":
                    episode = 10000
                else:
                    episode = int(episode)
                data[episode] = d
        except:
            pass
    return data

def plot_diagram(data, title,min = 250, max = 10000):
    reward_data = sorted(data.items(), key=lambda x: x[0])
    reward_data = [x[1] for x in reward_data]
    reward_data = np.concatenate(reward_data)
    x = np.arange(min, max + (max // len(reward_data)), max // len(reward_data))
    plt.plot(x, reward_data)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(title)
    plt.show()
    
def preprocess_data(data, dir, min = 250, max = 10000):
    reward_data = sorted(data.items(), key=lambda x: x[0])
    reward_data = [x[1] for x in reward_data]
    reward_data = np.concatenate(reward_data)
    with open(f"{dir}", "w") as file1:
        for e, r in enumerate(reward_data, max // len(reward_data)):
            file1.write(f"{e}, {r}\n")
    

if __name__ == '__main__':
    
    reward_data1 = load_reward_data("model1/out/reward")
    plot_diagram(reward_data1, "Resnet LSTM Model 1")
    
    reward_data2 = load_reward_data("model2/out/reward")
    plot_diagram(reward_data2, "Resnet LSTM Model 2")
    
    # with open("model1/out/reward/model1_data.txt", "w") as file1:
    #     for e, r in sorted(reward_data1.items(), key=lambda x: x[0]):
    #         file1.write(f"{e}, {r}\n")
    #
    #
    # with open("model2/out/reward/model2_data.txt", "w") as file1:
    #     for e, r in sorted(reward_data2.items(), key=lambda x: x[0]):
    #         file1.write(f"{e}, {r}\n")
    
    preprocess_data(reward_data1, "model1/out/reward/model1_data.txt")
    preprocess_data(reward_data2, "model2/out/reward/model2_data.txt")
    
    
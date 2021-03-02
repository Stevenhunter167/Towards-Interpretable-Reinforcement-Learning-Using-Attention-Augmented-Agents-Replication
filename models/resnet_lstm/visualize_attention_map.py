from torchvision import transforms
import torch
from collections import defaultdict

import os, sys
import pickle


def save_status(A, round_num, dir):
    A = A.reshape(14,10, 1).repeat(1,1,3).transpose(0,2)
    
    transforms.Normalize(torch.mean(A), torch.std(A))
    transforms.ToPILImage()(A.detach().cpu()).save(f'{dir}/{round_num}.png')
    
def load_attention_tensor(dir):
    data = defaultdict(torch.tensor)
    for file in os.listdir(dir):
        try:
            with open(f"{dir}/{file}", "rb") as f:
                d = pickle.load(f)
                d = d.reshape((14, 10, 4)).sum(2).unsqueeze(2)
                # d = d.reshape((14, 10, 4))
                episode = file.split('.')[0]
                data[episode] = d
        except EOFError:
            pass
    return data
    


if __name__ == '__main__':
    
    attention_data = load_attention_tensor("model1/out/attention")
    for name, data in attention_data.items():
        save_status(data, name, "model1/out/visualization")

    attention_data = load_attention_tensor("model2/out/attention")
    for name, data in attention_data.items():
        save_status(data, name, "model2/out/visualization")
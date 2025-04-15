import torch

def positional_encoding(i, pos, d_model):
    value = pos / 10000 ** (i / d_model)
    value = torch.tensor(value)  
    if i % 2 == 0:
        return torch.sin(value)
    else:
        return torch.cos(value)
    
i = 2
pos = 10000
d_model = 1
print(positional_encoding(i, pos, d_model))
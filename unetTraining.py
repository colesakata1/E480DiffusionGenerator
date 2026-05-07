import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader 
from DiffusionGenerator.DDPMScheduler import DDPMScheduler
import unet
from timm.utils import ModelEmaV3 
from tqdm import tqdm 
import matplotlib.pyplot as plt 
import torch.optim as optim
from einops import rearrange 
from typing import List
import random
import math

def train(batch_size: int=16,
          num_time_steps: int=1000,
          num_epochs: int=100,
          seed: int=-1,
          ema_decay: float=0.9999,  
          lr=2e-5,
          checkpoint_path: str=None):
    DDPMScheduler.set_seed(random.randint(0, 2**32-1)) if seed == -1 else DDPMScheduler.set_seed(seed)

    train_dataset = datasets.Flowers102(root='./DiffusionGenerator/', split='train', download=True, transform=transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor()]))
    #train_dataset = datasets.MNIST(root='./data', train=True, download=False,transform=transforms.ToTensor())
    #nworkers = os.cpu_count() if os.cpu_count() is not None else 0
    nworkers=0
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=nworkers)

    scheduler = DDPMScheduler(num_time_steps=num_time_steps)
    model = unet.unet()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    ema = ModelEmaV3(model, decay=ema_decay)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['weights'])
        ema.load_state_dict(checkpoint['ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    criterion = nn.MSELoss(reduction='mean')

    for i in range(num_epochs):
        total_loss = 0
        for bidx, (x,_) in enumerate(tqdm(train_loader, desc=f"Epoch {i+1}/{num_epochs}")):
            t = torch.randint(0,num_time_steps,(batch_size,))
            e = torch.randn_like(x, requires_grad=False)
            a = scheduler.alpha[t].view(batch_size,1,1,1)
            x = (torch.sqrt(a)*x) + (torch.sqrt(1-a)*e)
            output = model(x, t)
            optimizer.zero_grad()
            loss = criterion(output, e)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            ema.update(model)
        print(f'Epoch {i+1} | Loss {total_loss / (len(train_loader)/batch_size):.5f}')

        checkpoint = {
        'weights': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'ema': ema.state_dict()
        }
        torch.save(checkpoint, f'./DiffusionGenerator/checkpoints/flwrs_checkpoint5.{i+1}')


def tensor_to_image(tensor: torch.Tensor):
    x = tensor.detach().cpu()
    if x.ndim == 4:
        x = x.squeeze(0)
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    x = x.permute(1, 2, 0)
    x = x - x.min()
    x = x / (x.max() + 1e-7)
    x = x.clamp(0, 1)
    return x.numpy()


def save_image_array(tensor: torch.Tensor, path: str):
    x = tensor_to_image(tensor)
    plt.imsave(path, x)


def display_reverse(images: List):
    if len(images) == 0:
        return
    fig, axes = plt.subplots(1, len(images), figsize=(len(images), 1))
    if len(images) == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        x = tensor_to_image(images[i])
        ax.imshow(x)
        ax.axis('off')
    plt.show()

def inference(checkpoint_path: str=None,
              num_time_steps: int=1000,
              ema_decay: float=0.9999,
              save_images: int=1,
              save_dir: str='./DiffusionGenerator/generated_images'):
    checkpoint = torch.load(checkpoint_path)
    model = unet.unet()
    model.load_state_dict(checkpoint['weights'])
    ema = ModelEmaV3(model, decay=ema_decay)
    ema.load_state_dict(checkpoint['ema'])
    scheduler = DDPMScheduler(num_time_steps=num_time_steps)
    times = [0,15,50,100,200,300,400,550,700,999]
    images = []

    final_dir = os.path.join(save_dir, 'final')
    snapshot_dir = os.path.join(save_dir, 'snapshots')
    os.makedirs(final_dir, exist_ok=True)
    os.makedirs(snapshot_dir, exist_ok=True)

    with torch.no_grad():
        model = ema.module.eval()
        for i in range(15):
            z = torch.randn(1, 3, 64, 64)
            for t in reversed(range(1, num_time_steps)):
                t = [t]
                temp = (scheduler.beta[t]/((torch.sqrt(1 - scheduler.alpha[t]))*(torch.sqrt(1 - scheduler.beta[t]))))
                z = (1/(torch.sqrt(1 - scheduler.beta[t])))*z - (temp*model(z,t).cpu())
                if t[0] in times:
                    images.append(z)
                e = torch.randn(1, 3, 64, 64)
                z = z + (e*torch.sqrt(scheduler.beta[t]))
            temp = scheduler.beta[0]/((torch.sqrt(1 - scheduler.alpha[0]))*(torch.sqrt(1 - scheduler.beta[0])))
            x = (1/(torch.sqrt(1 - scheduler.beta[0])))*z - (temp*model(z,[0]).cpu())

            images.append(x)
            if save_images == 1:
                final_path = os.path.join(final_dir, f'generated_{i:02d}_final.png')
                save_image_array(x, final_path)

                for idx, snapshot in enumerate(images):
                    snapshot_path = os.path.join(snapshot_dir, f'generated_{i:02d}_step_{idx:02d}.png')
                    save_image_array(snapshot, snapshot_path)

                x_image = tensor_to_image(x)
                plt.imshow(x_image)
                plt.axis('off')
                plt.show()


            display_reverse(images)
            images = []

def main(): 
    train(checkpoint_path='./DiffusionGenerator/checkpoints/flwrs_checkpoint4.14', lr=2e-5, num_epochs=100)
    inference('./DiffusionGenerator/checkpoints/flwrs_checkpoint5.100')

if __name__ == '__main__':
    main()
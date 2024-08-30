import sys
sys.path.append('..')

import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary
import model_zoo
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from . import kornia_transform
import numpy as np

class TranNetWrapper:
    def __init__(self, datapath):
        self.datapath = datapath

        self.aug_library = {
            'g': [
                kornia_transform.RandomPerspective(0.5, "nearest", align_corners=True, same_on_batch=False, keepdim=False, p=1.0),
                kornia_transform.RandomThinPlateSpline(0.3, align_corners=True, same_on_batch=False, keepdim=False, p=1.0),
                kornia_transform.RandomResize(0.9, p=1.0), # interpolation
                kornia_transform.RandomAffine((-1.0, 5.0), (0.3, 1.0), (0.4, 1.3), 0.5, resample="nearest", # resample
                    padding_mode="reflection", align_corners=True, same_on_batch=False, keepdim=False, p=1.0,),
                kornia_transform.RandomShear((-5., 2., 5., 10.), same_on_batch=False, keepdim=False, p=1.0),
                kornia_transform.RandomRotation(15.0, "nearest", align_corners=True, same_on_batch=False, keepdim=False, p=1.0), # interpolation
            ],
            'c': [
                kornia_transform.RandomHorizontalFlip(same_on_batch=False, keepdim=False, p=1.0),
                kornia_transform.RandomVerticalFlip(same_on_batch=False, keepdim=False, p=1.0),
                kornia_transform.RandomErasing(scale=(0.01, 0.04), ratio=(0.3, 1.0), value=1, same_on_batch=False, keepdim=False, p=1.0),
                kornia_transform.RandomGaussianBlur((21, 21), (0.2, 1.3), 'reflect', same_on_batch=False, keepdim=False, p=1.0),
                kornia_transform.RandomGaussianNoise(mean=0.1, std=0.1, same_on_batch=False, keepdim=False, p=1.0),
                kornia_transform.RandomMotionBlur((7, 7), 35.0, 0.5, 'reflect', 'nearest', same_on_batch=False, keepdim=False, p=1.0),
            ]
        }

        self.composers = [
            transforms.Compose([
            transforms.Resize([299, 299]),
            transforms.ToTensor()
            ]),

            transforms.Compose([
                transforms.Resize([299, 299]),
                transforms.ToTensor()
            ])
        ]

        self.dataset = datasets.ImageFolder(self.datapath, self.composers[0])
        self.candidates = np.arange(len(self.aug_library['g']))

    def trans_model(self, hiddens=[16, 3]):
        layers = []
        in_channels = 3
        for i, hidden in enumerate(hiddens):
            layer = nn.Conv2d(in_channels, hidden, kernel_size=(3, 3), padding=(1, 1))
            if i + 1 < len(hiddens):
                act = nn.LeakyReLU()
            else:
                act = nn.Sigmoid()

            layers.append(layer)
            layers.append(act)
            in_channels = hidden

        model = nn.Sequential(
            *layers
        )
        return model

    def freeze(self, model):
        for named, params in model.named_parameters():
            params.requires_grad = False

    def train(self, surrogate_model, tmodel, epochs=100, batch_size=64, k=1, lr=0.01):
        data_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(tmodel.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        for epoch in range(epochs):
            for imgs, labs in data_loader:
                imgs = imgs.cuda()
                # calculate targets
                choices = np.random.choice(self.candidates, (k,), replace=False)
                with torch.no_grad():
                    tgt_logits = 0
                    for choice in choices:
                        new_imgs = self.aug_library['g'][choice](imgs)
                        new_imgs = self.aug_library['c'][choice](new_imgs)
                        tgt_logits = tgt_logits + surrogate_model(new_imgs)

                        new_imgs = self.aug_library['c'][choice](imgs)
                        new_imgs = self.aug_library['g'][choice](new_imgs)
                        tgt_logits = tgt_logits + surrogate_model(new_imgs)
                        tgt_logits = tgt_logits / (2 * k)

                new_imgs = tmodel(imgs)
                pred_logits = surrogate_model(new_imgs)
                loss = loss_fn(pred_logits, tgt_logits)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if epoch % 10 == 0 or (epoch + 1) == epochs:
                print('epoch:', epoch, 'loss:', loss.item())

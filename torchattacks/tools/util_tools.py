#!/usr/bin/env python
# coding=utf-8
import torch

def is_on_gpu(model):
    device = next(model.parameters()).device
    return device.type == 'cuda'

def model_device(model):
    device = next(model.parameters()).device
    return device

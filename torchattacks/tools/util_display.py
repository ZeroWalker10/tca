#!/usr/bin/env python
# coding=utf-8
import torch
import cv2
import numpy as np

def visualize_cam(mask, img):
    # mask (channel, h, w)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask.squeeze(0).permute(1, 2, 0)), cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b])
    
    result = heatmap + img.cpu()
    result = result.div(result.max()).squeeze()
    
    return heatmap, result

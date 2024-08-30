#!/usr/bin/env python
# coding=utf-8
import sys
sys.path.append('..')

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from ..attack import Attack
from ..tools import kornia_transform, translation_map, transform_selection
import pdb, pickle

class TCA(Attack):
    def __init__(self, model, aug_library=None, eps=8/255, alpha=2/255, steps=10, decay=1.0, n=10, 
            tran_map_path=[], fix1=None, fix2=None, fix3=None, T=0.2, tran_mode=None, r=1, d=1,
            depth_T={}, depth_probs={}):
        super().__init__("TCA", model)
        self.fix1 = fix1
        self.fix2 = fix2
        self.fix3 = fix3
        self.eps = eps
        self.steps = steps
        self.alpha = alpha 
        self.supported_mode = ["default", "targeted"]
        self.decay = decay
        self.alpha = alpha
        self.epsilon = 1e-8

        self.n = n 
        self.T = T
        self.tran_map_path = tran_map_path
        self.tran_mode = tran_mode 
        self.r = r
        self.d = d
        self.tran_selection = transform_selection.TransformSelection(p=1.0)
        self.tran_lib = self.tran_selection.aug_library['tran']

        tran_seqs = {}
        if len(self.tran_map_path) > 0:
            for tpath in self.tran_map_path:
                with open(tpath, 'rb') as fp:
                    tseqs = pickle.load(fp)

                for d, nodes in tseqs.items():
                    if d in tran_seqs:
                        tran_seqs[d].update(nodes)
                    else:
                        tran_seqs[d] = nodes.copy()

            self.tmap = translation_map.TranslationMap(tran_seqs, self.tran_selection.aug_library['tran'], T=T,
                    depth_T=depth_T)
        else:
            self.tmap = translation_map.TranslationMap(trans_lib=self.tran_selection.aug_library['tran'],
                    T=T, depth_T=depth_T)

        self.depth_prob_choices = [k for k, v in depth_probs.items()]
        self.depth_prob_dists = [v for k, v in depth_probs.items()]

    def _sigvar(self, x, alpha=1.0):
        return 1.0 / (1.0 + torch.exp(-x * alpha))

    def coeff(self, values, a=1.0):
        values_exp = torch.exp(values.abs())
        return a * (values_exp - 1.0) / (values_exp + 1.0)

    def forward(self, images, labels, *args, **kwargs):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)
            source_labels = kwargs.get('source_labels', None)
        else:
            source_labels = labels


        pred_loss = torch.nn.CrossEntropyLoss()

        adv_images = images.clone().detach()
        for i in range(len(images)):
            src_image = images[i]
            adv_image = adv_images[i].unsqueeze(0)

            source_label = source_labels[i]
            target_label = target_labels[i]
            s_label = source_label.unsqueeze(0)
            t_label = target_label.unsqueeze(0)
            pos_labels = torch.LongTensor([int(target_label)] * (self.n)).to(self.device)
            neg_labels = torch.LongTensor([int(source_label)] * (self.n)).to(self.device)

            src_delta_lower_bound = -torch.minimum(src_image.cpu(), torch.Tensor([self.eps])).to(self.device)
            src_delta_upper_bound = torch.minimum(1.0 - src_image.cpu(), torch.Tensor([self.eps])).to(self.device)
            src_delta_bound = (src_delta_upper_bound - src_delta_lower_bound).unsqueeze(0)
            uniform_dist = torch.distributions.Uniform(src_delta_lower_bound, src_delta_upper_bound)

            delta = uniform_dist.sample().unsqueeze(0)
            m, v = 0.0, 0.0
            for step in range(self.steps):
                delta = delta.detach()
                delta.requires_grad = True
                adv_image = src_image + delta

                if self.fix1 is None and self.fix2 is None:
                    aug_adv_images = []
                    pick_tran_seqs = []
                    for k in range(self.n):
                        if self.tran_mode == 'random':
                            aug_adv_image, tran_seq = self.tmap.random_translate(adv_image, r=self.r, have_list=pick_tran_seqs)
                        elif self.tran_mode == 'index':
                            aug_adv_image, tran_seq = self.tmap.index_translate(adv_image, k, have_list=pick_tran_seqs)
                        elif self.tran_mode == 'priority':
                            aug_adv_image, tran_seq = self.tmap.priority_translate(adv_image, have_list=pick_tran_seqs)
                        elif self.tran_mode == 'hamburger':
                            depth = np.random.choice(self.depth_prob_choices,
                                    p=self.depth_prob_dists)
                            self.tmap.set_depth(depth)

                            aug_adv_image, tran_seq = self.tmap.hamburger_translate(adv_image, have_list=pick_tran_seqs)
                        elif self.tran_mode == 'depth':
                            aug_adv_image, tran_seq = self.tmap.depth_translate(adv_image, d=self.d, have_list=pick_tran_seqs)
                        else:
                            aug_adv_image, tran_seq = self.tmap.translate(adv_image, have_list=pick_tran_seqs)

                        aug_adv_images.append(aug_adv_image)
                        pick_tran_seqs.append(tran_seq)

                    aug_adv_images = torch.cat(aug_adv_images)
                    '''
                    if self.tran_mode == 'hamburger':
                        for di in range(1, len(pick_tran_seqs[0])):
                            aug_adv_images = torch.cat([
                                self.tmap.translate(aug_adv_images[[k]],
                                    (-1, pick_tran_seqs[k][di]))[0]
                                for k in range(self.n)
                                ])
                    '''

                    new_aug_adv_images = aug_adv_images
                elif self.fix1 is not None and self.fix2 is None:
                    new_aug_adv_images = torch.cat([
                        self.tran_lib[self.fix1](adv_image.clone())
                        for k in range(self.n)
                    ])
                else:
                    if self.fix3 is not None:
                        aug_adv_images = torch.cat([
                            self.tran_lib[self.fix3](adv_image.clone())
                            for k in range(self.n)
                        ])
                    else:
                        aug_adv_images = torch.cat([
                            adv_image.clone()
                            for k in range(self.n)
                        ])

                    aug_adv_images = torch.cat([
                        self.tran_lib[self.fix1](aug_adv_images[[k]])
                        for k in range(self.n)
                    ])

                    new_aug_adv_images = torch.cat([
                        self.tran_lib[self.fix2](aug_adv_images[[k]])
                        for k in range(self.n)
                    ])

                logits = self.model(new_aug_adv_images)
                if self.targeted: 
                    # good
                    fused_logit = logits.mean(0, keepdim=True)
                    pred_cost = pred_loss(fused_logit, t_label)

                    cost = -pred_cost

                grad = torch.autograd.grad(cost, delta, 
                     retain_graph=False, create_graph=False)[0]
                with torch.no_grad():
                    grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)

                    m = grad + m * self.decay
                    delta = delta + self.alpha * m
                    delta = delta.clip(src_delta_lower_bound, src_delta_upper_bound)

            adv_image = torch.clamp(src_image + delta, 0, 1)
            adv_images[i] = adv_image.squeeze(0)
        return adv_images

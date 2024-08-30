import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms as v2
import numpy as np
import pdb
from . import kornia_transform 

class TransformSelection:
    def __init__(self, aug_library=None, p=0.8, alpha=2/255, eps=0.07, check_times=1):
        self.p = p
        self.alpha = alpha
        self.eps = eps
        self.check_times = check_times

        if aug_library is None:
            self.aug_library = {
                'tran': [
                    # kornia_transform.RandomPerspective(0.5, "nearest", align_corners=True, same_on_batch=False, keepdim=False, p=self.aug_p),
                    # kornia_transform.RandomThinPlateSpline(0.3, align_corners=True, same_on_batch=False, keepdim=False, p=self.aug_p),
                    # kornia_transform.RandomResize(0.9, p=self.aug_p), # interpolation
                    # kornia_transform.RandomAffine((-1.0, 5.0), (0.3, 1.0), (0.4, 1.3), 0.5, resample="nearest", # resample
                    #     padding_mode="reflection", align_corners=True, same_on_batch=False, keepdim=False, p=self.aug_p,),
                    # kornia_transform.RandomShear((-5., 2., 5., 10.), same_on_batch=False, keepdim=False, p=self.aug_p),
                    # kornia_transform.RandomRotation(15.0, "nearest", align_corners=True, same_on_batch=False, keepdim=False, p=self.aug_p), # interpolation
                    # kornia_transform.RandomHorizontalFlip(same_on_batch=False, keepdim=False, p=self.aug_p),
                    # kornia_transform.RandomVerticalFlip(same_on_batch=False, keepdim=False, p=self.aug_p, p_batch=self.aug_p),
                    # kornia_transform.RandomErasing(scale=(0.01, 0.04), ratio=(0.3, 1.0), value=1, same_on_batch=False, keepdim=False, p=self.aug_p),
                    # kornia_transform.RandomGaussianBlur((21, 21), (0.2, 1.3), 'reflect', same_on_batch=False, keepdim=False, p=self.aug_p),
                    # kornia_transform.RandomGaussianNoise(mean=0.1, std=0.1, same_on_batch=False, keepdim=False, p=self.aug_p),
                    # kornia_transform.RandomMotionBlur((7, 7), 35.0, 0.5, 'reflect', 'nearest', same_on_batch=False, keepdim=False, p=self.aug_p),
                    self._random_spline(self.p), # 0
                    self._random_hflip(self.p),  # 1
                    self._random_vflip(self.p),  # 2
                    self._random_rotate(self.p), # 3
                    self._random_perspective(self.p), # 4
                    self._random_resize(self.p),      # 5
                    self._random_gaussian_noise(self.p), # 6
                    self._random_gaussian_blur(self.p),  # 7
                    self._random_erasing(self.p),        # 8
                ],
            }
        else:
            self.aug_library = aug_library

    def _random_spline(self, p=0.5):
        return kornia_transform.RandomThinPlateSpline(0.3, align_corners=True, same_on_batch=False, keepdim=False, p=p)

    def _random_hflip(self, p=0.5):
        return v2.RandomHorizontalFlip(p)

    def _random_vflip(self, p=0.5):
        return v2.RandomVerticalFlip(p) 
    
    def _random_rotate(self, p=0.5, degree=30):
        return v2.RandomRotation(degrees=30) 

    def _random_perspective(self, p=0.5, scale=0.15):
        return v2.RandomPerspective(distortion_scale=scale, p=p)

    def _random_gaussian_blur(self, p=0.5, kernel_size=(21, 21)):
        return v2.GaussianBlur(kernel_size=kernel_size)

    def _random_erasing(self, p=0.5, scale=(0.01, 0.04), ratio=(0.3, 1.0)):
        return v2.RandomErasing(p=p, scale=scale, ratio=ratio)

    def _random_resize(self, p=0.5, scale_factor=0.8):
        def resize(imgs):
            if np.random.rand() < p:
                img_size = int(imgs.shape[-1] * scale_factor)
                img_resize = imgs.shape[-1]

                rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
                new_imgs = F.interpolate(
                    imgs, size=[rnd, rnd], mode="bilinear", align_corners=False
                )
                h_rem = img_resize - rnd
                w_rem = img_resize - rnd
                pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
                pad_bottom = h_rem - pad_top
                pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
                pad_right = w_rem - pad_left

                new_imgs = F.pad(
                    new_imgs,
                    [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()],
                    value=0,
                )
                return new_imgs
            else:
                return imgs
        return resize

    def _random_gaussian_noise(self, p=0.5, mean=0.0, std=0.1):
        def noise(imgs):
            if np.random.rand() < p:
                new_imgs = imgs + (torch.randn_like(imgs) * std + mean)
                return torch.clamp(new_imgs, 0, 1)
            else:
                return imgs
        return noise

    def _random_shearx(self, p=0.5, mag=15.0):
        def shearx(imgs):
            if np.random.rand() < p:
                return v2.functional.affine(imgs,
                        angle=0.0, translate=[0, 0],
                        scale=1.0, shear=[mag, 0.0])
            else:
                return imgs

        return shearx

    def _random_sheary(self, p=0.5, mag=15.0):
        def sheary(imgs):
            if np.random.rand() < p:
                return v2.functional.affine(imgs,
                        angle=0.0, translate=[0, 0],
                        scale=1.0, shear=[0.0, mag])
            else:
                return imgs

        return sheary


    def combine(self, n, d):
        for i in range(n):
            if d == 1:
                yield (i,)
            else:
                for j in self.combine(n, d-1):
                    yield (i,) + j

    def translate(self, inputs, tran_seq):
        new_inputs = inputs.clone()
        for tran_i in tran_seq:
            new_inputs = self.aug_library['tran'][tran_i](new_inputs)
        return new_inputs

    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def hanming_similarity(self, a, b):
        return np.count_nonzero(a.astype(np.int32) ^ b.astype(np.int32))

    def trend_similarity(self, a):
        a_min = a.min()
        if a_min < 0.0:
            a_comp = a - a_min
        else:
            a_comp = a
        a_length = len(a)
        trend_base = np.arange(a_length)
        return self.cosine_similarity(a_comp, trend_base)

    '''
    def translation_score(self, model, check_model, inputs, tlabels, n_epoch, tran_seq):
        prev_logits = model(inputs)
        adv_inputs = inputs.detach()
        loss_fn = nn.CrossEntropyLoss()
        m = 0
        score_history = []
        for i in range(n_epoch):
            adv_inputs.requires_grad = True
            logits = model(self.translate(adv_inputs, tran_seq))
            loss = -loss_fn(F.softmax(logits, dim=1), tlabels)
            grad = torch.autograd.grad(loss, adv_inputs, retain_graph=False, create_graph=False)[0]
            grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            m = m + grad
            adv_inputs = adv_inputs + self.alpha * m.sign()
            delta = torch.clamp(adv_inputs - inputs, -self.eps, self.eps)
            adv_inputs = torch.clamp(inputs + delta, 0, 1).detach()

            with torch.no_grad():
                delta_logits = logits - prev_logits
                prev_logits = logits.detach()
                tgt_score = delta_logits[torch.arange(len(tlabels)), tlabels]
                all_score = delta_logits.sum(dim=1)
                score = 2 * tgt_score - all_score
                score_history.append(score.mean().item())

        if tran_seq[0] == 8:
            pdb.set_trace()
        sim = self.trend_similarity(np.array(score_history, dtype=np.float32))
        return sim
    '''

    def _f_beta(self, a, b, beta=0.9):
        return (1 + beta * beta) * a * b / (beta * beta * a + b + 1e-10)

    def translation_score(self, model, check_model, inputs, tlabels, n_epoch, tran_seq, ngrp):
        loss_fn = nn.CrossEntropyLoss()
        score_history = []

        adv_inputs = []
        for inp, tlabel in zip(inputs, tlabels):
            org_inp = inp.unsqueeze(0)
            adv_tlabel = tlabel.unsqueeze(0)

            src_delta_lower_bound = -torch.minimum(org_inp, torch.Tensor([self.eps]).to(org_inp.device))
            src_delta_upper_bound = torch.minimum(1.0 - org_inp, torch.Tensor([self.eps]).to(org_inp.device))
            src_delta_bound = (src_delta_upper_bound - src_delta_lower_bound)
            uniform_dist = torch.distributions.Uniform(src_delta_lower_bound, src_delta_upper_bound)

            delta = uniform_dist.sample()
            m = 0
            single_score_history = []
            for i in range(n_epoch):
                delta = delta.detach()
                delta.requires_grad = True
                adv_inp = org_inp + delta
                adv_inps = torch.cat([
                      self.translate(adv_inp.clone(), tran_seq)
                      for _ in range(ngrp)
                   ])
                logits = model(adv_inps)
                loss = -loss_fn(logits.mean(0, keepdims=True), adv_tlabel)
                grad = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]
                grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
                m = m + grad
                delta = delta + self.alpha * m * 4.0
                delta = delta.clip(src_delta_lower_bound, src_delta_upper_bound)
            adv_inp = torch.clamp(org_inp + delta, 0, 1)
            adv_inputs.append(adv_inp)
        adv_inputs = torch.cat(adv_inputs)

        '''
        adv_inputs = inputs.clone().detach() 
        m = 0.0
        for i in range(n_epoch):
            adv_inputs.requires_grad = True
            logits = model(self.translate(adv_inputs, tran_seq))
            loss = -loss_fn(logits, tlabels)
            grad = torch.autograd.grad(loss, adv_inputs, retain_graph=False, create_graph=False)[0]
            grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            m = m + grad
            adv_inputs = adv_inputs + self.alpha * m.sign()
            delta = torch.clamp(adv_inputs - inputs, -self.eps, self.eps)
            adv_inputs = torch.clamp(inputs + delta, 0, 1).detach()
        '''

        adv_preds = F.softmax(model(adv_inputs), dim=1).argmax(dim=1)
        init_acc = torch.sum(adv_preds == tlabels) / len(tlabels)
        # begin to check
        check_accs = []
        sum_check_acc = 0.0
        sum_check_margin = 0.0
        for _ in range(self.check_times):
            with torch.no_grad():
                check_logits = check_model(adv_inputs)
                check_preds = F.softmax(check_logits, dim=1).argmax(dim=1)
                corrects = check_preds == tlabels
                check_acc = torch.sum(corrects) / len(tlabels)

                check_target_logits = check_logits[torch.arange(len(tlabels)), tlabels]
                all_logits = check_logits.sum(dim=1)
                check_margin = (2 * check_target_logits - all_logits).mean()

                # best_logits = check_logits.argsort(dim=1, descending=True)[:, 0]
                # second_best_logits = check_logits.argsort(dim=1, descending=True)[:, 1]
                # other_best_logits = best_logits != tlabels
                # other_second_best_logits = second_best_logits != tlabels
                # other_labels = torch.zeros_like(tlabels, dtype=torch.int64)
                # other_labels[other_best_logits] = best_logits[other_best_logits]
                # other_labels[~other_best_logits & other_second_best_logits] = second_best_logits[~other_best_logits & other_second_best_logits]
                # check_other_best_logits = check_logits[torch.arange(len(tlabels)), other_labels]
                # check_margin = (check_target_logits - check_other_best_logits).mean()

                check_accs.append(check_acc)
                sum_check_acc = sum_check_acc + check_acc
                sum_check_margin = sum_check_margin + check_margin
        avg_check_acc = sum_check_acc / self.check_times
        avg_check_margin = torch.maximum(sum_check_margin / self.check_times, 
                torch.zeros_like(sum_check_margin))
        check_score = avg_check_margin / (avg_check_margin + 1)
        score = check_score * self._f_beta(init_acc, avg_check_acc, beta=0.5) 
        # score = (1.0 + check_score) * (init_acc + avg_check_acc) / 4.0
        # score_history = np.array(score_history, dtype=np.float32)
        # score = np.mean(score_history, axis=0)
        # sim = self.trend_similarity(score)
        # return sim
        return score.item()

    def select_bak(self, model, check_model, inputs, slabels, tlabels, n_epoch, n_tran, topk=5, ngrp=10):
        score_list, tran_list = [], []
        checked = {}
        for tran_seq in self.combine(len(self.aug_library['tran']), n_tran):
            if tran_seq in checked or tuple(reversed(tran_seq)) in checked:
                continue
            score = self.translation_score(model, check_model, inputs, tlabels, n_epoch, tran_seq, ngrp)
            score_list.append(score)
            tran_list.append(tran_seq)

            checked[tran_seq] = score
        # topk_score_indices = np.argsort(score_list)[:topk] # choose to kth smallest
        topk_score_indices = np.argsort(score_list)[-topk:] # choose to kth biggest 
        return [tran_list[indice] for indice in topk_score_indices]

    def select(self, model, check_model, inputs, slabels, tlabels, n_epoch, n_tran, topk=1, threshold_k=3, 
            threshold=0.6, ngrp=10):
        score_list, tran_list = [], []
        for i in range(n_tran):
            tmp_score_list = []
            tmp_tran_list = []
            checked = {}
            for tran_i in range(len(self.aug_library['tran'])):
                if len(tran_list) == 0:
                    tran_seq = (tran_i,)
                    if tran_seq in checked or reversed(tran_seq) in checked:
                        continue

                    score = self.translation_score(model, check_model, inputs, tlabels, n_epoch, tran_seq, ngrp)
                    tmp_score_list.append(score)
                    tmp_tran_list.append(tran_seq)
                    checked[tran_seq] = score
                else:
                    for tran_seq in tran_list:
                        if tran_i == tran_seq[-1]:
                            continue

                        if (tran_i == 1 or tran_i == 2) and tran_i in tran_seq:
                            # hflip, vflip only once
                            continue

                        tran_seq = tran_seq + (tran_i,)
                        if tran_seq in checked or reversed(tran_seq) in checked:
                            continue

                        score = self.translation_score(model, check_model, inputs, tlabels, n_epoch, tran_seq, ngrp)
                        tmp_score_list.append(score)
                        tmp_tran_list.append(tran_seq)
                        checked[tran_seq] = score

            score_list = tmp_score_list
            tran_list = tmp_tran_list
            if i + 1 < n_tran:
                indice_list_threshold = [indice
                        for indice, score in enumerate(score_list)
                        if score > threshold]
                if len(indice_list_threshold) == 0:
                    topk_score_indices = np.argsort(score_list)[-topk:] # choose to kth biggest 
                    tran_list = [tran_list[indice] for indice in topk_score_indices]
                else:
                    score_list_threshold = [score_list[indice]
                                            for indice in indice_list_threshold
                                           ]
                    tran_list_threshold = [tran_list[indice]
                                            for indice in indice_list_threshold
                                           ]
                    topk_score_indices = np.argsort(score_list_threshold)[-threshold_k:] # choose to kth biggest 
                    tran_list = [tran_list_threshold[indice] for indice in topk_score_indices]
        return tran_list, score_list 


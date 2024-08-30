#!/usr/bin/env python
# coding=utf-8
import sys
sys.path.append('..')

import os, json
import torch
from torch import nn
from model_zoo import ModelZoo
from dataset_zoo import DatasetZoo
from configure import *
import pdb
from attack_utils import save_one_img
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import pickle
from torchattacks import attacks
from torchattacks.tools.seed import random_seed
import time

def main():
    mzoo = ModelZoo()
    dzoo = DatasetZoo()
    with open(attack_book, 'r') as fp:
        attack_targets = json.load(fp)

    for eps in [0.07]:
        for mname in ['resnet50',]:
            print('model {} generates adversarial examples...'.format(mname))
            adv_output_dir = os.path.join(eps_output_path, str(eps), mname)
            if not os.path.exists(adv_output_dir):
                os.makedirs(adv_output_dir)
    
            for (dname, dpath), (fbname, fbpath) in zip(victim_datasets, feature_libraries):
    
                adv_output_dir = os.path.join(eps_output_path, str(eps), mname, dname)
                if not os.path.exists(adv_output_dir):
                    os.makedirs(adv_output_dir)
    
                print('1. dataset {} is attacked...'.format(dname)) 
                ds = dzoo.load_dataset(dname, dpath)
                label_space = list(ds.class_to_idx.values())
    
                model = mzoo.pick_model(mname, dataset=dname)
                model = model.cuda()
                model.eval()
    
                for i, (attack_name, attack_args) in enumerate(baseline_attack_methods.items()):
                    # random_seed()

                    attack_args['eps'] = eps
                    alpha = 2.0 / 255.0

                    # skip some methods
                    if attack_name not in ['TCA-t2', 'TCA-t3', 'TCA-t4', 'TCA-t5']:
                        continue
    
                    adv_output_dir = os.path.join(eps_output_path, str(eps), mname, dname, attack_name)
                    if not os.path.exists(adv_output_dir):
                        os.makedirs(adv_output_dir)
        
                    print('2.{} attack method {} is attacking...'.format(i, attack_name))
                    if attack_name == 'DI-FGSM':
                        attack = attacks.difgsm.DIFGSM(model,
                                eps=attack_args['eps'],
                                steps=attack_args['max_iter'],
                                decay=attack_args['decay_factor'],
                                alpha=alpha,
                                diversity_prob=attack_args['diversity_prob'])
                        # targeted
                        attack.set_mode_targeted_by_label()
                    elif attack_name in ['TCA-t2', 'TCA-t3', 'TCA-t4', 'TCA-t5']:
                        d = int(attack_name[-1])

                        depth_T = {2: 0.3, 3: 0.01, 4: 0.01, 5: 0.03}
                        depth_probs = {2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}

                        depth_T[d] = attack_args['temperature'] 
                        depth_probs[d] = 1.0

                        attack = attacks.tca.TCA(model,
                            eps=attack_args['eps'],
                            steps=attack_args['max_iter'],
                            alpha=alpha,
                            decay=attack_args['decay_factor'],
                            n=10,
                            tran_map_path=[
                                '/home/zero/zero/adversarial_examples/adversarial_examples/libraries/adversarial-attacks-pytorch-master/twin_test/new-resnet50-res34-2.pickle',
                                '/home/zero/zero/adversarial_examples/adversarial_examples/libraries/adversarial-attacks-pytorch-master/twin_test/new-resnet50-res34-3.pickle',
                                '/home/zero/zero/adversarial_examples/adversarial_examples/libraries/adversarial-attacks-pytorch-master/twin_test/new-resnet50-res34-4.pickle',
                                '/home/zero/zero/adversarial_examples/adversarial_examples/libraries/adversarial-attacks-pytorch-master/twin_test/new-resnet50-res34-5.pickle',
                            ], 
                            tran_mode='hamburger', depth_T=depth_T, depth_probs=depth_probs)
                        # targeted
                        attack.set_mode_targeted_by_label()
                    else:
                        # raise 'Invalid attack method!!!'
                        continue
    
                    # begin to attack
                    adv_confidences = {} 
                    start = time.time()
                    for (feature, label), (fname, _) in tqdm(zip(ds, ds.imgs)):
                        feature = feature.unsqueeze(0).cuda()
                        source = torch.LongTensor([label]).cuda()
    
                        fname_basename = os.path.basename(fname)
                        (_, target) = attack_targets[fname_basename]
                        target = torch.LongTensor([target]).cuda()
                        adv_output_file = os.path.join(adv_output_dir, fname_basename)
    
                        adv_feature = attack(feature, target, source_labels=source) 
                        save_one_img(adv_feature.detach().cpu(), adv_output_file)
    
                        adv_confidence = F.softmax(model(adv_feature), dim=1)
                        adv_confidences[fname_basename] = adv_confidence.detach().cpu().numpy()
                    end = time.time()
    
                    adv_output_time = os.path.join(adv_output_dir, 'time.npy')
                    with open(adv_output_time, 'w') as f:
                        f.write(str(end-start) + '\n')

                    adv_output_confidence = os.path.join(adv_output_dir, 'confidence.npy')
                    with open(adv_output_confidence, 'wb') as fp:
                        pickle.dump(adv_confidences, fp)

if __name__ == '__main__':
    main()

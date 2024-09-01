#!/o0usr/bin/env python
# coding=utf-8
model_names = ['resnet50', 'densenet121', 'wide_resnet50_2', 'vgg19']
victim_model_names = ['resnet50', 'vgg19', 'inception_v3', 'densenet121', 'wide_resnet50_2']
victim_datasets = [('imagenet', '~/zero/split_dp/dataset/imagenet/new_adv_1k')]
feature_libraries = [('imagenet', '~/zero/split_dp/dataset/imagenet/feature_library')]
clean_libraries = ('imagenet', '~/zero/split_dp/dataset/imagenet/train_by_class')
eps_output_path = '~/zero/split_dp/dataset/imagenet/teps_outputs'
attack_book = './new_attack_book_1k.json'

eps_evaluation_file = 'evaluation/teps_evaluation.csv'
baseline_attack_methods = {
    'DI-FGSM': {
        'max_iter': 10,            # iterations
        'decay_factor': 1.0,          # decay factor
        'eps': 0.07,    # perturbation
        'diversity_prob': 0.7,
        'feature_model': False,
    },
    'TCA-t2': {
        'max_iter': 10,            # iterations
        'eps': 0.07,    # perturbation
        'alpha': 2/255.0,    # step size 
        'decay_factor': 1.0,          # decay factor
        'n': 10,        
        'feature_model': True,
        'temperature': 0.6,
    },
    'TCA-t3': {
        'max_iter': 10,            # iterations
        'eps': 0.07,    # perturbation
        'alpha': 2/255.0,    # step size 
        'decay_factor': 1.0,          # decay factor
        'n': 10,       
        'feature_model': True,
        'temperature': 0.3,
    },
    'TCA-t4': {
        'max_iter': 10,            # iterations
        'eps': 0.07,    # perturbation
        'alpha': 2/255.0,    # step size 
        'decay_factor': 1.0,          # decay factor
        'n': 10,      
        'feature_model': True,
        'temperature': 0.1,
    },
    'TCA-t5': {
        'max_iter': 10,            # iterations
        'eps': 0.07,    # perturbation
        'alpha': 2/255.0,    # step size 
        'decay_factor': 1.0,          # decay factor
        'n': 10,          
        'feature_model': True,
        'temperature': 0.03,
    },
}

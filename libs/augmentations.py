import torch
from timm.data.mixup import Mixup
from timm.data.random_erasing import RandomErasing
import random


def erase(inputs, classes, label_smoothing=0.0):
    erase_fn = RandomErasing()
    return erase_fn(inputs), classes


def mixup_cutmix(inputs,
                 classes,
                 num_classes,
                 prob=1,
                 mixup_alpha=[.2, 1.],
                 cutmix_alpha=[.2, 1.],
                 label_smoothing=0):
    mixup_alpha_choice = random.uniform(mixup_alpha[0], mixup_alpha[1])
    mixup_cutmix_choice = random.uniform(cutmix_alpha[0], cutmix_alpha[1])
    mixup_args = {
        'mixup_alpha': mixup_alpha_choice,
        'cutmix_alpha': mixup_cutmix_choice,
        'cutmix_minmax': None,
        'prob': prob,
        'switch_prob': 0.5,
        'mode': 'batch',
        'label_smoothing': label_smoothing,
        'num_classes': num_classes
    }
    mixup_fn = Mixup(**mixup_args)
    return mixup_fn(inputs, classes)


def smooth_one_hot(true_labels, smoothing=0.0):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method

    """
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    with torch.no_grad():
        true_dist = torch.empty(true_labels.shape, device=true_labels.device)
        fill = smoothing / (true_labels.shape[1] - 1)
        true_dist.fill_(fill)
    return true_dist + (true_labels * confidence - true_labels * fill)

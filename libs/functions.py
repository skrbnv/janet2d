import torch
from tqdm import tqdm
from numpy import mean
import libs.augmentations as _aug
from libs.data import encode_classes
from glob import glob
import os
import re
import random
from string import ascii_lowercase


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Torch is using device:", device)
    return device


def evaluate(model, loader, criterion):
    device = next(model.parameters()).device
    model.eval()
    losses = []
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in (pbar := tqdm(loader)):
            labels = encode_classes(labels, 10)
            outputs = model(inputs.to(device))
            correct += torch.sum(
                torch.argmax(outputs.clone().detach().cpu(), axis=1) ==
                torch.argmax(labels, axis=1)).item()
            total += labels.size(0)
            loss = criterion(outputs, labels.to(device))
            pbar.set_description(f'{loss.item():.4f} {correct/total:.4f}')
            losses.append(loss.item())
    loss = mean(losses)
    model.train()
    return loss, correct / total


def train_loop(inputs, labels, model, criterion, optimizer):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    return loss


def train(model,
          loader,
          criterion,
          optimizer,
          augmentations=[''],
          label_smoothing=.1,
          num_classes=10):
    losses = []
    device = next(model.parameters()).device
    for augmentation in augmentations:
        for inputs, labels in (pbar := tqdm(loader)):
            inputs = inputs.to(device)
            labels = encode_classes(labels.to(device), 10)
            if augmentation == 'mix':
                inputs, labels = _aug.mixup_cutmix(
                    inputs,
                    torch.nonzero(labels, as_tuple=True)[1], num_classes)
            elif augmentation == 'erase':
                inputs, labels = _aug.erase(inputs, labels)
            if label_smoothing > 0:
                labels = _aug.smooth_one_hot(labels, label_smoothing)
            loss = train_loop(inputs, labels, model, criterion, optimizer)
            pbar.set_description(f'{loss.item():.4f}')
            losses.append(loss.item())
    return mean(losses)


def checkpoint(id, data, path='./checkpoints', keep=3):
    removables = glob(os.path.join(path, f'{id}*'))
    if len(removables) > 0:
        latest = os.path.basename(max(removables, key=os.path.getctime))
        try:
            current = int(re.search(r'(\d{3})\.dict$', latest).groups()[0]) + 1
        except Exception:
            current = 0
    else:
        current = 0
    chkptfname = os.path.join(path, f'{id}{(current):03}.dict')
    if not os.path.isdir(path):
        os.mkdir(path)
    torch.save(data, chkptfname)
    print(f'Checkpoint {chkptfname} saved')
    for rm in removables:
        if int(os.path.basename(rm)[-8:-5]) <= current - keep:
            os.remove(rm)
    return True


def get_random_hash():
    return ''.join(random.choice(ascii_lowercase) for i in range(10))

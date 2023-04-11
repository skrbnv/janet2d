import torch
from torchvision import transforms as tf
import yaml
from models.resnet_planar import rn18
from torchvision.datasets import ImageFolder
import wandb
import torchinfo
import argparse
from libs.data import Dataset
from libs.functions import train, evaluate, checkpoint, get_random_hash
import datetime

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb',
                        action='store_true',
                        default=False,
                        help='sync with W&B')
    parser.add_argument('--resume',
                        action='store_true',
                        default=False,
                        help='resume')
    parser.add_argument('--config',
                        action='store',
                        default='config.yaml',
                        help='config filename')
    args = parser.parse_args()
    WANDB, RESUME, path = args.wandb, args.resume, args.config
    with open(path) as stream:
        CFG = yaml.safe_load(stream)

    laststate = torch.load(CFG['checkpoint']) if RESUME else None
    initial_epoch = laststate['epoch'] + 1 if RESUME else 0
    '''
    if RESUME:
        RID = os.path.basename(CFG['checkpoint']).rstrip('.dict')[:-3]
        if WANDB:
            print(f"Your run id is {RID} with checkpoint {CFG['checkpoint']}")
            input("Press any key if you want to continue >>")
            wprj = wandb.init(id=RID,
                              project=CFG['wandb']['project'],
                              resume='must',
                              config=CFG)
    else:  # not RESUME
    '''
    for i in range(1):
        print(f'{i}-th run')
        if WANDB:
            wprj = wandb.init(project=CFG['wandb']['project'],
                              resume=False,
                              config=CFG,
                              name=f"{CFG['wandb']['name']}{i}",
                              reinit=True)
            RID = wprj.id
        else:
            RID = get_random_hash()

        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        print("Torch is using device:", device)

        model = rn18()  # Model(3, 10)
        initial_epoch = 0
        model.float()
        model.to(device)
        if RESUME:
            model.load_state_dict(laststate['state_dict'])
            print("Model state dict loaded from checkpoint")
        print(model)
        torchinfo.summary(model, tuple(CFG['general']['torchinfo_shape']))
        if WANDB:
            wandb.watch(model)

        trainset = Dataset(CFG['dataset']['train'])
        testset = Dataset(CFG['dataset']['test'])

        stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        train_tfms = tf.Compose([
            tf.RandomCrop(32, padding=4, padding_mode='reflect'),
            tf.RandomHorizontalFlip(),
            tf.ToTensor(),
            tf.Normalize(*stats, inplace=True)
        ])
        valid_tfms = tf.Compose([tf.ToTensor(), tf.Normalize(*stats)])
        trainset = ImageFolder(CFG['dataset']['train'], train_tfms)
        testset = ImageFolder(CFG['dataset']['test'], valid_tfms)

        classnames = CFG['dataset']['classnames']

        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=CFG['batch_size'],
                                                  shuffle=True,
                                                  num_workers=2)
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=CFG['batch_size'],
                                                 shuffle=False,
                                                 num_workers=2)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

        if RESUME and laststate['optimizer'] is not None:
            optimizer.load_state_dict(laststate['optimizer'])
            print("Optimizer state dict loaded from checkpoint")
        times = []
        for epoch in range(initial_epoch, CFG['epochs']):
            print(f'==================== Epoch: {epoch} ====================')
            start = datetime.datetime.now()
            train(model=model,
                  loader=trainloader,
                  criterion=criterion,
                  optimizer=optimizer,
                  augmentations=CFG['augmentations'],
                  label_smoothing=CFG['label_smoothing'],
                  num_classes=10)
            loss_train, acc_train = evaluate(model, trainloader, criterion)
            loss_test, acc_test = evaluate(model, testloader, criterion)
            end = datetime.datetime.now()
            times.append(end - start)
            print(f'  Training loss: {loss_train:.4f}')
            print(f'  Training acc:  {acc_train*100:.2f}%')
            print(f'  Testing loss:  {loss_test:.4f}')
            print(f'  Testing acc:   {acc_test*100:.2f}%')
            print(f'  Time:          {end-start}')

            if WANDB:
                wandb.log({
                    "Training loss": loss_train,
                    "Training acc": acc_train,
                    "Testing loss": loss_test,
                    "Testing acc": acc_test,
                })

            checkpoint(RID,
                       data={
                           'epoch': epoch,
                           'state_dict': model.state_dict(),
                           'optimizer': optimizer.state_dict()
                       })

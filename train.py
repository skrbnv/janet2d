import torch
from torchvision import transforms as tf
import yaml
from torchvision.datasets import ImageFolder
import wandb
import torchinfo
import argparse
from libs.functions import train, evaluate, checkpoint, get_random_hash

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
                        default='flowers.yaml',
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
        module = __import__(CFG['model_file'], fromlist=[CFG['model_name']])
        model_constructor = getattr(module, CFG['model_name'])
        model = model_constructor()  # Model(3, 10)
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

        stats = CFG['dataset']['stats']
        train_tfms = tf.Compose([
            tf.RandomCrop(CFG['dataset']['crop'], padding=4, padding_mode='reflect'),
            tf.RandomHorizontalFlip(),
            tf.ToTensor(),
            tf.Normalize(*stats, inplace=True)
        ])
        valid_tfms = tf.Compose([tf.ToTensor(), tf.Normalize(*stats)])
        trainset = ImageFolder(CFG['dataset']['train'], train_tfms)
        testset = ImageFolder(CFG['dataset']['test'], valid_tfms)
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
        for epoch in range(initial_epoch, CFG['epochs']):
            print(f'==================== Epoch: {epoch} ====================')
            train(model=model,
                  loader=trainloader,
                  criterion=criterion,
                  optimizer=optimizer,
                  augmentations=CFG['augmentations'],
                  label_smoothing=CFG['label_smoothing'],
                  num_classes=CFG['dataset']['num_classes'])
            loss_train, acc_train = evaluate(model, trainloader, criterion, CFG['dataset']['num_classes'])
            loss_test, acc_test = evaluate(model, testloader, criterion, CFG['dataset']['num_classes'])
            print(f'  Training loss: {loss_train:.4f}')
            print(f'  Training acc:  {acc_train*100:.2f}%')
            print(f'  Testing loss:  {loss_test:.4f}')
            print(f'  Testing acc:   {acc_test*100:.2f}%')

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

import torch
import torchvision
from torchvision import transforms
import yaml
from models.cnn import Model
from tqdm import tqdm
from numpy import mean
import wandb
import torchinfo
import argparse
path = 'config.yaml'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb',
                        action='store_true',
                        default=False,
                        help='sync with W&B')
    args = parser.parse_args()
    WANDB = args.wandb
    with open(path) as stream:
        CFG = yaml.safe_load(stream)
    if WANDB:
        wprj = wandb.init(project=CFG['wandb']['project'],
                        resume=False,
                        config=CFG)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Torch is using device:", device)

    model = Model()
    model.float()
    model.to(device)
    print(model)
    torchinfo.summary(model,
                      tuple(CFG['general']['torchinfo_shape']))
    if WANDB:
        wandb.watch(model)

    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=CFG['batch_size'],
                                          shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=CFG['batch_size'],
                                         shuffle=False, num_workers=2)


    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)



    for epoch in range(2):
        print(f'Epoch: {epoch}')
        epoch_losses_train = []
        for i, (inputs, labels) in (pbar := tqdm(enumerate(trainloader))):
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            pbar.set_description(f'{loss.item():.4f}')
            epoch_losses_train.append(loss.item())
        print(f'Training loss: {mean(epoch_losses_train):.4f}')


        epoch_losses_test = []
        model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in (pbar := tqdm(enumerate(testloader))):
                outputs = model(inputs.to(device))
                loss = criterion(outputs, labels.to(device))
                pbar.set_description(f'{loss.item():.4f}')
                epoch_losses_test.append(loss.item())
        model.train()
        print(f'Testing loss: {mean(epoch_losses_test):.4f}')

        if WANDB:
            wandb.log({
                "Loss": mean(epoch_losses_train),
                "Test loss": mean(epoch_losses_test),
            })
import torch
import torchvision

print(torch.utils.data.DataLoader)

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('..\MNIST', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_train, shuffle=True)

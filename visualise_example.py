import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import matplotlib.pyplot as plt
from torchvision.utils import make_grid



# class Extractor(nn.Module):
#     def __init__(self):
#         super(Extractor, self).__init__()
#         self.extractor = nn.Sequential(
#             nn.Conv2d(1, 3, 3, 1, 1),
#             nn.MaxPool2d(2),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(3, 6, 3, 1, 1),
#             nn.MaxPool2d(2),
#             nn.ReLU(inplace=True),
#         )
#
#         self.conv_trans1 = nn.ConvTranspose2d(6, 3, 4, 2, 1)
#         self.conv_trans2 = nn.ConvTranspose2d(3, 1, 4, 2, 1)
#
#     def forward(self, x):
#         x = self.extractor(x)
#         x = F.relu(self.conv_trans1(x))
#         x = self.conv_trans2(x)
#         return x


class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=0),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=0),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 5, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 6, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 6, 2)
        )

    def forward(self, x):
        x = self.extractor(x)
        x = self.decoder(x)
        return x

dataset = datasets.MNIST(
    root='/home/zhenyue-qin/Research/Project-Katyusha-Multi-Label-VAE/Katyusha-CVAE/data',
    transform=transforms.ToTensor()
)
loader = DataLoader(
    dataset,
    num_workers=2,
    batch_size=512,
    shuffle=True
)

model = Extractor()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 1
# for epoch in range(epochs):
#     for batch_idx, (data, target) in enumerate(loader):
#         optimizer.zero_grad()
#         output = model(data)
#         loss = criterion(output, data)
#         loss.backward()
#         optimizer.step()
#
#         print('Epoch {}, Batch idx {}, loss {}'.format(
#             epoch, batch_idx, loss.item()))


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model = Extractor()
model.extractor[0].register_forward_hook(get_activation('ext_conv1'))
x = torch.randn(1, 3, 96, 96)
output = model(x)
print(output.shape)
torch.Size([1, 3, 96, 96])

act = activation['ext_conv1'].squeeze()
num_plot = 4
fig, axarr = plt.subplots(min(act.size(0), num_plot))
for idx in range(min(act.size(0), num_plot)):
    axarr[idx].imshow(act[idx])

from torchvision.utils import make_grid

kernels = model.extractor[0].weight.detach().clone()
kernels = kernels - kernels.min()
kernels = kernels / kernels.max()
img = make_grid(kernels)
plt.imshow(img.permute(1, 2, 0))

plt.show()

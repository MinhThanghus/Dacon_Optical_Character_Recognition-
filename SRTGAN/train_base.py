import os
import cv2
import random

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision.models import vgg19_bn

from albumentations.pytorch.transforms import ToTensorV2
from tqdm import tqdm

from sklearn.model_selection import train_test_split


class CFG:
    traindata = './data/train/'
    valdata = './data/val/'
    device = 'cuda'

    # srgan recommand lr 0.001
    lr = 1e-4
    batch_size = 16

    mseLoss = nn.MSELoss()
    bceLoss = nn.BCELoss()
    epochs = 100

    HR_patch_size = 64
    LR_patch_size = 16

    weights_path = './weights/'


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19_bn(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features[:6])).to(CFG.device, dtype=torch.float)

    def forward(self, img):
        return self.feature_extractor(img)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=4, bias=False)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1, bias=False)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, padding=4, bias=False)

        self.bn = nn.BatchNorm2d(64)
        self.ps = nn.PixelShuffle(2)
        self.prelu = nn.PReLU()

    def forward(self, x):
        block1 = self.first_block(x)

        block2 = self.residual_block(block1)
        block2 = self.residual_block(block2)
        block2 = self.residual_block(block2)
        block2 = self.residual_block(block2)
        block3 = self.third_block(block2, block1)
        block4 = self.fourth_block(block3)

        return block4

    def first_block(self, x):
        return self.prelu(self.conv1(x))

    def residual_block(self, x):
        return torch.add(self.bn(self.conv2(self.prelu(self.bn(self.conv2(x))))), x)

    def third_block(self, x, skip_data):
        return torch.add(self.bn(self.conv2(x)), skip_data)

    def fourth_block(self, x):
        x = self.prelu(self.ps(self.conv3(x)))
        x = self.prelu(self.ps(self.conv3(x)))
        # Rearranges elements in a tensor of shape (B, C*r^2,H,W) -> (B, C, H*r, W*r)
        x = self.conv4(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = self.CNNblock(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = self.CNNblock(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv3 = self.CNNblock(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4 = self.CNNblock(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv5 = self.CNNblock(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv6 = self.CNNblock(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv7 = self.CNNblock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv8 = self.CNNblock(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, bias=False)

        self.leakyrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x).flatten()
        result = self.MLPblock(x, in_features=x.size(dim=0))
        return result

    def CNNblock(self, in_channels, out_channels, kernel_size, padding, stride, bias):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding,
                      stride=stride, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def MLPblock(self, x, in_features, out_features=1024):
        model = nn.Linear(in_features=in_features, out_features=out_features).to(CFG.device)
        x = self.leakyrelu(model(x))
        model = nn.Linear(in_features=out_features, out_features=1).to(CFG.device)
        x = self.sigmoid(model(x))

        return x


class ImageDataset(Dataset):
    def __init__(self, dataPath):
        self.dataPath = dataPath
        self.dataList = []

        for path in os.listdir(self.dataPath):
            self.dataList.append(self.dataPath + path)

    def __len__(self):
        return len(self.dataList)

    def __getitem__(self, index):
        HRimage = cv2.imread(self.dataList[index], cv2.COLOR_BGR2RGB)
        HRimage = cv2.resize(HRimage, dsize=(CFG.HR_patch_size, CFG.HR_patch_size))
        h, w = HRimage.shape[:2]
        LRimage = cv2.resize(HRimage, dsize=(round(w / 4), round(h / 4)))

        HRimage = ToTensorV2()(image=HRimage)['image'].to(CFG.device, dtype=torch.float)
        LRimage = ToTensorV2()(image=LRimage)['image'].to(CFG.device, dtype=torch.float)

        return HRimage, LRimage


def PSNR(HRimage, LRimage):
    r = 255
    mse = nn.MSELoss()
    mseloss = mse(HRimage, LRimage)
    psnr = 20 * torch.log10(r / torch.sqrt(mseloss))
    return psnr


def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(random_seed)


def train_one_epoch(G_model, D_model, G_optimizer, D_optimizer, dataloader, epoch):
    G_model.train()
    D_model.train()

    dataset_size = 0
    running_loss = 0

    mseLoss = CFG.mseLoss
    bceLoss = CFG.bceLoss

    bar = tqdm(enumerate(dataloader), total=len(dataloader))

    for step, data in bar:
        HRimages = data[0]
        LRimages = data[1]

        # Discriminator Loss part
        G_outputs = G_model(LRimages)
        fakeLabel = D_model(G_outputs.detach())
        realLabel = D_model(HRimages)

        d1Loss = torch.mean(bceLoss(fakeLabel, torch.zeros_like(fakeLabel, dtype=torch.float)))
        d2Loss = torch.mean(bceLoss(realLabel, torch.ones_like(realLabel, dtype=torch.float)))
        dLoss = d1Loss + d2Loss

        D_optimizer.zero_grad()
        d2Loss.backward()
        d1Loss.backward(retain_graph=True)
        D_optimizer.step()

        G_optimizer.zero_grad()

        # Generator Loss part
        g1Loss = bceLoss(fakeLabel.detach(), torch.ones_like(fakeLabel))
        g2Loss = mseLoss(feature_extractor(G_outputs).detach(), feature_extractor(HRimages).detach())
        g3Loss = mseLoss(G_outputs, HRimages)

        gLoss = g1Loss + g2Loss + g3Loss

        gLoss.backward()
        G_optimizer.step()

        running_loss += ((gLoss.item() + dLoss.item()) / 2) * CFG.batch_size
        dataset_size += CFG.batch_size
        epoch_loss = running_loss / dataset_size

        bar.set_postfix(EPOCH=epoch, TRAIN_LOSS=epoch_loss, PSNR=PSNR(HRimages, G_outputs))


def val_one_epoch(G_model, D_model, dataloader, epoch):
    G_model.eval()
    D_model.eval()
    with torch.no_grad():
        dataset_size = 0
        running_loss = 0

        mseLoss = CFG.mseLoss
        bceLoss = CFG.bceLoss

        bar = tqdm(enumerate(dataloader), total=len(dataloader))

        for step, data in bar:
            HRimages = data[0]
            LRimages = data[1]

            # Discriminator Loss part
            G_outputs = G_model(LRimages)
            fakeLabel = D_model(G_outputs.detach())
            realLabel = D_model(HRimages)

            d1Loss = torch.mean(bceLoss(fakeLabel, torch.zeros_like(fakeLabel, dtype=torch.float)))
            d2Loss = torch.mean(bceLoss(realLabel, torch.ones_like(realLabel, dtype=torch.float)))
            dLoss = d1Loss + d2Loss

            # Generator Loss part
            g1Loss = bceLoss(fakeLabel.detach(), torch.ones_like(fakeLabel))
            g2Loss = mseLoss(feature_extractor(G_outputs).detach(), feature_extractor(HRimages).detach())
            g3Loss = mseLoss(G_outputs, HRimages)

            gLoss = g1Loss + g2Loss + g3Loss

            running_loss += ((gLoss.item() + dLoss.item()) / 2) * CFG.batch_size
            dataset_size += CFG.batch_size
            epoch_loss = running_loss / dataset_size

            bar.set_postfix(EPOCH=epoch, VAL_LOSS=epoch_loss, PSNR=PSNR(HRimages, G_outputs))


if __name__ == "__main__":
    set_seed(42)

    G_Model = Generator().to(CFG.device)
    D_Model = Discriminator().to(CFG.device)

    G_optimizer = optim.Adam(G_Model.parameters(), lr=CFG.lr)
    D_optimizer = optim.Adam(D_Model.parameters(), lr=CFG.lr)

    train_dataset = ImageDataset(CFG.traindata)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=CFG.batch_size)

    val_dataset = ImageDataset(CFG.valdata)
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=CFG.batch_size)

    feature_extractor = FeatureExtractor()
    feature_extractor.eval()

    for epoch in range(CFG.epochs):
        train_one_epoch(G_Model, D_Model, G_optimizer, D_optimizer, train_loader, epoch)
        print('\n')
        val_one_epoch(G_Model, D_Model, val_loader, epoch)
        print('\n')
        torch.save(G_Model.state_dict(), CFG.weights_path + f'Generator_epochs_{epoch}.pt')
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, num_classes=1):
        super(UNet, self).__init__()
        self.d1 = DoubleConv(3, 64)
        self.p1 = nn.MaxPool2d(2)

        self.d2 = DoubleConv(64, 128)
        self.p2 = nn.MaxPool2d(2)

        self.d3 = DoubleConv(128, 256)
        self.p3 = nn.MaxPool2d(2)

        self.d4 = DoubleConv(256, 512)
        self.p4 = nn.MaxPool2d(2)

        self.d5 = DoubleConv(512, 1024)

        self.u4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dc4 = DoubleConv(1024, 512)

        self.u3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dc3 = DoubleConv(512, 256)

        self.u2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dc2 = DoubleConv(256, 128)

        self.u1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dc1 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        c1 = self.d1(x)
        p1 = self.p1(c1)

        c2 = self.d2(p1)
        p2 = self.p2(c2)

        c3 = self.d3(p2)
        p3 = self.p3(c3)

        c4 = self.d4(p3)
        p4 = self.p4(c4)

        c5 = self.d5(p4)

        u4 = self.u4(c5)
        u4 = torch.cat([u4, c4], dim=1)
        c6 = self.dc4(u4)

        u3 = self.u3(c6)
        u3 = torch.cat([u3, c3], dim=1)
        c7 = self.dc3(u3)

        u2 = self.u2(c7)
        u2 = torch.cat([u2, c2], dim=1)
        c8 = self.dc2(u2)

        u1 = self.u1(c8)
        u1 = torch.cat([u1, c1], dim=1)
        c9 = self.dc1(u1)

        out = self.out(c9)
        return torch.sigmoid(out)

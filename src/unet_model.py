""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNetDummy(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 2))
        self.down1 = (Down(2, 4))
        self.down2 = (Down(4, 8))
        self.down3 = (Down(8, 16))
        factor = 2 if bilinear else 1
        self.down4 = (Down(16, 32 // factor))
        self.up1 = (Up(32, 16 // factor, bilinear))
        self.up2 = (Up(16, 8 // factor, bilinear))
        self.up3 = (Up(8, 4 // factor, bilinear))
        self.up4 = (Up(4, 2, bilinear))
        self.outc = (OutConv(2, n_classes))

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = x.float()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return logits


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, dropout=0.2):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))
        self.dropout = torch.nn.Dropout(p=dropout)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.float()
        x1 = self.inc(x)
        # x1 = self.dropout(x1)
        x2 = self.down1(x1)
        # x2 = self.dropout(x2)
        x3 = self.down2(x2)
        # x3 = self.dropout(x3)
        x4 = self.down3(x3)
        # x4 = self.dropout(x4)
        x5 = self.down4(x4)
        # x5 = self.dropout(x5)

        x = self.up1(x5, x4)
        # x = self.dropout(x)
        x = self.up2(x, x3)
        # x = self.dropout(x)
        x = self.up3(x, x2)
        # x = self.dropout(x)
        x = self.up4(x, x1)
        # x = self.dropout(x)
        logits = self.outc(x)#.squeeze(1)

        return logits


class EnsembleUNet(nn.Module):
    def __init__(self, T: int) -> None:
        super().__init__()
        self.T = T
        self.models = nn.ModuleList([UNet(n_channels=1, n_classes=2) for t in range (T)])

    def forward(self, x):
        lung_logits = self.lung_segmenter(x)

        return
""" Full assembly of the parts to form the complete network """

from .unet_parts import *
import torch

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, first_channel=64, bilinear=False, dropout=0):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, first_channel))
        self.down1 = (Down(first_channel, first_channel * 2))
        self.down2 = (Down(first_channel * 2, first_channel * 4))
        self.down3 = (Down(first_channel * 4, first_channel * 8))
        factor = 2 if bilinear else 1
        self.down4 = (Down(first_channel * 8, first_channel * 16 // factor))
        self.up1 = (Up(first_channel * 16, first_channel * 8 // factor, bilinear))
        self.up2 = (Up(first_channel * 8, first_channel * 4// factor, bilinear))
        self.up3 = (Up(first_channel * 4, first_channel * 2 // factor, bilinear))
        self.up4 = (Up(first_channel * 2, first_channel, bilinear))
        self.outc = (OutConv(first_channel, n_classes))

        self.dropout = torch.nn.Dropout(p=dropout)

        if n_classes == 1:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = x.float()
        x1 = self.inc(x)
        x1 = self.dropout(x1)
        x2 = self.down1(x1)
        x2 = self.dropout(x2)
        x3 = self.down2(x2)
        x3 = self.dropout(x3)
        x4 = self.down3(x3)
        x4 = self.dropout(x4)
        x5 = self.down4(x4)
        x5 = self.dropout(x5)

        x = self.up1(x5, x4)
        x = self.dropout(x)
        x = self.up2(x, x3)
        x = self.dropout(x)
        x = self.up3(x, x2)
        x = self.dropout(x)
        x = self.up4(x, x1)
        x = self.dropout(x)
        logits = self.outc(x).squeeze(1)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
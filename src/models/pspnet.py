from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from torchvision.models import resnet50, resnet18
from .ppm import PPM


class PSPNet(nn.Module):
    """
    The final feature map size is 1/8 of the input image.

    Use the dilated network strategy
    ResNet-50 has 4 blocks, and those 4 blocks have [3, 4, 6, 3] layers, respectively.
    """

    def __init__(
        self,
        layers: int=18,
        bins=(1, 2, 3, 6),
        dropout: float = 0.1,
        num_classes: int=4,
        zoom_factor: int=8,
        use_ppm: bool=True,
        criterion=nn.CrossEntropyLoss(ignore_index=255),
        pretrained: bool=True,
        deep_base: bool=True,
    ) -> None:
        """
        Args:
            layers: int = 50,
            bins: list of grid dimensions for PPM, e.g. (1,2,3) means to create (1x1), (2x2), and (3x3) grids
            dropout: float representing probability of dropping out data
            num_classes: number of classes
            zoom_factor: scale value used to upsample the model output's (HxW) size to (H * zoom_factor, W * zoom_factor)
            use_ppm: boolean representing whether to use the Pyramid Pooling Module
            criterion: loss function module
            pretrained: boolean representing pretrain status
        """
        super().__init__()
        # assert layers == 50
        assert 2048 % len(bins) == 0
        assert num_classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.dropout = dropout
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        self.criterion = criterion

        self.layer0 = None
        self.layer1 = None
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None
        self.ppm = None
        self.cls = None
        self.aux = None

        #############################################################################
        # Initialize ResNet backbone, and set the layers                            #
        # layer0, layer1, layer2, layer3, layer4. Note: layer0 should be sequential #
        #############################################################################
        
        if layers == 18:
            self.resnet = resnet18(pretrained=pretrained) 
        elif layers == 50:   
            self.resnet = resnet50(pretrained=pretrained)
        self.layer0 = nn.Sequential(
                                        self.resnet.conv1,
                                        self.resnet.bn1,
                                        self.resnet.relu,
                                        # self.resnet.conv2,
                                        # self.resnet.bn2,
                                        # self.resnet.relu,
                                        # self.resnet.conv3,
                                        # self.resnet.bn3,
                                        # self.resnet.relu,
                                        self.resnet.maxpool,
                                    )
        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4

        self.__replace_conv_with_dilated_conv()

        ###########################################################################
        # Initialize the PPM. The reduction_dim should be equal to the            #
        # output number of ResNet feature maps, divided by the number of PPM bins #
        # Afterwards, set fea_dim to the updated feature dimension to be passed   #
        # to the classifier
        ###########################################################################

        fea_dim = 512
        if use_ppm:
            self.ppm = PPM(in_dim=fea_dim, reduction_dim=(fea_dim // len(bins)), bins=bins)
            fea_dim *= 2
        self.cls = self.__create_classifier(in_feats=fea_dim, out_feats=fea_dim // 2, num_classes=num_classes)
        self.aux = self.__create_classifier(in_feats=fea_dim // 4, out_feats=fea_dim // 8, num_classes=num_classes)

    def __replace_conv_with_dilated_conv(self):
        """Increase the receptive field by reducing stride and increasing dilation.
        In Layer3, in every `Bottleneck`, we will change the 3x3 `conv2`. We will
        replace the conv layer that had stride=2, dilation=1, and padding=1 with a
        new conv layer, that instead has stride=1, dilation=2, and padding=2.

        In the `downsample` block, we'll also need to hardcode the stride to 1, instead of 2.
        In Layer4, for every `Bottleneck`, we will make the same changes, except we'll
        change the dilation to 4 and padding to 4.

        Hint: you can iterate over each layer's modules using the .named_modules()
        attribute, and check the name to see if it's the one you want to edit.
        Then you can edit the dilation, padding, and stride attributes of the module.
        """

        for name, param in self.layer3.named_modules():
            if 'conv2' in name:
                param.dilation, param.padding, param.stride = (2, 2), (2, 2), (1, 1)
            # elif 'downsample.0' in name:
            #     param.stride = (1, 1)
        for name, param in self.layer4.named_modules():
            if 'conv2' in name:
                param.dilation, param.padding, param.stride = (4, 4), (4, 4), (1, 1)
            # elif 'downsample.0' in name:
            #     param.stride = (1, 1)

    def __create_classifier(self, in_feats: int, out_feats: int, num_classes: int) -> nn.Module:
        """Implement the final PSPNet classifier over the output categories.

        Args:
            in_feats: number of channels in input feature map
            out_feats: number of filters for classifier's conv layer
            num_classes: number of output categories

        Returns:
            cls: A sequential block of 3x3 convolution, 2d Batch norm, ReLU,
                2d dropout, and a final 1x1 conv layer over the number of
                output classes. The 3x3 conv layer's padding should preserve
                the height and width of the feature map. The specified dropout
                is defined in `self.dropout`.
                2d dropout, and a final 1x1 conv layer over the number of output classes.
                The 3x3 conv layer's padding should preserve the height and width of the
                feature map. The specified dropout is defined in `self.dropout`.
        """

        cls = nn.Sequential(
                            nn.Conv2d(in_channels=in_feats, out_channels=out_feats, kernel_size=3, stride=1, padding=1, bias=False),
                            nn.BatchNorm2d(num_features=out_feats),
                            nn.ReLU(),
                            nn.Dropout2d(p=self.dropout),
                            nn.Conv2d(in_channels=out_feats, out_channels=num_classes, kernel_size=1)
                            )

        return cls

    def forward(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Forward pass of the network.

        Feed the input through the network, upsample the aux output (from layer 3)
        and main output (from layer4) to the ground truth resolution (based on zoom_factor), and then
        compute the loss and auxiliary loss.
        The aux classifier should operate on the output of layer3.
        The PPM should operate on the output of layer4.

        Note that you can return a tensor of dummy values for the auxiliary loss
        if the model is set to inference mode. Note that nn.Module() has a
         `self.training` attribute, which is set to True depending upon whether
        the model is in in training or evaluation mode.
        https://pytorch.org/docs/stable/generated/torch.nn.Module.html#module

        comments on zoom_factor:
            If the input to the network is of shape (N,C,H,W) and the zoom_factor is 2,
            the computed logits has shape (N,num_classes,H/2,W/2), and yhat has shape (N,H/2,W/2).
            Also in this case the associated ground truth labels is of shape (N,H/2,W/2).

            Note that this has nothing to do with the dilated convolutions, in which zoom_factor
            is only used to scale down the size of the output segmented image.

            We will be testing your zoom_factor for values of [1, 2, 4, 8] and assume
            that the ground truth labels will have already beeen scaled down to the zoom_factor.

            When scaling the dimenions round up to the nearest integer value.

            Use Pytorch's functional interpolate for downsampling the outputs to the correct (H/zoom_factor, W/zoom_factor) shape.

        Args:
            x: tensor of shape (N, C, H, W) representing batch of normalized input image
            y: tensor of shape (N, H/zoom_factor, W/zoom_factor) represnting batch of ground truth labels

        Returns:
            logits: tensor of shape (N,num_classes,H/zoom_factor,W/zoom_factor) representing class scores at each pixel
            yhat: tensor of shape (N,H/zoom_factor,W/zoom_factor) representing predicted labels at each pixel
            main_loss: loss computed on output of final classifier if y is provided,
               else return None if no ground truth is passed in
            aux_loss:loss computed on output of auxiliary classifier (from intermediate output)
               if y is provided, else return None if no ground truth is passed in
        """
        # print(self.layer2)
        y = y.long()
        x = x.float()
        B, C, H, W = x.shape
        # print(x_size)
        # print(x_size[2] - 1)
        # assert (H - 1) % 8 == 0 and (W - 1) % 8 == 0

        h = (H) // 8 * self.zoom_factor
        w = (W) // 8 * self.zoom_factor

        if C == 1:
            x = torch.tile(x, (1, 3, 1, 1))
        # print('Original', x.shape)
        x = self.layer0(x)
        # print('Layer 0', x.shape)
        x = self.layer1(x)
        # print('Layer 1', x.shape)
        x = self.layer2(x)
        # print('Layer 2', x.shape)
        aux_x = self.layer3(x)
        # print('Aux', aux_x.shape)
        x = self.layer4(aux_x)
        # print('Layer 4', x.shape)

        if self.use_ppm:
            x = self.ppm(x)
            # print('PPM', x.shape)

        logits = self.cls(x)
        # print('logits', logits.shape)
        aux_logits = self.aux(aux_x)
        # print('aux logits', aux_logits.shape)

        if self.zoom_factor != 1:
            logits = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=True)
            aux_logits = F.interpolate(aux_logits, size=(h, w), mode='bilinear', align_corners=True)
        yhat = torch.argmax(logits, dim=1)

        if y is not None:
            main_loss = self.criterion(logits, y)
            aux_loss = self.criterion(aux_logits, y)
        else:
            main_loss = None
            aux_loss = None

        return logits, yhat, main_loss, aux_loss
    

def psp_model_optimizer(
        layers: int=18,
        bins=(1, 2, 3, 6),
        dropout: float=0.1,
        num_classes: int=4,
        zoom_factor: int=8,
        use_ppm: bool=True,
        criterion=nn.CrossEntropyLoss(ignore_index=255),
        pretrained: bool=True,
        deep_base: bool=True,
        lr: float=0.01,
        weight_decay: float=0.0001,
        momentum: float=0.9
    ) -> None:

    model = PSPNet(
                    layers=18,
                    bins=(1, 2, 3, 6),
                    dropout=0.1,
                    num_classes=4,
                    zoom_factor=8,
                    use_ppm=True,
                    criterion=nn.CrossEntropyLoss(ignore_index=255),
                    pretrained=True,
                    deep_base=True,
                )
    
    layer0_params = {'params': model.layer0.parameters(), 'lr': lr, 'weight_decay': weight_decay, 'momentum': momentum}
    layer1_params = {'params': model.layer1.parameters(), 'lr': lr, 'weight_decay': weight_decay, 'momentum': momentum}
    layer2_params = {'params': model.layer2.parameters(), 'lr': lr, 'weight_decay': weight_decay, 'momentum': momentum}
    layer3_params = {'params': model.layer3.parameters(), 'lr': lr, 'weight_decay': weight_decay, 'momentum': momentum}
    layer4_params = {'params': model.layer4.parameters(), 'lr': lr, 'weight_decay': weight_decay, 'momentum': momentum}
    if model.use_ppm:
        ppm_params = {'params': model.ppm.parameters(), 'lr': 10*lr, 'weight_decay': weight_decay, 'momentum': momentum}
    cls_params = {'params': model.cls.parameters(), 'lr': 10*lr, 'weight_decay': weight_decay, 'momentum': momentum}
    aux_params = {'params': model.aux.parameters(), 'lr': 10*lr, 'weight_decay': weight_decay, 'momentum': momentum}
    if model.use_ppm:
        optimizer = torch.optim.SGD([
                                        layer0_params,
                                        layer1_params,
                                        layer2_params,
                                        layer3_params,
                                        layer4_params,
                                        ppm_params,
                                        cls_params,
                                        aux_params
                                    ], lr=lr, weight_decay=weight_decay, momentum=momentum)
    else:
        optimizer = torch.optim.Adam([
                                        layer0_params,
                                        layer1_params,
                                        layer2_params,
                                        layer3_params,
                                        layer4_params,
                                        cls_params,
                                        aux_params
                                    ], lr=lr, weight_decay=weight_decay, momentum=momentum)

    return model, optimizer
import torch
import torch.nn as nn
# from torchsummary import summary
from misc.utils import initialize_weights
from torchvision import models


class DilateModule(nn.Module):
    def __init__(self):
        super(DilateModule, self).__init__()
        self.branch1 = make_layer(512, 256, 1, 0, 1)
        self.conv1 = make_layer(512, 128, 1, 0, 1)
        self.dconv1 = make_layer(128, 64, 3, 2, 1, 2)
        self.dconv2 = make_layer(64, 32, 3, 2, 1, 2)
        self.dconv3 = make_layer(32, 16, 3, 2, 1, 2)
        self.conv2 = make_layer(16, 256, 1, 0, 1)

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.conv2(self.dconv3(self.dconv2(self.dconv1(self.conv1(x)))))
        x = torch.cat((branch1, branch2), dim=1)
        return x

class Output(nn.Module):
    def __init__(self):
        super(Output, self).__init__()
        self.upconv1 = make_contrans_layer(512, 256, 4, 1, 2)
        self.upconv2 = make_contrans_layer(256, 128, 4, 1, 2)
        self.upconv3 = make_contrans_layer(128, 64, 4, 1, 2)
        self.output = make_layer(64, 1, 1, 0, 1)

    def forward(self, x):
        output = self.upconv3(self.upconv2(self.upconv1(x)))
        output = self.output(output)
        return output


class CL_DCNN(nn.Module):
    def __init__(self):
        super(CL_DCNN, self).__init__()
        
        self.features = nn.Sequential(
            *(list(list(models.vgg16_bn(False).children())[0].children())[0:33])
        )
        self.dilate_module = DilateModule()
        self.decoder = Output()
        initialize_weights(self.modules())
        
    def forward(self, x):
        # Frontend
        features = self.features(x)

        # Dilate module
        d1 = self.dilate_module(features)

        # Output
        output = self.decoder(d1)

        return output


def make_layer(in_channel, out_channel, kernel_size, padding, stride, dilate=1, is_bn=True):
    layers = []
    conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilate)
    if is_bn:
        layers += [conv, nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True)]
    else:
        layers += [conv, nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


def make_contrans_layer(in_channel, out_channel, kernel_size, padding, stride, is_bn=True):
    layers = []
    conv = nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, padding=padding,
                     stride=stride)
    if is_bn:
        layers += [conv, nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True)]
    else:
        layers += [conv, nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)

# input = torch.randn((1, 3, 128, 128))
# model = CL_DCNN()
# output = model(input)
# print(model)
# print(output.shape)
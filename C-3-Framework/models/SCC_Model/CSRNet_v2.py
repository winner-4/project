import torch.nn as nn
import torch
from torchvision import models
import torch.nn.functional as F
from torchsummary import summary
class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        self.seen = 0
        self.frontend_feat = [('conv3', 32), ('conv3', 32), ('dconv', 32), ('conv1', 16),
                             ('conv3', 64), ('conv3', 64), ('dconv', 64), ('conv1', 32),
                             ('conv3', 128), ('conv3', 128), ('conv3', 128), ('dconv', 128),
                             ('conv1', 64), ('conv3', 256)]
        self.frontend = make_layers(self.frontend_feat, batch_norm=True)
        self.output_layer = nn.Conv2d(256, 1, kernel_size=1)
        if not load_weights:
            self._initialize_weights()
    def forward(self,x):
        x = self.frontend(x)
        x = self.output_layer(x)
        return x
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
                
def make_layers(cfg, in_channels = 3,batch_norm=False):
    layers = []
    for val in cfg:
        shape, v = val
        if shape == 'conv3':
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, dilation = 1)
        elif shape == 'conv1':
            conv2d = nn.Conv2d(in_channels, v, kernel_size=1, dilation = 1)
        elif shape == 'dconv':
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=2, dilation = 2)
            
        if batch_norm:
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
        else:
            layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = v
        
    return nn.Sequential(*layers)                



class DSNet(nn.Module):
    def __init__(self):
        super(DSNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 9, (1, 1))
        self.dw1 = nn.Sequential(
            nn.Conv2d(9, 9, kernel_size=(5, 5),
                                stride=(1, 1), padding=1, groups=9, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(9),
            nn.Conv2d(9, 27, kernel_size=(1,1), padding=1, stride=(1,1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(27),
        )

        self.max_pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

        self.dw2 = nn.Sequential(
            nn.Conv2d(72, 72, kernel_size=(3, 3),
                      stride=(1, 1), groups=72, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(72),
            nn.Conv2d(72, 144, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(144),
        )

        self.dw3 = nn.Sequential(
            nn.Conv2d(216, 216, kernel_size=(3, 3),
                      stride=(1, 1), padding=1, groups=216, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(216),
            nn.Conv2d(216, 216, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(216),
        )
        
        self.output = nn.Conv2d(216, 1, (1,1))
        
        

    def forward(self, x):

        x1 = self.conv1(x)
        x2 = self.dw1(x1)

        x3 = self.max_pool(torch.cat((x1, x2), dim=1))
        out1 = self.max_pool(x1)
        out2 = self.max_pool(x2)

        x4 = self.dw2(torch.cat((out1, out2, x3), dim=1))

        x = self.dw3(torch.cat([out1, out2, x3, x4], dim=1))
        
        x = self.output(x)
        x = F.upsample(x,scale_factor=2)

        return x


if __name__ == '__main__':
    x = torch.randn(1, 3, 128, 128)
    model = DSNet().cuda()
#     summary(model, input_size=(3, 128, 128))
    print(model(x).size())
# x = torch.Tensor(1, 3, 64, 64)
# net = CSRNet()
# print(net)
# y = net(x)
# print(y.shape)
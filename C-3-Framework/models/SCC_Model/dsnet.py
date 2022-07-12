import torch
import torch.nn as nn
import torch.nn.functional as F

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
                      stride=(1, 1), groups=216, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(216),
            nn.Conv2d(216, 216, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(216),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(216, 25)
        )

    def forward(self, x):

        x1 = self.conv1(x)
        x2 = self.dw1(x1)

        x3 = self.max_pool(torch.cat((x1, x2), dim=1))
        out1 = self.max_pool(x1)
        out2 = self.max_pool(x2)

        x4 = self.dw2(torch.cat((out1, out2, x3), dim=1))

        x = self.dw3(torch.concat([out1, out2, x3, x4], dim=1))

        x = self.avg_pool(x)
        x = nn.Flatten()(x)
        x = self.classifier(x)

        return x


if __name__ == '__main__':
    x = torch.randn(32, 3, 128, 128)
    model = DSNet()
    print(model(x).size())

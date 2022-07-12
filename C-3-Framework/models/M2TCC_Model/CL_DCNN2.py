import torch
import torch.nn as nn
#from torchsummary import summary
class CL_DCNN(nn.Module):
    def __init__(self):
        super(CL_DCNN, self).__init__()
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2)
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(64, 128, (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2)
        )
        self.conv3_3 = nn.Sequential(
            nn.Conv2d(128, 256, (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2)
        )
        self.conv4_3 = nn.Sequential(
            nn.Conv2d(256, 512, (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.dconv1 = nn.Sequential(
            nn.Conv2d(512, 256, (3, 3), stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.c1 = nn.Sequential(
            nn.Conv2d(512, 256, (1, 1), stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.Upconv1 = nn.Upsample(scale_factor=2)
        # self.Upconv1 = nn.Sequential(
        #     nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU()
        # )
        self.dconv2 = nn.Sequential(
            nn.Conv2d(256, 128, (3, 3), stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(256, 128, (1, 1), stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.Upconv2 = nn.Upsample(scale_factor=2)
        # self.Upconv2 = nn.Sequential(
        #     nn.ConvTranspose2d(128, 128, (3, 3), stride=2, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU()
        # )
        self.dconv3 = nn.Sequential(
            nn.Conv2d(128, 64, (3, 3), stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.c3 = nn.Sequential(
            nn.Conv2d(128, 64, (1, 1), stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.Upconv3 = nn.Upsample(scale_factor=2)
        # self.Upconv3 = nn.Sequential(
        #     nn.ConvTranspose2d(64, 64, (3, 3), stride=2, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU()
        # )

        self.dconv4 = nn.Sequential(
            nn.Conv2d(64, 32, (3, 3), stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, (3, 3), stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8, (3, 3), stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )


        self.Output = nn.Conv2d(64, 1, (1, 1), stride=1, padding=0)

    def forward(self, x):
        # Frontend
        c12 = self.conv1_2(x)
        c22 = self.conv2_2(c12)
        c33 = self.conv3_3(c22)
        c43 = self.conv4_3(c33)

        # Backend
        dc1 = self.dconv1(c43)
        ac1 = c33 + dc1
        # ac1 = torch.cat((c33, dc1), dim=1)
        # ac1 = self.c1(ac1)

        uc1 = self.Upconv1(ac1)
        dc2 = self.dconv2(uc1)
        ac2 = c22 + dc2
        # ac2 = torch.cat((c22, dc2), dim=1)
        # ac2 = self.c2(ac2)


        uc2 = self.Upconv2(ac2)
        dc3 = self.dconv3(uc2)
        ac3 = c12 + dc3
        # ac3 = torch.cat((c12, dc3), dim=1)
        # ac3 = self.c3(ac3)

        uc3 = self.Upconv3(ac3)

        # dc4 = self.dconv4(uc3)

        # Output
        output = self.Output(uc3)
        return output




# input = torch.randn((1, 3, 640, 640))
# model = CL_DCNN()
# summary(model, input_size=(3, 640, 640))
# output = model(input)
# print(model)
# print(output.shape)
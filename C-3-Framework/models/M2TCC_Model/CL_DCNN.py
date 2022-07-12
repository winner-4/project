import torch
import torch.nn as nn
# from torchsummary import summary
from misc.utils import initialize_weights


# 1_1_0
# class CL_DCNN(nn.Module):
#     def __init__(self):
#         super(CL_DCNN, self).__init__()
#         self.conv1_2 = nn.Sequential(
#             nn.Conv2d(3, 64, (3, 3), stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, (3, 3), stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d((2, 2), stride=2)
#         )
#         self.conv2_2 = nn.Sequential(
#             nn.Conv2d(64, 128, (3, 3), stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.Conv2d(128, 128, (3, 3), stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d((2, 2), stride=2)
#         )
#         self.conv3_3 = nn.Sequential(
#             nn.Conv2d(128, 256, (3, 3), stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.Conv2d(256, 256, (3, 3), stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.Conv2d(256, 256, (3, 3), stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.MaxPool2d((2, 2), stride=2)
#         )
#         self.conv4_3 = nn.Sequential(
#             nn.Conv2d(256, 512, (3, 3), stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.Conv2d(512, 512, (3, 3), stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.Conv2d(512, 512, (3, 3), stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU()
#         )
#         self.dconv1 = nn.Sequential(
#             nn.Conv2d(512, 256, (3, 3), stride=1, padding=2, dilation=2),
#             nn.BatchNorm2d(256),
#             nn.ReLU()
#         )
#         self.c1 = nn.Sequential(
#             nn.Conv2d(512, 256, (1, 1), stride=1, padding=0),
#             nn.BatchNorm2d(256),
#             nn.ReLU()
#         )

#         # self.Upconv1 = nn.Upsample(scale_factor=2)
#         self.Upconv1 = nn.Sequential(
#             nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU()
#         )
#         self.dconv2 = nn.Sequential(
#             nn.Conv2d(256, 128, (3, 3), stride=1, padding=2, dilation=2),
#             nn.BatchNorm2d(128),
#             nn.ReLU()
#         )
#         self.c2 = nn.Sequential(
#             nn.Conv2d(256, 128, (1, 1), stride=1, padding=0),
#             nn.BatchNorm2d(128),
#             nn.ReLU()
#         )

#         # self.Upconv2 = nn.Upsample(scale_factor=2)
#         self.Upconv2 = nn.Sequential(
#             nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU()
#         )
#         self.dconv3 = nn.Sequential(
#             nn.Conv2d(128, 64, (3, 3), stride=1, padding=2, dilation=2),
#             nn.BatchNorm2d(64),
#             nn.ReLU()
#         )
#         self.c3 = nn.Sequential(
#             nn.Conv2d(128, 64, (1, 1), stride=1, padding=0),
#             nn.BatchNorm2d(64),
#             nn.ReLU()
#         )

#         # self.Upconv3 = nn.Upsample(scale_factor=2)
#         self.Upconv3 = nn.Sequential(
#             nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU()
#         )

#         # self.dconv4 = nn.Sequential(
#         #     nn.Conv2d(64, 32, (3, 3), stride=1, padding=2, dilation=2),
#         #     nn.BatchNorm2d(32),
#         #     nn.ReLU(),
#         #     nn.Conv2d(32, 16, (3, 3), stride=1, padding=2, dilation=2),
#         #     nn.BatchNorm2d(16),
#         #     nn.ReLU(),
#         #     nn.Conv2d(16, 8, (3, 3), stride=1, padding=2, dilation=2),
#         #     nn.BatchNorm2d(8),
#         #     nn.ReLU()
#         # )


#         # self.Output = nn.Conv2d(8, 1, (1, 1), stride=1, padding=0)
#         self.Output = nn.Conv2d(64, 1, (1, 1), stride=1, padding=0)

#     def forward(self, x):
#         # Frontend
#         c12 = self.conv1_2(x)
#         c22 = self.conv2_2(c12)
#         c33 = self.conv3_3(c22)
#         c43 = self.conv4_3(c33)

#         # Backend
#         dc1 = self.dconv1(c43)
#         ac1 = torch.cat((c33, dc1), dim=1)
#         ac1 = self.c1(ac1)

#         uc1 = self.Upconv1(ac1)
#         dc2 = self.dconv2(uc1)
#         ac2 = torch.cat((c22, dc2), dim=1)
#         ac2 = self.c2(ac2)


#         uc2 = self.Upconv2(ac2)
#         dc3 = self.dconv3(uc2)
#         ac3 = torch.cat((c12, dc3), dim=1)
#         ac3 = self.c3(ac3)

#         uc3 = self.Upconv3(ac3)

#         # dc4 = self.dconv4(uc3)

#         # Output
#         output = self.Output(uc3)
#         return output


# 1_1_1 ~ 1_1_3
# class CL_DCNN(nn.Module):
#     def __init__(self):
#         super(CL_DCNN, self).__init__()
#         self.conv1_2 = nn.Sequential(
#             nn.Conv2d(3, 64, (3, 3), stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, (3, 3), stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d((2, 2), stride=2)
#         )
#         self.conv2_2 = nn.Sequential(
#             nn.Conv2d(64, 128, (3, 3), stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.Conv2d(128, 128, (3, 3), stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d((2, 2), stride=2)
#         )
#         self.conv3_3 = nn.Sequential(
#             nn.Conv2d(128, 256, (3, 3), stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.Conv2d(256, 256, (3, 3), stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.Conv2d(256, 256, (3, 3), stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.MaxPool2d((2, 2), stride=2)
#         )
#         self.conv4_3 = nn.Sequential(
#             nn.Conv2d(256, 512, (3, 3), stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.Conv2d(512, 512, (3, 3), stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.Conv2d(512, 512, (3, 3), stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU()
#         )
#         self.dconv1 = nn.Sequential(
#             nn.Conv2d(512, 256, (3, 3), stride=1, padding=2, dilation=2),
#             nn.BatchNorm2d(256),
#             nn.ReLU()
#         )
#         self.c1 = nn.Sequential(
#             nn.Conv2d(512, 256, (1, 1), stride=1, padding=0),
#             nn.BatchNorm2d(256),
#             nn.ReLU()
#         )

#         
#         self.Upconv1 = nn.Sequential(
#             nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU()
#         )
#         self.dconv2 = nn.Sequential(
#             nn.Conv2d(128, 64, (3, 3), stride=1, padding=2, dilation=2),
#             nn.BatchNorm2d(64),
#             nn.ReLU()
#         )
#         self.c2 = nn.Sequential(
#             nn.Conv2d(192, 96, (1, 1), stride=1, padding=0),
#             nn.BatchNorm2d(96),
#             nn.ReLU()
#         )

#         
#         self.Upconv2 = nn.Sequential(
#             nn.ConvTranspose2d(96, 48, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(48),
#             nn.ReLU()
#         )
#         self.dconv3 = nn.Sequential(
#             nn.Conv2d(48, 24, (3, 3), stride=1, padding=2, dilation=2),
#             nn.BatchNorm2d(24),
#             nn.ReLU()
#         )
#         self.c3 = nn.Sequential(
#             nn.Conv2d(88, 44, (1, 1), stride=1, padding=0),
#             nn.BatchNorm2d(44),
#             nn.ReLU()
#         )

#         
#         self.Upconv3 = nn.Sequential(
#             nn.ConvTranspose2d(44, 22, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(22),
#             nn.ReLU()
#         )

#         
#         self.Output = nn.Conv2d(22, 1, (1, 1), stride=1, padding=0)
#         initialize_weights(self.modules())
        
#     def forward(self, x):
#         # Frontend
#         c12 = self.conv1_2(x)
#         c22 = self.conv2_2(c12)
#         c33 = self.conv3_3(c22)
#         c43 = self.conv4_3(c33)

#         # Backend
#         dc1 = self.dconv1(c43)
#         ac1 = torch.cat((c33, dc1), dim=1)
#         ac1 = self.c1(ac1)

#         uc1 = self.Upconv1(ac1)
#         dc2 = self.dconv2(uc1)
#         ac2 = torch.cat((c22, dc2), dim=1)
#         ac2 = self.c2(ac2)


#         uc2 = self.Upconv2(ac2)
#         dc3 = self.dconv3(uc2)
#         ac3 = torch.cat((c12, dc3), dim=1)
#         ac3 = self.c3(ac3)

#         uc3 = self.Upconv3(ac3)

#         # Output
#         output = self.Output(uc3)
#         return output


# 1_1_4 参数太多 10,631,825
# class CL_DCNN(nn.Module):
#     def __init__(self):
#         super(CL_DCNN, self).__init__()
        
#         self.maxpool = nn.MaxPool2d((2, 2), stride=2)
        
#         self.conv1_2 = nn.Sequential(
#             nn.Conv2d(3, 64, (3, 3), stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, (3, 3), stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU()
#         )
#         self.conv2_2 = nn.Sequential(
#             nn.Conv2d(64, 128, (3, 3), stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.Conv2d(128, 128, (3, 3), stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU()
#         )
#         self.conv3_3 = nn.Sequential(
#             nn.Conv2d(128, 256, (3, 3), stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.Conv2d(256, 256, (3, 3), stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.Conv2d(256, 256, (3, 3), stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU()
#         )
#         self.conv4_3 = nn.Sequential(
#             nn.Conv2d(256, 512, (3, 3), stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.Conv2d(512, 512, (3, 3), stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.Conv2d(512, 512, (3, 3), stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU()
#         )
        
#         self.dconv1 = nn.Sequential(
#             nn.Conv2d(512, 256, (3, 3), stride=1, padding=2, dilation=2),
#             nn.BatchNorm2d(256),
#             nn.ReLU()
#         )
#         self.Upconv1 = nn.Sequential(
#             nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU()
#         )

        
        
#         self.dconv2 = nn.Sequential(
#             nn.Conv2d(384, 192, (3, 3), stride=1, padding=2, dilation=2),
#             nn.BatchNorm2d(192),
#             nn.ReLU()
#         )
#         self.Upconv2 = nn.Sequential(
#             nn.ConvTranspose2d(192, 96, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(96),
#             nn.ReLU()
#         )
        
        
        
#         self.dconv3 = nn.Sequential(
#             nn.Conv2d(224, 112, (3, 3), stride=1, padding=2, dilation=2),
#             nn.BatchNorm2d(112),
#             nn.ReLU()
#         )
#         self.Upconv3 = nn.Sequential(
#             nn.ConvTranspose2d(112, 56, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(56),
#             nn.ReLU()
#         )
        
        
#         self.Output = nn.Conv2d(120, 1, (1, 1), stride=1, padding=0)
# #         initialize_weights(self.modules())
        
        
        
#     def forward(self, x):
#         # Frontend
#         c12 = self.conv1_2(x)
#         c12_m = self.maxpool(c12)
#         c22 = self.conv2_2(c12_m)
#         c22_m = self.maxpool(c22)
#         c33 = self.conv3_3(c22_m)
#         c33_m = self.maxpool(c33)
#         c43 = self.conv4_3(c33_m)

#         # Backend
#         dc1 = self.dconv1(c43)
#         uc1 = self.Upconv1(dc1)
#         ac1 = torch.cat((c33, uc1), dim=1)

        
#         dc2 = self.dconv2(ac1)
#         uc2 = self.Upconv2(dc2)
#         ac2 = torch.cat((c22, uc2), dim=1)
        
        
#         dc3 = self.dconv3(ac2)
#         uc3 = self.Upconv3(dc3)
#         ac3 = torch.cat((c12, uc3), dim=1)
        
#         # Output
#         output = self.Output(ac3)
#         return output


# 1_1_4 改良版的 9,726,453
class CL_DCNN(nn.Module):
    def __init__(self):
        super(CL_DCNN, self).__init__()
        
        self.maxpool = nn.MaxPool2d((2, 2), stride=2)
        
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(64, 128, (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
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
            nn.ReLU()
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
        self.Upconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        
        self.c1 = nn.Sequential(
            nn.Conv2d(384, 192, (1, 1), stride=1, padding=0),
            nn.BatchNorm2d(192),
            nn.ReLU()
        )
        self.dconv2 = nn.Sequential(
            nn.Conv2d(192, 96, (3, 3), stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(96),
            nn.ReLU()
        )
        self.Upconv2 = nn.Sequential(
            nn.ConvTranspose2d(96, 48, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )
        
        
        self.c2 = nn.Sequential(
            nn.Conv2d(176, 88, (1, 1), stride=1, padding=0),
            nn.BatchNorm2d(88),
            nn.ReLU()
        )
        self.dconv3 = nn.Sequential(
            nn.Conv2d(88, 44, (3, 3), stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(44),
            nn.ReLU()
        )
        self.Upconv3 = nn.Sequential(
            nn.ConvTranspose2d(44, 22, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(22),
            nn.ReLU()
        )
        
        
        self.Output = nn.Conv2d(86, 1, (1, 1), stride=1, padding=0)
        initialize_weights(self.modules())
        
        
        
    def forward(self, x):
        # Frontend
        c12 = self.conv1_2(x)
        c12_m = self.maxpool(c12)
        c22 = self.conv2_2(c12_m)
        c22_m = self.maxpool(c22)
        c33 = self.conv3_3(c22_m)
        c33_m = self.maxpool(c33)
        c43 = self.conv4_3(c33_m)

        # Backend
        dc1 = self.dconv1(c43)
        uc1 = self.Upconv1(dc1)
        ac1 = torch.cat((c33, uc1), dim=1)

        c1 = self.c1(ac1)
        dc2 = self.dconv2(c1)
        uc2 = self.Upconv2(dc2)
        ac2 = torch.cat((c22, uc2), dim=1)
        
        c2 = self.c2(ac2)
        dc3 = self.dconv3(c2)
        uc3 = self.Upconv3(dc3)
        ac3 = torch.cat((c12, uc3), dim=1)
        
        # Output
        output = self.Output(ac3)
        return output    
    
# input = torch.randn((1, 3, 128, 128)).cuda()
# model = CL_DCNN().cuda()
# summary(model, input_size=(3, 128, 128))
# output = model(input)
# print(model)
# print(output.shape)
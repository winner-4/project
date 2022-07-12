# 1.模型输出层形状

```python
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 64, 640, 640]           1,792
       BatchNorm2d-2         [-1, 64, 640, 640]             128
              ReLU-3         [-1, 64, 640, 640]               0
            Conv2d-4         [-1, 64, 640, 640]          36,928
       BatchNorm2d-5         [-1, 64, 640, 640]             128
              ReLU-6         [-1, 64, 640, 640]               0
         MaxPool2d-7         [-1, 64, 320, 320]               0
            Conv2d-8        [-1, 128, 320, 320]          73,856
       BatchNorm2d-9        [-1, 128, 320, 320]             256
             ReLU-10        [-1, 128, 320, 320]               0
           Conv2d-11        [-1, 128, 320, 320]         147,584
      BatchNorm2d-12        [-1, 128, 320, 320]             256
             ReLU-13        [-1, 128, 320, 320]               0
        MaxPool2d-14        [-1, 128, 160, 160]               0
           Conv2d-15        [-1, 256, 160, 160]         295,168
      BatchNorm2d-16        [-1, 256, 160, 160]             512
             ReLU-17        [-1, 256, 160, 160]               0
           Conv2d-18        [-1, 256, 160, 160]         590,080
      BatchNorm2d-19        [-1, 256, 160, 160]             512
             ReLU-20        [-1, 256, 160, 160]               0
           Conv2d-21        [-1, 256, 160, 160]         590,080
      BatchNorm2d-22        [-1, 256, 160, 160]             512
             ReLU-23        [-1, 256, 160, 160]               0
        MaxPool2d-24          [-1, 256, 80, 80]               0
           Conv2d-25          [-1, 512, 80, 80]       1,180,160
      BatchNorm2d-26          [-1, 512, 80, 80]           1,024
             ReLU-27          [-1, 512, 80, 80]               0
           Conv2d-28          [-1, 512, 80, 80]       2,359,808
      BatchNorm2d-29          [-1, 512, 80, 80]           1,024
             ReLU-30          [-1, 512, 80, 80]               0
           Conv2d-31          [-1, 512, 80, 80]       2,359,808
      BatchNorm2d-32          [-1, 512, 80, 80]           1,024
             ReLU-33          [-1, 512, 80, 80]               0
           Conv2d-34          [-1, 256, 80, 80]       1,179,904
      BatchNorm2d-35          [-1, 256, 80, 80]             512
             ReLU-36          [-1, 256, 80, 80]               0
           Conv2d-37          [-1, 256, 80, 80]         131,328
      BatchNorm2d-38          [-1, 256, 80, 80]             512
             ReLU-39          [-1, 256, 80, 80]               0
         Upsample-40        [-1, 256, 160, 160]               0
           Conv2d-41        [-1, 128, 160, 160]         295,040
      BatchNorm2d-42        [-1, 128, 160, 160]             256
             ReLU-43        [-1, 128, 160, 160]               0
           Conv2d-44        [-1, 128, 160, 160]          32,896
      BatchNorm2d-45        [-1, 128, 160, 160]             256
             ReLU-46        [-1, 128, 160, 160]               0
         Upsample-47        [-1, 128, 320, 320]               0
           Conv2d-48         [-1, 64, 320, 320]          73,792
      BatchNorm2d-49         [-1, 64, 320, 320]             128
             ReLU-50         [-1, 64, 320, 320]               0
           Conv2d-51         [-1, 64, 320, 320]           8,256
      BatchNorm2d-52         [-1, 64, 320, 320]             128
             ReLU-53         [-1, 64, 320, 320]               0
         Upsample-54         [-1, 64, 640, 640]               0
           Conv2d-55         [-1, 32, 640, 640]          18,464
      BatchNorm2d-56         [-1, 32, 640, 640]              64
             ReLU-57         [-1, 32, 640, 640]               0
           Conv2d-58         [-1, 16, 640, 640]           4,624
      BatchNorm2d-59         [-1, 16, 640, 640]              32
             ReLU-60         [-1, 16, 640, 640]               0
           Conv2d-61          [-1, 8, 640, 640]           1,160
      BatchNorm2d-62          [-1, 8, 640, 640]              16
             ReLU-63          [-1, 8, 640, 640]               0
           Conv2d-64          [-1, 1, 640, 640]               9
================================================================
Total params: 9,388,017
Trainable params: 9,388,017
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 4.69
Forward/backward pass size (MB): 3965.62
Params size (MB): 35.81
Estimated Total Size (MB): 4006.12
----------------------------------------------------------------
CL_DCNN(
  (conv1_2): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU()
    (6): MaxPool2d(kernel_size=(2, 2), stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (conv2_2): Sequential(
    (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU()
    (6): MaxPool2d(kernel_size=(2, 2), stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (conv3_3): Sequential(
    (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU()
    (6): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (8): ReLU()
    (9): MaxPool2d(kernel_size=(2, 2), stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (conv4_3): Sequential(
    (0): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU()
    (6): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (8): ReLU()
  )
  (dconv1): Sequential(
    (0): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2))
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
  )
  (c1): Sequential(
    (0): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
  )
  (Upconv1): Upsample(scale_factor=2.0, mode=nearest)
  (dconv2): Sequential(
    (0): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2))
    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
  )
  (c2): Sequential(
    (0): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
  )
  (Upconv2): Upsample(scale_factor=2.0, mode=nearest)
  (dconv3): Sequential(
    (0): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
  )
  (c3): Sequential(
    (0): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
  )
  (Upconv3): Upsample(scale_factor=2.0, mode=nearest)
  (dconv4): Sequential(
    (0): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2))
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2))
    (4): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU()
    (6): Conv2d(16, 8, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2))
    (7): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (8): ReLU()
  )
  (Output): Conv2d(8, 1, kernel_size=(1, 1), stride=(1, 1))
)
```

# 2.实验

## AUTODL服务器

```shell
# AUTODL服务器
# ssh -p 端口号 用户名@Id
ssh -p 12881 root@region-11.autodl.com 
ssh -p 10365 root@region-11.autodl.com 

# 连接xshell，命令行输入
ssh root@region-11.autodl.com 12881
ssh root@region-11.autodl.com 10365

# 连接FileZilla
文件 --> 站点管理器 --> 选择SFTP协议 --> 输入相应参数
```

![image-20220506090517469](../../../../AppData/Roaming/Typora/typora-user-images/image-20220506090517469.png)

## 安装C3 Framework框架

```shell
# 基础环境
python 3.8.10 + pytorch 1.7.0 + cuda 11.0

git clone https://github.com/gjy3035/C-3-Framework.git

# 安装第三方库
pip install -r requirements.txt
```

在本地下载好预处理过的[数据集](https://mailnwpueducn-my.sharepoint.com/personal/gjy3035_mail_nwpu_edu_cn/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fgjy3035%5Fmail%5Fnwpu%5Fedu%5Fcn%2FDocuments%2F%E8%AE%BA%E6%96%87%E5%BC%80%E6%BA%90%E6%95%B0%E6%8D%AE%2FC3Data)，放在../ProcessedData文件夹下

## 运行自己代码

### 1）确定自己的网络是单/双输出、单/双损失

| 输出（单/双） | 损失（单/双） | 网络类型 |
| :-----------: | :-----------: | :------: |
|      单       |      单       |   SCC    |
|      单       |      双       |  M2TCC   |
|      双       |      双       |   CMTL   |

CL-DCNN属于第二种

在train.py 53行添加上自己的网络

​                               ![image-20220505151225198](../../../../AppData/Roaming/Typora/typora-user-images/image-20220505151225198.png)

 在trainer_for_M2TCC.py 26行添加上自己的网络

![image-20220505151242843](../../../../AppData/Roaming/Typora/typora-user-images/image-20220505151242843.png)

### 2） 修改config.py相关的参数

### 3）  运行代码

python train.py 



# 3.数据集

| 数据集 |
| :----: |
|  SHHA  |
|  SHHB  |
|  MALL  |
|  QNRF  |
| UCF50  |

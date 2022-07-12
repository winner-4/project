from matplotlib import pyplot as plt

import matplotlib
import os
import random
import torch
from torch.autograd import Variable
import torchvision.transforms as standard_transforms
import misc.transforms as own_transforms
import pandas as pd

from models.CC import CrowdCounter
from config import cfg
from misc.utils import *
import scipy.io as sio
from PIL import Image, ImageOps
from qualitycc import calc_psnr, calc_ssim
from datasets.SHHA.loading_data import loading_data 
from datasets.SHHA.setting import cfg_data 
    
    
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True


train_loader, val_loader, restore_transform = loading_data()

mean_std = ([0.446139603853, 0.409515678883, 0.395083993673], 
            [0.288205742836, 0.278144598007, 0.283502370119])
factor = 1
LOG_PARA = 100.0


img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
gt_transform = standard_transforms.Compose([
        own_transforms.GTScaleDown(factor),  # GT下采样
        own_transforms.LabelNormalize(LOG_PARA) # GT*放大因子100
    ])
restore = standard_transforms.Compose([
        own_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])
pil_to_tensor = standard_transforms.ToTensor()


dataRoot = '../ProcessedData/Shanghai_proA'

# model_path = 'exp/24_05-23_10-10_SHHA_CL_DCNN_1e-05/latest_state.pth'
best_model_path = 'exp/24_05-23_10-10_SHHA_CL_DCNN_1e-05/all_ep_118_mae_55.2_mse_93.3.pth'



def test(val_gt_file, model_path):

    num_samples = len(os.listdir(val_gt_file))
    print('Total smaples nums:', num_samples-1)
    print('length of val_loader:', len(val_loader))
    ssim = psnr = 0
    
    # net相关
    net = CrowdCounter(cfg.GPU_ID, 'CL_DCNN')
    net.cuda()
    net.load_state_dict(torch.load(best_model_path))
    net.eval()
    
    
    for vi, data in enumerate(val_loader, 0):
        print(vi)
        img, gt_map = data
        
        with torch.no_grad():
            img = Variable(img).cuda()
            gt_map = Variable(gt_map).cuda()
            
            pred_map = net.forward(img, gt_map)
            
            pred_map = pred_map.cpu().data.numpy()[0,0,:,:]
#             pred_map = pred_map.data.cpu().numpy()
            gt_map = gt_map.squeeze().data.cpu().numpy()
            
            print(f'gt_map.shape:{np.shape(gt_map)}, pred_map.shape:{np.shape(pred_map)}')
            psnr += calc_psnr(gt_map, pred_map)
            ssim += calc_ssim(gt_map, pred_map)
            
    # 输出最后结果
    print('avg_psnr:', psnr / num_samples)
    print('avg_ssim:', ssim / num_samples)
                
    
    
    
#     for i, gt_csv in enumerate(sorted(os.listdir(val_gt_file))):    
        
#         print(i+1)
#         # gt相关
#         if gt_csv.endswith('.csv'):
#             gt_csv_path = os.path.join(val_gt_file, gt_csv)
#             gt_csv = pd.read_csv(gt_csv_path, engine='python', sep=',',header=None).values
# #             print(np.shape(gt_csv))
#             gt_map = gt_csv.astype(np.float32, copy=False)    
#             gt_map = Image.fromarray(gt_map)  
#             gt_map = gt_transform(gt_map)
#             # img相关
#             imgname = gt_csv_path.replace('den', 'img').replace('.csv', '.jpg')
#             img = Image.open(imgname)
#             if img.mode == 'L':
#                 img = img.convert('RGB')
#             img = img_transform(img)[None, :, :, :]
            
            
#             print(f'img.shape:{np.shape(img)}, gt_map.shape:{np.shape(gt_map)}')
#             # 前向传播
#             with torch.no_grad():
#                 img = Variable(img).cuda()
#                 gt_map = Variable(gt_map).cuda()
#                 pred_map = net(img, gt_map)
                
            
#             # 计算ssim、psnr
#             pred_map = pred_map.cpu().data.numpy()[0,0,:,:]
            
#             gt_map = gt_map.cpu().data.numpy()
#             psnr += calc_psnr(gt_map, pred_map)
#             ssim += calc_ssim(gt_map, pred_map)
        
    
    



def main():
    print('Start cal')
    print('-'*40)
    val_gt_file = os.path.join(dataRoot, 'test', 'den')
    test(val_gt_file, best_model_path)
    print('-'*40)
    print('End cal')

if __name__ == '__main__':
    main()

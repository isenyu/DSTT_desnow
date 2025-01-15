
import time
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset_processing.train_data import  K12_dataset
from utils import *
from models.finalNet import final_net
import cv2
import os
import random

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for GridDehazeNet')
parser.add_argument('-train_data_dir', help='taraining data path', default='/Datasets/kitti2012/training/')
parser.add_argument('-checkpoints_dir', help='training checkpoints path', default='./checkpoint/K12/')
parser.add_argument('-output_img_dir', help='desonowing output path', default='./output_img/K12/')


parser.add_argument('-learning_rate', help='Set the learning rate', default=0.0002, type=float)
parser.add_argument('-crop_size', help='Set the crop_size', default=240, nargs='+', type=int)
parser.add_argument('-train_batch_size', help='Set the training batch size', default=1, type=int)
parser.add_argument('-frame_num', help='Set the clip long', default=3, type=int)
parser.add_argument('-step', help='Set the sample step', default=1, type=int)
parser.add_argument('-checkpoint_name', help='Set the saved checkpoint name', default='/checkpoint_K12_epoch_{0}', type=str)



parser.add_argument('-network_height', help='Set the network height (row)', default=3, type=int)
parser.add_argument('-network_width', help='Set the network width (column)', default=6, type=int)
parser.add_argument('-num_dense_layer', help='Set the number of dense layer in RDB', default=4, type=int)
parser.add_argument('-window_size', help='Set window size', default=8, type=int)


args = parser.parse_args()

frame_num = args.frame_num
step = args.step
train_data_dir = args.train_data_dir
checkpoints_dir = args.checkpoints_dir
output_img_dir = args.output_img_dir
checkpoint_name = args.checkpoint_name
window_size = args.window_size

if  not os.path.exists(train_data_dir):
    print('Can not find training data!!!')
if not os.path.exists(checkpoints_dir):
    os.makedirs(checkpoints_dir)
if not os.path.exists(output_img_dir):
    os.makedirs(output_img_dir)


def set_seed(seed):
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False

# set_seed(1)
learning_rate = args.learning_rate
crop_size = args.crop_size
train_batch_size = args.train_batch_size

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device_ids = [Id for Id in range(torch.cuda.device_count())]
print(device_ids)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
height, width = crop_size, crop_size

## model
model = final_net(height=height, width=width, window_size=window_size)
model.to(device)
model = nn.DataParallel(model, device_ids=device_ids)

## build optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Load the data
train_data_loader = DataLoader(K12_dataset(crop_size=crop_size, frame_num=frame_num, step=step, root=train_data_dir), batch_size=train_batch_size, shuffle=True, num_workers=10, drop_last=True)

epoch = 0
num_epochs = 100
while epoch < num_epochs:
    adjust_learning_rate(optimizer, epoch)

    for batch_id, train_data in enumerate(train_data_loader):
        left_running_psnr = []
        right_running_psnr = []
        haze, gt, haze2, gt2 = train_data
        haze = haze.to(device)
        gt = gt.to(device)
        haze2 = haze2.to(device)
        gt2 = gt2.to(device)
        # --- Zero the parameter gradients --- #
        optimizer.zero_grad()
        dehaze, dehaze2 = model(haze, haze2)
        dehaze, _, _ = cubes_2_maps(dehaze)
        dehaze2, _, _ = cubes_2_maps(dehaze2)
        gt, _, _ = cubes_2_maps(gt)
        gt2, _, _ = cubes_2_maps(gt2)

        total_loss = 0
        smooth_loss = F.smooth_l1_loss(dehaze, gt)
        smooth_loss2 = F.smooth_l1_loss(dehaze2, gt2)
        total_loss = smooth_loss + smooth_loss2
        total_loss.backward()
        optimizer.step()
        left_running_psnr.extend(to_psnr(dehaze, gt))
        right_running_psnr.extend(to_psnr(dehaze2, gt2))
        if batch_id % 100 == 0:
            print('Epoch: {0}, Iteration: {1}, loss = {2}'.format(epoch, batch_id, total_loss))
            print('left_psnr = {0}, right_psnr ={1}, '.format(sum(left_running_psnr)/len(left_running_psnr), sum(right_running_psnr)/len(right_running_psnr)))
    epoch += 1
    if epoch % 5 == 0:
        torch.save(model, checkpoints_dir + checkpoint_name.format(epoch))


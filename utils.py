# --- Imports --- #
import time
import torch
import torch.nn.functional as F
import math
from math import log10
from skimage import measure
import numpy as np
import cv2
from  skimage import measure



def cubes_2_maps(cubes):
    b, c, d, h, w = cubes.shape
    cubes = cubes.permute(0, 2, 1, 3, 4)

    return cubes.contiguous().view(b*d, c, h, w), b, d


def to_psnr(dehaze, gt):
   # _,_, dehaze = cubes_2_maps(dehaze)
   # _,_, gt = cubes_2_maps(gt)
    #print(dehaze.shape, gt.shape)
    # dehaze = np.array(dehaze.detach().cpu())
    # gt = np.array(gt.detach().cpu())
    # print(dehaze[3,:,:,:], gt[3,:,:,:])
    # mse_list = [np.mean((dehaze[i, :, :, :] - gt[i, :, :, :]) ** 2) for i in range(dehaze.shape[0])]
    mse = F.mse_loss(dehaze, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]
    intensity_max = 1.0
    # print(mse_list)
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list


def to_ssim_skimage(dehaze, gt):
    dehaze_list = torch.split(dehaze, 1, dim=0)
    gt_list = torch.split(gt, 1, dim=0)
    dehaze_list_np = [dehaze_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(dehaze_list))]
    gt_list_np = [gt_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(dehaze_list))]
    ssim_list = [measure.compare_ssim(dehaze_list_np[ind],  gt_list_np[ind], data_range=1, multichannel=True) for ind in range(len(dehaze_list))]
    return ssim_list


def save_image_tensor2cv2(input_tensor: torch.Tensor, filename):
    """
    Save the tensor in CV2 format
    :param input_tensor: tensor saved
    :param filename : filename saved
    """
    input_tensor = input_tensor[:1]
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    input_tensor = input_tensor.clone().detach()
    input_tensor = input_tensor.to(torch.device('cpu'))
    input_tensor = input_tensor.squeeze()
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, input_tensor)


def stereo_validation_video(net, val_data_loader, save_dir, device):
    psnr_list = []
    ssim_list = []
    psnr_list2 = []
    ssim_list2 = []
    window_size = 8
    for batch_id, data in enumerate(val_data_loader):
        with torch.no_grad():
            haze, gt, haze2, gt2 = data
            haze = haze.cuda()
            haze2 = haze2.cuda()
            gt = gt.cuda()
            gt2 = gt2.cuda()
            _, _, _, h_old, w_old = haze.size()
            h_pad = (h_old // window_size +1) * window_size -h_old
            w_pad = (w_old//window_size+1) * window_size - w_old
            haze = torch.cat([haze, torch.flip(haze, [3])], 3)[:, :, :h_old+h_pad, :]
            haze = torch.cat([haze, torch.flip(haze, [4])], 4)[:, :,:, :w_old+w_pad]

            haze2 = torch.cat([haze2, torch.flip(haze2, [3])], 3)[:, :, :h_old+h_pad, :]
            haze2 = torch.cat([haze2, torch.flip(haze2, [4])], 4)[:, :, :, :w_old+w_pad]
            # gt = torch.cat([gt, torch.flip(gt, [2])], 2)[:, :, :h_old+h_pad, :]
            # gt = torch.cat([gt, torch.flip(gt, [3])], 3)[:, :, :, :w_old+w_pad]
            #
            # gt2 = torch.cat([gt2, torch.flip(gt2, [2])], 2)[:, :, :h_old + h_pad, :]
            # gt2 = torch.cat([gt2, torch.flip(gt2, [3])], 3)[:, :, :, :w_old + w_pad]

            #print(haze.shape, haze2.shape)
            dehaze, dehaze2 = net(haze, haze2)

            dehaze = dehaze[:, :, :, :h_old, :w_old]
            dehaze2 = dehaze2[:, :, :, :h_old, :w_old]
            haze, _, _ = cubes_2_maps(haze)
            haze2, _, _ = cubes_2_maps(haze2)
            dehaze, _, _ = cubes_2_maps(dehaze)
            dehaze2, _, _ = cubes_2_maps(dehaze2)
            gt, _, _ = cubes_2_maps(gt)
            gt2, _, _ = cubes_2_maps(gt2)
            #print(dehaze.shape, dehaze2.shape)
        psnr_list.append(to_psnr(dehaze, gt))
        ssim_list.append(to_ssim_skimage(dehaze, gt))
        psnr_list2.append(to_psnr(dehaze2, gt2))
        ssim_list2.append(to_ssim_skimage(dehaze2, gt2))

        # save result
        # save_image_tensor2cv2(dehaze, save_dir  + str(batch_id) +'left.png')
        # save_image_tensor2cv2(dehaze2, save_dir +  str(batch_id)+'right.png')

        if batch_id % 300 == 0:
            print("processed %d images" % batch_id)
    return np.mean(psnr_list), np.mean(ssim_list), np.mean(psnr_list2), np.mean(ssim_list2)
def adjust_learning_rate(optimizer, epoch, lr_decay=0.5):
    # --- Decay learning rate --- #
    step = 10

    if not epoch % step and epoch > 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
            print('Learning rate sets to {}.'.format(param_group['lr']))
    else:
        for param_group in optimizer.param_groups:
            print('Learning rate sets to {}.'.format(param_group['lr']))

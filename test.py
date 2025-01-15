
import time
import torch
import argparse
from torch.utils.data import DataLoader
from utils import *
from dataset_processing.test_data import K12_dataset_test


# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for GridDehazeNet')
parser.add_argument('-test_data_dir', help='training data path', default='/Datasets/kitti2012/test/')
parser.add_argument('-checkpoint_dir', help='training checkpoints path', default='./checkpoint/K12/checkpoint_K12')
parser.add_argument('-output_img_dir', help='desonowing output path', default='./output_img/K12/')
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)


args = parser.parse_args()

ckpt_path = args.checkpoint_dir
test_data_dir = args.test_data_dir
output_img_dir = args.output_img_dir
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device_ids = [Id for Id in range(torch.cuda.device_count())]
print(device_ids)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load(ckpt_path)
model.to(device)
print('--- backbone weight loaded ---')

# --- Set category-specific hyper-parameters  --- #
val_data_loader = DataLoader(K12_dataset_test(root=test_data_dir, frame_num=1, step=1), batch_size=1, shuffle=False, num_workers=0, drop_last=False)

# --- Use the evaluation model in testing --- #
model.eval()
print('--- Testing starts! ---')
print("checkpoint:", ckpt_path)
val_psnr, val_ssim, val_psnr2, val_ssim2 = stereo_validation_video(model, val_data_loader, output_img_dir, device)
print(val_psnr)
print(val_ssim)
print(val_psnr2)
print(val_ssim2)


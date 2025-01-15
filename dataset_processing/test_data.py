# --- Imports --- #
import torch.utils.data as data
from PIL import Image
import random
from random import randrange
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop
import os
import torch
from collections import Counter


def pil_loader(path):
    img = Image.open(path).convert('RGB')

    return img

def load_frames(frameList, path):
    video = []
    #print("frameLits", frameList)
    for i in range(0, len(frameList)):
        if os.path.exists(os.path.join(path,frameList[i])):
            video.append(pil_loader(os.path.join(path,frameList[i])))
            #print("frame:", pil_loader(frameList[i]))
        else:
            raise ValueError('File {} not exists.'.format(frameList))
    return video

def get_dataset(dir):
    if not os.path.isdir(dir):
        raise Exception('check' + dir)
    image_list = next(os.walk(dir))[2]
    image_list.sort()

    VideoID_list = []
    for img in image_list:
        VideoID_list.append(img.split('_')[0])
    videos_all = []
    for (key, value) in dict(Counter(VideoID_list)).items():
        images_video = []
        for image in image_list:
            if image.split('_')[0] == key:
                images_video.append(image)
        videos_all.append(images_video)
    return videos_all

def select_video_clips(frame_num, step, videos_list):
    video_clips = []
    for video in videos_list:
        for i in range(0, len(video)-frame_num+1, step):
            video_clip = video[i:i+frame_num]
            video_clips.append(video_clip)
    return video_clips


def video_clip(frame_num, step, gt_images, rain_images, gt_images2, rain_images2):

    video_clips_left_gt = select_video_clips(frame_num, step, gt_images)
    print(len(video_clips_left_gt))
    video_clips_left_rain = select_video_clips(frame_num, step, rain_images)
    video_clips_right_gt = select_video_clips(frame_num, step, gt_images2)

    video_clips_right_rain = select_video_clips(frame_num, step, rain_images2)
    data = {'input1': video_clips_left_rain, 'gt1': video_clips_left_gt, 'input2': video_clips_right_rain, 'gt2': video_clips_right_gt}
    return data

class K12_dataset_test(data.Dataset):
    """Some Information about K12(K15)_dataset"""

    def __init__(self, root, frame_num, step, apply_crop=True):
        super(K12_dataset_test, self).__init__()
        # ---- root == './' ---
        self.path_gt_images = root + '/image_2'
        self.path_rain_images = root + '/image_2_snow'
        self.path_gt_images2 = root + '/image_3'
        self.path_rain_images2 = root + '/image_3_snow'

        self.gt_images = get_dataset(self.path_gt_images)
        #print(self.gt_images[0])
        self.rain_images = get_dataset(self.path_rain_images)
        self.gt_images2 = get_dataset(self.path_gt_images2)
        self.rain_images2 = get_dataset(self.path_rain_images2)
        # self.crop_size = crop_size
        self.root = root
        self.apply_crop = apply_crop
        #self.frame_num = frame_num
        self.data = video_clip(frame_num, step, self.gt_images, self.rain_images, self.gt_images2, self.rain_images2)
        #print(self.data['input1'][0])
    def __getitem__(self, idx):
        ip_data1 = self.data['input1'][idx]
        gt_data1 = self.data['gt1'][idx]
        ip_data2 = self.data['input2'][idx]
        gt_data2 = self.data['gt2'][idx]


        ip_frames1 = load_frames(ip_data1, path=self.path_rain_images)
        gt_frames1 = load_frames(gt_data1, path=self.path_gt_images)
        ip_frames2 = load_frames(ip_data2, path=self.path_rain_images2)
        gt_frames2 = load_frames(gt_data2, path=self.path_gt_images2)
        ip_tensors1, gt_tensors1, ip_tensors2, gt_tensors2 = [], [], [], []
        # the whole clip should apply same transform.
        for ip_frame1, gt_frame1, ip_frame2, gt_frame2 in zip(ip_frames1, gt_frames1, ip_frames2, gt_frames2):
            ip_frame1, gt_frame1, ip_frame2, gt_frame2 = self.apply_transform_stereo(ip_frame1, gt_frame1, ip_frame2, gt_frame2)
            ip_tensors1.append(ip_frame1)
            gt_tensors1.append(gt_frame1)
            ip_tensors2.append(ip_frame2)
            gt_tensors2.append(gt_frame2)
        ip_tensors1 = torch.stack(ip_tensors1, 0).permute(1, 0, 2, 3)
        gt_tensors1 = torch.stack(gt_tensors1, 0).permute(1, 0, 2, 3)

        ip_tensors2 = torch.stack(ip_tensors2, 0).permute(1, 0, 2, 3)
        gt_tensors2 = torch.stack(gt_tensors2, 0).permute(1, 0, 2, 3)

        # tensors shape: Channel (C) x temporal window size (T) x H x W
        return ip_tensors1, gt_tensors1, ip_tensors2, gt_tensors2

    def get_transform_params(self, w, h):
        x0 = random.randint(0, w - self.crop_size)
        y0 = random.randint(0, h - self.crop_size)

        hflip_rnd = random.uniform(0, 1)
        vflip_rnd = random.uniform(0, 1)
        degree = random.choice([0, 90, 180, 270])

        return {'x0': x0,
                'y0': y0,
                'hflip_rnd': hflip_rnd,
                'vflip_rnd': vflip_rnd,
                'degree': degree
                }
    def apply_transform_stereo(self, ip_frame, gt_frame, ip_frame2, gt_frame2):
   
        transform = Compose([ToTensor()])
        # gt_frame = Compose(ToTensor())
        ip_frame = transform(ip_frame)
        gt_frame = transform(gt_frame)
        ip_frame2 = transform(ip_frame2)
        gt_frame2 = transform(gt_frame2)
        return ip_frame, gt_frame, ip_frame2, gt_frame2
    def apply_transform(self, ip_frame, gt_frame, params):
        x0, y0, hflip_rnd, vflip_rnd, deg = params['x0'], params['y0'], params['hflip_rnd'], params['vflip_rnd'], \
                                            params['degree']


        # random cropping
        if self.apply_crop:
            x1 = x0 + self.crop_size
            y1 = y0 + self.crop_size

            ip_frame = ip_frame.crop((x0, y0, x1, y1))
            gt_frame = gt_frame.crop((x0, y0, x1, y1))
            transform = Compose([ToTensor()])
            # gt_frame = Compose(ToTensor())
            ip_frame = transform(ip_frame)
            gt_frame = transform(gt_frame)
   
        return ip_frame, gt_frame


    def __len__(self):
        return len(self.data['gt1'])





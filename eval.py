import cv2
import os
import sys
import math
import time
import json
import glob
import argparse
import urllib.request
from PIL import Image, ImageFilter
from numpy import random
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as F
import torch.multiprocessing as mp

from core.i3d import InceptionI3d
from core import metric as module_metric
from core.metric import get_fid_score
from core.transform import (
  GroupScale, Stack, ToTorchFormatTensor,
  GroupRandomHorizontalFlip
)

parser = argparse.ArgumentParser(description='PyTorch Template')
parser.add_argument('-c', '--config', default='configs/iccv19-vos-fixed.json', type=str)
parser.add_argument('-r', '--resume', default=None, type=str)
args = parser.parse_args()


# set random seed 
seed = 2020
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

default_fps = 6
ngpus = torch.cuda.device_count()
_to_tensors = transforms.Compose([
  Stack(),
  ToTorchFormatTensor()])


def main():
  orig_names = list(glob.glob('{}/*/orig.avi'.format(args.resume)))
  comp_names = list(glob.glob('{}/*/comp.avi'.format(args.resume)))
  orig_names.sort()
  comp_names.sort()
  orig_videos = []
  comp_videos = []
  # metrics prepare for image assesments
  metrics = {met: getattr(module_metric, met) for met in ['mse', 'psnr', 'ssim']}
  evaluation_scores = {key: 0 for key,val in metrics.items()}
  # infer through videos
  for vi, (orig_vname, comp_vname) in enumerate(zip(orig_names, comp_names)):
    orig_frames = read_frame_from_videos(orig_vname)
    comp_frames = read_frame_from_videos(comp_vname)
    orig_videos.append(orig_frames)
    comp_videos.append(comp_frames)
    # calculating image quality assessments
    for key, val in metrics.items():
      evaluation_scores[key] += val(orig_frames, comp_frames)
    print('{}/{} from {} : {}'.format(vi+1, len(orig_names), args.resume,
      ' '.join(['{}: {:5f},'.format(key, val/(vi+1)) for key,val in evaluation_scores.items()])))
  # metrics prepare for video assesments
  print('start evaluation for FID scores ...')
  stime = time.time()
  i3d_model_weight = '../libs/model_weights/i3d_rgb_imagenet.pt'
  if not os.path.exists(i3d_model_weight):
      os.makedirs(os.path.dirname(i3d_model_weight), exist_ok=True)
      urllib.request.urlretrieve('http://www.cmlab.csie.ntu.edu.tw/~zhe2325138/i3d_rgb_imagenet.pt', i3d_model_weight)
  i3d_model = InceptionI3d(400, in_channels=3, final_endpoint='Logits')
  i3d_model.load_state_dict(torch.load(i3d_model_weight))
  i3d_model = i3d_model.cuda()
  output_i3d_activations = []
  real_i3d_activations = []
  # i3d_model=nn.DataParallel(i3d_model,device_ids=[i for i in range(ngpus)])
  for index in range(len(comp_videos)):
    # calculating video quality assessments
    targets = list_to_batch(orig_videos[index:index+1]).cuda()
    outputs = list_to_batch(comp_videos[index:index+1]).cuda()
    with torch.no_grad():
      output_i3d_feat = i3d_model.extract_features(outputs.transpose(1, 2), 'Logits')
      output_i3d_feat = output_i3d_feat.view(output_i3d_feat.size(0), -1)
      output_i3d_activations.append(output_i3d_feat.cpu().numpy())
      real_i3d_feat = i3d_model.extract_features(targets.transpose(1, 2), 'Logits')
      real_i3d_feat = real_i3d_feat.view(real_i3d_feat.size(0), -1)
      real_i3d_activations.append(real_i3d_feat.cpu().numpy())
  output_i3d_activations = np.concatenate(output_i3d_activations, axis=0)
  real_i3d_activations = np.concatenate(real_i3d_activations, axis=0)
  fid_score = get_fid_score(real_i3d_activations, output_i3d_activations)
  print('finished calculating fid score of {}: {:5f}  in {:2f}s'.format(args.resume, fid_score, time.time()-stime))


def list_to_batch(video_list):
  tensor_list = []
  for frames in video_list:
    tensor_list.append(_to_tensors([Image.fromarray(cv2.cvtColor(m, cv2.COLOR_BGR2RGB)) for m in frames]).unsqueeze(0))
  if len(video_list) > 1:
    return torch.cat(tensor_list, dim=0)
  else:
    return tensor_list[0]


def read_frame_from_videos(vname):
  frames = []
  vidcap = cv2.VideoCapture(vname)
  success, image = vidcap.read()
  count = 0
  while success:
    frames.append(image)
    success,image = vidcap.read()
    count += 1
  return frames


if __name__ == '__main__':
  main()

      

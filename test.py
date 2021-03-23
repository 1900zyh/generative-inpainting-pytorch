# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models
import torch.multiprocessing as mp
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import make_grid, save_image

import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import time
import os
import argparse
import copy
import datetime
import random
import sys
import json
import glob

### My libs
from core.utils import set_device, postprocess, ZipReader, set_seed
from core.utils import postprocess
from core.dataset import Dataset
from model.ca import Generator
from tqdm import tqdm 

parser = argparse.ArgumentParser(description="CA")
parser.add_argument("-c", "--config", type=str, required=True)
parser.add_argument("-l", "--level", type=int, required=True)
parser.add_argument("-s", "--size", type=int, default=512)
parser.add_argument("-m", "--mask", type=str, default="pconv")
parser.add_argument("-p", "--port", type=str, default="23451")
args = parser.parse_args()

BATCH_SIZE = 4

def main_worker(gpu, ngpus_per_node, config):
  torch.cuda.set_device(gpu)
  set_seed(config['seed'])

  # Model and version
  model = set_device(Generator())
  latest_epoch = open(os.path.join(config['save_dir'], 'latest.ckpt'), 'r').read().splitlines()[-1]
  path = os.path.join(config['save_dir'], 'gen_{}.pth'.format(latest_epoch))
  data = torch.load(path, map_location = lambda storage, loc: set_device(storage)) 
  model.load_state_dict(data['netG'])
  model.eval()

  # prepare dataset
  dataset = Dataset(config['data_loader'], debug=False, split='test', level=args.level)
  step = math.ceil(len(dataset) / ngpus_per_node)
  dataset.set_subset(gpu*step, min(gpu*step+step, len(dataset)))
  dataloader = DataLoader(dataset, batch_size= BATCH_SIZE, shuffle=False, num_workers=config['trainer']['num_workers'], pin_memory=True)

  
  path = os.path.join('results', os.path.basename(args.config).split('.')[0])
  os.makedirs(os.path.join(path, f'comp_level0{args.level}'), exist_ok=True)
  os.makedirs(os.path.join(path, f'mask_level0{args.level}'), exist_ok=True)
  # iteration through datasets
  for images, masks, names in tqdm(dataloader, desc=f'ca-id-{gpu}-level {args.level}'):
    inpts = images*(1-masks)
    images, inpts, masks = set_device([images, inpts, masks])
    with torch.no_grad():
      output, _ = model(inpts, masks)
    orig_imgs = postprocess(images)
    comp_imgs = postprocess((1-masks)*images+masks*output)
    mask_imgs = postprocess(inpts)
    for i in range(len(orig_imgs)):
      Image.fromarray(comp_imgs[i]).save(os.path.join(path, f'comp_level0{args.level}', names[i].split('.')[0]+'.png'))
  print('Finish in {}'.format(path))



if __name__ == '__main__':
  ngpus_per_node = torch.cuda.device_count()
  config = json.load(open(args.config))
  if args.mask is not None:
    config['data_loader']['mask'] = args.mask
  if args.size is not None:
    config['data_loader']['w'] = config['data_loader']['h'] = args.size
  config['save_dir'] = os.path.join(config['save_dir'], '{}{}'.format(config['data_loader']['name'], config['data_loader']['w']))

  print('using {} GPUs for testing ... '.format(ngpus_per_node))
  # setup distributed parallel training environments
  ngpus_per_node = torch.cuda.device_count()
  config['world_size'] = ngpus_per_node
  config['init_method'] = 'tcp://127.0.0.1:'+ args.port 
  config['distributed'] = True
  mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))
 

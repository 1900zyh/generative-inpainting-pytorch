import random
from random import shuffle
import os 
import math 
import numpy as np 
from PIL import Image, ImageFilter
from glob import glob

import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


from core.utils import ZipReader


class Dataset(torch.utils.data.Dataset):
  def __init__(self, data_args, debug=False, split='train', level=None):
    super(Dataset, self).__init__()
    self.split = split
    self.level = level
    self.w, self.h = data_args['w'], data_args['h']
    # self.data = [os.path.join(data_args['zip_root'], data_args['name'], i) 
    #   for i in np.genfromtxt(os.path.join(data_args['flist_root'], data_args['name'], split+'.flist'), dtype=np.str, encoding='utf-8')]
    self.data = glob('/data05/t-yazen/data/places2/centercrop_512/*.png')
    self.data.sort()

    self.mask = [os.path.join(data_args['zip_root'], 'mask/{}.png'.format(str(i).zfill(5))) for i in range(self.level*2000, (self.level+1)*2000)]
    self.mask = self.mask*(len(self.data)//len(self.mask)) + self.mask[:len(self.data)%len(self.mask)]
    print(f'data:{len(self.data)}, mask:{len(self.mask)}')

  def __len__(self):
    return len(self.data)
  
  def set_subset(self, start, end):
    self.mask = self.mask[start:end]
    self.data = self.data[start:end] 

  def __getitem__(self, index):
    try:
      item = self.load_item(index)
    except:
      print('loading error: ' + self.data[index])
      item = self.load_item(0)
    return item

  def load_item(self, index):
    # load image
    img = Image.open(self.data[index]).convert('RGB')
    img_name = os.path.basename(self.data[index])
    # load mask 
    mask_path = os.path.dirname(self.mask[index]) + '.zip'
    mask_name = os.path.basename(self.mask[index])
    mask = ZipReader.imread(mask_path, mask_name).convert('L')
    
    img = img.resize((self.w, self.h))
    mask = mask.resize((self.w, self.h), Image.NEAREST)
    return F.to_tensor(img)*2-1., F.to_tensor(mask), img_name

  def create_iterator(self, batch_size):
    while True:
      sample_loader = DataLoader(dataset=self,batch_size=batch_size,drop_last=True)
      for item in sample_loader:
        yield item

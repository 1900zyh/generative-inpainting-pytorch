import os
import time
import math
import glob
import shutil
import datetime
import numpy as np
from PIL import Image
from math import log10
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter

from core.dataset import Dataset
from core.utils import set_seed, set_device
from core import metric as module_metric



class BaseTrainer():
  def __init__(self, config, debug=False):
    self.config = config
    self.epoch = 0
    self.iteration = 0

    # setup data set and data loader
    self.train_dataset = Dataset(config['data_loader'], debug=debug, split='train')
    self.valid_dataset = Dataset(config['data_loader'], debug=debug, split='test')
    worker_init_fn = partial(set_seed, base=config['seed'])
    self.train_sampler = None
    self.valid_sampler = None
    if config['distributed']:
      self.train_sampler = DistributedSampler(self.train_dataset, 
        num_replicas=config['world_size'], rank=config['global_rank'])
      self.valid_sampler = DistributedSampler(self.valid_dataset, 
        num_replicas=config['world_size'], rank=config['local_rank'])
    self.train_loader = DataLoader(self.train_dataset, 
      batch_size= config['data_loader']['batch_size'] // config['world_size'],
      shuffle=(self.train_sampler is None), num_workers=config['data_loader']['num_workers'],
      pin_memory=True, sampler=self.train_sampler, worker_init_fn=worker_init_fn)
    self.valid_loader = DataLoader(self.valid_dataset, 
      batch_size= 1, shuffle=None, num_workers=config['data_loader']['num_workers'],
      pin_memory=True, sampler=self.valid_sampler, worker_init_fn=worker_init_fn)

    # set tup matrics
    self.metrics = {met: getattr(module_metric, met) for met in config['metrics']}
    self.dis_writer = None
    self.gen_writer = None
    if self.config['global_rank'] == 0 or (not config['distributed']):
      self.dis_writer = SummaryWriter(os.path.join(config['save_dir'], 'dis'))
      self.gen_writer = SummaryWriter(os.path.join(config['save_dir'], 'gen'))
    self.samples_path = os.path.join(config['save_dir'], 'samples')
    self.results_path = os.path.join(config['save_dir'], 'results')
    
    # other args
    self.log_args = self.config['logger']
    self.train_args = self.config['trainer']


  def add_summary(self, writer, name, val):
    if writer is not None and self.iteration % self.log_args['log_step'] == 0:
      writer.add_scalar(name, val, self.iteration)

  # get current learning rate
  def get_lr(self):
    return self.optimG.param_groups[0]['lr']

  def train(self):
    while True:
      self.epoch += 1
      if self.config['distributed']:
        self.train_sampler.set_epoch(self.epoch)
      self._train_epoch()
      if self.iteration > self.config['trainer']['iterations']:
        break
    print('\nEnd training....')

  # save parameters every eval_epoch
  def _save(self, it):
    if self.config['global_rank'] == 0:
      gen_path = os.path.join(self.config['save_dir'], 'gen_{}.pth'.format(str(it).zfill(5)))
      dis_path = os.path.join(self.config['save_dir'], 'dis_{}.pth'.format(str(it).zfill(5)))
      print('\nsaving model to {} ...'.format(gen_path))
      if isinstance(self.netG, torch.nn.DataParallel) or isinstance(self.netG, DDP):
        netG, localD, globalD = self.netG.module, self.localD.module, self.globalD.module
      else:
        netG, localD, globalD = self.netG.module, self.localD.module, self.globalD.module
      torch.save({'netG': netG.state_dict()}, gen_path)
      torch.save({'epoch': self.epoch, 'iteration': self.iteration,
                  'localD': localD.state_dict(), 
                  'globalD': globalD.state_dict(),
                  'optimG': self.optimG.state_dict(),
                  'optimD': self.optimD.state_dict()}, dis_path)
      os.system('echo {} > {}'.format(str(it).zfill(5), os.path.join(self.config['save_dir'], 'latest.ckpt')))

  # load model parameters
  def _load(self):
    model_path = self.config['save_dir']
    if os.path.isfile(os.path.join(model_path, 'latest.ckpt')):
      latest_epoch = open(os.path.join(model_path, 'latest.ckpt'), 'r').read().splitlines()[-1]
    else:
      ckpts = [os.path.basename(i).split('.pth')[0] for i in glob.glob(os.path.join(model_path, '*.pth'))]
      ckpts.sort()
      latest_epoch = ckpts[-1] if len(ckpts)>0 else None
    if latest_epoch is not None:
      gen_path = os.path.join(model_path, 'gen_{}.pth'.format(str(latest_epoch).zfill(5)))
      dis_path = os.path.join(model_path, 'dis_{}.pth'.format(str(latest_epoch).zfill(5)))
      if self.config['global_rank'] == 0:
        print('Loading model from {}...'.format(gen_path))
      data = torch.load(gen_path, map_location = lambda storage, loc: set_device(storage)) 
      self.netG.load_state_dict(data['netG'])
      data = torch.load(dis_path, map_location = lambda storage, loc: set_device(storage)) 
      self.optimG.load_state_dict(data['optimG'])
      self.optimD.load_state_dict(data['optimD'])
      self.localD.load_state_dict(data['localD'])
      self.globalD.load_state_dict(data['globalD'])
      self.epoch = data['epoch']
      self.iteration = data['iteration']
    else:
      if self.config['global_rank'] == 0:
        print('Warnning: There is no trained model found. An initialized model will be used.')
      

  # process input and calculate loss every training epoch
  def _train_epoch(self,):
    """
    Training logic for an epoch
    :param epoch: Current epoch number
    """
    raise NotImplementedError

  def _eval_epoch(self,ep):
    """
    Training logic for an epoch
    :param epoch: Current epoch number
    """
    raise NotImplementedError
  
  def adjust_learning_rate(self,):
    raise NotImplementedError

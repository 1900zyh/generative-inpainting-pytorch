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

from core import loss as module_loss
from core import metric as module_metric



class BaseTrainer():
  def __init__(self, config, debug=False):
    self.config = config
    self.epoch = 0
    self.iteration = 0
    self.vgg =  VGG16FeatureExtractor()
    self.vgg = set_device(self.vgg)

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

    # set loss functions and evaluation metrics
    self.losses = {entry['name']: (
        getattr(module_loss, entry['name']),
        entry['weight'], 
        entry['input']
      ) for entry in config['losses']}
    self.metrics = {met: getattr(module_metric, met) for met in config['metrics']}

    # setup models 
    self.model = set_device(PConvUNet())
    self.optim_args = self.config['optimizer']
    self.optim = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
      lr = self.optim_args['lr'])
    self._load()
    if config['distributed']:
      self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
      self.model = DDP(self.model, device_ids=[config['global_rank']], output_device=config['global_rank'], 
        broadcast_buffers=True)#, find_unused_parameters=False)
    
    # set summary writer
    self.writer = None
    if self.config['global_rank'] == 0 or (not config['distributed']):
      self.writer = SummaryWriter(os.path.join(config['save_dir'], 'logs'))
    self.samples_path = os.path.join(config['save_dir'], 'samples')
    self.results_path = os.path.join(config['save_dir'], 'results')
    
    # other args
    self.log_args = self.config['logger']
    self.train_args = self.config['trainer']

  # get current learning rate
  def get_lr(self):
    return self.optim.param_groups[0]['lr']

  def add_summary(self, writer, name, val):
    if writer is not None and self.iteration % self.log_args['log_step'] == 0:
      writer.add_scalar(name, val, self.iteration)

  def get_non_gan_loss(self, outputs, videos, masks):
    non_gan_losses = []
    outputs_feat = self.vgg(outputs)
    videos_feat = self.vgg(videos)
    for loss_name, (loss_instance, loss_weight, input_type) in self.losses.items():
      if loss_weight > 0.0:
        if input_type == 'RGB':
          loss = loss_instance(outputs, videos, masks)
        elif input_type == 'feat':
          loss = loss_instance(outputs_feat, videos_feat, masks)
        loss *= loss_weight
        self.add_summary(self.writer, f'loss/{loss_name}', loss.item())
        non_gan_losses.append(loss)
    return sum(non_gan_losses)


  def train(self):
    while True:
      self.epoch += 1
      if self.config['distributed']:
        self.train_sampler.set_epoch(self.epoch)
      self._train_epoch()
      if self.iteration > self.config['trainer']['iterations']:
        break
    print('\nEnd training....')


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
      path = os.path.join(model_path, latest_epoch+'.pth')
      if self.config['global_rank'] == 0:
        print('Loading model from {}...'.format(path))
      data = torch.load(path, map_location = lambda storage, loc: set_device(storage)) 
      model_dict = self.model.state_dict()
      pretrained_dict = {k:v for k,v in data['model'].items() if k in model_dict}
      model_dict.update(pretrained_dict)
      self.model.load_state_dict(model_dict)
      self.optim.load_state_dict(data['optim'])
      self.epoch = data['epoch']
      self.iteration = data['iteration']
    else:
      if self.config['global_rank'] == 0:
        print('Warnning: There is no trained model found. An initialized model will be used.')
      
  # save parameters every eval_epoch
  def _save(self, it):
    if self.config['global_rank'] == 0:
      path = os.path.join(self.config['save_dir'], str(it).zfill(5)+'.pth')
      print('\nsaving model to {} ...'.format(path))
      if isinstance(self.model, torch.nn.DataParallel) or isinstance(self.model, DDP):
        model = self.model.module
      else:
        model = self.model
      torch.save({'model': model.state_dict(), 
        'optim': self.optim.state_dict(),
        'epoch': self.epoch, 'iteration': self.iteration, }, path)
      os.system('echo {} > {}'.format(str(it).zfill(5), os.path.join(self.config['save_dir'], 'latest.ckpt')))

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

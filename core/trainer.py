import os
import time
import datetime
import glob
from PIL import Image
from functools import partial

import torch
import torch.nn as nn
from torch import autograd
import torch.distributed as dist
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import make_grid, save_image

from model.ca import Generator, LocalDis, GlobalDis
from core.utils import local_patch, spatial_discounting_mask
from core.utils import random_bbox, mask_image, postprocess
from core.utils import set_seed, set_device, Progbar
from core.dataset import Dataset


class Trainer():
  def __init__(self, config, debug=False):
    self.config = config
    self.epoch = 0
    self.iteration = 0
    if debug:
      self.config['trainer']['save_freq'] = 5
      self.config['trainer']['valid_freq'] = 5

    # setup data set and data loader
    self.train_dataset = Dataset(config['data_loader'], debug=debug, split='train')
    worker_init_fn = partial(set_seed, base=config['seed'])
    self.train_sampler = None
    if config['distributed']:
      self.train_sampler = DistributedSampler(self.train_dataset, 
        num_replicas=config['world_size'], rank=config['global_rank'])
    self.train_loader = DataLoader(self.train_dataset, 
      batch_size= config['trainer']['batch_size'] // config['world_size'],
      shuffle=(self.train_sampler is None), num_workers=config['trainer']['num_workers'],
      pin_memory=True, sampler=self.train_sampler, worker_init_fn=worker_init_fn)

    # set up metrics
    self.dis_writer = None
    self.gen_writer = None
    self.summary = {}
    if self.config['global_rank'] == 0 or (not config['distributed']):
      self.dis_writer = SummaryWriter(os.path.join(config['save_dir'], 'dis'))
      self.gen_writer = SummaryWriter(os.path.join(config['save_dir'], 'gen'))
    self.train_args = self.config['trainer']
    
    # setup models 
    self.netG = set_device(Generator())
    self.localD = set_device(LocalDis(config))
    self.globalD = set_device(GlobalDis(config))
    self.optimG = torch.optim.Adam(self.netG.parameters(), lr=self.config['trainer']['lr'],
      betas=(self.config['trainer']['beta1'], self.config['trainer']['beta2']))
    self.optimD = torch.optim.Adam(list(self.localD.parameters()) + list(self.globalD.parameters()), lr=config['trainer']['lr'],
      betas=(self.config['trainer']['beta1'], self.config['trainer']['beta2']))
    self._load()
    if config['distributed']:
      self.netG = DDP(self.netG, device_ids=[config['global_rank']], output_device=config['global_rank'], 
        broadcast_buffers=True)#, find_unused_parameters=False)
      self.localD = DDP(self.localD, device_ids=[config['global_rank']], output_device=config['global_rank'], 
        broadcast_buffers=True)#, find_unused_parameters=False)
      self.globalD = DDP(self.globalD, device_ids=[config['global_rank']], output_device=config['global_rank'], 
        broadcast_buffers=True)#, find_unused_parameters=False)
    

  # get current learning rate
  def get_lr(self, type='G'):
    if type == 'G':
      return self.optimG.param_groups[0]['lr']
    return self.optimD.param_groups[0]['lr']
  
 # learning rate scheduler, step
  def adjust_learning_rate(self):
    decay = 0.1**(min(self.iteration, self.config['trainer']['niter_steady']) // self.config['trainer']['niter']) 
    new_lr = self.config['trainer']['lr'] * decay
    if new_lr != self.get_lr():
      for param_group in self.optimG.param_groups:
        param_group['lr'] = new_lr
      for param_group in self.optimD.param_groups:
       param_group['lr'] = new_lr

  def add_summary(self, writer, name, val):
    if name not in self.summary:
      self.summary[name] = 0
    self.summary[name] += val
    if writer is not None and self.iteration % 100 == 0:
      writer.add_scalar(name, self.summary[name]/100, self.iteration)

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
      opt_path = os.path.join(self.config['save_dir'], 'opt_{}.pth'.format(str(it).zfill(5)))
      print('\nsaving model to {} ...'.format(gen_path))
      if isinstance(self.netG, torch.nn.DataParallel) or isinstance(self.netG, DDP):
        netG, localD, globalD = self.netG.module, self.localD.module, self.globalD.module
      else:
        netG, localD, globalD = self.netG, self.localD, self.globalD
      torch.save({'netG': netG.state_dict()}, gen_path)
      torch.save({'localD': localD.state_dict(), 
                  'globalD': globalD.state_dict()}, dis_path)
      torch.save({'epoch': self.epoch, 'iteration': self.iteration,
                  'optimG': self.optimG.state_dict(),
                  'optimD': self.optimD.state_dict()}, opt_path)
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
      opt_path = os.path.join(self.config['save_dir'], 'opt_{}.pth'.format(str(it).zfill(5)))
      if self.config['global_rank'] == 0:
        print('Loading model from {}...'.format(gen_path))
      data = torch.load(gen_path, map_location = lambda storage, loc: set_device(storage)) 
      self.netG.load_state_dict(data['netG'])
      data = torch.load(dis_path, map_location = lambda storage, loc: set_device(storage)) 
      self.localD.load_state_dict(data['localD'])
      self.globalD.load_state_dict(data['globalD'])
      data = torch.load(opt_path, map_location = lambda storage, loc: set_device(storage)) 
      self.optimG.load_state_dict(data['optimG'])
      self.optimD.load_state_dict(data['optimD'])
      self.epoch = data['epoch']
      self.iteration = data['iteration']
    else:
      if self.config['global_rank'] == 0:
        print('Warnning: There is no trained model found. An initialized model will be used.')


  def _train_epoch(self):
    progbar = Progbar(len(self.train_dataset), width=20, stateful_metrics=['epoch', 'iter'])
    mae = 0
    for ground_truth, _, _ in self.train_loader:
      self.iteration += 1
      end = time.time()
      ground_truth = set_device(ground_truth)
      bboxes = random_bbox(self.config, batch_size=ground_truth.size(0))
      x, masks = mask_image(ground_truth, bboxes, self.config)

      losses = {}
      x1, x2 = self.netG(x, masks)
      local_patch_gt = local_patch(ground_truth, bboxes)
      x1_inpaint = x1 * masks + x * (1. - masks)
      x2_inpaint = x2 * masks + x * (1. - masks)
      local_patch_x1_inpaint = local_patch(x1_inpaint, bboxes)
      local_patch_x2_inpaint = local_patch(x2_inpaint, bboxes)

      # compute losses
      local_patch_real_pred, local_patch_fake_pred = self.dis_forward(self.localD, local_patch_gt, local_patch_x2_inpaint.detach())
      global_real_pred, global_fake_pred = self.dis_forward(self.globalD, ground_truth, x2_inpaint.detach())
      losses['wgan_d'] = torch.mean(local_patch_fake_pred - local_patch_real_pred) \
                          + torch.mean(global_fake_pred - global_real_pred) * self.config['losses']['global_wgan_loss_alpha']
      # gradients penalty loss
      local_penalty = self.calc_gradient_penalty(self.localD, local_patch_gt, local_patch_x2_inpaint.detach())
      global_penalty = self.calc_gradient_penalty(self.globalD, ground_truth, x2_inpaint.detach())
      losses['wgan_gp'] = local_penalty + global_penalty

      ## G part
      if self.iteration % self.config['trainer']['n_critic']:
        sd_mask = spatial_discounting_mask(self.config)
        losses['l1'] = nn.L1Loss()(local_patch_x1_inpaint * sd_mask, local_patch_gt * sd_mask) \
                        * self.config['losses']['coarse_l1_alpha'] \
                        + nn.L1Loss()(local_patch_x2_inpaint * sd_mask, local_patch_gt * sd_mask)
        losses['ae'] = nn.L1Loss()(x1 * (1. - masks), ground_truth * (1. - masks)) \
                        * self.config['losses']['coarse_l1_alpha'] \
                        + nn.L1Loss()(x2 * (1. - masks), ground_truth * (1. - masks))
        # wgan g loss
        local_patch_real_pred, local_patch_fake_pred = self.dis_forward(self.localD, local_patch_gt, local_patch_x2_inpaint)
        global_real_pred, global_fake_pred = self.dis_forward(self.globalD, ground_truth, x2_inpaint)
        losses['wgan_g'] = - torch.mean(local_patch_fake_pred) - torch.mean(global_fake_pred) * self.config['losses']['global_wgan_loss_alpha']

      # Scalars from different devices are gathered into vectors
      for k in losses.keys():
        if not losses[k].dim() == 0:
          losses[k] = torch.mean(losses[k])

      ###### Backward pass ######
      # Update D
      self.optimD.zero_grad()
      losses['d'] = losses['wgan_d'] + losses['wgan_gp'] * self.config['losses']['wgan_gp_lambda']
      losses['d'].backward()
      self.optimD.step()
      # Update G
      if self.iteration % self.config['trainer']['n_critic']:
        self.optimG.zero_grad()
        losses['g'] = losses['l1'] * self.config['losses']['l1_loss_alpha'] \
                      + losses['ae'] * self.config['losses']['ae_loss_alpha'] \
                      + losses['wgan_g'] * self.config['losses']['gan_loss_alpha']
        losses['g'].backward()
        self.optimG.step()
    
      # logs
      new_mae = (torch.mean(torch.abs(ground_truth - x2)) / torch.mean(masks)).item()
      mae = new_mae if mae == 0 else (new_mae+mae)/2
      speed = ground_truth.size(0)/(time.time() - end)*self.config['world_size']
      logs = [("epoch", self.epoch),("iter", self.iteration),("lr", self.get_lr()),
        ('mae', mae), ('samples/s', speed)]
      if self.config['global_rank'] == 0:
        progbar.add(len(ground_truth)*self.config['world_size'], values=logs \
          if self.train_args['verbosity'] else [x for x in logs if not x[0].startswith('l_')])

      # saving and evaluating
      if self.iteration % self.train_args['save_freq'] == 0:
        self._save(self.iteration//self.train_args['save_freq'])
      if self.iteration % self.train_args['valid_freq'] == 0:
        self._test_epoch(self.iteration//self.train_args['save_freq'])
        if self.config['global_rank'] == 0:
          print('[**] Training till {} in Rank {}\n'.format(
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), self.config['global_rank']))
      if self.iteration > self.config['trainer']['iterations']:
        break


  def dis_forward(self, netD, ground_truth, x_inpaint):
    assert ground_truth.size() == x_inpaint.size()
    batch_size = ground_truth.size(0)
    batch_data = torch.cat([ground_truth, x_inpaint], dim=0)
    batch_output = netD(batch_data)
    real_pred, fake_pred = torch.split(batch_output, batch_size, dim=0)

    return real_pred, fake_pred

  # Calculate gradient penalty
  def calc_gradient_penalty(self, netD, real_data, fake_data):
    batch_size, channel, height, width = real_data.size()
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(real_data.nelement() // batch_size)).contiguous() \
        .view(batch_size, channel, height, width)
    alpha = set_device(alpha)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates)

    grad_outputs = torch.ones(disc_interpolates.size())
    grad_outputs = set_device(grad_outputs)
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
      grad_outputs=grad_outputs, create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

  def _test_epoch(self, it):
    if self.config['global_rank'] == 0:
      print('[**] Testing in backend ...')
      model_path = self.config['save_dir']
      result_path = '{}/results_{}_level_03'.format(model_path, str(it).zfill(5))
      log_path = os.path.join(model_path, 'valid.log')
      try: 
        os.popen('python test.py -c {} -n {} -l 3 > valid.log;'
          'CUDA_VISIBLE_DEVICES=1 python eval.py -r {} >> {};'
          'rm -rf {}'.format(self.config['config'], self.config['model_name'], result_path, log_path, result_path))
      except (BrokenPipeError, IOError):
        pass

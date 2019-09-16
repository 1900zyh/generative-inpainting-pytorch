import os
import time
import datetime
from PIL import Image

import torch
import torch.nn as nn
from torch import autograd
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from core.base_trainer import BaseTrainer
from core.model import Generator, LocalDis, GlobalDis
from core.utils import local_patch, spatial_discounting_mask
from core.utils import random_bbox, mask_image, postprocess
from torchvision.utils import make_grid, save_image

from core.utils import set_device, Progbar



class Trainer(BaseTrainer):
  def __init__(self, config, debug=False):
    super().__init__(config, debug=debug)
    # setup models 
    self.netG = set_device(Generator(self.config['netG']))
    self.localD = set_device(LocalDis(self.config['netD']))
    self.globalD = set_device(GlobalDis(self.config['netD']))
    self.optimG = torch.optim.Adam(self.netG.parameters(), lr=self.config['optimizer']['lr'],
      betas=(self.config['optimizer']['beta1'], self.config['optimizer']['beta2']))
    self.optimD = torch.optim.Adam(list(self.localD.parameters()) + list(self.globalD.parameters()), lr=config['optimizer']['lr'],
      betas=(self.config['optimizer']['beta1'], self.config['optimizer']['beta2']))
    self._load()
    if config['distributed']:
      self.netG = DDP(self.netG, device_ids=[config['global_rank']], output_device=config['global_rank'], 
        broadcast_buffers=True)#, find_unused_parameters=False)
      self.localD = DDP(self.localD, device_ids=[config['global_rank']], output_device=config['global_rank'], 
        broadcast_buffers=True)#, find_unused_parameters=False)
      self.globalD = DDP(self.globalD, device_ids=[config['global_rank']], output_device=config['global_rank'], 
        broadcast_buffers=True)#, find_unused_parameters=False)
    if debug:
      self.config['trainer']['save_freq'] = 5
      self.config['trainer']['valid_freq'] = 5
    

  def _train_epoch(self):
    progbar = Progbar(len(self.train_dataset), width=20, stateful_metrics=['epoch', 'iter'])
    mae = 0
    for ground_truth, _, names in self.train_loader:
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
        self._eval_epoch(self.iteration//self.train_args['save_freq'])
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

  def _eval_epoch(self, it):
    self.valid_sampler.set_epoch(it)
    path = os.path.join(self.config['save_dir'], 'samples_{}'.format(str(it).zfill(5)))
    os.makedirs(path, exist_ok=True)
    if self.config['global_rank'] == 0:
      print('start evaluating ...')
    evaluation_scores = {key: 0 for key,val in self.metrics.items()}
    index = 0
    for images, masks, names in self.valid_loader:
      inpts = images*masks
      images, inpts, masks = set_device([images, inpts, masks])
      with torch.no_grad():
        output, _ = self.netG(inpts, masks)
      grid_img = make_grid(torch.cat([(images+1)/2, ((1-masks)*images+1)/2,
        (output+1)/2, ((1-masks)*images+masks*output+1)/2], dim=0), nrow=4)
      save_image(grid_img, os.path.join(path, '{}_stack.png'.format(names[0].split('.')[0])))
      orig_imgs = postprocess(images)
      comp_imgs = postprocess((1-masks)*images+masks*output)
      Image.fromarray(orig_imgs[0]).save(os.path.join(path, '{}_orig.png'.format(names[0].split('.')[0])))
      Image.fromarray(comp_imgs[0]).save(os.path.join(path, '{}_comp.png'.format(names[0].split('.')[0])))
      for key, val in self.metrics.items():
        evaluation_scores[key] += val(orig_imgs, comp_imgs)
      index += 1
    for key, val in evaluation_scores.items():
      tensor = set_device(torch.FloatTensor([val/index]))
      dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
      evaluation_scores[key] = tensor.cpu().item()
    evaluation_message = ' '.join(['{}: {:5f},'.format(key, val/self.config['world_size']) \
                        for key,val in evaluation_scores.items()])
    if self.config['global_rank'] == 0:
      print('[**] Evaluation: {}'.format(evaluation_message))


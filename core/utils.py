import os
import time
import io
import sys
import zipfile 
import torch
import yaml
import random
import numpy as np
from PIL import Image

import torch.nn.functional as F


# set random seed 
def set_seed(seed, base=0, is_set=True):
  seed += base
  assert seed >=0, '{} >= {}'.format(seed, 0)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  random.seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = True

class ZipReader(object):
  file_dict = dict()
  def __init__(self):
    super(ZipReader, self).__init__()

  @staticmethod
  def build_file_dict(path):
    file_dict = ZipReader.file_dict
    if path in file_dict:
      return file_dict[path]
    else:
      file_handle = zipfile.ZipFile(path, mode='r', allowZip64=True)
      file_dict[path] = file_handle
      return file_dict[path]

  @staticmethod
  def imread(path, image_name):
    zfile = ZipReader.build_file_dict(path)
    data = zfile.read(image_name)
    im = Image.open(io.BytesIO(data))
    return im

# set parameter to gpu or cpu
def set_device(args):
  if torch.cuda.is_available():
    if isinstance(args, list):
      return (item.cuda() for item in args)
    else:
      return args.cuda()
  return args

def postprocess(img):
  img = (img+1.)/2*255
  img = img.permute(0,2,3,1)
  img = img.int().cpu().numpy().astype(np.uint8)
  return img


class Progbar(object):
  """Displays a progress bar.

  Arguments:
    target: Total number of steps expected, None if unknown.
    width: Progress bar width on screen.
    verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
    stateful_metrics: Iterable of string names of metrics that
      should *not* be averaged over time. Metrics in this list
      will be displayed as-is. All others will be averaged
      by the progbar before display.
    interval: Minimum visual progress update interval (in seconds).
  """

  def __init__(self, target, width=25, verbose=1, interval=0.05, stateful_metrics=None):
    self.target = target
    self.width = width
    self.verbose = verbose
    self.interval = interval
    if stateful_metrics:
      self.stateful_metrics = set(stateful_metrics)
    else:
      self.stateful_metrics = set()

    self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
      sys.stdout.isatty()) or 'ipykernel' in sys.modules or 'posix' in sys.modules)
    self._total_width = 0
    self._seen_so_far = 0
    # We use a dict + list to avoid garbage collection
    # issues found in OrderedDict
    self._values = {}
    self._values_order = []
    self._start = time.time()
    self._last_update = 0

  def update(self, current, values=None):
    """Updates the progress bar.
    Arguments:
      current: Index of current step.
      values: List of tuples:
        `(name, value_for_last_step)`.
        If `name` is in `stateful_metrics`,
        `value_for_last_step` will be displayed as-is.
        Else, an average of the metric over time will be displayed.
    """
    values = values or []
    for k, v in values:
      if k not in self._values_order:
        self._values_order.append(k)
      if k not in self.stateful_metrics:
        if k not in self._values:
          self._values[k] = [v * (current - self._seen_so_far), current - self._seen_so_far]
        else:
          self._values[k][0] += v * (current - self._seen_so_far)
          self._values[k][1] += (current - self._seen_so_far)
      else:
        self._values[k] = v
    self._seen_so_far = current

    now = time.time()
    info = ' - %.0fs' % (now - self._start)
    if self.verbose == 1:
      if (now - self._last_update < self.interval and 
        self.target is not None and current < self.target):
          return

      prev_total_width = self._total_width
      if self._dynamic_display:
        sys.stdout.write('\b' * prev_total_width)
        sys.stdout.write('\r')
      else:
        sys.stdout.write('\n')

      if self.target is not None:
        numdigits = int(np.floor(np.log10(self.target))) + 1
        barstr = '%%%dd/%d [' % (numdigits, self.target)
        bar = barstr % current
        prog = float(current) / self.target
        prog_width = int(self.width * prog)
        if prog_width > 0:
          bar += ('=' * (prog_width - 1))
          if current < self.target:
            bar += '>'
          else:
            bar += '='
        bar += ('.' * (self.width - prog_width))
        bar += ']'
      else:
        bar = '%7d/Unknown' % current
      self._total_width = len(bar)
      sys.stdout.write(bar)
      if current:
        time_per_unit = (now - self._start) / current
      else:
        time_per_unit = 0
      if self.target is not None and current < self.target:
        eta = time_per_unit * (self.target - current)
        if eta > 3600:
          eta_format = '%d:%02d:%02d' % (eta // 3600, (eta % 3600) // 60, eta % 60)
        elif eta > 60:
          eta_format = '%d:%02d' % (eta // 60, eta % 60)
        else:
          eta_format = '%ds' % eta
        info = ' - ETA: %s' % eta_format
      else:
        if time_per_unit >= 1:
          info += ' %.0fs/step' % time_per_unit
        elif time_per_unit >= 1e-3:
          info += ' %.0fms/step' % (time_per_unit * 1e3)
        else:
          info += ' %.0fus/step' % (time_per_unit * 1e6)

      for k in self._values_order:
        info += ' - %s:' % k
        if isinstance(self._values[k], list):
          avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
          if abs(avg) > 1e-3:
            info += ' %.4f' % avg
          else:
            info += ' %.4e' % avg
        else:
          info += ' %s' % self._values[k]

      self._total_width += len(info)
      if prev_total_width > self._total_width:
        info += (' ' * (prev_total_width - self._total_width))
      if self.target is not None and current >= self.target:
        info += '\n'
      sys.stdout.write(info)
      sys.stdout.flush()
    elif self.verbose == 2:
      if self.target is None or current >= self.target:
        for k in self._values_order:
          info += ' - %s:' % k
          avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
          if avg > 1e-3:
            info += ' %.4f' % avg
          else:
            info += ' %.4e' % avg
        info += '\n'
        sys.stdout.write(info)
        sys.stdout.flush()
    self._last_update = now

  def add(self, n, values=None):
    self.update(self._seen_so_far + n, values)

#  original implementation 


def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows-1)*strides[0]+effective_k_row-rows)
    padding_cols = max(0, (out_cols-1)*strides[1]+effective_k_col-cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images


def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    batch_size, channel, height, width = images.size()

    if padding == 'same':
        images = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))

    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             dilation=rates,
                             padding=0,
                             stride=strides)
    patches = unfold(images)
    return patches  # [N, C*k*k, L], L is the total number of such blocks


def random_bbox(config, batch_size):
    """Generate a random tlhw with configuration.

    Args:
        config: Config should have configuration including img

    Returns:
        tuple: (top, left, height, width)

    """
    img_height, img_width = config['data_loader']['h'], config['data_loader']['w']
    h = w = config['data_loader']['w'] // 2
    margin_height = margin_width = 0
    maxt = img_height - margin_height - h
    maxl = img_width - margin_width - w
    bbox_list = []
    t = np.random.randint(margin_height, maxt)
    l = np.random.randint(margin_width, maxl)
    bbox_list.append((t, l, h, w))
    bbox_list = bbox_list * batch_size
    
    return torch.tensor(bbox_list, dtype=torch.int64)


def bbox2mask(bboxes, height, width, max_delta_h, max_delta_w):
    batch_size = bboxes.size(0)
    mask = torch.zeros((batch_size, 1, height, width), dtype=torch.float32)
    delta_h = np.random.randint(max_delta_h // 2 + 1)
    delta_w = np.random.randint(max_delta_w // 2 + 1)
    for i in range(batch_size):
      bbox = bboxes[i]
      mask[i, :, bbox[0] + delta_h:bbox[0] + bbox[2] - delta_h, bbox[1] + delta_w:bbox[1] + bbox[3] - delta_w] = 1.
    return mask


def local_patch(x, bbox_list):
    assert len(x.size()) == 4
    patches = []
    for i, bbox in enumerate(bbox_list):
        t, l, h, w = bbox
        patches.append(x[i, :, t:t + h, l:l + w])
    return torch.stack(patches, dim=0)


def mask_image(x, bboxes, config):
    height, width = config['data_loader']['h'], config['data_loader']['w']
    max_delta_h = max_delta_w = 32
    mask = bbox2mask(bboxes, height, width, max_delta_h, max_delta_w)
    mask = set_device(mask)
    result = x * (1. - mask)
    return result, mask


def spatial_discounting_mask(config):
    """Generate spatial discounting mask constant.

    Spatial discounting mask is first introduced in publication:
        Generative Image Inpainting with Contextual Attention, Yu et al.

    Args:
        config: Config should have configuration including HEIGHT, WIDTH,
            DISCOUNTED_MASK.

    Returns:
        tf.Tensor: spatial discounting mask

    """
    gamma = config['losses']['spatial_discounting_gamma']
    height = width = config['data_loader']['w'] // 2
    shape = [1, 1, height, width]
    mask_values = np.ones((height, width))
    for i in range(height):
        for j in range(width):
            mask_values[i, j] = max(
                gamma ** min(i, height - i),
                gamma ** min(j, width - j))
    mask_values = np.expand_dims(mask_values, 0)
    mask_values = np.expand_dims(mask_values, 0)
    spatial_discounting_mask_tensor = torch.tensor(mask_values, dtype=torch.float32)
    spatial_discounting_mask_tensor = set_device(spatial_discounting_mask_tensor)
    return spatial_discounting_mask_tensor


def reduce_mean(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.mean(x, dim=i, keepdim=keepdim)
    return x


def reduce_std(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.std(x, dim=i, keepdim=keepdim)
    return x


def reduce_sum(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.sum(x, dim=i, keepdim=keepdim)
    return x


def flow_to_image(flow):
    """Transfer flow map to image.
    Part of code forked from flownet.
    """
    out = []
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    maxrad = -1
    for i in range(flow.shape[0]):
        u = flow[i, :, :, 0]
        v = flow[i, :, :, 1]
        idxunknow = (abs(u) > 1e7) | (abs(v) > 1e7)
        u[idxunknow] = 0
        v[idxunknow] = 0
        maxu = max(maxu, np.max(u))
        minu = min(minu, np.min(u))
        maxv = max(maxv, np.max(v))
        minv = min(minv, np.min(v))
        rad = np.sqrt(u ** 2 + v ** 2)
        maxrad = max(maxrad, np.max(rad))
        u = u / (maxrad + np.finfo(float).eps)
        v = v / (maxrad + np.finfo(float).eps)
        img = compute_color(u, v)
        out.append(img)
    return np.float32(np.uint8(out))


def pt_flow_to_image(flow):
    """Transfer flow map to image.
    Part of code forked from flownet.
    """
    out = []
    maxu = torch.tensor(-999)
    maxv = torch.tensor(-999)
    minu = torch.tensor(999)
    minv = torch.tensor(999)
    maxrad = torch.tensor(-1)
    if torch.cuda.is_available():
        maxu = maxu.cuda()
        maxv = maxv.cuda()
        minu = minu.cuda()
        minv = minv.cuda()
        maxrad = maxrad.cuda()
    for i in range(flow.shape[0]):
        u = flow[i, 0, :, :]
        v = flow[i, 1, :, :]
        idxunknow = (torch.abs(u) > 1e7) + (torch.abs(v) > 1e7)
        u[idxunknow] = 0
        v[idxunknow] = 0
        maxu = torch.max(maxu, torch.max(u))
        minu = torch.min(minu, torch.min(u))
        maxv = torch.max(maxv, torch.max(v))
        minv = torch.min(minv, torch.min(v))
        rad = torch.sqrt((u ** 2 + v ** 2).float()).to(torch.int64)
        maxrad = torch.max(maxrad, torch.max(rad))
        u = u / (maxrad + torch.finfo(torch.float32).eps)
        v = v / (maxrad + torch.finfo(torch.float32).eps)
        # TODO: change the following to pytorch
        img = pt_compute_color(u, v)
        out.append(img)

    return torch.stack(out, dim=0)


def highlight_flow(flow):
    """Convert flow into middlebury color code image.
    """
    out = []
    s = flow.shape
    for i in range(flow.shape[0]):
        img = np.ones((s[1], s[2], 3)) * 144.
        u = flow[i, :, :, 0]
        v = flow[i, :, :, 1]
        for h in range(s[1]):
            for w in range(s[1]):
                ui = u[h, w]
                vi = v[h, w]
                img[ui, vi, :] = 255.
        out.append(img)
    return np.float32(np.uint8(out))


def pt_highlight_flow(flow):
    """Convert flow into middlebury color code image.
        """
    out = []
    s = flow.shape
    for i in range(flow.shape[0]):
        img = np.ones((s[1], s[2], 3)) * 144.
        u = flow[i, :, :, 0]
        v = flow[i, :, :, 1]
        for h in range(s[1]):
            for w in range(s[1]):
                ui = u[h, w]
                vi = v[h, w]
                img[ui, vi, :] = 255.
        out.append(img)
    return np.float32(np.uint8(out))


def compute_color(u, v):
    h, w = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0
    # colorwheel = COLORWHEEL
    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)
    rad = np.sqrt(u ** 2 + v ** 2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1) + 1
    k0 = np.floor(fk).astype(int)
    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0
    for i in range(np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)
        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))
    return img


def pt_compute_color(u, v):
    h, w = u.shape
    img = torch.zeros([3, h, w])
    if torch.cuda.is_available():
        img = img.cuda()
    nanIdx = (torch.isnan(u) + torch.isnan(v)) != 0
    u[nanIdx] = 0.
    v[nanIdx] = 0.
    # colorwheel = COLORWHEEL
    colorwheel = pt_make_color_wheel()
    if torch.cuda.is_available():
        colorwheel = colorwheel.cuda()
    ncols = colorwheel.size()[0]
    rad = torch.sqrt((u ** 2 + v ** 2).to(torch.float32))
    a = torch.atan2(-v.to(torch.float32), -u.to(torch.float32)) / np.pi
    fk = (a + 1) / 2 * (ncols - 1) + 1
    k0 = torch.floor(fk).to(torch.int64)
    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0.to(torch.float32)
    for i in range(colorwheel.size()[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1]
        col1 = tmp[k1 - 1]
        col = (1 - f) * col0 + f * col1
        idx = rad <= 1. / 255.
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = (idx != 0)
        col[notidx] *= 0.75
        img[i, :, :] = col * (1 - nanIdx).to(torch.float32)
    return img


def make_color_wheel():
    RY, YG, GC, CB, BM, MR = (15, 6, 4, 11, 13, 6)
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros([ncols, 3])
    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY
    # YG
    colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG
    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC
    # CB
    colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB
    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += + BM
    # MR
    colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255
    return colorwheel


def pt_make_color_wheel():
    RY, YG, GC, CB, BM, MR = (15, 6, 4, 11, 13, 6)
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = torch.zeros([ncols, 3])
    col = 0
    # RY
    colorwheel[0:RY, 0] = 1.
    colorwheel[0:RY, 1] = torch.arange(0, RY, dtype=torch.float32) / RY
    col += RY
    # YG
    colorwheel[col:col + YG, 0] = 1. - (torch.arange(0, YG, dtype=torch.float32) / YG)
    colorwheel[col:col + YG, 1] = 1.
    col += YG
    # GC
    colorwheel[col:col + GC, 1] = 1.
    colorwheel[col:col + GC, 2] = torch.arange(0, GC, dtype=torch.float32) / GC
    col += GC
    # CB
    colorwheel[col:col + CB, 1] = 1. - (torch.arange(0, CB, dtype=torch.float32) / CB)
    colorwheel[col:col + CB, 2] = 1.
    col += CB
    # BM
    colorwheel[col:col + BM, 2] = 1.
    colorwheel[col:col + BM, 0] = torch.arange(0, BM, dtype=torch.float32) / BM
    col += BM
    # MR
    colorwheel[col:col + MR, 2] = 1. - (torch.arange(0, MR, dtype=torch.float32) / MR)
    colorwheel[col:col + MR, 0] = 1.
    return colorwheel


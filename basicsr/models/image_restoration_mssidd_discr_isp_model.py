# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
import numpy as np

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.dist_util import get_dist_info

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')


class RawISPProcessing(nn.Module):
    def __init__(self, awb_en=1, ccm_en=1, clip=0):
        super(RawISPProcessing, self).__init__()
        self.awb_en = awb_en
        self.ccm_en = ccm_en
        self.clip = clip

    def gamma_compression(self, images, gamma=2.2):
        """Converts from linear to gamma space."""
        images = torch.clamp(images, min=1e-8)
        return images ** (1.0 / gamma)
    
    def demosiac(self, bayer_images):
        # bayer_images: bs*sn,4,256,256
        H, W = bayer_images.shape[-2:]

        red = bayer_images[:, 0:1, :, :]   # bs*sn,1,256,256
        red = F.interpolate(red, size=[H*2, W*2], mode="bilinear")  # bs*sn,1,512,512 

        green_red = bayer_images[:, 1:2, :, :]   # bs*sn,1,256,256
        green_red = torch.flip(green_red, dims=(-1,))
        green_red = F.interpolate(green_red, size=[H*2, W*2], mode="bilinear")   # bs*sn,1,512,512
        green_red = torch.flip(green_red, dims=(-1,))
        green_red = F.pixel_unshuffle(green_red, downscale_factor=2)   # bs*sn,4,256,256

        green_blue = bayer_images[:, 2:3, :, :]   # bs*sn,1,256,256
        green_blue = torch.flip(green_blue, dims=(-2,))
        green_blue = F.interpolate(green_blue, size=[H*2, W*2], mode="bilinear")   # bs*sn,1,512,512
        green_blue = torch.flip(green_blue, dims=(-2,))
        green_blue = F.pixel_unshuffle(green_blue, downscale_factor=2)   # bs*sn,4,256,256

        green_at_red = (green_red[:,0,:,:] + green_blue[:,0,:,:]) / 2    # bs*sn,256,256
        green_at_green_red = green_red[:,1,:,:]   # bs*sn,256,256
        green_at_green_blue = green_blue[:,2,:,:]   # bs*sn,256,256
        green_at_blue = (green_red[:,3,:,:] + green_blue[:,3,:,:]) / 2   # bs*sn,256,256
        green_planes = [green_at_red, green_at_green_red, green_at_green_blue, green_at_blue]
        green = torch.stack(green_planes, dim=1)   # bs*sn,4,256,256
        green = F.pixel_shuffle(green, upscale_factor=2)   # bs*sn,1,512,512

        blue = bayer_images[:, 3:4, :, :]   # bs*sn,1,256,256
        blue = torch.flip(blue, dims=(-1,))
        blue = F.interpolate(blue, size=[H*2, W*2], mode="bilinear")   # bs*sn,1,512,512
        blue = torch.flip(blue, dims=(-1,))

        rgb_images = torch.cat([red, green, blue], dim=1)

        return rgb_images

    def isp(self, x, dgain, awb_gains=None, ccms=None):
        # x: bs*sn,3,512,512  or  bs*sn,4,256,256
        # dgain: bs*sn,1     awb_gains: bs*sn,3,3   ccms: bs*sn,3,3

        if x.shape[1] == 4:
            x = self.demosiac(x)  # bs*sn,4,256,256 -> bs*sn,3,512,512

        # x = x * dgain[:, :, None, None]
        x = x * 2

        x = x[:, [2, 1, 0], ...]
        if self.awb_en == 1:
            x = torch.einsum('ijwh,ijk->ikwh', x, awb_gains)
            if self.clip:
                x = torch.clamp(x, min=1e-8, max=1.0)
        if self.ccm_en == 1:
            x = torch.einsum('ijwh,ijk->ikwh', x, ccms)
            if self.clip:
                x = torch.clamp(x, min=1e-8, max=1.0)
        x = x[:, [2, 1, 0], ...]

        x = self.gamma_compression(x)
        return x

    def forward(self, pred, gt, awb, cam2rgb, rgb_gain):
        x = self.isp(pred, rgb_gain, awb_gains=awb, ccms=cam2rgb)
        y = self.isp(gt, rgb_gain, awb_gains=awb, ccms=cam2rgb)
        return x, y

class ImageRestorationMSSIDDDiscrISPModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageRestorationMSSIDDDiscrISPModel, self).__init__(opt)

        # define network
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

        self.scale = int(opt['scale'])

        self.isp_pipeline = RawISPProcessing(awb_en=1, ccm_en=1, clip=0)

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.total_iter = train_opt['total_iter']

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            percep_type = train_opt['perceptual_opt'].pop('type')
            cri_perceptual_cls = getattr(loss_module, percep_type)
            self.cri_perceptual = cri_perceptual_cls(
                **train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
        #         if k.startswith('module.offsets') or k.startswith('module.dcns'):
        #             optim_params_lowlr.append(v)
        #         else:
                optim_params.append(v)
            # else:
            #     logger = get_root_logger()
            #     logger.warning(f'Params {k} will not be optimized.')
        # print(optim_params)
        # ratio = 0.1

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam([{'params': optim_params}],
                                                **train_opt['optim_g'])
        elif optim_type == 'SGD':
            self.optimizer_g = torch.optim.SGD(optim_params,
                                               **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW([{'params': optim_params}],
                                                **train_opt['optim_g'])
            pass
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data, is_val=False):
        self.lq = data['lq'].to(self.device)   # bs,sn,4,256,256
        B, SN, _, _, _ = self.lq.shape
        self.batch_size = B
        self.sensor_num = SN
        self.lq = self.lq.reshape(B*SN, *self.lq.shape[-3:])  # bs*sn,4,256,256

        if 'gt' in data:
            self.gt = data['gt'].to(self.device)     # bs,sn,3,512,512  or  bs,sn,4,256,256
            self.gt = self.gt.reshape(B*SN, *self.gt.shape[-3:])   # bs*sn,3,512,512  or  bs*sn,4,256,256

        if 'var' in data:
            self.var = data['var'].to(self.device)  # bs,sn,4,256,256

        if 'awb' in data and 'cam2rgb' in data and 'rgb_gain' in data:
            self.awb = data['awb'].to(self.device)   # bs,sn,3,3
            self.awb = self.awb.reshape(B*SN, *self.awb.shape[-2:])  # bs*sn,3,3
            self.cam2rgb = data['cam2rgb'].to(self.device)  # bs,sn,3,3
            self.cam2rgb = self.cam2rgb.reshape(B*SN, *self.cam2rgb.shape[-2:])  # bs*sn,3,3
            self.rgb_gain = data['rgb_gain'].to(self.device)   # bs,sn,1
            self.rgb_gain = self.rgb_gain.reshape(B*SN, 1)  # bs*sn,1


    def optimize_parameters(self, current_iter, tb_logger):
        self.optimizer_g.zero_grad()

        # lq: bs,sn,4,256,256 -> bs*sn,4,256,256
        # gt: bs,sn,3,512,512 -> bs*sn,3,512,512  or  bs,sn,4,256,256 -> bs*sn,4,256,256

        if self.opt['train'].get('mixup', False):
            self.mixup_aug()

        p = float(current_iter / self.total_iter)
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        preds, sensor_pred = self.net_g(self.lq, alpha)

        # apply isp
        preds_isp, self.gt = self.isp_pipeline(preds, self.gt, self.awb, self.cam2rgb, self.rgb_gain)

        if not isinstance(preds_isp, list):
            preds_isp = [preds_isp]

        self.output = preds_isp[-1]

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = 0.
            for pred in preds_isp:
                l_pix += self.cri_pix(pred, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style


        l_total = l_total + 0. * sum(p.sum() for p in self.net_g.parameters())

        l_total.backward()
        use_grad_clip = self.opt['train'].get('use_grad_clip', True)
        if use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()


        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            n = len(self.lq)
            outs = []
            m = self.opt['val'].get('max_minibatch', n)
            i = 0
            while i < n:
                j = i + m
                if j >= n:
                    j = n
                pred, _ = self.net_g(self.lq[i:j], 0)
                if isinstance(pred, list):
                    pred = pred[-1]
                outs.append(pred.detach())
                i = j

            self.output = torch.cat(outs, dim=0)
        self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }

        rank, world_size = get_dist_info()
        if rank == 0:
            pbar = tqdm(total=len(dataloader), unit='image')

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            if idx % world_size != rank:
                continue

            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            self.feed_data(val_data, is_val=True)

            self.test()

            # apply isp
            self.output, self.gt = self.isp_pipeline(self.output, self.gt, self.awb, self.cam2rgb, self.rgb_gain)

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if sr_img.shape[2] == 6:
                    L_img = sr_img[:, :, :3]
                    R_img = sr_img[:, :, 3:]

                    # visual_dir = osp.join('visual_results', dataset_name, self.opt['name'])
                    visual_dir = osp.join(self.opt['path']['visualization'], dataset_name)

                    imwrite(L_img, osp.join(visual_dir, f'{img_name}_L.png'))
                    imwrite(R_img, osp.join(visual_dir, f'{img_name}_R.png'))
                else:
                    if self.opt['is_train']:

                        save_img_path = osp.join(self.opt['path']['visualization'],
                                                 img_name,
                                                 f'{img_name}_{current_iter}.png')

                        save_gt_img_path = osp.join(self.opt['path']['visualization'],
                                                 img_name,
                                                 f'{img_name}_{current_iter}_gt.png')
                    else:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}.png')
                        save_gt_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_gt.png')

                    imwrite(sr_img, save_img_path)
                    imwrite(gt_img, save_gt_img_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            cnt += 1
            if rank == 0:
                for _ in range(world_size):
                    pbar.update(1)
                    pbar.set_description(f'Test {img_name}')
        if rank == 0:
            pbar.close()

        # current_metric = 0.
        collected_metrics = OrderedDict()
        if with_metrics:
            for metric in self.metric_results.keys():
                collected_metrics[metric] = torch.tensor(self.metric_results[metric]).float().to(self.device)
            collected_metrics['cnt'] = torch.tensor(cnt).float().to(self.device)

            self.collected_metrics = collected_metrics
        
        keys = []
        metrics = []
        for name, value in self.collected_metrics.items():
            keys.append(name)
            metrics.append(value)
        metrics = torch.stack(metrics, 0)
        torch.distributed.reduce(metrics, dst=0)
        if self.opt['rank'] == 0:
            metrics_dict = {}
            cnt = 0
            for key, metric in zip(keys, metrics):
                if key == 'cnt':
                    cnt = float(metric)
                    continue
                metrics_dict[key] = float(metric)

            for key in metrics_dict:
                metrics_dict[key] /= cnt

            self._log_validation_metric_values(current_iter, dataloader.dataset.opt['name'],
                                               tb_logger, metrics_dict)
        return 0.

    def nondist_validation(self, *args, **kwargs):
        logger = get_root_logger()
        logger.warning('nondist_validation is not implemented. Run dist_validation.')
        self.dist_validation(*args, **kwargs)


    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger, metric_dict):
        log_str = f'Validation {dataset_name}, \t'
        for metric, value in metric_dict.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)

        log_dict = OrderedDict()
        # for name, value in loss_dict.items():
        for metric, value in metric_dict.items():
            log_dict[f'm_{metric}'] = value

        self.log_dict = log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)

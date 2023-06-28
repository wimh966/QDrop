# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import copy
import os.path as osp
import time
import warnings
import torch
import logging
import torch.distributed as dist
import torch.nn as nn
import mmcv
from mmcv.utils import get_logger
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmdet.apis import multi_gpu_test, single_gpu_test, set_random_seed
from mmdet.models import build_detector
from mmdet.utils import (build_ddp, build_dp, compat_cfg, get_device,
                         replace_cfg_vals, rfnext_init_model, get_root_logger,
                         setup_multi_processes, update_data_root)
from qdrop.model.quant_model import specials
from qdrop.quantization.state import enable_calibration_woquantization, enable_quantization, disable_all
from qdrop.quantization.quantized_module import QuantizedLayer, QuantizedBlock
from qdrop.quantization.fake_quant import QuantizeBase
from qdrop.quantization.observer import ObserverBase
from recon import reconstruction
import utils
logger = None


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument('--seed', type=int, default=1005, help='random seed')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpus',
        type=int,
        help='(Deprecated, please use --gpu-id) number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--q_config',
        type=str,
        default=None,
        help='quantization config files')

    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    global logger
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    # code of configs
    cfg = Config.fromfile(args.config)
    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)
    # update data root according to MMDET_DATASETS
    update_data_root(cfg)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg = compat_cfg(cfg)

    # seed
    set_random_seed(args.seed, deterministic=True)
    cfg.seed = args.seed

    # work_dir and logger
    # create work_dir
    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_logger(name='qdrop', log_file=log_file, log_level=logging.INFO)
    json_file = osp.join(args.work_dir, f'eval_{timestamp}.json')

    # set multi-process settings
    setup_multi_processes(cfg)
    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn('`--gpus` is deprecated because we only support '
                      'single GPU mode in non-distributed training. '
                      'Use `gpus=1` now.')
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed training. Use the first GPU '
                      'in `gpu_ids` now.')
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        rank, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)
    cfg.device = get_device()

    if args.q_config:
        q_config = utils.parse_config(args.q_config)

    val_loader, val_dataset, cali_data = utils.load_data(cfg, distributed, q_config.calibrate)
    # build model
    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    # init rfnext if 'RFSearchHook' is defined in cfg
    rfnext_init_model(model, cfg=cfg)

    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = val_dataset.CLASSES

    if args.q_config:
        for para in model.parameters():
            para.requires_grad = False
        model = quantize_model(model, q_config)
        model.cuda()
        model.eval()
        fp_model = copy.deepcopy(model)
        calibrate(model, cali_data)
        if hasattr(q_config, 'recon'):
            if rank == 0:
                logger.info('begin to do reconstruction')
            recon_model(model, fp_model, cali_data, q_config.recon)
        enable_quantization(model)
    # begin to evaluate
    model = build_ddp(
        model,
        cfg.device,
        device_ids=[int(os.environ['LOCAL_RANK'])],
        broadcast_buffers=False)
    outputs = multi_gpu_test(
        model, val_loader, args.tmpdir, args.gpu_collect
        or cfg.evaluation.get('gpu_collect', False))

    if rank == 0:
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule', 'dynamic_intervals'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            metric = val_dataset.evaluate(outputs, **eval_kwargs)
            metric_dict = dict(config=args.config, metric=metric)
            if args.work_dir is not None and rank == 0:
                mmcv.dump(metric_dict, json_file)


def quantize_model(model, config_quant):

    def replace_module(module, w_qconfig, a_qconfig, qoutput=True):
        for name, child_module in module.named_children():
            if 'fpn_convs' in name:
                qoutput = False
            if type(child_module) in specials:
                setattr(module, name, specials[type(child_module)](child_module, w_qconfig, a_qconfig))
            elif isinstance(child_module, (nn.Conv2d, nn.Linear)):
                setattr(module, name, QuantizedLayer(child_module, None, w_qconfig, a_qconfig, qoutput))
                prev_quantmodule = getattr(module, name)
            elif isinstance(child_module, (nn.ReLU, nn.ReLU6)):
                if prev_quantmodule is not None:
                    prev_quantmodule.activation = child_module
                    setattr(module, name, nn.Identity())
                else:
                    pass
            elif isinstance(child_module, nn.Identity):
                pass
            else:
                replace_module(child_module, w_qconfig, a_qconfig, qoutput)


    replace_module(model.backbone, config_quant.w_qconfig, config_quant.a_qconfig)
    replace_module(model.neck, config_quant.w_qconfig, config_quant.a_qconfig)
    'set first layer\'s weight to 8-bit'
    w_list, a_list = [], []
    for name, module in model.named_modules():
        if isinstance(module, QuantizeBase) and 'weight' in name:
            w_list.append(module)
        if isinstance(module, QuantizeBase) and 'act' in name:
            a_list.append(module)
    w_list[0].set_bit(8)
    return model


@torch.no_grad()
def calibrate(model, cali_data):
    st = time.time()
    enable_calibration_woquantization(model, quantizer_type='act_fake_quant')
    for i in range(len(cali_data)):
        model.extract_feat(cali_data[i].cuda())
    rank, world_size = get_dist_info()
    for name, module in model.named_modules():
        if isinstance(module, ObserverBase):
            module.min_val.data /= world_size
            module.max_val.data /= world_size
            dist.all_reduce(module.min_val.data)
            dist.all_reduce(module.max_val.data)
    enable_calibration_woquantization(model, quantizer_type='weight_fake_quant')  
    model.extract_feat(cali_data[0].cuda())
    ed = time.time()
    if rank == 0:
        logger.info('the calibration time is {}'.format(ed - st))


def recon_model(model, fp_model, cali_data, recon_config):
    enable_quantization(model)
    from mmdet.models.necks.fpn import FPN
    def _recon_model(module, fp_module):
        for name, child_module in module.named_children():
            if isinstance(child_module, (QuantizedLayer, QuantizedBlock)):
                logger.info('begin reconstruction for module:\n{}'.format(str(child_module)))
                reconstruction(model, fp_model, child_module, getattr(fp_module, name), cali_data, recon_config)
            else:
                _recon_model(child_module, getattr(fp_module, name))
    # Start reconstruction
    _recon_model(model, fp_model)


if __name__ == '__main__':
    main()

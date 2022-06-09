import numpy as np  # noqa: F401
import copy
import time
import torch
import torch.nn as nn
import logging
import argparse
import imagenet_utils
from recon import reconstruction
from fold_bn import search_fold_and_remove_bn, StraightThrough
from qdrop.model import load_model, specials
from qdrop.quantization.state import enable_calibration_woquantization, enable_quantization, disable_all
from qdrop.quantization.quantized_module import QuantizedLayer, QuantizedBlock
from qdrop.quantization.fake_quant import QuantizeBase
from qdrop.quantization.observer import ObserverBase
logger = logging.getLogger('qdrop')
logging.basicConfig(level=logging.INFO, format='%(message)s')


def quantize_model(model, config_quant):

    def replace_module(module, w_qconfig, a_qconfig, qoutput=True):
        childs = list(iter(module.named_children()))
        st, ed = 0, len(childs)
        prev_quantmodule = None
        while(st < ed):
            tmp_qoutput = qoutput if st == ed - 1 else True
            name, child_module = childs[st][0], childs[st][1]
            if type(child_module) in specials:
                setattr(module, name, specials[type(child_module)](child_module, w_qconfig, a_qconfig, tmp_qoutput))
            elif isinstance(child_module, (nn.Conv2d, nn.Linear)):
                setattr(module, name, QuantizedLayer(child_module, None, w_qconfig, a_qconfig, qoutput=tmp_qoutput))
                prev_quantmodule = getattr(module, name)
            elif isinstance(child_module, (nn.ReLU, nn.ReLU6)):
                if prev_quantmodule is not None:
                    prev_quantmodule.activation = child_module
                    setattr(module, name, StraightThrough())
                else:
                    pass
            elif isinstance(child_module, StraightThrough):
                pass
            else:
                replace_module(child_module, w_qconfig, a_qconfig, tmp_qoutput)
            st += 1
    replace_module(model, config_quant.w_qconfig, config_quant.a_qconfig, qoutput=False)
    w_list, a_list = [], []
    for name, module in model.named_modules():
        if isinstance(module, QuantizeBase) and 'weight' in name:
            w_list.append(module)
        if isinstance(module, QuantizeBase) and 'act' in name:
            a_list.append(module)
    w_list[0].set_bit(8)
    w_list[-1].set_bit(8)
    'the image input has already been in 256, set the last layer\'s input to 8-bit'
    a_list[-1].set_bit(8)
    logger.info('finish quantize model:\n{}'.format(str(model)))
    return model


def get_cali_data(train_loader, num_samples):
    cali_data = []
    for batch in train_loader:
        cali_data.append(batch[0])
        if len(cali_data) * batch[0].size(0) >= num_samples:
            break
    return torch.cat(cali_data, dim=0)[:num_samples]


def main(config_path):
    config = imagenet_utils.parse_config(config_path)
    imagenet_utils.set_seed(config.process.seed)
    'cali data'
    train_loader, val_loader = imagenet_utils.load_data(**config.data)
    cali_data = get_cali_data(train_loader, config.quant.calibrate)
    'model'
    model = load_model(config.model)
    search_fold_and_remove_bn(model)
    if hasattr(config, 'quant'):
        model = quantize_model(model, config.quant)
    model.cuda()
    model.eval()
    fp_model = copy.deepcopy(model)
    disable_all(fp_model)
    for name, module in model.named_modules():
        if isinstance(module, ObserverBase):
            module.set_name(name)

    # calibrate first
    with torch.no_grad():
        st = time.time()
        enable_calibration_woquantization(model, quantizer_type='act_fake_quant')
        model(cali_data[: 256].cuda())
        enable_calibration_woquantization(model, quantizer_type='weight_fake_quant')
        model(cali_data[: 2].cuda())
        ed = time.time()
        logger.info('the calibration time is {}'.format(ed - st))

    if hasattr(config.quant, 'recon'):
        enable_quantization(model)

        def recon_model(module: nn.Module, fp_module: nn.Module):
            """
            Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
            """
            for name, child_module in module.named_children():
                if isinstance(child_module, (QuantizedLayer, QuantizedBlock)):
                    logger.info('begin reconstruction for module:\n{}'.format(str(child_module)))
                    reconstruction(model, fp_model, child_module, getattr(fp_module, name), cali_data, config.quant.recon)
                else:
                    recon_model(child_module, getattr(fp_module, name))
        # Start reconstruction
        recon_model(model, fp_model)
    enable_quantization(model)
    imagenet_utils.validate_model(val_loader, model)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='configuration',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config', default='config.yaml', type=str)
    args = parser.parse_args()
    main(args.config)

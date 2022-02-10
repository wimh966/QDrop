import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union


class StraightThrough(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input


def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred - tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred - tgt).abs().pow(p).mean()


class UniformAffineQuantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.

    :param n_bits: number of bit for quantization
    :param symmetric: if True, the zero_point should always be 0
    :param channel_wise: if True, compute scale and zero_point in each channel
    :param scale_method: determines the quantization scale and zero point
    :param prob: for qdrop;
    """

    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False,
                 scale_method: str = 'minmax',
                 leaf_param: bool = False, prob: float = 1.0):
        super(UniformAffineQuantizer, self).__init__()
        self.sym = symmetric
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta = 1.0
        self.zero_point = 0.0
        self.inited = True

        '''if leaf_param, use EMA to set scale'''
        self.leaf_param = leaf_param
        self.channel_wise = channel_wise
        self.scale_method = scale_method

        '''for activation quantization'''
        self.running_min = None
        self.running_max = None

        '''do like dropout'''
        self.prob = prob
        self.is_training = False

    def set_inited(self, inited: bool = True):  # inited manually
        self.inited = inited

    def update_quantize_range(self, x_min: float, x_max: float):
        if self.running_min is None:
            self.running_min = x_min
            self.running_max = x_max
        self.running_min = 0.1 * x_min + 0.9 * self.running_min
        self.running_max = 0.1 * x_max + 0.9 * self.running_max
        x_min = self.running_min
        x_max = self.running_max
        return x_min, x_max

    def forward(self, x: torch.Tensor):
        if self.inited is False:
            if self.leaf_param:
                self.delta, self.zero_point = self.init_quantization_scale(x.clone().detach(), self.channel_wise)
                # self.delta=torch.nn.Parameter(torch.tensor(delta).type_as(x))
                # self.zero_point=torch.nn.Parameter(torch.tensor(zero_point).type_as(x))
            else:
                self.delta, self.zero_point = self.init_quantization_scale(x.clone().detach(), self.channel_wise)

        # start quantization
        x_int = round_ste(x / self.delta) + self.zero_point
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - self.zero_point) * self.delta
        if self.is_training and self.prob < 1.0:
            x_ans = torch.where(torch.rand_like(x) < self.prob, x_dequant, x)
        else:
            x_ans = x_dequant
        return x_ans

    def get_x_min_x_max(self, x, x_min: float, x_max: float):
        if 'max' in self.scale_method:
            if 'scale' in self.scale_method:
                x_min = x_min * (self.n_bits + 2) / 8
                x_max = x_max * (self.n_bits + 2) / 8
            if self.leaf_param:
                x_min, x_max = self.update_quantize_range(x_min, x_max)
            x_absmax = max(abs(x_min), x_max)
            if self.sym:
                x_min, x_max = -x_absmax if x_min < 0 else 0, x_absmax
            return x_min, x_max
        elif self.scale_method == 'mse':
            best_score = 1e+10
            best_min, best_max = x_min, x_max
            for i in range(80):
                new_max = x_max * (1.0 - (i * 0.01))
                new_min = x_min * (1.0 - (i * 0.01))
                x_q = self.quantize(x, new_max, new_min)
                # L_p norm minimization as described in LAPQ
                # https://arxiv.org/abs/1911.07190
                score = lp_loss(x, x_q, 2.4, reduction='all')
                if score < best_score:
                    best_score = score
                    best_min, best_max = new_min, new_max
            x_min, x_max = best_min, best_max
            if self.leaf_param:
                x_min, x_max = self.update_quantize_range(x_min, x_max)
            return x_min, x_max
        else:
            raise NotImplementedError

    def init_quantization_scale_channel(self, x: torch.Tensor):
        x_min, x_max = x.min().item(), x.max().item()
        x_min, x_max = self.get_x_min_x_max(x, x_min, x_max)
        delta = (x_max - x_min) / (2 ** self.n_bits - 1)
        delta = max(delta, 1e-8)
        zero_point = round(-x_min / delta)
        return delta, zero_point

    def init_quantization_scale(self, x_clone: torch.Tensor, channel_wise: bool = False):
        if channel_wise:
            n_channels = x_clone.shape[0]
            if len(x_clone.shape) == 4:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            else:
                x_max = x_clone.abs().max(dim=-1)[0]
            delta = x_max.clone()
            zero_point = x_max.clone()
            # determine the scale and zero point channel-by-channel
            for c in range(n_channels):
                delta[c], zero_point[c] = self.init_quantization_scale_channel(x_clone[c])
            if len(x_clone.shape) == 4:
                delta = delta.view(-1, 1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1, 1)
            else:
                delta = delta.view(-1, 1)
                zero_point = zero_point.view(-1, 1)
        else:
            delta, zero_point = self.init_quantization_scale_channel(x_clone)

        return delta, zero_point

    def quantize(self, x: torch.Tensor, x_max: float, x_min: float):
        delta = (x_max - x_min) / (2 ** self.n_bits - 1)
        delta = max(delta, 1e-8)
        zero_point = round(-x_min / delta)
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q

    def bitwidth_refactor(self, refactored_bit: int):
        assert 2 <= refactored_bit <= 8, 'bitwidth not supported'
        self.n_bits = refactored_bit
        self.n_levels = 2 ** self.n_bits

    @torch.jit.export
    def extra_repr(self):
        return 'bit={}, is_training={}, inited={}'.format(
            self.n_bits, self.is_training, self.inited
        )


class QuantModule(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """

    def __init__(self, org_module: Union[nn.Conv2d, nn.Linear], weight_quant_params: dict = {},
                 act_quant_params: dict = {}, disable_act_quant=False):
        super(QuantModule, self).__init__()
        if isinstance(org_module, nn.Conv2d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv2d
        else:
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear
        self.weight = org_module.weight
        self.org_weight = org_module.weight.data.clone()
        if org_module.bias is not None:
            self.bias = org_module.bias
            self.org_bias = org_module.bias.data.clone()
        else:
            self.bias = None
            self.org_bias = None
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        # initialize quantizer
        self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params)
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

        self.activation_function = StraightThrough()
        self.ignore_reconstruction = False
        self.disable_act_quant = disable_act_quant

    def forward(self, input: torch.Tensor):
        if self.use_weight_quant:
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.org_weight
            bias = self.org_bias
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        # disable act quantization is designed for convolution before elemental-wise operation,
        # in that case, we apply activation function and quantization after ele-wise op.
        out = self.activation_function(out)
        if self.disable_act_quant:
            return out
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    @torch.jit.export
    def extra_repr(self):
        return 'weight_quantizer={}, act_quantizer={}, disable_act_quant={}'.format(
            self.weight_quantizer, self.act_quantizer, self.disable_act_quant
        )

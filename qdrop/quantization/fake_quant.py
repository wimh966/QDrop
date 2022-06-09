import torch
import torch.nn as nn
from .observer import ObserverBase
from .util_quant import (
    fake_quantize_per_channel_affine,
    fake_quantize_per_tensor_affine,
    fake_quantize_learnable_per_tensor_affine_training,
    fake_quantize_learnable_per_channel_affine_training,
    fake_quantize_learnableplus_per_channel_affine_training,
    fake_quantize_learnableplus_per_tensor_affine_training,
)


class QuantizeBase(nn.Module):

    def __init__(self, observer=ObserverBase, bit=8, symmetric=False, ch_axis=-1):
        super().__init__()
        self.observer = observer(bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        self.bit = bit
        self.symmetric = symmetric
        self.ch_axis = ch_axis
        self.observer_enabled = 0
        self.fake_quant_enabled = 0
        self.quant_min = self.observer.quant_min
        self.quant_max = self.observer.quant_max
        self.drop_prob = 1.0

    def set_bit(self, bit):
        self.observer.set_bit(bit)
        self.bit = bit
        self.quant_min = self.observer.quant_min
        self.quant_max = self.observer.quant_max

    def set_name(self, name):
        self.name = name

    @torch.jit.export
    def calculate_qparams(self):
        return self.observer.calculate_qparams()

    @torch.jit.export
    def disable_observer(self):
        self.observer_enabled = 0

    @torch.jit.export
    def enable_observer(self):
        self.observer_enabled = 1

    @torch.jit.export
    def disable_fake_quant(self):
        self.fake_quant_enabled = 0

    @torch.jit.export
    def enable_fake_quant(self):
        self.fake_quant_enabled = 1

    @torch.jit.export
    def extra_repr(self):
        return 'fake_quant_enabled={}, observer_enabled={}, ' \
               'symmetric={}, bit={}, ch_axis={}, quant_min={}, quant_max={}'.format(
                   self.fake_quant_enabled, self.observer_enabled,
                   self.symmetric, self.bit, self.ch_axis,
                   self.quant_min, self.quant_max)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # We cannot currently register scalar values as buffers, so need to manually
        # specify serialization here.
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'scale'] = self.scale
        destination[prefix + 'zero_point'] = self.zero_point

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Removing this function throws an error that the the size of the loaded tensor does not match the original size
        # i.e., These buffers start out with numel 0 and become numel 1 once they have their first forward pass.
        local_state = ['scale', 'zero_point']
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                if name == 'scale':
                    if isinstance(self.scale, nn.Parameter):
                        self.scale.data = torch.ones_like(val.to(self.scale.device))
                    else:
                        self.scale.resize_(val.shape)
                else:
                    assert name == 'zero_point'
                    if isinstance(self.zero_point, nn.Parameter):
                        self.zero_point.data = torch.ones_like(val.to(self.zero_point.device))
                    else:
                        self.zero_point.resize_(val.shape)
                # For torchscript module we need to update the attributes here since we do not
                # call the `_load_from_state_dict` function defined module.py
                if torch.jit.is_scripting():
                    if name == 'scale':
                        self.scale.copy_(val)
                    else:
                        assert name == 'zero_point'
                        self.zero_point.copy_(val)
            elif strict:
                missing_keys.append(key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)


class FixedFakeQuantize(QuantizeBase):

    def __init__(self, observer, bit=8, symmetric=False, ch_axis=-1):
        super().__init__(observer, bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        self.register_buffer('scale', torch.tensor([1.0], dtype=torch.float))
        self.register_buffer('zero_point', torch.tensor([0], dtype=torch.int))
        self.drop_prob = 1.0

    def forward(self, X):
        if self.observer_enabled == 1:
            self.observer(X.detach())
            _scale, _zero_point = self.observer.calculate_qparams(self.observer.min_val, self.observer.max_val)
            _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
            if self.scale.shape != _scale.shape:
                self.scale.resize_(_scale.shape)
                self.zero_point.resize_(_zero_point.shape)
            self.scale.copy_(_scale)
            self.zero_point.copy_(_zero_point)

        if self.fake_quant_enabled == 1:
            if self.drop_prob < 1.0:
                x_orig = X

            if self.ch_axis != -1:
                X = fake_quantize_per_channel_affine(
                    X, self.scale.data, self.zero_point.data.int(), self.ch_axis,
                    self.quant_min, self.quant_max)
            else:
                X = fake_quantize_per_tensor_affine(
                    X, self.scale.item(), self.zero_point.item(),
                    self.quant_min, self.quant_max)
            if self.drop_prob < 1.0:
                x_prob = torch.where(torch.rand_like(X) < self.drop_prob, X, x_orig)
                return x_prob

        return X


class LSQFakeQuantize(QuantizeBase):

    def __init__(self, observer, bit=8, symmetric=False, ch_axis=-1, use_grad_scaling=True):
        super().__init__(observer, bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        self.scale = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float))
        self.register_buffer('zero_point', torch.tensor([0], dtype=torch.int))
        self.register_buffer('eps', torch.tensor([torch.finfo(torch.float32).eps]))
        self.use_grad_scaling = use_grad_scaling
        self.drop_prob = 1.0

    def forward(self, X):
        if self.observer_enabled == 1:
            self.observer(X.detach())
            _scale, _zero_point = self.observer.calculate_qparams(self.observer.min_val, self.observer.max_val)
            _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
            if self.ch_axis != -1:
                self.scale.data = torch.ones_like(_scale)
                self.zero_point.resize_(_zero_point.shape)

            self.scale.data.copy_(_scale)
            self.zero_point.copy_(_zero_point)
        else:
            self.scale.data.abs_()
            self.scale.data.clamp_(min=self.eps.item())

        if self.fake_quant_enabled == 1:
            if self.drop_prob < 1.0:
                x_orig = X
            if self.ch_axis != -1:
                if self.use_grad_scaling:
                    grad_factor = 1.0 / (X.numel() / X.shape[self.ch_axis] * self.quant_max) ** 0.5
                else:
                    grad_factor = 1.0
                X = fake_quantize_learnable_per_channel_affine_training(
                    X, self.scale, self.zero_point.data.int(), self.ch_axis,
                    self.quant_min, self.quant_max, grad_factor)
            else:
                if self.use_grad_scaling:
                    grad_factor = 1.0 / (X.numel() * self.quant_max) ** 0.5
                else:
                    grad_factor = 1.0
                X = fake_quantize_learnable_per_tensor_affine_training(
                    X, self.scale, self.zero_point.item(), self.quant_min, self.quant_max, grad_factor)
            if self.drop_prob < 1.0:
                x_prob = torch.where(torch.rand_like(X) < self.drop_prob, X, x_orig)
                return x_prob
        return X


class LSQPlusFakeQuantize(QuantizeBase):

    def __init__(self, observer, bit=8, symmetric=False, ch_axis=-1, use_grad_scaling=True):
        super().__init__(observer, bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        self.scale = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float))
        self.zero_point = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float))
        self.register_buffer('eps', torch.tensor([torch.finfo(torch.float32).eps]))
        self.use_grad_scaling = use_grad_scaling
        self.drop_prob = 1.0

    def forward(self, X):
        if self.observer_enabled == 1:
            self.observer(X.detach())
            _scale, _zero_point = self.observer.calculate_qparams(self.observer.min_val, self.observer.max_val)
            _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
            if self.ch_axis != -1:
                self.scale.data = torch.ones_like(_scale)
                self.zero_point.data = torch.zeros_like(_zero_point.float())
            self.scale.data.copy_(_scale)
            self.zero_point.data.copy_(_zero_point.float())
        else:
            self.scale.data.abs_()
            self.scale.data.clamp_(min=self.eps.item())
            self.zero_point.data.clamp_(self.quant_min, self.quant_max).float()

        if self.fake_quant_enabled == 1:
            if self.drop_prob < 1.0:
                x_orig = X
            if self.ch_axis != -1:
                if self.use_grad_scaling:
                    grad_factor = 1.0 / (X.numel() / X.shape[self.ch_axis] * self.quant_max) ** 0.5
                else:
                    grad_factor = 1.0
                X = fake_quantize_learnableplus_per_channel_affine_training(
                    X, self.scale, self.zero_point, self.ch_axis,
                    self.quant_min, self.quant_max, grad_factor)
            else:
                if self.use_grad_scaling:
                    grad_factor = 1.0 / (X.numel() * self.quant_max) ** 0.5
                else:
                    grad_factor = 1.0
                X = fake_quantize_learnableplus_per_tensor_affine_training(
                    X, self.scale, self.zero_point, self.quant_min, self.quant_max, grad_factor)
            if self.drop_prob < 1.0:
                x_prob = torch.where(torch.rand_like(X) < self.drop_prob, X, x_orig)
                return x_prob
        return X


class AdaRoundFakeQuantize(QuantizeBase):
    """
    self.adaround=True: turn on up or down forward
    self.adaround=False: turn on round-to-nearest forward
    based on the FixedFakeQuantize
    """

    def __init__(self, observer, bit=8, symmetric=False, ch_axis=-1):
        super().__init__(observer, bit=bit, symmetric=symmetric, ch_axis=ch_axis)
        self.register_buffer('scale', torch.tensor([1.0], dtype=torch.float))
        self.register_buffer('zero_point', torch.tensor([0], dtype=torch.int))
        self.adaround = False
        self.gamma, self.zeta = -0.1, 1.1

    def init(self, weight_tensor: torch.Tensor, round_mode):
        self.adaround = True
        self.round_mode = round_mode
        self.init_alpha(x=weight_tensor.data.clone().detach())

    def init_alpha(self, x: torch.Tensor):
        if self.ch_axis != -1:
            new_shape = [1] * len(x.shape)
            new_shape[self.ch_axis] = x.shape[self.ch_axis]
            scale = self.scale.data.reshape(new_shape)
        else:
            scale = self.scale.data
        x_floor = torch.floor(x / scale)
        if self.round_mode == 'learned_hard_sigmoid':
            rest = (x / scale) - x_floor  # rest of rounding [0, 1)
            alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)  # => sigmoid(alpha) = rest
            self.alpha = torch.nn.Parameter(alpha)
        else:
            raise NotImplementedError

    def rectified_sigmoid(self):
        """generate rounding mask.
        """
        return ((self.zeta - self.gamma) * torch.sigmoid(self.alpha) + self.gamma).clamp(0, 1)

    def adaround_forward(self, X, hard_value=False):
        if self.ch_axis != -1:
            new_shape = [1] * len(X.shape)
            new_shape[self.ch_axis] = X.shape[self.ch_axis]
            scale = self.scale.data.reshape(new_shape)
            zero_point = self.zero_point.data.int().reshape(new_shape)
        else:
            scale = self.scale.item()
            zero_point = self.zero_point.item()
        X = torch.floor(X / scale)
        if hard_value:
            X += (self.alpha >= 0).float()
        else:
            X += self.rectified_sigmoid()
        X += zero_point
        X = torch.clamp(X, self.quant_min, self.quant_max)
        X = (X - zero_point) * scale
        return X

    def get_hard_value(self, X):
        X = self.adaround_forward(X, hard_value=True)
        return X

    def forward(self, X):
        if self.observer_enabled == 1:
            self.observer(X.detach())
            _scale, _zero_point = self.observer.calculate_qparams(self.observer.min_val, self.observer.max_val)
            _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
            if self.scale.shape != _scale.shape:
                self.scale.resize_(_scale.shape)
                self.zero_point.resize_(_zero_point.shape)
            self.scale.copy_(_scale)
            self.zero_point.copy_(_zero_point)

        if self.fake_quant_enabled == 1:
            if not self.adaround:
                if self.ch_axis != -1:
                    X = fake_quantize_per_channel_affine(
                        X, self.scale.data, self.zero_point.data.int(), self.ch_axis,
                        self.quant_min, self.quant_max)
                else:
                    X = fake_quantize_per_tensor_affine(
                        X, self.scale.item(), self.zero_point.item(),
                        self.quant_min, self.quant_max)
            else:
                if not hasattr(self, 'alpha'):
                    raise NotImplementedError
                if self.round_mode == 'learned_hard_sigmoid':
                    X = self.adaround_forward(X)
                else:
                    raise NotImplementedError
        return X

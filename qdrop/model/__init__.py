import torch.nn as nn
import torch
from .resnet import BasicBlock, Bottleneck, resnet18, resnet50  # noqa: F401
from .regnet import ResBottleneckBlock, regnetx_600m, regnetx_3200m  # noqa: F401
from .mobilenetv2 import InvertedResidual, mobilenetv2  # noqa: F401
from .mnasnet import _InvertedResidual, mnasnet  # noqa: F401
from qdrop.quantization.quantized_module import QuantizedLayer, QuantizedBlock, Quantizer   # noqa: F401


class QuantBasicBlock(QuantizedBlock):
    """
    Implementation of Quantized BasicBlock used in ResNet-18 and ResNet-34.
    """
    def __init__(self, org_module: BasicBlock, w_qconfig, a_qconfig, qoutput=True):
        super().__init__()
        self.qoutput = qoutput
        self.conv1_relu = QuantizedLayer(org_module.conv1, org_module.relu1, w_qconfig, a_qconfig)
        self.conv2 = QuantizedLayer(org_module.conv2, None, w_qconfig, a_qconfig, qoutput=False)
        if org_module.downsample is None:
            self.downsample = None
        else:
            self.downsample = QuantizedLayer(org_module.downsample[0], None, w_qconfig, a_qconfig, qoutput=False)
        self.activation = org_module.relu2
        if self.qoutput:
            self.block_post_act_fake_quantize = Quantizer(None, a_qconfig)

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1_relu(x)
        out = self.conv2(out)
        out += residual
        out = self.activation(out)
        if self.qoutput:
            out = self.block_post_act_fake_quantize(out)
        return out


class QuantBottleneck(QuantizedBlock):
    """
    Implementation of Quantized Bottleneck Block used in ResNet-50, -101 and -152.
    """
    def __init__(self, org_module: Bottleneck, w_qconfig, a_qconfig, qoutput=True):
        super().__init__()
        self.qoutput = qoutput
        self.conv1_relu = QuantizedLayer(org_module.conv1, org_module.relu1, w_qconfig, a_qconfig)
        self.conv2_relu = QuantizedLayer(org_module.conv2, org_module.relu2, w_qconfig, a_qconfig)
        self.conv3 = QuantizedLayer(org_module.conv3, None, w_qconfig, a_qconfig, qoutput=False)

        if org_module.downsample is None:
            self.downsample = None
        else:
            self.downsample = QuantizedLayer(org_module.downsample[0], None, w_qconfig, a_qconfig, qoutput=False)
        self.activation = org_module.relu3
        if self.qoutput:
            self.block_post_act_fake_quantize = Quantizer(None, a_qconfig)

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1_relu(x)
        out = self.conv2_relu(out)
        out = self.conv3(out)
        out += residual
        out = self.activation(out)
        if self.qoutput:
            out = self.block_post_act_fake_quantize(out)
        return out


class QuantResBottleneckBlock(QuantizedBlock):
    """
    Implementation of Quantized Bottleneck Blockused in RegNetX (no SE module).
    """
    def __init__(self, org_module: ResBottleneckBlock, w_qconfig, a_qconfig, qoutput=True):
        super().__init__()
        self.qoutput = qoutput
        self.conv1_relu = QuantizedLayer(org_module.f.a, org_module.f.a_relu, w_qconfig, a_qconfig)
        self.conv2_relu = QuantizedLayer(org_module.f.b, org_module.f.b_relu, w_qconfig, a_qconfig)
        self.conv3 = QuantizedLayer(org_module.f.c, None, w_qconfig, a_qconfig, qoutput=False)
        if org_module.proj_block:
            self.downsample = QuantizedLayer(org_module.proj, None, w_qconfig, a_qconfig, qoutput=False)
        else:
            self.downsample = None
        self.activation = org_module.relu
        if self.qoutput:
            self.block_post_act_fake_quantize = Quantizer(None, a_qconfig)

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1_relu(x)
        out = self.conv2_relu(out)
        out = self.conv3(out)
        out += residual
        out = self.activation(out)
        if self.qoutput:
            out = self.block_post_act_fake_quantize(out)
        return out


class QuantInvertedResidual(QuantizedBlock):
    """
    Implementation of Quantized Inverted Residual Block used in MobileNetV2.
    Inverted Residual does not have activation function.
    """
    def __init__(self, org_module: InvertedResidual, w_qconfig, a_qconfig, qoutput=True):
        super().__init__()
        self.qoutput = qoutput
        self.use_res_connect = org_module.use_res_connect
        if org_module.expand_ratio == 1:
            self.conv = nn.Sequential(
                QuantizedLayer(org_module.conv[0], org_module.conv[2], w_qconfig, a_qconfig),
                QuantizedLayer(org_module.conv[3], None, w_qconfig, a_qconfig, qoutput=False),
            )
        else:
            self.conv = nn.Sequential(
                QuantizedLayer(org_module.conv[0], org_module.conv[2], w_qconfig, a_qconfig),
                QuantizedLayer(org_module.conv[3], org_module.conv[5], w_qconfig, a_qconfig),
                QuantizedLayer(org_module.conv[6], None, w_qconfig, a_qconfig, qoutput=False),
            )
        if self.qoutput:
            self.block_post_act_fake_quantize = Quantizer(None, a_qconfig)

    def forward(self, x):
        if self.use_res_connect:
            out = x + self.conv(x)
        else:
            out = self.conv(x)
        if self.qoutput:
            out = self.block_post_act_fake_quantize(out)
        return out


class _QuantInvertedResidual(QuantizedBlock):
    # mnasnet
    def __init__(self, org_module: _InvertedResidual, w_qconfig, a_qconfig, qoutput=True):
        super().__init__()
        self.qoutput = qoutput
        self.apply_residual = org_module.apply_residual
        self.conv = nn.Sequential(
            QuantizedLayer(org_module.layers[0], org_module.layers[2], w_qconfig, a_qconfig),
            QuantizedLayer(org_module.layers[3], org_module.layers[5], w_qconfig, a_qconfig),
            QuantizedLayer(org_module.layers[6], None, w_qconfig, a_qconfig, qoutput=False),
        )
        if self.qoutput:
            self.block_post_act_fake_quantize = Quantizer(None, a_qconfig)

    def forward(self, x):
        if self.apply_residual:
            out = x + self.conv(x)
        else:
            out = self.conv(x)
        if self.qoutput:
            out = self.block_post_act_fake_quantize(out)
        return out


specials = {
    BasicBlock: QuantBasicBlock,
    Bottleneck: QuantBottleneck,
    ResBottleneckBlock: QuantResBottleneckBlock,
    InvertedResidual: QuantInvertedResidual,
    _InvertedResidual: _QuantInvertedResidual,
}


def load_model(config):
    config['kwargs'] = config.get('kwargs', dict())
    model = eval(config['type'])(**config['kwargs'])
    checkpoint = torch.load(config.path, map_location='cpu')
    if config.type == 'mobilenetv2':
        checkpoint = checkpoint['model']
    model.load_state_dict(checkpoint)
    return model

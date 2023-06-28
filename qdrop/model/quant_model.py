import torch.nn as nn
from mmdet.models.backbones.resnet import BasicBlock, Bottleneck
from mmdet.models.backbones.mobilenet_v2 import InvertedResidual
from qdrop.quantization.quantized_module import QuantizedLayer, QuantizedBlock, Quantizer   # noqa: F401


class QuantBasicBlock(QuantizedBlock):
    """
    Implementation of Quantized BasicBlock used in ResNet-18 and ResNet-34.
    """
    def __init__(self, org_module: BasicBlock, w_qconfig, a_qconfig, qoutput=True):
        super().__init__()
        self.qoutput = qoutput
        self.conv1_relu = QuantizedLayer(org_module.conv1, nn.ReLU(), w_qconfig, a_qconfig)
        self.conv2 = QuantizedLayer(org_module.conv2, None, w_qconfig, a_qconfig, qoutput=False)
        if org_module.downsample is None:
            self.downsample = None
        else:
            self.downsample = QuantizedLayer(org_module.downsample[0], None, w_qconfig, a_qconfig, qoutput=False)
        self.activation = nn.ReLU()
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
        self.conv1_relu = QuantizedLayer(org_module.conv1, nn.ReLU(), w_qconfig, a_qconfig)
        self.conv2_relu = QuantizedLayer(org_module.conv2, nn.ReLU(), w_qconfig, a_qconfig)
        self.conv3 = QuantizedLayer(org_module.conv3, None, w_qconfig, a_qconfig, qoutput=False)

        if org_module.downsample is None:
            self.downsample = None
        else:
            self.downsample = QuantizedLayer(org_module.downsample[0], None, w_qconfig, a_qconfig, qoutput=False)
        self.activation = nn.ReLU()
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
        self.with_res_shortcut = org_module.with_res_shortcut
        if org_module.with_expand_conv:
            self.conv = nn.Sequential(
                QuantizedLayer(org_module.expand_conv.conv, org_module.expand_conv.activate, w_qconfig, a_qconfig),
                QuantizedLayer(org_module.depthwise_conv.conv, org_module.depthwise_conv.activate, w_qconfig, a_qconfig),
                QuantizedLayer(org_module.linear_conv.conv, None, w_qconfig, a_qconfig, qoutput=False),
            )
        else:
            self.conv = nn.Sequential(
                QuantizedLayer(org_module.depthwise_conv.conv, org_module.depthwise_conv.activate, w_qconfig, a_qconfig),
                QuantizedLayer(org_module.linear_conv.conv, None, w_qconfig, a_qconfig, qoutput=False),
            )
        if self.qoutput:
            self.block_post_act_fake_quantize = Quantizer(None, a_qconfig)

    def forward(self, x):
        if self.with_res_shortcut:
            out = x + self.conv(x)
        else:
            out = self.conv(x)
        if self.qoutput:
            out = self.block_post_act_fake_quantize(out)
        return out


specials = {
    BasicBlock: QuantBasicBlock,
    Bottleneck: QuantBottleneck,
    InvertedResidual: QuantInvertedResidual,
}
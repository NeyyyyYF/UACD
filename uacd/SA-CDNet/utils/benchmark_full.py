import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class MCDropout2d(nn.Dropout2d):
    def __init__(self, p=0.5, inplace=False):
        super(MCDropout2d, self).__init__(p, inplace)

    def forward(self, x):
        return F.dropout2d(x, self.p, True, self.inplace)


class Space_Attention(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4):
        super(Space_Attention, self).__init__()
        self.SA = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.BatchNorm2d(in_channels // reduction, momentum=0.95),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels // reduction, out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.SA(x)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None: identity = self.downsample(x)
        out += identity
        return self.relu(out)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels_high, in_channels_low, out_channels):
        super(_DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels_high, in_channels_high, kernel_size=2, stride=2)
        self.decode = nn.Sequential(
            conv3x3(in_channels_high + in_channels_low, out_channels),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            conv3x3(out_channels, out_channels),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )

    def forward(self, x, low_feat):
        x = self.up(x)
        x = torch.cat((x, low_feat), dim=1)
        return self.decode(x)


class conv_block_nested(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        identity = x
        x = self.activation(self.bn1(x))
        x = self.bn2(self.conv2(x))
        return self.activation(x + identity)


class up(nn.Module):
    def __init__(self, in_ch):
        super(up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)

    def forward(self, x):
        return self.up(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_ = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmod(avg + max_)


class Mix(nn.Module):
    def __init__(self, m=-0.8):
        super(Mix, self).__init__()
        self.w = nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        self.mix = nn.Sigmoid()

    def forward(self, fea1, fea2):
        factor = self.mix(self.w)
        # Ensure that the dimensions match. If the dimensions are inconsistent due to simulation errors, force interpolation alignment
        if fea1.shape != fea2.shape:
            fea2 = F.interpolate(fea2, size=fea1.shape[-2:], mode='bilinear', align_corners=True)
        return fea1 * factor + fea2 * (1 - factor)

class MockFastSAM(nn.Module):
    """Simulate the output of FastSAM without loading the real weights, only ensuring the correct shape of the Tensor"""

    def __init__(self):
        super().__init__()

    def forward(self, x, **kwargs):
        B = x.shape[0]
        H, W = x.shape[-2:]
        return [
            torch.randn(B, 320, H // 8, W // 8).to(x.device),
            torch.randn(B, 640, H // 16, W // 16).to(x.device),
            torch.randn(B, 640, H // 32, W // 32).to(x.device),
            torch.randn(B, 160, H // 4, W // 4).to(x.device)
        ]


class SAM_CD_Base(nn.Module):
    """The base class containing the initialization of the public layer"""

    def __init__(self):
        super(SAM_CD_Base, self).__init__()
        self.model = MockFastSAM() 

        self.Adapter32 = nn.Sequential(nn.Conv2d(640, 160, 1, bias=False), nn.BatchNorm2d(160), nn.ReLU())
        self.Adapter16 = nn.Sequential(nn.Conv2d(640, 160, 1, bias=False), nn.BatchNorm2d(160), nn.ReLU())
        self.Adapter8 = nn.Sequential(nn.Conv2d(320, 80, 1, bias=False), nn.BatchNorm2d(80), nn.ReLU())
        self.Adapter4 = nn.Sequential(nn.Conv2d(160, 40, 1, bias=False), nn.BatchNorm2d(40), nn.ReLU())

        self.Dec2 = _DecoderBlock(160, 160, 80)
        self.Dec1 = _DecoderBlock(80, 80, 40)
        self.Dec0 = _DecoderBlock(40, 40, 64)
        self.segmenter = nn.Conv2d(64, 8, 1)

        filters = [32, 64, 128, 256]
        self.pool = nn.MaxPool2d(2, 2)
        self.Up1_0 = up(filters[1]);
        self.Up2_0 = up(filters[2]);
        self.Up3_0 = up(filters[3])
        self.Up1_1 = up(filters[1]);
        self.Up2_1 = up(filters[2]);
        self.Up3_1 = up(filters[3])
        self.Up1_2 = up(filters[1]);
        self.Up2_2 = up(filters[2]);
        self.Up1_3 = up(filters[1])

        self.conv0_0 = conv_block_nested(40, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(80, filters[1], filters[1])
        self.conv2_0 = conv_block_nested(160, filters[2], filters[2])
        self.conv3_0 = conv_block_nested(160, filters[3], filters[3])

        self.conv0_1 = conv_block_nested(filters[0] * 2 + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] * 2 + filters[2], filters[1], filters[1])
        self.conv2_1 = conv_block_nested(filters[2] * 2 + filters[3], filters[2], filters[2])
        self.conv3_1 = conv_block_nested(filters[3] * 2, filters[3], filters[3])

        self.conv0_2 = conv_block_nested(filters[0] * 3 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1] * 3 + filters[2], filters[1], filters[1])
        self.conv2_2 = conv_block_nested(filters[2] * 3 + filters[3], filters[2], filters[2])

        self.conv0_3 = conv_block_nested(filters[0] * 4 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1] * 4 + filters[2], filters[1], filters[1])
        self.conv0_4 = conv_block_nested(filters[0] * 5 + filters[1], filters[0], filters[0])

        self.ca = ChannelAttention(filters[0] * 4)
        self.ca1 = ChannelAttention(filters[0], ratio=4)
        self.conv_final = nn.Conv2d(filters[0] * 4, 1, 1)
        self.mix = Mix()

        self.SA = Space_Attention(16, 16, 4)
        self._make_resCD()
        self.headC = nn.Sequential(nn.Conv2d(128, 16, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU())
        self.segmenterC = nn.Conv2d(16, 1, 1)

    def _make_resCD(self):
        layers = [ResBlock(128, 128) for _ in range(6)]
        self.resCD = nn.Sequential(MCDropout2d(0.1), *layers)

    def run_encoder(self, x):
        return self.model(x)

    def common_forward(self, x1, x2):
        """Perform the calculation of the first 90% shared by the two cases"""
        input_shape = x1.shape[-2:]
        featsA = self.run_encoder(x1)
        featsB = self.run_encoder(x2)

        # Adapters
        featA_s4 = self.Adapter4(featsA[3]);
        featB_s4 = self.Adapter4(featsB[3])
        featA_s8 = self.Adapter8(featsA[0]);
        featB_s8 = self.Adapter8(featsB[0])
        featA_s16 = self.Adapter16(featsA[1]);
        featB_s16 = self.Adapter16(featsB[1])
        featA_s32 = self.Adapter32(featsA[2]);
        featB_s32 = self.Adapter32(featsB[2])

        # Decoders
        decA_2 = self.Dec2(featA_s32, featA_s16)
        decA_1 = self.Dec1(decA_2, featA_s8)
        decA_0 = self.Dec0(decA_1, featA_s4)
        outA = self.segmenter(decA_0)

        decB_2 = self.Dec2(featB_s32, featB_s16)
        decB_1 = self.Dec1(decB_2, featB_s8)
        decB_0 = self.Dec0(decB_1, featB_s4)
        outB = self.segmenter(decB_0)

        # UNet++ Nested Logic
        x0_0A = self.conv0_0(featA_s4);
        x0_0B = self.conv0_0(featB_s4)

        h, w = input_shape
        feat_shape = (h // 4, w // 4)
        out = torch.randn(x1.shape[0], 128, *feat_shape).to(x1.device)
        out = self.conv_final(out)

        A = self.SA(torch.cat([outA, outB], dim=1))
        featC = torch.cat([decA_0, decB_0], 1)

        return out, A, featC


class Case1_Model(SAM_CD_Base):
    def forward(self, x1, x2):
        out, A, featC = self.common_forward(x1, x2)

        out_samples = []
        for _ in range(5):
            out_samples.append(self.segmenterC(self.headC(self.resCD(featC)) * A))
        outs = torch.stack(out_samples, dim=0)

        outC = self.segmenterC(self.headC(self.resCD(featC)) * A)

        outF = self.mix(out, outC)
        return outC


class Case2_Model(SAM_CD_Base):
    def forward(self, x1, x2):
        out, A, featC = self.common_forward(x1, x2)

        out_samples = []
        for _ in range(5):
            out_samples.append(self.segmenterC(self.headC(self.resCD(featC)) * A))
        outs = torch.stack(out_samples, dim=0)

        outC = torch.mean(outs, dim=0)

        outF = self.mix(out, outC)
        return outC

def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Perform the calculation of the first 90% shared by the two cases: {device}")

    # Initialize the model
    model1 = Case1_Model().to(device).eval()
    model2 = Case2_Model().to(device).eval()

    # Initialize the input
    B, C, H, W = 2, 3, 256, 256  # Batch Size 2
    x1 = torch.randn(B, C, H, W).to(device)
    x2 = torch.randn(B, C, H, W).to(device)

    iterations = 50 

    print(f"\n Begins the test (Batch Size={B}, Iterations={iterations})...")
    print("This might take a few seconds...")

    for _ in range(5): _ = model1(x1, x2)

    if device.type == 'cuda': torch.cuda.synchronize()
    start = time.time()
    for i in range(iterations):
        _ = model1(x1, x2)
    if device.type == 'cuda': torch.cuda.synchronize()
    end = time.time()
    avg_time1 = (end - start) / iterations

    for _ in range(5): _ = model2(x1, x2)

    if device.type == 'cuda': torch.cuda.synchronize()
    start = time.time()
    for i in range(iterations):
        _ = model2(x1, x2)
    if device.type == 'cuda': torch.cuda.synchronize()
    end = time.time()
    avg_time2 = (end - start) / iterations

    print("\n" + "=" * 40)
    print("       Performance comparison results (each forward)")
    print("=" * 40)
    print(f"Case 1 (Additional reasoning): {avg_time1:.4f} s / step")
    print(f"Case 2 (calculated mean value): {avg_time2:.4f} s / step")
    print("-" * 40)

    diff = avg_time1 - avg_time2
    percent = (diff / avg_time1) * 100

    print(f"Case 2 Save time at every step: {diff:.4f} s")
    print(f"Case 2 Relative speed-up ratio: {percent:.2f}%")
    print("=" * 40)


if __name__ == '__main__':
    with torch.no_grad():
        run_benchmark()
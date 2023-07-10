import math

import torch
import torch.nn as nn


def make_divisible(x, divisor):
    # Returns nearest x divisible by divisor
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


def make_anchors(device, shapes=(64, 32, 16), strides=(8, 16, 32), grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []

    for shape, stride in zip(shapes, strides):
        x = torch.arange(end=shape, dtype=torch.float) + grid_cell_offset  # shift x
        y = torch.arange(end=shape, dtype=torch.float) + grid_cell_offset  # shift y
        y, x = torch.meshgrid(y, x, indexing='ij')

        anchor_points.append(torch.stack((x, y), -1).view(-1, 2))
        stride_tensor.append(torch.full((shape * shape, 1), stride, dtype=torch.float))

    return torch.cat(anchor_points).transpose(0, 1).to(device), torch.cat(stride_tensor).transpose(0, 1).to(device)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = torch.split(distance, 2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb

    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox

    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p



class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


# Integral module of Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
# https://ieeexplore.ieee.org/document/9792391
class DFL(nn.Module):
    def __init__(self, c1=16):
        super().__init__()

        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


class Detect(nn.Module):
    # YOLOv8 Detect head for detection models
    def __init__(self, stride, nc=80):  # detection layer
        super().__init__()

        ch = [64, 128, 256]

        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = stride  # strides computed during build
        self.inplace = True

        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], self.nc)  # channels

        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3),
                          Conv(c2, c2, 3),
                          nn.Conv2d(c2, 4 * self.reg_max, 1))
            for x in ch
        )

        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3),
                          Conv(c3, c3, 3),
                          nn.Conv2d(c3, self.nc, 1))
            for x in ch
        )

        self.bias_init()

    def forward(self, x):
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        return x

    def bias_init(self):
        for a, b, s in zip(self.cv2, self.cv3, self.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:self.nc] = math.log(5 / self.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True


# yolo v8
class ModelYOLO(nn.Module):
    def __init__(self, nc):
        super().__init__()

        # backbone
        self.cnv1 = Conv(c1=1, c2=16, k=3, s=2)
        self.cnv2 = Conv(c1=16, c2=32, k=3, s=2)
        self.cnf3 = C2f(c1=32, c2=32, n=1, shortcut=True)
        self.cnv4 = Conv(c1=32, c2=64, k=3, s=2)
        self.cnf5 = C2f(c1=64, c2=64, n=2, shortcut=True)
        self.cnv6 = Conv(c1=64, c2=128, k=3, s=2)
        self.cnf7 = C2f(c1=128, c2=128, n=2, shortcut=True)
        self.cnv8 = Conv(c1=128, c2=256, k=3, s=2)
        self.cnf9 = C2f(c1=256, c2=256, n=1, shortcut=True)
        self.spf10 = SPPF(c1=256, c2=256, k=5)

        # head
        self.ups11 = nn.modules.upsampling.Upsample(None, 2, 'nearest')
        self.cat12 = Concat(dimension=1)
        self.cnf13 = C2f(c1=384, c2=128, n=1, shortcut=False)

        self.ups14 = nn.modules.upsampling.Upsample(None, 2, 'nearest')
        self.cat15 = Concat(dimension=1)
        self.cnf16 = C2f(c1=192, c2=64, n=1, shortcut=False)

        self.cnv17 = Conv(c1=64, c2=64, k=3, s=2)
        self.cat18 = Concat(dimension=1)
        self.cnf19 = C2f(c1=192, c2=128, n=1, shortcut=False)

        self.cnv20 = Conv(c1=128, c2=128, k=3, s=2)
        self.cat21 = Concat(dimension=1)
        self.cnf22 = C2f(c1=384, c2=256, n=1, shortcut=False)

        # detect
        self.detect = Detect(stride=torch.tensor([8, 16, 32]), nc=nc)

    def forward(self, x):
        # backbone
        x1 = self.cnv1(x)
        x2 = self.cnv2(x1)
        x3 = self.cnf3(x2)
        x4 = self.cnv4(x3)
        x5 = self.cnf5(x4)
        x6 = self.cnv6(x5)
        x7 = self.cnf7(x6)
        x8 = self.cnv8(x7)
        x9 = self.cnf9(x8)
        x10 = self.spf10(x9)

        # head
        x11 = self.ups11(x10)
        x12 = self.cat12([x11, x7])
        x13 = self.cnf13(x12)

        x14 = self.ups14(x13)
        x15 = self.cat15([x14, x5])
        x16 = self.cnf16(x15)

        x17 = self.cnv17(x16)
        x18 = self.cat18([x17, x13])
        x19 = self.cnf19(x18)

        x20 = self.cnv20(x19)
        x21 = self.cat21([x20, x10])
        x22 = self.cnf22(x21)

        # detect
        x = self.detect([x16, x19, x22])

        return x

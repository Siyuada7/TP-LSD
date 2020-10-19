import torch
import torch.nn as nn
import torch.nn.functional as F
from .official import resnet34
from .DCNv2.dcn_v2 import DCN

task = {'center':1, 'dis':4, 'line':1}
class _sigmoid():
    def __init__(self):
        pass
    def __call__(self, x):
      y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
      return y

def prob_sigmoid(x):
    y = torch.clamp(x, min=1e-4, max=1 - 1e-4)
    return y

class double_conv(nn.Module):
    # (conv => BN => ReLU) * 2
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class outconv_dis(nn.Module):
    def __init__(self, in_ch, out_ch, activation=True):
        super(outconv_dis, self).__init__()
        self.activation = activation
        self.conv1 = double_conv(in_ch, in_ch)
        self.conv2 = nn.Conv2d(in_ch, out_ch, 1)
        if self.activation:
            self.sigmoid = _sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        if self.activation:
            x = self.sigmoid(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch, activation=True):
        super(outconv, self).__init__()
        self.activation = activation
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),)
        if self.activation:
            self.sigmoid = _sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        if self.activation:
            x = self.sigmoid(x)
        return x

class outDCNconv(nn.Module):
    def __init__(self, in_ch, out_ch, activation=True):
        super(outDCNconv, self).__init__()
        self.activation = activation
        self.conv1 = DCN(in_ch, in_ch, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=1)
        self.convh = nn.Conv2d(in_ch, in_ch, kernel_size=(3, 3), dilation=(2, 2), stride=1, padding=(2,2))
        self.convw = nn.Conv2d(in_ch, in_ch, kernel_size=(3, 3), dilation=(2, 2), stride=1, padding=(2,2))
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(in_ch)
        self.bn3 = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
        if self.activation:
            self.sigmoid = _sigmoid()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.convh(x)))
        x = self.relu(self.bn3(self.convw(x)))
        x = self.conv3(x)
        if self.activation:
            x = self.sigmoid(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_ch1, in_ch2, out_ch):
        super(DecoderBlock, self).__init__()
        self.conv1_1 = nn.Conv2d(in_ch1, out_ch // 2, kernel_size=1, bias=False)
        self.conv1_2 = nn.Conv2d(in_ch2, out_ch // 2, kernel_size=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(out_ch // 2)
        self.bn1_2 = nn.BatchNorm2d(out_ch // 2)

        self.conv_block = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, x2.size()[2:4], mode='bilinear', align_corners=True)
        x1 = self.conv1_1(x1)
        x1 = self.bn1_1(x1)
        x1 = self.relu(x1)
        x2 = self.conv1_2(x2)
        x2 = self.bn1_2(x2)
        x2 = self.relu(x2)
        x = torch.cat((x1, x2), dim=1)
        residual = x
        out = self.conv_block(x)
        out += residual
        out = self.relu(out)

        return out


class Res320(nn.Module):
    def __init__(self, task_dim=task):
        super(Res320, self).__init__()
        self.task_dim = task_dim
        self.resnet = resnet34()

        self.up1 = DecoderBlock(512, 256, 256)
        self.up2 = DecoderBlock(256, 128, 128)
        self.up3 = DecoderBlock(128, 64, 64)
        self.up4 = DecoderBlock(64, 64, 64)

        self.line_conv = nn.Conv2d(1, 1, 1)
        self.center_conv = nn.Conv2d(2, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

        self.head_center = outDCNconv(65, self.task_dim['center'])  # , activation=False)

        self.head_d_dcn = DCN(65, 64, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=1)
        self.head_dis = outconv_dis(64, self.task_dim['dis'], activation=False)
        self.head_line = outconv(64, self.task_dim['line'])

        self._init_weight()

    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.interpolate(x, [H, W], mode='bilinear', align_corners=True) + y

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        outs = []
        x, low_level_feature = self.resnet(x)
        x = self.up1(x, low_level_feature[3])
        x = self.up2(x, low_level_feature[2])
        x = self.up3(x, low_level_feature[1])
        share_feature = self.up4(x, low_level_feature[0])

        line = self.head_line(share_feature)
        line_cat = self.tanh(self.line_conv(line))
        x = torch.cat([share_feature, line_cat], dim=1)

        center = self.head_center(x)
        center_cat = torch.cat([line, center], dim=1)
        center_cat = self.tanh(self.center_conv(center_cat))

        x = torch.cat([share_feature, center_cat], dim=1)
        tmp_dis = self.head_d_dcn(x)
        dis = self.head_dis(tmp_dis)
        outs.append({'center': center, 'dis': dis, 'line': line,})

        return outs

class Res160(nn.Module):
    def __init__(self, task_dim=task, size=320):
        super(Res160, self).__init__()
        self.task_dim = task_dim
        self.size = size

        self.resnet = resnet34()

        self.up1 = DecoderBlock(512, 256, 256)
        self.up2 = DecoderBlock(256, 128, 128)
        self.up3 = DecoderBlock(128, 64, 64)

        self.line_conv = nn.Conv2d(1, 1, 1)
        self.center_conv = nn.Conv2d(2, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

        self.head_center = outDCNconv(65, self.task_dim['center'])  # , activation=False)

        self.head_d_dcn = DCN(65, 64, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=1)
        self.head_dis = outconv_dis(64, self.task_dim['dis'], activation=False)
        self.head_dis_1 = nn.ConvTranspose2d(4, 4, 2, stride=2)

        self.head_line = outconv(64, self.task_dim['line'])

        self._init_weight()

    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.interpolate(x, [H, W], mode='bilinear', align_corners=True) + y

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        outs = []

        x, low_level_feature = self.resnet(x)
        x = self.up1(x, low_level_feature[3])
        x = self.up2(x, low_level_feature[2])
        feature160 = self.up3(x, low_level_feature[1])

        line_use = self.head_line(feature160)
        line = prob_sigmoid(F.interpolate(line_use, size=[self.size, self.size], mode="bilinear"))

        line_cat = self.tanh(self.line_conv(line_use))
        x = torch.cat([feature160, line_cat], dim=1)

        center_use = self.head_center(x)
        center = prob_sigmoid(F.interpolate(center_use, size=[self.size, self.size], mode="bilinear"))
        center_cat = torch.cat([line_use, center_use], dim=1)
        center_cat = self.tanh(self.center_conv(center_cat))

        x = torch.cat([feature160, center_cat], dim=1)
        tmp_dis = self.head_d_dcn(x)
        dis_160 = self.head_dis(tmp_dis)
        dis = self.head_dis_1(dis_160)

        outs.append({'center': center, 'dis': dis, 'line': line})
        return outs

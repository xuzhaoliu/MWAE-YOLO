import torch
import torch.nn as nn
import torch.nn.functional as F 

class DepthwiseSeparableConv(nn.Module):
    # 深度可分离卷积
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, padding=padding, groups=in_ch, bias=bias)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class LightWaveletConv(nn.Module):
    # 轻量化小波卷积
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.num_branches = 3
        self.ch_per_branch = out_channels // self.num_branches
        if self.ch_per_branch * self.num_branches != out_channels:
            self.ch_per_branch = out_channels // self.num_branches
            last_ch = out_channels - self.ch_per_branch * (self.num_branches - 1)
        else:
            last_ch = self.ch_per_branch

        # 注册Haar滤波器
        self.register_buffer('haar_low', torch.tensor([[1, 1], [1, 1]], dtype=torch.float32)/2.0)
        self.register_buffer('haar_high_h', torch.tensor([[1, -1], [1, -1]], dtype=torch.float32)/2.0)
        self.register_buffer('haar_high_v', torch.tensor([[1, 1], [-1, -1]], dtype=torch.float32)/2.0)

        self.convs = nn.ModuleList([
            DepthwiseSeparableConv(in_channels, self.ch_per_branch, 3, 1)
            for _ in range(self.num_branches - 1)
        ])
        self.convs.append(DepthwiseSeparableConv(in_channels, last_ch, 3, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.branch_logits = nn.Parameter(torch.ones(self.num_branches))

    def forward(self, x):
        filters = [self.haar_low, self.haar_high_h, self.haar_high_v]
        branch_outs = []
        for i, filt in enumerate(filters):
            weight = filt.unsqueeze(0).unsqueeze(0).expand(x.shape[1], 1, 2, 2)
            out = F.conv2d(x, weight, stride=2, padding=0, groups=x.shape[1])
            out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
            out = self.convs[i](out)
            branch_outs.append(out)
        weights = F.softmax(self.branch_logits, dim=0)
        weighted = [w * b for w, b in zip(weights, branch_outs)]
        out = torch.cat(weighted, dim=1)
        out = self.bn(out)
        return self.relu(out)


class LightMultiDilatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates=[1, 2, 4]):
        super().__init__()
        self.dilation_rates = dilation_rates
        self.num_branches = len(dilation_rates)
        self.ch_per_branch = out_channels // self.num_branches
        if self.ch_per_branch * self.num_branches != out_channels:
            last_ch = out_channels - self.ch_per_branch * (self.num_branches - 1)
        else:
            last_ch = self.ch_per_branch

        self.branches = nn.ModuleList()
        for i, rate in enumerate(dilation_rates):
            ch = self.ch_per_branch if i < (self.num_branches - 1) else last_ch
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_channels, ch, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True)
            ))
        self.scales = nn.Parameter(torch.ones(self.num_branches))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x): 
        branch_outs = []
        for i, (rate, branch) in enumerate(zip(self.dilation_rates, self.branches)):
            out = branch(x)
            out = self.scales[i]*out
            branch_outs.append(out)
        out = torch.cat(branch_outs, dim=1)
        out = self.bn(out)
        return self.relu(out)


class ECA(nn.Module):
    # 轻量化通道注意力ECA
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).squeeze(-1)
        y = y.unsqueeze(1)
        y = self.conv(y)
        y = y.squeeze(1).unsqueeze(-1).unsqueeze(-1)
        return x * self.sigmoid(y)


class LightSpatialAttention(nn.Module):
    # 空间注意力
    def __init__(self, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = torch.mean(x, dim=1, keepdim=True)
        y = self.conv(y)
        return x * self.sigmoid(y)


class MWAF(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates=[1, 2, 4]):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.wavelet_conv = LightWaveletConv(out_channels, out_channels)
        self.multi_dilated_conv = LightMultiDilatedConv(out_channels, out_channels, dilation_rates)
        self.channel_attn = ECA(out_channels, kernel_size=3)
        self.spatial_attn = LightSpatialAttention(kernel_size=3)
        self.skip_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.gate_conv = nn.Conv2d(out_channels, out_channels, 1, bias=True)

    def forward(self, x):
        identity = self.skip_conv(x)
        x = self.input_proj(x)
        wavelet_out = self.wavelet_conv(x)
        dilated_out = self.multi_dilated_conv(x)
        fused = wavelet_out + dilated_out
        attn_c = self.channel_attn(fused)
        attn = self.spatial_attn(attn_c)
        gate = torch.sigmoid(self.gate_conv(attn))
        out = attn * gate + identity * (1 - gate)
        return out


if __name__ == '__main__':
    input = torch.randn(2, 64, 32, 32)
    model = MWAF(in_channels=64, out_channels=64, dilation_rates=[1, 2, 4])
    output = model(input)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"输入形状: {input.shape} → 输出形状: {output.shape}")
    print(f"总参数量: {total_params:,}")

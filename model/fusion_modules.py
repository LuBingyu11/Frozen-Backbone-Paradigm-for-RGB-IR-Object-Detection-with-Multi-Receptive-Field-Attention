import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from functools import reduce

#################################################MRF##############################################################



class Channel_Attention(nn.Module):
    def __init__(self, channel):
        super(Channel_Attention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv3x3 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, groups=channel, bias=False),
            nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0, bias=False)
        )
        self.conv5x5 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=5, stride=1, padding=2, groups=channel, bias=False),
            nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0, bias=False)
        )

        self.combine = nn.Conv2d(channel, channel, kernel_size=1)

        # 可学习权重 (初始化为 1/3)
        self.raw_weights = nn.Parameter(torch.tensor([1/3, 1/3, 1/3], dtype=torch.float32), requires_grad=True)

        # 可学习权重 (初始化为 1/2)
        # self.raw_weights = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32), requires_grad=True)

    def forward(self, x):
        return self._forward_se(x)

    def _forward_se(self, x):
        b, c, _, _ = x.size()

        # 多尺度卷积输出
        f1 = self.conv1x1(x)  # small
        f2 = self.conv3x3(x)  # medium
        f3 = self.conv5x5(x)  # large

        # 可学习权重 + softmax
        weights = F.softmax(self.raw_weights, dim=0)
        w1, w2, w3 = weights[0], weights[1], weights[2]
        # weights = F.softmax(self.raw_weights, dim=0)
        # w1, w2 = weights[0], weights[1]
        # print("w1:",w1)
        # print("w2:",w2)
        # print("w3:",w3)
        # 融合
        fused = w1 * f1 + w2 * f2 +w3 * f3

        # 通道注意力
        x_avg = self.avg_pool(fused)
        x_max = self.max_pool(fused)

        y_avg = torch.sigmoid(x_avg)
        y_max = torch.sigmoid(x_max)
        y = y_avg + y_max

        feat_attended = x * y

        out = self.combine(feat_attended) + x
        # self.last_channel_weights = F.adaptive_avg_pool2d(out, 1).view(b, c)  # ✅ shape = [B, C]
        return out

class Fused_module(torch.nn.Module):

      def __init__(self):
          super().__init__()

          self.sigmoid = torch.nn.Sigmoid()

      def forward(self, rgb_feats, ir_feats):
          # get shape

          B, C, H, W = rgb_feats.shape

          # channel concatenation (x_cat -> B,2C,H,W)
          x_cat = torch.cat((rgb_feats, ir_feats), axis=1)

          # create w_avg attention map (w_avg -> B,1,H,W)
          cap = torch.mean(x_cat, dim=1)
          w_avg = self.sigmoid(cap)
          w_avg = w_avg.unsqueeze(1)

          # create w_max attention maps (w_max -> B,1,H,W)
          cmp = torch.max(x_cat, dim=1)[0]
          w_max = self.sigmoid(cmp)
          w_max = w_max.unsqueeze(1)

          # weighted feature map (x_cat_w -> B,2C,H,W)
          x_cat_w = x_cat * w_avg * w_max

          # split weighted feature map (x_ir_w, x_rgb_w -> B,C,H,W)
          x_rgb_w = x_cat_w[:, :C, :, :]
          x_ir_w = x_cat_w[:, C:, :, :]

          # fuse feature maps (x_fused -> B,H,W,C)
          x_fused = (x_ir_w + x_rgb_w) / 2

          return x_fused

#
class MRF(torch.nn.Module):

  def __init__(self, in_channel=128):
    super().__init__()
    self.fm = Fused_module()
    self.ca_rgb= Channel_Attention(channel=in_channel)
    self.ca_ir = Channel_Attention(channel=in_channel)
  def forward(self, rgb_input, ir_input):

    rgb_feats=self.ca_rgb(rgb_input)
    ir_feats=self.ca_ir(ir_input)

    fused_feats=self.fm(rgb_feats, ir_feats)
    return fused_feats

################################################# CBAM ############################################################
class CBAMLayer(nn.Module):

    def __init__(self, channel, reduction=16):
        super(CBAMLayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
        )
        self.combine = nn.Conv2d(channel, int(channel/2), kernel_size=1)
        self.assemble = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        x = self._forward_se(x)
        x = self._forward_spatial(x)
        return x

    def _forward_se(self, x):
        # Channel attention module (SE with max-pool and average-pool)
        b, c, _, _ = x.size()
        x_avg = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        x_max = self.fc(self.max_pool(x).view(b, c)).view(b, c, 1, 1)

        y = torch.sigmoid(x_avg + x_max)

        # plot_y = y[0,:,0,0].cpu().numpy()
        # plot_y = (plot_y - np.nanmin(plot_y)) / (np.nanmax(plot_y) - np.nanmin(plot_y))
        # plot_x = np.arange(256)
        # fig, ax = plt.subplots()
        # markerline1, stemlines, _ = plt.stem(plot_x[:128], plot_y[:128], 'k')
        # plt.setp(markerline1, 'color', 'k', 'markerfacecolor', 'k', 'mec', 'k')
        # markerline2, stemlines, _ = plt.stem(plot_x[128:], plot_y[128:], 'crimson')
        # plt.setp(markerline2, 'color', 'crimson', 'markerfacecolor', 'crimson', 'mec', 'crimson')
        # plt.savefig('cam/{i}.png'.format(i=x.shape[-2]))

        return self.combine(x * y)

    def _forward_spatial(self, x):
        # Spatial attention module
        x_avg = torch.mean(x, 1, True)
        x_max, _ = torch.max(x, 1, True)
        y = torch.cat((x_avg, x_max), 1)
        y = torch.sigmoid(self.assemble(y))

        return x * y




################################### ECA Attention #####################################

class channel_attention_block(nn.Module):

    """ Implements a Channel Attention Block """

    def __init__(self,in_channels):

        super(channel_attention_block, self).__init__()
        
        adaptive_k = self.channel_att_kernel_calc(in_channels)
        

        self.pool_types = ["max","avg"]

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.conv = nn.Conv1d(1,1,kernel_size=adaptive_k,padding=(adaptive_k-1)//2,bias=False)
        
        self.combine = nn.Conv2d(in_channels, int(in_channels/2), kernel_size=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self,x):

        channel_att_sum = None

        for pool_type in self.pool_types:

            if pool_type == "avg":

                avg_pool = self.avg_pool(x)
                channel_att_raw = self.conv(avg_pool.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

            elif pool_type == "max":

                max_pool = self.max_pool(x)
                channel_att_raw = self.conv(max_pool.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

            if channel_att_sum is None:

                channel_att_sum = channel_att_raw

            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        gate = self.sigmoid(channel_att_sum).expand_as(x)

        return self.combine(x*gate)
    
    
    def channel_att_kernel_calc(self,num_channels,gamma=2,b=1):
        b=1
        gamma = 2
        t = int(abs((math.log(num_channels,2)+b)/gamma))
        k = t if t%2 else t+1
        
        return k


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )



class spatial_attention_block(nn.Module):

    """ Implements a Spatial Attention Block """

    def __init__(self):

        super(spatial_attention_block,self).__init__()

        kernel_size = 7

        self.compress = ChannelPool()

        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self,x):

        x_compress = self.compress(x)

        x_out = self.spatial(x_compress)

        gate = self.sigmoid(x_out)

        return x*gate

class attention_block(nn.Module):

    def __init__(self,in_channels):

        super(attention_block,self).__init__()

        self.channel_attention_block = channel_attention_block(in_channels=in_channels)

        self.spatial_attention_block = spatial_attention_block()

    def forward(self,x):

        x_out = self.channel_attention_block(x)
        x_out_1 = self.spatial_attention_block(x_out)

        return x_out_1
    
################################################# Shuffle Attention ############################################################

class shuffle_attention_block(nn.Module):
    """Constructs a Channel Spatial Group module.
    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, in_channels, groups=16):
        super(shuffle_attention_block, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = nn.parameter.Parameter(torch.zeros(1, in_channels // (2 * groups), 1, 1))
        self.cbias =  nn.parameter.Parameter(torch.ones(1, in_channels // (2 * groups), 1, 1))
        self.sweight =  nn.parameter.Parameter(torch.zeros(1, in_channels // (2 * groups), 1, 1))
        self.sbias =  nn.parameter.Parameter(torch.ones(1, in_channels // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(in_channels // (2 * groups), in_channels // (2 * groups))
        
        self.combine = nn.Conv2d(in_channels, int(in_channels/2), kernel_size=1)

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        
        
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)

        x_0, x_1 = x.chunk(2, dim=1)

        # channel attention
        xn = self.avg_pool(x_0)

        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)
        
        # Reduce the Channels
        out = self.combine(out)
        
        return out


################################### sknet #####################################
class SKNet1(nn.Module):
    def __init__(self, in_channels=128, out_channels=128, stride=1, M=2, r=16, L=32):


        super(SKNet1, self).__init__()
        self.in_channels = in_channels
        d = max(in_channels // r, L)
        self.M = M
        self.out_channels = out_channels
        self.conv = nn.ModuleList()
        for i in range(M):
            self.conv.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride, padding=1 + i, dilation=1 + i, groups=32, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)))
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = nn.Sequential(nn.Conv2d(out_channels, d, 1, bias=False),
                                 nn.BatchNorm2d(d),
                                 nn.ReLU(inplace=True))  # 降维
        self.fc2 = nn.Conv2d(d, out_channels * M, 1, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, input):
        batch_size = input.size(0)
        output = []
        for i, conv in enumerate(self.conv):
            output.append(conv(input))
        U = reduce(lambda x, y: x + y, output)
        s = self.global_pool(U)
        z = self.fc1(s)
        a_b = self.fc2(z)
        a_b = a_b.reshape(batch_size, self.M, self.out_channels, -1)
        a_b = self.softmax(a_b)
        a_b = list(a_b.chunk(self.M, dim=1))
        a_b = list(map(lambda x: x.reshape(batch_size, self.out_channels, 1, 1),
                       a_b))
        V = list(map(lambda x, y: x * y, output,
                     a_b))
        V = reduce(lambda x, y: x + y,
                   V)
        return V


class Fused_module1(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, rgb_feats, ir_feats):
        # get shape

        B, C, H, W = rgb_feats.shape

        # channel concatenation (x_cat -> B,2C,H,W)
        x_cat = torch.cat((rgb_feats, ir_feats), axis=1)

        # create w_avg attention map (w_avg -> B,1,H,W)
        cap = torch.mean(x_cat, dim=1)
        w_avg = self.sigmoid(cap)
        w_avg = w_avg.unsqueeze(1)

        # create w_max attention maps (w_max -> B,1,H,W)
        cmp = torch.max(x_cat, dim=1)[0]
        w_max = self.sigmoid(cmp)
        w_max = w_max.unsqueeze(1)

        # weighted feature map (x_cat_w -> B,2C,H,W)
        x_cat_w = x_cat * w_avg * w_max

        # split weighted feature map (x_ir_w, x_rgb_w -> B,C,H,W)
        x_rgb_w = x_cat_w[:, :C, :, :]
        x_ir_w = x_cat_w[:, C:, :, :]

        # fuse feature maps (x_fused -> B,H,W,C)
        x_fused = (x_ir_w + x_rgb_w) / 2

        return x_fused


#
class SKNet(torch.nn.Module):

    def __init__(self, in_channel=128):
        super().__init__()
        self.fm = Fused_module1()
        self.ca_rgb = SKNet1(in_channels=in_channel,out_channels=in_channel)
        self.ca_ir = SKNet1(in_channels=in_channel,out_channels=in_channel)

    def forward(self, rgb_input, ir_input):
        rgb_feats = self.ca_rgb(rgb_input)
        ir_feats = self.ca_ir(ir_input)

        fused_feats = self.fm(rgb_feats, ir_feats)
        return fused_feats
import torch
import math
from torch import nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import math
from module.mamba2 import Mamba2_1
from mamba_ssm import Mamba2

class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Sequential(nn.Conv2d(in_channels,
                                                 in_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 padding=padding,
                                                 groups=in_channels),  # 深度卷积
                                       nn.SiLU())
                                       # nn.LeakyReLU(negative_slope=0.2,inplace=True))

        self.pointwise = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1),
                                       nn.SiLU())

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class SCSSD(nn.Module):
    def __init__(self, input_dim, output_dim,ssd_config,split_nums=8,d_state=64, d_conv=3, expand=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.split_nums = split_nums
        # self.mamba = Mamba2(
        #     d_model=input_dim // 8 ,  # Model dimension d_model
        #     d_state=d_state,  # SSM state expansion factor
        #     d_conv=d_conv, # Local convolution width
        #     expand=expand,  # Block expansion factor
        #     headdim=8,
        # )
        self.mamba = Mamba2_1(
            d_model=input_dim // self.split_nums,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            headdim=8,
            **ssd_config
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)

        B, C = x.shape[:2]
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        if self.split_nums==8:
            x1, x2, x3, x4, x5, x6, x7, x8 = torch.chunk(x_norm, self.split_nums, dim=2)
            w = int(math.sqrt(x1.shape[1]))
            x_mamba1 = self.mamba(x1,w,w) + self.skip_scale * x1
            x_mamba2 = self.mamba(x2,w,w) + self.skip_scale * x2
            x_mamba3 = self.mamba(x3,w,w) + self.skip_scale * x3
            x_mamba4 = self.mamba(x4,w,w) + self.skip_scale * x4
            x_mamba5 = self.mamba(x5,w,w) + self.skip_scale * x5
            x_mamba6 = self.mamba(x6,w,w) + self.skip_scale * x6
            x_mamba7 = self.mamba(x7,w,w) + self.skip_scale * x7
            x_mamba8 = self.mamba(x8,w,w) + self.skip_scale * x8
            x_mamba = torch.cat([x_mamba1, x_mamba2, x_mamba3, x_mamba4,x_mamba5, x_mamba6, x_mamba7, x_mamba8], dim=2)
        elif self.split_nums==4:
            x1, x2, x3, x4= torch.chunk(x_norm, self.split_nums, dim=2)
            w = int(math.sqrt(x1.shape[1]))
            x_mamba1 = self.mamba(x1,w,w) + self.skip_scale * x1
            x_mamba2 = self.mamba(x2,w,w) + self.skip_scale * x2
            x_mamba3 = self.mamba(x3,w,w) + self.skip_scale * x3
            x_mamba4 = self.mamba(x4,w,w) + self.skip_scale * x4
            x_mamba = torch.cat([x_mamba1, x_mamba2, x_mamba3, x_mamba4], dim=2)
        elif self.split_nums==1:
            w = int(math.sqrt(x_norm.shape[1]))
            x_mamba = self.mamba(x_norm,w,w) + self.skip_scale * x_norm

        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        return out

class SCSSD_Layer(nn.Module):
    def __init__(self, in_channel_list,out_channel_list,ssd_config,split_nums):
        super(SCSSD_Layer, self).__init__()

        self.blocks = nn.ModuleList([SCSSD(in_channel_list[i],
                                           out_channel_list[i],
                                           ssd_config,
                                           split_nums)
                                     for i in range(len(in_channel_list))])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=8):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.SiLU = nn.SiLU()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.SiLU(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.SiLU = nn.SiLU()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.SiLU(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel, ssd_config, split_nums):
        super(CBAM, self).__init__()
        self.split_nums = split_nums
        self.ssd_config = ssd_config
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()
        self.gs = nn.Sequential(nn.GroupNorm(self.split_nums,channel),
                                nn.SiLU())
    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return self.gs(out)


class MSCB(nn.Module):
    def __init__(self, in_channels, out_channels, ssd_config, split_nums):
        super(MSCB, self).__init__()

        self.split_nums = split_nums
        self.ssd_config = ssd_config
        self.act = nn.SiLU()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1),
                                   nn.SiLU())

        self.dwcovn1 = nn.Conv2d(out_channels,
                                 out_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 groups=out_channels)  # 深度卷积

        self.dwcovn2 = nn.Conv2d(out_channels,
                                 out_channels,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1,
                                 groups=out_channels)  # 深度卷积

        self.dwcovn3 = nn.Conv2d(out_channels,
                                 out_channels,
                                 kernel_size=5,
                                 stride=1,
                                 padding=2,
                                 groups=out_channels)  # 深度卷积

        self.pointwise = nn.Sequential(nn.Conv2d(3 * out_channels,
                                                 in_channels,
                                                 kernel_size=1),
                                       nn.SiLU())

        self.cbam = CBAM(in_channels, self.ssd_config, self.split_nums)

    def forward(self, x):
        if self.ssd_config['MSCB'] and self.ssd_config['CBAM']:
            x_init = x
            x = self.conv1(x)
            x1 = self.dwcovn1(x)
            x2 = self.dwcovn2(x)
            x3 = self.dwcovn3(x)
            x = torch.cat((x1,x2,x3),dim=1)
            x = self.pointwise(x) + x_init
            x = self.act(x)
            x = self.cbam(x)
            return x

        elif self.ssd_config['MSCB'] and not self.ssd_config['CBAM']:
            x_init = x
            x = self.conv1(x)
            x1 = self.dwcovn1(x)
            x2 = self.dwcovn2(x)
            x3 = self.dwcovn3(x)
            x = torch.cat(x1,x2,x3)
            x = self.pointwise(x) + x_init
            x = self.act(x)
            return x

        elif not self.ssd_config['MSCB'] and self.ssd_config['CBAM']:
            return self.cbam(x)
        else:
            return None



class LU_M2SR(nn.Module):

    def __init__(self, input_channels=3, out_channels=3,rs_factor=2,
                 c_list=[], res=True,split_nums=8,atten_config={},ssd_config={}):
        super().__init__()

        self.rs_factor = rs_factor
        self.c_list = c_list
        self.out_channels = out_channels
        self.input_channels = input_channels
        self.split_nums = split_nums
        self.atten_config = atten_config
        self.ssd_config = ssd_config
        self.res = res

        # encoder_stage1
        self.stage1_en = DepthwiseSeparableConv2d(input_channels,c_list[0])
        self.att_1 = MSCB(self.c_list[0],3,self.atten_config,self.split_nums)
        # encoder_stage2
        self.stage2_en = SCSSD_Layer(in_channel_list=[self.c_list[0]],
                                     out_channel_list=[self.c_list[1]],
                                     ssd_config=self.ssd_config,
                                     split_nums=self.split_nums) #1层
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.gs_2 = nn.Sequential(nn.GroupNorm(self.split_nums,self.c_list[1]),
                                 nn.SiLU())
        self.att_2 = MSCB(self.c_list[1],3,self.atten_config,self.split_nums)
        # encoder_stage3
        self.stage3_en = SCSSD_Layer(in_channel_list=[self.c_list[1],self.c_list[2]],
                                     out_channel_list=[self.c_list[2],self.c_list[3]],
                                     ssd_config=self.ssd_config,
                                     split_nums=self.split_nums
                                     )#2层
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.att_3 = MSCB(self.c_list[3],3,self.atten_config,self.split_nums)
        self.gs_3 = nn.Sequential(nn.GroupNorm(self.split_nums, self.c_list[3]),
                                  nn.SiLU())
        # encoder_stage4
        self.stage4_en = SCSSD_Layer(in_channel_list=[self.c_list[3],self.c_list[4],self.c_list[5],self.c_list[6]],
                                     out_channel_list=[self.c_list[4],self.c_list[5],self.c_list[6],self.c_list[7]],
                                     ssd_config=self.ssd_config,
                                     split_nums=self.split_nums
                                     )#4层
        self.max_pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.att_4 = MSCB(self.c_list[7],3,self.atten_config,self.split_nums)
        self.gs_4 = nn.Sequential(nn.GroupNorm(self.split_nums, self.c_list[7]),
                                  nn.SiLU())
        # Bottleneck Block
        self.bottleneck_block = nn.Sequential(nn.Conv2d(self.c_list[7], self.c_list[7], kernel_size=3, stride=1, padding=1,
                                              groups=self.c_list[7]),
                                              nn.SiLU())
        self.stage4_de = SCSSD_Layer(in_channel_list=[self.c_list[7],self.c_list[6],
                                                      self.c_list[5],self.c_list[4]],
                                     out_channel_list=[self.c_list[6],self.c_list[5],
                                                       self.c_list[4],self.c_list[3]],
                                     ssd_config=self.ssd_config,
                                     split_nums=self.split_nums
                                     )#4层
        self.up_4 = nn.ConvTranspose2d(self.c_list[3], self.c_list[3], kernel_size=2, stride=2)
        self.gs_5 = nn.Sequential(nn.GroupNorm(self.split_nums, self.c_list[3]),
                                  nn.SiLU())
        # print("stage3_de,{0}   {1}".format(self.c_list[3],self.c_list[2]))
        self.stage3_de = SCSSD_Layer(in_channel_list=[self.c_list[3], self.c_list[2]],
                                     out_channel_list=[self.c_list[2], self.c_list[1]],
                                     ssd_config=self.ssd_config,
                                     split_nums=self.split_nums
                                     )  # 2层
        # self.stage3_de = SCSSD_Layer(in_channel_list=[128, 96],
        #                              out_channel_list=[96, 64])  # 2层
        self.up_3 = nn.ConvTranspose2d(self.c_list[1], self.c_list[1], kernel_size=2, stride=2)
        self.gs_6 = nn.Sequential(nn.GroupNorm(self.split_nums, self.c_list[1]),
                                  nn.SiLU())

        self.stage2_de = SCSSD_Layer(in_channel_list=[self.c_list[1]],
                                     out_channel_list=[self.c_list[0]],
                                     ssd_config=self.ssd_config,
                                     split_nums=self.split_nums
                                     )  # 1层
        self.up_2 = nn.ConvTranspose2d(self.c_list[0], self.c_list[0], kernel_size=2, stride=2)
        self.gs_7 = nn.Sequential(nn.GroupNorm(self.split_nums, self.c_list[0]),
                                  nn.SiLU())
        self.reconstruct = nn.Sequential(nn.Conv2d(self.c_list[0], self.out_channels, 3, 1, 1),
                                         Upsample(self.rs_factor, 3),
                                         nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1))
        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        base = F.interpolate(x, scale_factor=self.rs_factor, mode='bicubic')

        if not self.atten_config['MSCB'] and not self.atten_config['CBAM']:
            out1 = self.stage1_en(x)

            out2 = self.stage2_en(out1)
            out2 = self.max_pool_2(out2)
            out2 = self.gs_2(out2)

            out3 = self.stage3_en(out2)
            out3 = self.max_pool_3(out3)
            out3 = self.gs_3(out3)

            out4 = self.stage4_en(out3)
            out4 = self.max_pool_4(out4)
            out4 = self.gs_4(out4)


            out5 = self.stage4_de(out4)
            out5 = self.up_4(out5)
            out5 = self.gs_5(out5)

            out6 = self.stage3_de(out5)
            out6 = self.up_3(out6)
            out6 = self.gs_6(out6)

            out7 = self.stage2_de(out6)
            out7 = self.up_2(out7)
            out7 = self.gs_7(out7)

            out8 = self.reconstruct(out7)

        else:
            out1 = self.stage1_en(x)
            attn_1 =  self.att_1(out1)

            out2 = self.stage2_en(out1)
            out2 = self.max_pool_2(out2)
            out2 = self.gs_2(out2)
            attn_2 = self.att_2(out2)

            out3 = self.stage3_en(out2)
            out3 = self.max_pool_3(out3)
            out3 = self.gs_3(out3)
            attn_3 = self.att_3(out3)

            out4 = self.stage4_en(out3)
            out4 = self.max_pool_4(out4)
            out4 = self.gs_4(out4)
            attn_4 = self.att_4(out4)

            # outb = self.bottleneck_block(out4)

            # out5 = self.stage4_de(torch.add(outb , attn_4))
            out5 = self.stage4_de(torch.add(out4, attn_4))
            out5 = self.up_4(out5)
            out5 = self.gs_5(out5)

            out6 = self.stage3_de(torch.add(out5 , attn_3))
            out6 = self.up_3(out6)
            out6 = self.gs_6(out6)

            out7 = self.stage2_de(torch.add(out6 , attn_2))
            out7 = self.up_2(out7)
            out7 = self.gs_7(out7)

            out8 = self.reconstruct(torch.add(out7 , attn_1))

        if self.res:
            out8 = torch.add(out8,base) / 2

        return out8

# class LU_M2SR(nn.Module):
#
#     def __init__(self, input_channels=3, out_channels=3,rs_factor=2,
#                  c_list=[64, 80, 24, 128, 48, 64], bridge=True,res=True):
#         super().__init__()
#
#         self.rs_factor = rs_factor
#         self.bridge = bridge
#         if self.bridge:
#             print('~~~~~~~~~~~~~~~CBAM was used~~~~~~~~~~~~~~~')
#
#         self.res = res
#         # encoder_stage1
#         self.stage1_en = DepthwiseSeparableConv2d(input_channels,c_list[0])
#         self.att_1 = CBAM(c_list[0])
#         # encoder_stage2
#         self.stage2_en = SCSSD_Layer(in_channel_list=[c_list[0],c_list[1]],out_channel_list=[c_list[1],c_list[1]]) #2层
#         self.max_pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.gs_2 = nn.Sequential(nn.GroupNorm(8,c_list[0]),
#                                  nn.SiLU())
#         self.att_2 = CBAM(c_list[1])
#         # encoder_stage3
#         self.stage3_en = SCSSD_Layer(in_channel_list=[c_list[1],c_list[1],c_list[2],c_list[2],c_list[3]],
#                                      out_channel_list=[c_list[1],c_list[2],c_list[2],c_list[3],c_list[3]])#5层
#         self.max_pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.att_3 = CBAM(c_list[3])
#         self.gs_3 = nn.Sequential(nn.GroupNorm(8, c_list[3]),
#                                   nn.SiLU())
#         # encoder_stage4
#         self.stage4_en = SCSSD_Layer(in_channel_list=[c_list[3],c_list[4]],out_channel_list=[c_list[4],c_list[5]])#2层
#         self.max_pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.att_4 = CBAM(c_list[5])
#         self.gs_4 = nn.Sequential(nn.GroupNorm(8, c_list[5]),
#                                   nn.SiLU())
#         # Bottleneck Block
#         self.bottleneck_block = nn.Sequential(nn.Conv2d(c_list[5], c_list[5], kernel_size=3, stride=1, padding=1,
#                                               groups=c_list[5]),
#                                               nn.SiLU())
#         self.stage4_de = SCSSD_Layer(in_channel_list=[c_list[5],c_list[5]],out_channel_list=[c_list[5],c_list[4]])#2层
#         self.up_4 = nn.ConvTranspose2d(c_list[4], c_list[4], kernel_size=2, stride=2)
#         self.gs_5 = nn.Sequential(nn.GroupNorm(8, c_list[4]),
#                                   nn.SiLU())
#         self.stage3_de = SCSSD_Layer(in_channel_list=[c_list[4], c_list[4],c_list[3], c_list[3],c_list[2]],
#                                      out_channel_list=[c_list[4], c_list[3], c_list[3], c_list[2], c_list[2]])  # 5层
#         self.up_3 = nn.ConvTranspose2d(c_list[2], c_list[2], kernel_size=2, stride=2)
#         self.gs_6 = nn.Sequential(nn.GroupNorm(8, c_list[2]),
#                                   nn.SiLU())
#
#         self.stage2_de = SCSSD_Layer(in_channel_list=[c_list[2], c_list[1]],
#                                      out_channel_list=[c_list[1], c_list[0]])  # 2层
#         self.up_2 = nn.ConvTranspose2d(c_list[0], c_list[0], kernel_size=2, stride=2)
#         self.gs_7 = nn.Sequential(nn.GroupNorm(8, c_list[0]),
#                                   nn.SiLU())
#         self.reconstruct = nn.Sequential(nn.Conv2d(c_list[0], out_channels, 3, 1, 1),
#                                          Upsample(self.rs_factor, 3),
#                                          nn.Conv2d(out_channels, out_channels, 3, 1, 1))
#
#
#         # self.apply(self._init_weights)
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.Conv1d):
#             n = m.kernel_size[0] * m.out_channels
#             m.weight.data.normal_(0, math.sqrt(2. / n))
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()
#
#     def forward(self, x):
#         base = F.interpolate(x, scale_factor=self.rs_factor, mode='bicubic')
#
#         out1 = self.stage1_en(x)
#         attn_1 =  self.att_1(out1)
#
#         out2 = self.stage2_en(out1)
#         out2 = self.max_pool_2(out2)
#         out2 = self.gs_2(out2)
#         attn_2 = self.att_2(out2)
#
#         out3 = self.stage3_en(out2)
#         out3 = self.max_pool_3(out3)
#         out3 = self.gs_3(out3)
#         attn_3 = self.att_3(out3)
#
#         out4 = self.stage4_en(out3)
#         out4 = self.max_pool_4(out4)
#         out4 = self.gs_4(out4)
#         attn_4 = self.att_4(out4)
#
#         outb = self.bottleneck_block(out4)
#
#         out5 = self.stage4_de(torch.add(outb + attn_4))
#         out5 = self.up_4(out5)
#         out5 = self.gs_5(out5)
#
#         out6 = self.stage3_de(torch.add(out5 + attn_3))
#         out6 = self.up_3(out6)
#         out6 = self.gs_6(out6)
#
#         out7 = self.stage3_de(torch.add(out6 + attn_2))
#         out7 = self.up_2(out7)
#         out7 = self.gs_7(out7)
#
#         out8 = self.reconstruct(torch.add(out7 + attn_1))
#
#         if self.res:
#             out8 = torch.add(out8,base) / 2
#
#         return out8

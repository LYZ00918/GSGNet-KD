import torch
import torch.nn as nn
import torch.nn.functional as F
from toolbox.backbone.ResNet import Backbone_ResNet50_in3,Backbone_ResNet50_in1
from torch.nn.parameter import Parameter
import math
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class TransBasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=2, stride=2, padding=0, dilation=1, bias=False):
        super(TransBasicConv2d, self).__init__()
        self.Deconv = nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class Cr_La(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Cr_La, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(out_channels,in_channels,kernel_size=1),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(in_channels),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels,in_channels,kernel_size=1),
            nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True),
            nn.BatchNorm2d(in_channels),
        )
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(in_channels*3)
        self.r1 = nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=1,dilation=1,stride=1)
        self.r3 = nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=3,dilation=3,stride=1)
        self.r5 = nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=5,dilation=5,stride=1)
        self.bcr = nn.Sequential(
            nn.Conv2d(in_channels*3,in_channels,kernel_size=3,padding=1,stride=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )
    def forward(self,x_cur,x_lat,pre=1):
        if pre==1:
            x_lat = self.conv2(x_lat)

        elif pre==2:
            x_lat = self.conv1(x_lat)

        x_mul = x_cur + x_lat
        x_mul = torch.mul(self.sa(x_mul), x_cur)
        x_cur = self.r1(x_cur)
        x_mul = self.r3(x_mul)
        x_lat = self.r5(x_lat)
        x_all = torch.cat((x_cur, x_mul, x_lat), dim=1)
        x_all_sum = torch.mul(self.ca(x_all), x_all)
        x_all_sum = self.bcr(x_all_sum)
        return x_all_sum

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # need modify by the batchsize/GPU_number
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        output = torch.matmul(input, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class ChannelAttention_diag(nn.Module):
    def __init__(self, in_channels, squeeze_ratio=2):
        super(ChannelAttention_diag, self).__init__()
        self.inter_channels = in_channels // squeeze_ratio
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.avg_fc = nn.Sequential(nn.Linear(in_channels, self.inter_channels, bias=False),
                           nn.ReLU(inplace=True))
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.max_fc = nn.Sequential(nn.Linear(in_channels, self.inter_channels, bias=False),
                           nn.ReLU(inplace=True))
    def forward(self, ftr):
        # ftr: [B, C, H, W]
        device = torch.device("cuda")
        B, C, H, W = ftr.size()
        M = self.inter_channels
        ftr_avg = self.avg_fc(self.avg_pool(ftr).squeeze())
        ftr_max = self.max_fc(self.max_pool(ftr).squeeze())
        cw = torch.sigmoid(ftr_avg + ftr_max).unsqueeze(-1)
        b = torch.unsqueeze(torch.eye(M, device=device), 0).expand(B, M, M).cuda()
        return torch.mul(b, cw)

    # def initialize(self):
    #     weight_init(self)
class GR(nn.Module):
    def __init__(self, in_channels, node_n, squeeze_ratio=4):
        super(GR, self).__init__()
        self.squeeze_ratio = squeeze_ratio
        inter_channels = in_channels // squeeze_ratio
        self.conv_k = nn.Sequential(nn.Conv2d(in_channels, inter_channels, kernel_size=1),
                      nn.ReLU(inplace=True))
        self.conv_v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.ca_diag = ChannelAttention_diag(in_channels,squeeze_ratio)
        self.GCN = GraphConvolution(in_channels, in_channels, bias=False)
        self.delta = nn.Parameter(torch.Tensor([0]))

    def forward(self, ftr):
        device = torch.device("cuda")
        B, C, H, W = ftr.size()
        HW = H * W
        M = C // 8
        b = torch.unsqueeze(torch.eye(HW, device=device), 0).expand(B, HW, HW).cuda()
        One = torch.ones(HW, 1, dtype=torch.float32, device=device).expand(B, HW, 1).cuda() # [B, HW, 1]
        diag = self.ca_diag(ftr) # [B, M, M]

        ftr_k = self.conv_k(ftr).view(B, -1, HW)  # [B, M, HW]
        ftr_q = ftr_k.permute(0, 2, 1)  # [B, HW, M]

        # 不减少计算的方法
        # 构造邻接矩阵
        D = torch.bmm(ftr_q, diag)
        D = torch.sigmoid(torch.bmm(D, ftr_k))
        # 获取度图
        D = torch.bmm(D, One)
        D = D ** (-1 / 2)
        D = torch.mul(b, D)

        P = torch.bmm(D, ftr_q)
        Pt = P.permute(0, 2, 1)
        # LX就是构造好的特征图
        X = ftr.view(B, -1, HW).permute(0, 2, 1)
        LX = torch.bmm(Pt, X)
        LX = torch.bmm(diag, LX)
        LX = torch.bmm(P, LX)
        LX = X - LX

        Y = (X + self.GCN(LX)).permute(0, 2, 1)

        Y = Y.view(B, C, H, W)
        return Y


class Graph_aspp(nn.Module):
    def __init__(self,cur_channels):
        super(Graph_aspp, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cur_channels,cur_channels//4,kernel_size=3,padding=1,stride=1),
        )
        self.up2 = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        self.down = nn.Conv2d(cur_channels,cur_channels,kernel_size=3,stride=2,padding=1)
        self.rate1 = nn.Conv2d(cur_channels,cur_channels,kernel_size=3,padding=1,stride=1,dilation=1)
        self.rate2 = nn.Conv2d(cur_channels,cur_channels,kernel_size=3,stride=1,padding=2,dilation=2)
        self.rate4 = nn.Conv2d(cur_channels,cur_channels,kernel_size=3,stride=1,padding=4,dilation=4)
        self.rate6 = nn.Conv2d(cur_channels,cur_channels,kernel_size=3,stride=1,padding=6,dilation=6)
        self.rate8 = nn.Conv2d(cur_channels,cur_channels,kernel_size=3,stride=1,padding=8,dilation=8)
        # 图卷积 未定义好 只对空间进行图卷积
        self.graph1 = GR(cur_channels,32*32,squeeze_ratio=4)
        self.graph2 = GR(cur_channels,16*16,squeeze_ratio=8)
        self.graph3 = GR(cur_channels,8*8,  squeeze_ratio=8)
    def forward(self,x_cur,x_lat,pre=1):
        if pre == 1:
            x_sum = self.up2(x_lat) * x_cur + x_cur

            feature0 = self.rate1(x_sum)
            x_0 = self.graph2(feature0)

            feature = feature0 + x_sum
            feature1 = self.rate2(feature)
            x_1 = self.graph2(feature1)
            # x_1 = torch.mul(x_1,x_cur)

            feature = feature1 + x_sum
            feature2 = self.rate4(feature)
            x_2 = self.graph2(feature2)
            # x_2 = torch.mul(x_2, x_cur)

            feature = feature2 + x_sum
            feature3 = self.rate4(feature)
            x_3 = self.graph2(feature3)
            # x_3 = torch.mul(x_3, x_cur)

            sum = x_0 + x_1 + x_2 + x_3


        else:
            x_sum = x_cur * self.down(x_lat) + x_cur

            feature0 = self.rate1(x_sum)
            x_0 = self.graph3(feature0)

            feature = feature0 + x_sum
            feature1 = self.rate2(feature)
            x_1 = self.graph3(feature1)
            # x_1 = torch.mul(x_1, x_cur)

            feature = feature1 + x_sum
            feature2 = self.rate4(feature)
            x_2 = self.graph3(feature2)
            # x_2 = torch.mul(x_2, x_cur)

            sum = x_0 + x_1 + x_2
        return sum

# 语义融合模块
class Se_A(nn.Module):
    def __init__(self,in_channels,scale1=2,scale2=4):
        super(Se_A, self).__init__()
        self.conv1 = nn.Conv2d(512,in_channels,kernel_size=3,padding=1,stride=1)
        self.conv2 = nn.Conv2d(512,in_channels,kernel_size=3,padding=1,stride=1)
        # 第二层语义进行上采样
        self.up1 = nn.Upsample(scale_factor=scale1,mode='bilinear',align_corners=True)
        # 第一层语义进行上采样
        self.up2 = nn.Upsample(scale_factor=scale2,mode='bilinear',align_corners=True)
    def forward(self,x1,x2):
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x_sum = self.up2(x2) + self.up1(x1)
        return x_sum

# rgb与dep融合
class Mu_fuse(nn.Module):
    def __init__(self,cur_channels):
        super(Mu_fuse, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(cur_channels)
        self.ca1 = ChannelAttention(cur_channels*2)
        self.bcr = nn.Sequential(
            nn.Conv2d(cur_channels, cur_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(cur_channels),
            nn.ReLU(),
        )
    def forward(self,x_rgb,x_d):
        x_m = self.ca(torch.mul(x_d,x_rgb))
        x_m_d = torch.mul(x_m,x_d)
        x_m_rgb = torch.mul(x_m,x_rgb)
        x_d_f = torch.mul(self.sa(x_m_d),x_m_d)
        x_d_rgb = torch.mul(self.sa(x_m_rgb),x_m_rgb)
        # x_c_all = torch.mul(self.ca1(torch.cat((x_d_f,x_d_rgb),dim=1)),torch.cat((x_d_f,x_d_rgb),dim=1))
        x_c_all = x_d_f + x_d_rgb
        x_c_all = self.bcr(x_c_all)
        return x_c_all
# 编写解码块
class decoder(nn.Module):

    def __init__(self):
        super(decoder, self).__init__()

        self.dp5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 512, kernel_size=1),
        )
        self.dp4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512,256,kernel_size=1),
        )
        self.dp3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256,128,kernel_size=1),
        )
        self.dp2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128,64,kernel_size=1),
        )
        self.dp1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64,32,kernel_size=1),
        )
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.se_a1 = Se_A(256,2,4)
        self.se_a2 = Se_A(128,4,8)
        self.se_a3 = Se_A(64,8,16)
        self.conv4 = nn.Sequential(
            nn.Conv2d(256,256,kernel_size=3,padding=1,stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.dconv4 = TransBasicConv2d(256,128)
        self.dconv3 = TransBasicConv2d(128,64)
        self.dconv2 = TransBasicConv2d(64,32)

        self.down1 = nn.Conv2d(256,128,kernel_size=1)
        self.S5 = nn.Conv2d(512, 6, 3, stride=1, padding=1)
        self.S4 = nn.Conv2d(512, 6, 3, stride=1, padding=1)
        self.S3 = nn.Conv2d(256, 6, 3, stride=1, padding=1)
        self.S2 = nn.Conv2d(128, 6, 3, stride=1, padding=1)
        self.S1 = nn.Conv2d(64, 6, 3, stride=1, padding=1)
        self.S0 = nn.Conv2d(32, 6, 3, stride=1, padding=1)

    def forward(self, x1_Accom,x2_Accom,x3_Accom,x4_Accom,x4_sma,x5_sma,x1_rgb,x2_rgb,x3_rgb):
        x4 = self.se_a1(x4_sma,x5_sma)
        x3 = self.se_a2(x4_sma,x5_sma)
        x2 = self.se_a3(x4_sma,x5_sma)
        x = x2_Accom + x3_Accom

        t6 = self.S5(x5_sma)
        t5 = self.S4(x4_sma)

        z4 = self.conv4(torch.mul(x4,x4_Accom) + x4_Accom) + x3_rgb
        t4 = self.S3(z4)

        z3 = self.conv3(torch.mul(x3,x) + x) + self.conv3(torch.mul(x3,self.dconv4(z4)) + self.dconv4(z4)) + x2_rgb
        t3 = self.S2(z3)

        z2 = self.conv2(torch.mul(x2,x1_Accom)) + self.conv2(torch.mul(x2,self.dconv3(z3))) + self.dconv3(z3) + x1_rgb
        t2 = self.S1(z2)

        z1 = self.dconv2(z2)
        t1 = self.S0(z1)
        return t1, t2, t3, t4, t5, t6

class GSGNet(nn.Module):
    def __init__(self):
        super(GSGNet, self).__init__()
        #Backbone model
        # ---- ResNet50 Backbone ----
        (
            self.encoder1,
            self.encoder2,
            self.encoder4,
            self.encoder8,
            self.encoder16,
        ) = Backbone_ResNet50_in3()
        self.depth_encoder2, self.depth_encoder4, self.depth_encoder8, self.depth_encoder16, self.depth_encoder32 = \
            Backbone_ResNet50_in1()

        # Lateral layers
        self.lateral_conv0 = BasicConv2d(64, 64, 3, stride=1, padding=1)
        self.lateral_conv1 = BasicConv2d(256, 128, 3, stride=1, padding=1)
        self.lateral_conv2 = BasicConv2d(512, 256, 3, stride=1, padding=1)
        self.lateral_conv3 = BasicConv2d(1024, 512, 3, stride=1, padding=1)
        self.lateral_conv4 = BasicConv2d(2048, 512, 3, stride=1, padding=1)
        # rgb dep 融合模块
        self.fuse1 = Mu_fuse(64)
        self.fuse2 = Mu_fuse(128)
        self.fuse3 = Mu_fuse(256)
        self.fuse4 = Mu_fuse(512)
        self.fuse5 = Mu_fuse(512)
        # 细节模块
        self.Accom1 = Cr_La(64,128)
        self.Accom2 = Cr_La(128,64)
        self.Accom3 = Cr_La(128,256)
        self.Accom4 = Cr_La(256,128)

        # 语义模块
        self.att1 = Graph_aspp(512)
        self.att2 = Graph_aspp(512)

        # 进行解码
        self.decoder_rgb = decoder()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x_rgb,x_d):
        x0 = self.encoder1(x_rgb)
        x1 = self.encoder2(x0)
        x2 = self.encoder4(x1)
        x3 = self.encoder8(x2)
        x4 = self.encoder16(x3)

        d1 = self.depth_encoder2(x_d)
        d2 = self.depth_encoder4(d1)
        d3 = self.depth_encoder8(d2)
        d4 = self.depth_encoder16(d3)
        d5 = self.depth_encoder32(d4)

        x1_rgb = self.lateral_conv0(x0)
        x2_rgb = self.lateral_conv1(x1)
        x3_rgb = self.lateral_conv2(x2)
        x4_rgb = self.lateral_conv3(x3)
        x5_rgb = self.lateral_conv4(x4)

        x1_d = self.lateral_conv0(d1)
        x2_d = self.lateral_conv1(d2)
        x3_d = self.lateral_conv2(d3)
        x4_d = self.lateral_conv3(d4)
        x5_d = self.lateral_conv4(d5)

        fuse1 = self.fuse1(x1_rgb,x1_d)
        fuse2 = self.fuse2(x2_rgb,x2_d)
        fuse3 = self.fuse3(x3_rgb,x3_d)
        fuse4 = self.fuse4(x4_rgb,x4_d)
        fuse5 = self.fuse5(x5_rgb,x5_d)

        x1_Accom = self.Accom1(fuse1,fuse2,pre=1)
        x2_Accom = self.Accom2(fuse2,fuse1,pre=2)
        x3_Accom = self.Accom3(fuse2,fuse3,pre=1)
        x4_Accom = self.Accom4(fuse3,fuse2,pre=2)

        x4_sma = self.att1(fuse4,fuse5,pre=1)
        x5_sma = self.att2(fuse5,fuse4,pre=2)

        s0,s1, s2, s3, s5, s6 = self.decoder_rgb(x1_Accom,x2_Accom,x3_Accom,x4_Accom,x4_sma,x5_sma,x1_rgb,x2_rgb,x3_rgb)
        # At test phase, we can use the HA to post-processing our saliency map


        return s0,s1, s2, s3, s5, s6



if __name__ == '__main__':
    image = torch.randn(3, 3, 256, 256).cuda()
    dep = torch.randn(3, 3, 256, 256).cuda()
    model = GSGNet().cuda()
    out = model(image,dep)
    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)
    print(out[3].shape)
    print(out[4].shape)
    print(out[5].shape)

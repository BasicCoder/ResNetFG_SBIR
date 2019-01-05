import torch as t
from torch import nn
import torch.nn.functional as F

from models_attn.resnet import resnet34, resnet50

class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(t.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B x (N) x C
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B x C x (*W*H)
        energy = t.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # B x (N) x (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B x C x N

        out = t.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention

class ResNet34_Attn(nn.Module):
    def __init__(self, pretrained=False, out_features=125):
        super(ResNet34_Attn, self).__init__()
        self.model = resnet34(pretrained=pretrained)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.attn_avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(in_features=512*2, out_features=out_features)
        self.attn = Self_Attn(512, 'relu')

    def forward(self, x):
        # shape [N, C, H, W]
        x = self.model(x)
        global_feature = self.avgpool(x)
        global_feature = global_feature.view(global_feature.size(0), -1)

        attn_feature, p = self.attn(x)

        attn_feature = self.attn_avgpool(attn_feature)
        attn_feature = attn_feature.view(attn_feature.size(0), -1)

        feat = t.cat([global_feature, attn_feature], 1)
        x = self.fc(feat)
        return x, feat



class ResNet50_Attn(nn.Module):
    def __init__(self, pretrained=False, out_features=125):
        super(ResNet50_Attn, self).__init__()
        self.model = resnet50(pretrained=pretrained)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.attn_avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(in_features=2048*2, out_features=out_features)
        self.attn = Self_Attn(2048, 'relu')

    def forward(self, x):
        # shape [N, C, H, W]
        x = self.model(x)
        global_feature = self.avgpool(x)
        global_feature = global_feature.view(global_feature.size(0), -1)

        attn_feature, p = self.attn(x)

        attn_feature = self.attn_avgpool(attn_feature)
        attn_feature = attn_feature.view(attn_feature.size(0), -1)

        feat = t.cat([global_feature, attn_feature], 1)
        x = self.fc(feat)
        return x, feat

class ResNet34_Attn_Single(nn.Module):
    def __init__(self, pretrained=False, out_features=125):
        super(ResNet34_Attn_Single, self).__init__()
        self.model = resnet34(pretrained=pretrained)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(in_features=512, out_features=out_features)
        self.attn = Self_Attn(512, 'relu')

    def forward(self, x):
        # shape [N, C, H, W]
        x = self.model(x)

        attn_feature, p = self.attn(x)
        attn_feature = self.avgpool(attn_feature)
        attn_feature = attn_feature.view(attn_feature.size(0), -1)

        x = self.fc(attn_feature)
        return x, attn_feature



class ResNet50_Attn_Single(nn.Module):
    def __init__(self, pretrained=False, out_features=125):
        super(ResNet50_Attn_Single, self).__init__()
        self.model = resnet50(pretrained=pretrained)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(in_features=2048*2, out_features=out_features)
        self.attn = Self_Attn(2048, 'relu')

    def forward(self, x):
        # shape [N, C, H, W]
        x = self.model(x)

        attn_feature, p = self.attn(x)
        attn_feature = self.avgpool(attn_feature)
        attn_feature = attn_feature.view(attn_feature.size(0), -1)

        x = self.fc(attn_feature)
        return x, attn_feature


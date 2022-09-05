import torch.nn as nn
import torch.nn.functional as F
import torch

class Residual(nn.Module):
    def __init__(self, in_size, out_size, ks = 3, stride = 1, padding = 1):
        super(Residual, self).__init__()
        self.conv_block1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, ks, stride, padding),
                nn.BatchNorm2d(out_size),
                nn.ReLU(),
        )

        self.conv_block2 = nn.Sequential(
                nn.Conv2d(out_size, out_size, ks, stride, padding),
                nn.BatchNorm2d(out_size),
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x + residual
        out = self.relu(x)
        return out

class MIL(nn.Module):
    def __init__(self, in_size, out_size, img_channels = 3, ks = 3, stride = 1, padding = 1):
        super(MIL, self).__init__()
        self.img_conv = nn.Sequential(
                nn.Conv2d(img_channels, out_size // 2, ks, stride, padding),
                nn.ReLU(),
        )
        self.con_conv = nn.Sequential(
                nn.Conv2d(out_size // 2 + in_size, out_size, ks, stride, padding),
                nn.ReLU(),
        )
        self.conv_block1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, ks, stride, padding),
                nn.BatchNorm2d(out_size),
                nn.ReLU(),
        )
        self.conv_block2 = nn.Sequential(
                nn.Conv2d(out_size, out_size, ks, stride, padding),
                nn.BatchNorm2d(out_size),
        )
        self.relu = nn.ReLU()


    def forward(self, x, img):
        img = F.interpolate(img, size = (x.shape[2], x.shape[3]), mode = 'bicubic')
        residual = self.img_conv(img)
        residual = self.con_conv(torch.concat((x, residual), 1))
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = residual + x
        out = self.relu(x)
        return out

class Dilated(nn.Module):
    def __init__(self, in_size, out_size, rate = 2, ks = 3, stride = 1, padding = 1):
        super(Dilated, self).__init__()
        self.conv_block1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, ks, stride, padding = rate, dilation = rate, padding_mode = 'reflect'),
                nn.BatchNorm2d(out_size),
                nn.ReLU(),
        )
        self.conv_block2 = nn.Sequential(
                nn.Conv2d(out_size, out_size, ks, stride, padding = rate, dilation = rate, padding_mode = 'reflect'),
                nn.BatchNorm2d(out_size),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = residual + x
        out = self.relu(x)
        return out

class ASPP(nn.Module):
        def __init__(self, in_size, ks = 3, stride = 1, padding = 1):
                super(ASPP, self).__init__()
                self.dilated_18 = nn.Sequential(
                        nn.Conv2d(in_size, in_size, ks, stride, 18, dilation = 18, padding_mode = 'reflect'),
                        nn.ReLU(),
                        nn.Conv2d(in_size, in_size, 1),
                        nn.ReLU(),
                        nn.Dropout(p = 0.5),
                        nn.Conv2d(in_size, 128, 1),
                        nn.ReLU(),
                )
                self.dilated_12 = nn.Sequential(
                        nn.Conv2d(in_size, in_size, ks, stride, 12, dilation = 12, padding_mode = 'reflect'),
                        nn.ReLU(),
                        nn.Conv2d(in_size, in_size, 1),
                        nn.ReLU(),
                        nn.Dropout(p = 0.5),
                        nn.Conv2d(in_size, 128, 1),
                        nn.ReLU(),
                )
                self.dilated_6 = nn.Sequential(
                        nn.Conv2d(in_size, in_size, ks, stride, 6, dilation = 6, padding_mode = 'reflect'),
                        nn.ReLU(),
                        nn.Conv2d(in_size, in_size, 1),
                        nn.ReLU(),
                        nn.Dropout(p = 0.5),
                        nn.Conv2d(in_size, 128, 1),
                        nn.ReLU(),
                )
                self.conv_block1 = nn.Sequential(
                        nn.Conv2d(in_size, in_size, 1),
                        nn.ReLU(),
                        nn.Conv2d(in_size, in_size, 1),
                        nn.ReLU(),
                        nn.Dropout(p = 0.5),
                        nn.Conv2d(in_size, 128, 1),
                        nn.ReLU(),
                )
                self.conv_block2 = nn.Sequential(
                        nn.AvgPool2d(2),
                        nn.Conv2d(in_size, in_size, 1),
                        nn.ReLU(),
                        nn.Dropout(p = 0.5),
                        nn.Conv2d(in_size, 128, 1),
                        nn.ReLU(),
                )

        def forward(self, x):
                shape = (x.shape[2], x.shape[3])
                x1 = self.dilated_18(x)
                x2 = self.dilated_12(x)
                x3 = self.dilated_6(x)
                x4 = self.conv_block1(x)
                x5 = self.conv_block2(x)
                x5 = F.interpolate(x5, size = shape, mode = 'bicubic')
                out = torch.concat((x1, x2, x3, x4, x5), 1)
                return out

class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm = True, n=2, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True),)
                setattr(self, 'conv%d'%i, conv)
                in_size = out_size

        else:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True),)
                setattr(self, 'conv%d'%i, conv)
                in_size = out_size

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n+1):
            conv = getattr(self, 'conv%d'%i)
            x = conv(x)

        return x

class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size+(n_concat-2)*out_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, padding=0)
        else:
            self.up = nn.Sequential(
                 nn.UpsamplingBilinear2d(scale_factor=2),
                 nn.Conv2d(in_size, out_size, 1))

    def forward(self, high_feature, *low_feature):
        outputs0 = self.up(high_feature)
        for feature in low_feature:
            c = feature.shape[2] - outputs0.shape[2]
            if c > 0:
                outputs0 = F.pad(outputs0, (c, 0, c, 0))
            outputs0 = torch.cat([outputs0, feature], 1)
        return self.conv(outputs0)

class AttentionBlock(nn.Module):
        def __init__(self, g_in = 64, x_in = 64):
                super(AttentionBlock, self).__init__()
                self.g_in = g_in
                self.x_in = x_in
                self.g_conv1 = nn.Conv2d(g_in, 128, 1)
                self.x_conv1 = nn.Conv2d(x_in, 128, 1)
                self.relu = nn.ReLU()
                self.phi = nn.Conv2d(128, 1, 1)
                self.sigmoid = nn.Sigmoid()

        def forward(self, g, x):
                g1 = self.g_conv1(g)
                x1 = self.x_conv1(x)
                out = self.relu(g1 + x1)
                out = self.phi(out)
                out = self.sigmoid(out)
                out = torch.mul(x, out)
                return out




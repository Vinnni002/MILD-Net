import torch.nn as nn
from .units import MIL, AttentionBlock, Dilated, Residual, ASPP, unetConv2, unetUp
import torch
import torch.nn.functional as F
from torch.nn import init

class MILD_Net(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 1):
        super(MILD_Net, self).__init__()
        self.conv_block1 = nn.Sequential(
                nn.Conv2d(in_channels, 64, 3, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
        )
        self.conv_block2 = nn.Sequential(
                nn.Conv2d(64, 64, 3, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
        )
        self.m_unit128 = MIL(64, 128)
        self.r_unit128 = Residual(128, 128)
        self.m_unit256 = MIL(128, 256)
        self.r_unit256 = Residual(256, 256)
        self.m_unit512 = MIL(256, 512)
        self.r_unit512 = Residual(512, 512)
        self.d_unit2_1 = Dilated(512, 512, rate = 2)
        self.d_unit2_2 = Dilated(512, 512, rate = 2)
        self.d_unit4_1 = Dilated(512, 512, rate = 4)
        self.d_unit4_2 = Dilated(512, 512, rate = 4)
        self.aspp = ASPP(512)
        self.ab1 = AttentionBlock(640, 640)
        self.ab2 = AttentionBlock(256, 256)
        self.ab3 = AttentionBlock(128, 128)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor = 2, mode = 'bilinear')
        self.low_conv1 = nn.Sequential(
                nn.Conv2d(256, 640, 1),
                nn.BatchNorm2d(640),
                nn.ReLU(),
        )
        self.low_conv2 = nn.Sequential(
                nn.Conv2d(128, 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
        )
        self.low_conv3 = nn.Sequential(
                nn.Conv2d(64, 128, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
        )
        self.conv_block3 = nn.Sequential(
                nn.Conv2d(1280, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
        )
        self.conv_block4 = nn.Sequential(
                nn.Conv2d(512, 128, 3, 1, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, 3, 1, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
        )
        self.object_map = nn.Sequential(
                nn.Conv2d(256, 64, 3, 1, 1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 64, 3, 1, 1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.Dropout(p = 0.5),
                nn.Conv2d(64, out_channels, 1),
                # nn.Softmax(dim = 1),
        )

        self.contour_map = nn.Sequential(
                nn.Conv2d(256, 64, 3, 1, 1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 64, 3, 1, 1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.Dropout(p = 0.5),
                nn.Conv2d(64, out_channels, 1),
                # nn.Softmax(dim = 1),
        )

#         self.aux_img = nn.Sequential(
#                 nn.Conv2d(512, 64, 3, 1, 1),
#                 nn.ReLU(),
# #                 nn.BatchNorm2d(64),
#                 nn.Conv2d(64, 64, 3, 1, 1),
#                 nn.ReLU(),
# #                 nn.BatchNorm2d(64),
#                 nn.Conv2d(64, out_channels, 1),
#         )

#         self.aux_obj = nn.Sequential(
#                 nn.Conv2d(512, 64, 3, 1, 1),
#                 nn.ReLU(),
# #                 nn.BatchNorm2d(64),
#                 nn.Conv2d(64, 64, 3, 1, 1),
#                 nn.ReLU(),
# #                 nn.BatchNorm2d(64),
#                 nn.Conv2d(64, out_channels, 1),
#         )

    def forward(self, input):
        conv_1 = self.conv_block1(input)
        conv_2 = self.conv_block2(conv_1)
        maxpool_1 = self.maxpool(conv_2)
        mil_1 = self.m_unit128(maxpool_1, input)
        res_1 = self.r_unit128(mil_1)
        maxpool_2 = self.maxpool(res_1)
        mil_2 = self.m_unit256(maxpool_2, input)
        res_2 = self.r_unit256(mil_2)
        maxpool_3 = self.maxpool(res_2)
        mil_3 = self.m_unit512(maxpool_3, input)
        res_3 = self.r_unit512(mil_3)
        d_1 = self.d_unit2_1(res_3)
        d_2 = self.d_unit2_2(d_1)
        # aux_img = self.aux_img(d_2)
        # aux_obj = self.aux_obj(d_2)
        d_3 = self.d_unit4_1(d_2)
        d_4 = self.d_unit4_2(d_3)
        aspp = self.aspp(d_4)
        u_1 = self.upsample(aspp)
        concat_feature_1 = self.ab1(u_1, self.low_conv1(res_2))
        concat_conv_1 = self.conv_block3(torch.concat((u_1, concat_feature_1), 1))
        u_2 = self.upsample(concat_conv_1)
        concat_feature_2 = self.ab2(u_2, self.low_conv2(res_1))
        concat_conv_2 = self.conv_block4(torch.concat((u_2, concat_feature_2), 1))
        u_3 = self.upsample(concat_conv_2)
        concat_feature_3 = self.ab3(u_3, self.low_conv3(conv_2))
        u_3 = torch.concat((u_3, concat_feature_3), 1)
        object_map = self.object_map(u_3)
        contour_map = self.contour_map(u_3)
        return object_map, contour_map

class UNet_Contour(nn.Module):

    def __init__(self, in_channels=1, n_classes=2, feature_scale=2, is_deconv=True, is_batchnorm=True, is_ds=True):
        super(UNet_Contour, self).__init__()
        self.in_channels = in_channels
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.is_ds = is_ds

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv00 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.conv10 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.conv20 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.conv30 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.conv40 = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat1 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat2 = unetUp(filters[3], filters[2], self.is_deconv, 2)
        self.up_concat3 = unetUp(filters[2], filters[1], self.is_deconv, 2)
        self.up_concat4 = unetUp(filters[1], filters[0], self.is_deconv, 2)
        
        self.up_concat11 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat12 = unetUp(filters[3], filters[2], self.is_deconv, 2)
        self.up_concat13 = unetUp(filters[2], filters[1], self.is_deconv, 2)
        self.up_concat14 = unetUp(filters[1], filters[0], self.is_deconv, 2)
        
        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)
        
        self.final1 = nn.Conv2d(filters[0], n_classes, 1)
        
    def forward(self, inputs):
        # column : 0
        X_00 = self.conv00(inputs)     
        maxpool0 = self.maxpool(X_00)    
        X_10= self.conv10(maxpool0)     
        maxpool1 = self.maxpool(X_10)   
        X_20 = self.conv20(maxpool1)   
        maxpool2 = self.maxpool(X_20)    
        X_30 = self.conv30(maxpool2)
        maxpool3 = self.maxpool(X_30)
        X_40 = self.conv40(maxpool3)
        
        # Map Generation
        
        # column : 1
        X_1 = self.up_concat1(X_40,X_30)
        # column : 2
        X_2 = self.up_concat2(X_1,X_20)
        # column : 3
        X_3 = self.up_concat3(X_2,X_10)
        # column : 4
        X_4 = self.up_concat4(X_3,X_00)

        # final layer
        object_map = self.final(X_4)

        # # Contour Generation
    
        # # column : 1
        # X_11 = self.up_concat11(X_40,X_30)
        # # column : 2
        # X_12 = self.up_concat12(X_11,X_20)
        # # column : 3
        # X_13 = self.up_concat13(X_12,X_10)
        # # column : 4
        # X_14 = self.up_concat14(X_13,X_00)

        # # final layer
        # contour_map = self.final1(X_14)
           
        return object_map
import pdb
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.op import MLPBlock, MotionBlock, GroupOperation
from models.pointlstm import PointLSTM
#from models.attention_test import SelfA
#from models.apesattention import N2PAttention
from models.endding import Endding

class Motion(nn.Module):
    def __init__(self, num_classes, pts_size, offsets, topk=16, downsample=(2, 2, 2),
                 knn=(16, 48, 48, 24)):
        super(Motion, self).__init__()
        self.stage1 = MLPBlock([4, 32, 64], 2)
        self.pool1 = nn.AdaptiveMaxPool2d((None, 1))
        self.stage2 = MotionBlock([128, 128, ], 2, 4)
        self.pool2 = nn.AdaptiveMaxPool2d((None, 1))
        self.stage1_3 = MotionBlock([64, 128, 256], 2, 4)
        self.stage3 = MotionBlock([256, 256, ], 2, 4)
        self.stage3_4 = MotionBlock([128, 256, ], 2, 4)
        self.pool3 = nn.AdaptiveMaxPool2d((None, 1))
        self.stage4 = MotionBlock([512, 512, ], 2, 4)
        self.pool4 = nn.AdaptiveMaxPool2d((None, 1))
        self.stage5 = MLPBlock([512, 1024], 2)
        self.pool5 = nn.AdaptiveMaxPool2d((1, 1))
        self.stage6 = MLPBlock([1024, num_classes], 2, with_bn=False)
        self.stage6_end = MLPBlock([1472, num_classes], 2, with_bn=False)

        self.stage6_fea1 = MLPBlock([64, num_classes], 2, with_bn=False)
        self.stage6_fea2 = MLPBlock([128, num_classes], 2, with_bn=False)
        self.stage6_fea3 = MLPBlock([256, num_classes], 2, with_bn=False)
        #self.stage6_fea1_lstm = MLPBlock([68, num_classes], 2, with_bn=False)
        #self.stage6_fea2_lstm = MLPBlock([132, num_classes], 2, with_bn=False)
        #self.stage6_fea3_lstm = MLPBlock([260, num_classes], 2, with_bn=False)

        self.global_bn = nn.BatchNorm2d(1472)
        self.knn = knn
        self.pts_size = pts_size
        self.downsample = downsample
        self.num_classes = num_classes
        self.group = GroupOperation()
        self.lstm = PointLSTM(offsets=offsets, pts_num=pts_size // downsample[0], in_channels=132, hidden_dim=256,
                              offset_dim=4, num_layers=1, topk=topk)
        
        #叠加xlstm层
        self.xlstm_same_first = PointLSTM(offsets=offsets, pts_num=pts_size, in_channels=68, hidden_dim=64,offset_dim=4, num_layers=1, topk=topk)
        self.xlstm_same_second = PointLSTM(offsets=offsets, pts_num=64, in_channels=132, hidden_dim=128,offset_dim=4, num_layers=1, topk=topk)

        #apesattention测试
        #self.xlstm_same_second = PointLSTM(offsets=offsets, pts_num=64, in_channels=136, hidden_dim=128,offset_dim=4, num_layers=1, topk=topk)
        
        self.xlstm_same_thired = PointLSTM(offsets=offsets, pts_num=32, in_channels=260, hidden_dim=256,offset_dim=4, num_layers=1, topk=topk)

        #self.selfa1 = SelfA(pts_num=pts_size, in_channels=132, qk_dim=64, v_dim=64, num_layers=1, topk=topk)
        #self.selfa2 = SelfA(pts_num=pts_size, in_channels=260, qk_dim=128, v_dim=128, num_layers=1, topk=topk)
        #self.apesattention = N2PAttention(pts_num=pts_size, in_channels=132, out_channels=132 , knn=topk)
        self.endding_first = Endding(in_channels = 68)
        self.endding_second = Endding(in_channels = 132)
        self.endding_thired = Endding(in_channels = 260)

    def mogrify_conv(self,xt,ht):
        for i in range(1,6): #for i in range(1,self.mog_iterations+1):
            if (i % 2 == 0):
                ht = (2*torch.sigmoid(self.conv_input_to_hidden(xt))) * ht
            else:
                xt = (2*torch.sigmoid(self.conv_hidden_to_input(ht))) * xt
        return xt, ht

    #def forward(self, inputs):
        # B * T * N * D,  e.g. 16 * 32 * 512 * 4
        inputs = inputs.permute(0, 3, 1, 2)
        if self.training:
            inputs = inputs[:, :, :, torch.randperm(inputs.shape[3])[:self.pts_size]]
        else:
            inputs = inputs[:, :, :, ::inputs.shape[3] // self.pts_size]# B * (4 + others) * 32 * 128
        inputs = inputs[:, :4]# B * 4 * 32 * 128
        batchsize, in_dims, timestep, pts_num = inputs.shape

        # stage 1: intra-frame
        ret_array1 = self.group.group_points(distance_dim=[0, 1, 2], array1=inputs, array2=inputs, knn=self.knn[0],
                                             dim=3)
        # B * 4 * 32 * 128 * 16
        ret_array1 = ret_array1.contiguous().view(batchsize, in_dims, timestep * pts_num, -1)
        # B * 4 * 4096 * 16
        fea1 = self.pool1(self.stage1(ret_array1)).view(batchsize, -1, timestep, pts_num)
        # B * 64 * 32 * 128
        fea1 = torch.cat((inputs, fea1), dim=1)
        # B * 68 * 32 * 128                               相当于将近邻16个点的坐标接在原始点坐标后面（但每个点使用四个点坐标，其中前三个为xyz，最后一个原始为0）

        #fea1_attention = self.selfa1(fea1)    #[b,c,t,n]

        # stage 2: inter-frame, early
        

        #in_dims = fea1.shape[1] * 2 - 4  #132
        in_dims = fea1.shape[1] * 2 - 4  #132
        pts_num //= self.downsample[0]   #64

        #fea1 = self.xlstm_same_first(fea_test.permute(0,2,1,3))
        #fea1 = self.xlstm_same_first(fea1_attention[0].permute(0,2,1,3))
        fea1 = self.xlstm_same_first(fea1.permute(0,2,1,3))
        fea1 = fea1[0][0].squeeze(-1).permute(0,2,1,3)                         #[8,68,32,128]
        ret_group_array2 = self.group.st_group_points(fea1, 3, [0, 1, 2], self.knn[1], 3)  # b * 132 * 32 * 128 * knn(24)
        ret_array2, inputs, _ = self.select_ind(ret_group_array2, inputs,  #b,132,2048,knn(24)//b,4,32,64//b,32,64
                                                batchsize, in_dims, timestep, pts_num)     # b , 132 ,32 , 64
        fea2 = self.pool2(self.stage2(ret_array2)).view(batchsize, -1, timestep, pts_num)  #b,128,32,64
        fea2 = torch.cat((inputs, fea2), dim=1)                                            #b,132,32,64
        #(b,132,32,64) (b,c,t,n) 132*2-4

        #fea2_a = self.apesattention(fea2)    #[b,c,t,n]
        #fea2_connect = torch.cat((inputs,fea2_attention[0][:,4:]+fea2[:,4:]),dim=1)

        # stage 3: inter-frame, middle, applying lstm in this stage
        in_dims = fea2.shape[1] * 2 - 4
        pts_num //= self.downsample[1]

        #fea2 = self.xlstm_same_first(fea2_connect.permute(0,2,1,3))
        #fea2 = self.xlstm_same_second(fea2_a.permute(0,2,1,3))
        fea2 = self.xlstm_same_second(fea2.permute(0,2,1,3))
        fea2 = fea2[0][0].squeeze(-1).permute(0,2,1,3)
        ret_group_array3 = self.group.st_group_points(fea2, 3, [0, 1, 2], self.knn[2], 3)
        ret_array3, inputs, _ = self.select_ind(ret_group_array3, inputs,
                                                batchsize, in_dims, timestep, pts_num)
        fea3 = self.pool3(self.stage3(ret_array3)).view(batchsize, -1, timestep, pts_num)
        fea3 = torch.cat((inputs, fea3), dim=1)

        #fea3_attention = self.selfa3(fea3)    #[b,c,t,n]

        # stage 4: inter-frame, late
        in_dims = fea3.shape[1] * 2 - 4
        pts_num //= self.downsample[2]
        fea3 = self.xlstm_same_thired(fea3.permute(0,2,1,3))
        fea3 = fea3[0][0].squeeze(-1).permute(0,2,1,3)
        ret_group_array4 = self.group.st_group_points(fea3, 3, [0, 1, 2], self.knn[3], 3)
        ret_array4, inputs, _ = self.select_ind(ret_group_array4, inputs,
                                                batchsize, in_dims, timestep, pts_num)
        fea4 = self.pool4(self.stage4(ret_array4)).view(batchsize, -1, timestep, pts_num)

        output = self.stage5(fea4)
        output = self.pool5(output)
        output = self.global_bn(output)
        output = self.stage6(output)
        return output.view(batchsize, self.num_classes)
    
    def forward(self, inputs):
        # B * T * N * D,  e.g. 16 * 32 * 512 * 4
        inputs = inputs.permute(0, 3, 1, 2)
        if self.training:
            inputs = inputs[:, :, :, torch.randperm(inputs.shape[3])[:self.pts_size]]
        else:
            inputs = inputs[:, :, :, ::inputs.shape[3] // self.pts_size]# B * (4 + others) * 32 * 128
        inputs = inputs[:, :4]# B * 4 * 32 * 128
        batchsize, in_dims, timestep, pts_num = inputs.shape

        # stage 1: intra-frame
        ret_array1 = self.group.group_points(distance_dim=[0, 1, 2], array1=inputs, array2=inputs, knn=self.knn[0],
                                             dim=3)
        # B * 4 * 32 * 128 * 16
        ret_array1 = ret_array1.contiguous().view(batchsize, in_dims, timestep * pts_num, -1)
        # B * 4 * 4096 * 16
        fea1 = self.pool1(self.stage1(ret_array1)).view(batchsize, -1, timestep, pts_num)
        # B * 64 * 32 * 128
        fea1 = torch.cat((inputs, fea1), dim=1)
        # B * 68 * 32 * 128                               相当于将近邻16个点的坐标接在原始点坐标后面（但每个点使用四个点坐标，其中前三个为xyz，最后一个原始为0）
        # stage 2: inter-frame, early
        

        #in_dims = fea1.shape[1] * 2 - 4  #132
        in_dims = fea1.shape[1] * 2 - 4  #132
        pts_num //= self.downsample[0]   #64


        fea1_lstm = self.xlstm_same_first(fea1.permute(0,2,1,3))
        fea1_lstm = fea1_lstm[0][0].squeeze(-1).permute(0,2,1,3)                         #[8,68,32,128]
        ret_group_array2 = self.group.st_group_points(fea1_lstm, 3, [0, 1, 2], self.knn[1], 3)  # b * 132 * 32 * 128 * knn(24)
        ret_array2, inputs, _ = self.select_ind(ret_group_array2, inputs,  #b,132,2048,knn(24)//b,4,32,64//b,32,64
                                                batchsize, in_dims, timestep, pts_num)     # b , 132 ,32 , 64
        fea2 = self.pool2(self.stage2(ret_array2)).view(batchsize, -1, timestep, pts_num)  #b,128,32,64
        fea2 = torch.cat((inputs, fea2), dim=1)                                            #b,132,32,64
        #(b,132,32,64) (b,c,t,n) 132*2-4

        # stage 3: inter-frame, middle, applying lstm in this stage
        in_dims = fea2.shape[1] * 2 - 4
        pts_num //= self.downsample[1]

        fea2_lstm = self.xlstm_same_second(fea2.permute(0,2,1,3))
        fea2_lstm = fea2_lstm[0][0].squeeze(-1).permute(0,2,1,3)
        ret_group_array3 = self.group.st_group_points(fea2_lstm, 3, [0, 1, 2], self.knn[2], 3)
        ret_array3, inputs, _ = self.select_ind(ret_group_array3, inputs,
                                                batchsize, in_dims, timestep, pts_num)
        fea3 = self.pool3(self.stage3(ret_array3)).view(batchsize, -1, timestep, pts_num)
        fea3 = torch.cat((inputs, fea3), dim=1)

        #fea3_attention = self.selfa3(fea3)    #[b,c,t,n]

        # stage 4: inter-frame, late
        in_dims = fea3.shape[1] * 2 - 4
        pts_num //= self.downsample[2]
        fea3_lstm = self.xlstm_same_thired(fea3.permute(0,2,1,3))
        fea3_lstm = fea3_lstm[0][0].squeeze(-1).permute(0,2,1,3)
        ret_group_array4 = self.group.st_group_points(fea3_lstm, 3, [0, 1, 2], self.knn[3], 3)
        ret_array4, inputs, _ = self.select_ind(ret_group_array4, inputs,
                                                batchsize, in_dims, timestep, pts_num)
        fea4 = self.pool4(self.stage4(ret_array4)).view(batchsize, -1, timestep, pts_num)


        output = self.stage5(fea4)
        output = self.pool5(output)

        #output_fea1 = self.stage6_fea1(fea1)
        #output_fea1 = self.pool5(output_fea1)
        #output_fea1 = self.stage6_fea1(fea1)

        #output_fea1_lstm = self.stage6_fea1_lstm(fea1_lstm)
        #output_fea1_lstm = self.pool5(output_fea1_lstm)
        #output_fea1_lstm = self.stage6_fea1(fea1_lstm)

        #output_fea2 = self.stage6_fea2(fea2)
        #output_fea2 = self.pool5(output_fea2)
        #output_fea2 = self.stage6_fea2(fea2)

        #output_fea2_lstm = self.stage6_fea2_lstm(fea2_lstm)
        #output_fea2_lstm = self.pool5(output_fea2_lstm)
        #output_fea2_lstm = self.stage6_fea2(fea2_lstm)

        #output_fea3 = self.stage6_fea3(fea3)
        #output_fea3 = self.pool5(output_fea3)
        #output_fea3 = self.stage6_fea3(fea3)

        #output_fea3_lstm = self.stage6_fea3_lstm(fea3_lstm)
        #output_fea3_lstm = self.pool5(output_fea3_lstm)
        #output_fea3_lstm = self.stage6_fea3(fea3_lstm)
        end1 = self.endding_first(fea1, fea1_lstm, 68)
        #output_fea1 = self.stage6_fea1(end1)

        end2 = self.endding_second(fea2, fea2_lstm, 132)


        end3 = self.endding_thired(fea3, fea3_lstm, 260)


        #output_end = torch.cat((output , end1),dim=1)
        output_end = torch.cat((output , end1, end2, end3),dim=1)
        output = self.global_bn(output_end)
        output = self.stage6_end(output)
        #output = self.global_bn(output)
        #output = self.stage6(output) - 0.02 * output_fea1 - 0.04 * output_fea2 - 0.08 * output_fea3 \
        #                             + 0.02 * output_fea1_lstm + 0.04 * output_fea2_lstm + 0.08 * output_fea3_lstm
        return output.view(batchsize, self.num_classes)
    


    #def forward(self, inputs):
        # B * T * N * D,  e.g. 16 * 32 * 512 * 4
        inputs = inputs.permute(0, 3, 1, 2)
        if self.training:
            inputs = inputs[:, :, :, torch.randperm(inputs.shape[3])[:self.pts_size]]
        else:
            inputs = inputs[:, :, :, ::inputs.shape[3] // self.pts_size]# B * (4 + others) * 32 * 128
        inputs = inputs[:, :4]# B * 4 * 32 * 128
        batchsize, in_dims, timestep, pts_num = inputs.shape

        # stage 1: intra-frame
        ret_array1 = self.group.group_points(distance_dim=[0, 1, 2], array1=inputs, array2=inputs, knn=self.knn[0],
                                             dim=3)
        # B * 4 * 32 * 128 * 16
        ret_array1 = ret_array1.contiguous().view(batchsize, in_dims, timestep * pts_num, -1)
        # B * 4 * 4096 * 16
        fea_block1 = self.pool1(self.stage1(ret_array1)).view(batchsize, -1, timestep, pts_num)
        # B * 64 * 32 * 128
        fea_block1 = torch.cat((inputs, fea_block1), dim=1)
        # B * 68 * 32 * 128                               相当于将近邻16个点的坐标接在原始点坐标后面（但每个点使用四个点坐标，其中前三个为xyz，最后一个原始为0）

        # stage 2: inter-frame, early
        in_dims = fea_block1.shape[1] * 2 - 4  #132
        pts_num //= self.downsample[0]   #64
        fea1 = self.xlstm_same_first(fea_block1.permute(0,2,1,3))
        fea1 = fea1[0][0].squeeze(-1).permute(0,2,1,3)
        fea1[:,4:] = fea1[:,4:] + fea_block1[:,4:]
        ret_group_array2 = self.group.st_group_points(fea1, 3, [0, 1, 2], self.knn[1], 3)  # b * 132 * 32 * 128 * knn(24)
        ret_array2, inputs, _ = self.select_ind(ret_group_array2, inputs,  #b,132,2048,knn(24)//b,4,32,64//b,32,64
                                                batchsize, in_dims, timestep, pts_num)     # b , 132 ,32 , 64
        fea_block2 = self.pool2(self.stage2(ret_array2)).view(batchsize, -1, timestep, pts_num)  #b,128,32,64
        fea_block2 = torch.cat((inputs, fea_block2), dim=1)                                            #b,132,32,64
        #(b,132,32,64) (b,c,t,n) 132*2-4


        # stage 3: inter-frame, middle, applying lstm in this stage
        in_dims = fea_block2.shape[1] * 2 - 4
        pts_num //= self.downsample[1]
        fea2 = self.xlstm_same_second(fea_block2.permute(0,2,1,3))
        fea2 = fea2[0][0].squeeze(-1).permute(0,2,1,3)
        fea2[:,4:] = fea2[:,4:] + fea_block2[:,4:]       
        ret_group_array3 = self.group.st_group_points(fea2, 3, [0, 1, 2], self.knn[2], 3)
        ret_array3, inputs, _ = self.select_ind(ret_group_array3, inputs,
                                                batchsize, in_dims, timestep, pts_num)
        fea_block3 = self.pool3(self.stage3(ret_array3)).view(batchsize, -1, timestep, pts_num)
        fea_block3 = torch.cat((inputs, fea_block3), dim=1)


        # stage 4: inter-frame, late
        in_dims = fea_block3.shape[1] * 2 - 4
        pts_num //= self.downsample[2]
        fea3 = self.xlstm_same_thired(fea_block3.permute(0,2,1,3))
        fea3 = fea3[0][0].squeeze(-1).permute(0,2,1,3)
        #fea3[:,4:] = fea3[:,4:] + fea_block3[:,4:]
        ret_group_array4 = self.group.st_group_points(fea3, 3, [0, 1, 2], self.knn[3], 3)
        ret_array4, inputs, _ = self.select_ind(ret_group_array4, inputs,
                                                batchsize, in_dims, timestep, pts_num)
        fea4 = self.pool4(self.stage4(ret_array4)).view(batchsize, -1, timestep, pts_num)

        output = self.stage5(fea4)
        output = self.pool5(output)
        output = self.global_bn(output)
        output = self.stage6(output)
        return output.view(batchsize, self.num_classes)

    #def forward(self, inputs):
        # B * T * N * D,  e.g. 16 * 32 * 512 * 4   纯lstm结构
        inputs = inputs.permute(0, 3, 1, 2)
        if self.training:
            inputs = inputs[:, :, :, torch.randperm(inputs.shape[3])[:self.pts_size]]
        else:
            inputs = inputs[:, :, :, ::inputs.shape[3] // self.pts_size]# B * (4 + others) * 32 * 128
        inputs = inputs[:, :4]# B * 4 * 32 * 128
        batchsize, in_dims, timestep, pts_num = inputs.shape

        # stage 1: intra-frame
        ret_array1 = self.group.group_points(distance_dim=[0, 1, 2], array1=inputs, array2=inputs, knn=self.knn[0],
                                             dim=3)
        # B * 4 * 32 * 128 * 16
        ret_array1 = ret_array1.contiguous().view(batchsize, in_dims, timestep * pts_num, -1)
        # B * 4 * 4096 * 16
        fea1 = self.pool1(self.stage1(ret_array1)).view(batchsize, -1, timestep, pts_num)
        # B * 64 * 32 * 128
        fea1 = torch.cat((inputs, fea1), dim=1)
        # B * 68 * 32 * 128                               相当于将近邻16个点的坐标接在原始点坐标后面（但每个点使用四个点坐标，其中前三个为xyz，最后一个原始为0）

        # stage 2: inter-frame, early
        in_dims = fea1.shape[1] * 2 - 4 #260
        pts_num //= self.downsample[0]  #32
        output = self.xlstm_same_first(fea1.permute(0, 2, 1, 3))   #output[0][0]b,32,260,64,1
        fea2 = output[0][0].squeeze(-1).permute(0, 2, 1, 3)  #b,260,32,64
        ret_group_array2 = self.group.st_group_points(fea1, 3, [0, 1, 2], self.knn[1], 3)  #b,260,32,64,knn(48)
        ret_array2, inputs, ind = self.select_ind(ret_group_array2, inputs, #b,260,1024,knn(48)//b,4,32,32//4,32,32
                                                  batchsize, in_dims, timestep, pts_num)
        fea2 = fea2.gather(-1, ind.unsqueeze(1).expand(-1, fea2.shape[1], -1, -1))  #b,260,32,32

        # stage 3: inter-frame, middle, applying lstm in this stage
        in_dims = fea2.shape[1] * 2 - 4 #260
        pts_num //= self.downsample[1]  #32
        output = self.xlstm_same_second(fea2.permute(0, 2, 1, 3))   #output[0][0]b,32,260,64,1
        fea3 = output[0][0].squeeze(-1).permute(0, 2, 1, 3)  #b,260,32,64
        ret_group_array3 = self.group.st_group_points(fea2, 3, [0, 1, 2], self.knn[2], 3)  #b,260,32,64,knn(48)
        ret_array3, inputs, ind = self.select_ind(ret_group_array3, inputs, #b,260,1024,knn(48)//b,4,32,32//4,32,32
                                                  batchsize, in_dims, timestep, pts_num)
        fea3 = fea3.gather(-1, ind.unsqueeze(1).expand(-1, fea3.shape[1], -1, -1))  #b,260,32,32

        # stage 4: inter-frame, late
        in_dims = fea3.shape[1] * 2 - 4 #260
        pts_num //= self.downsample[2]  #32
        output = self.xlstm_same_thired(fea3.permute(0, 2, 1, 3))   #output[0][0]b,32,260,64,1
        fea4 = output[0][0].squeeze(-1).permute(0, 2, 1, 3)  #b,260,32,64
        ret_group_array4 = self.group.st_group_points(fea3, 3, [0, 1, 2], self.knn[2], 3)  #b,260,32,64,knn(48)
        ret_array4, inputs, ind = self.select_ind(ret_group_array4, inputs, #b,260,1024,knn(48)//b,4,32,32//4,32,32
                                                  batchsize, in_dims, timestep, pts_num)
        fea4 = fea4.gather(-1, ind.unsqueeze(1).expand(-1, fea4.shape[1], -1, -1))  #b,260,32,32

        output = self.stage5(fea4)        #b,1024,32,16
        output = self.pool5(output)       #b,1024,1,1
        output = self.global_bn(output)   #b,1024,1,1
        output = self.stage6(output)      #b,class_num,1,1
        return output.view(batchsize, self.num_classes)


    #def forward(self, inputs):
        # B * T * N * D,  e.g. 16 * 32 * 512 * 4
        #inputs = inputs[:, :, :, :4]
        #if self.training:
        #    #inputs = inputs[:, :, :, torch.randperm(inputs.shape[3])[:self.pts_size]]
        #    inputs = self.group.fps_dim(inputs, self.pts_size, dim = 1)
        #else:
        #    inputs = inputs[:, : ,::inputs.shape[2] // self.pts_size]
        #inputs = inputs.permute(0, 3, 1, 2)

        inputs = inputs.permute(0, 3, 1, 2)
        if self.training:
            inputs = inputs[:, :, :, torch.randperm(inputs.shape[3])[:self.pts_size]]
        else:
            inputs = inputs[:, :, :, ::inputs.shape[3] // self.pts_size]# B * (4 + others) * 32 * 128
        inputs = inputs[:, :4]# B * 4 * 32 * 128
        batchsize, in_dims, timestep, pts_num = inputs.shape

        # stage 1: intra-frame
        ret_array1 = self.group.group_points(distance_dim=[0, 1, 2], array1=inputs, array2=inputs, knn=self.knn[0],
                                             dim=3)
        # B * 4 * 32 * 128 * 16
        ret_array1 = ret_array1.contiguous().view(batchsize, in_dims, timestep * pts_num, -1)
        # B * 4 * 4096 * 16
        fea1 = self.pool1(self.stage1(ret_array1)).view(batchsize, -1, timestep, pts_num)
        # B * 64 * 32 * 128
        fea1 = torch.cat((inputs, fea1), dim=1)
        # B * 68 * 32 * 128                               相当于将近邻16个点的坐标接在原始点坐标后面（但每个点使用四个点坐标，其中前三个为xyz，最后一个原始为0）

        # stage 2: inter-frame, early
        in_dims = fea1.shape[1] * 2 - 4  #132
        pts_num //= self.downsample[0]   #64

        ret_group_array2 = self.group.st_group_points(fea1, 3, [0, 1, 2], self.knn[1], 3)  # b * 132 * 32 * 128 * knn(24) 
        ret_array2, inputs, _ = self.select_ind(ret_group_array2, inputs,  #b,132,2048,knn(24)//b,4,32,64//b,32,64
                                                batchsize, in_dims, timestep, pts_num)     # b , 132 ,32 , 64
        fea2 = self.pool2(self.stage2(ret_array2)).view(batchsize, -1, timestep, pts_num)  #b,128,32,64
        fea2 = torch.cat((inputs, fea2), dim=1)                                            #b,132,32,64
        #(b,132,32,64) (b,c,t,n) 132*2-4

        # stage 3: inter-frame, middle, applying lstm in this stage
        in_dims = fea2.shape[1] * 2 - 4 #260
        pts_num //= self.downsample[1]  #32
        output = self.lstm(fea2.permute(0, 2, 1, 3))   #output[0][0]b,32,260,64,1
        fea3 = output[0][0].squeeze(-1).permute(0, 2, 1, 3)  #b,260,32,64
        ret_group_array3 = self.group.st_group_points(fea2, 3, [0, 1, 2], self.knn[2], 3)  #b,260,32,64,knn(48)
        ret_array3, inputs, ind = self.select_ind(ret_group_array3, inputs, #b,260,1024,knn(48)//b,4,32,32//4,32,32
                                                  batchsize, in_dims, timestep, pts_num)
        fea3 = fea3.gather(-1, ind.unsqueeze(1).expand(-1, fea3.shape[1], -1, -1))  #b,260,32,32

        # stage 4: inter-frame, late
        in_dims = fea3.shape[1] * 2 - 4  #516
        pts_num //= self.downsample[2]   #16
        ret_group_array4 = self.group.st_group_points(fea3, 3, [0, 1, 2], self.knn[3], 3)  #b,516,32,32,knn(12)
        ret_array4, inputs, _ = self.select_ind(ret_group_array4, inputs,  #b,516,512,knn(12)//b,4,32,16//b,32,16
                                                batchsize, in_dims, timestep, pts_num)
        fea4 = self.pool4(self.stage4(ret_array4)).view(batchsize, -1, timestep, pts_num) #b,512,32,16

        output = self.stage5(fea4)        #b,1024,32,16
        output = self.pool5(output)       #b,1024,1,1
        output = self.global_bn(output)   #b,1024,1,1
        output = self.stage6(output)      #b,class_num,1,1
        return output.view(batchsize, self.num_classes)
    

    

    def select_ind(self, group_array, inputs, batchsize, in_dim, timestep, pts_num):
        ind = self.weight_select(group_array, pts_num)                                     
        ret_group_array = group_array.gather(-2, ind.unsqueeze(1).unsqueeze(-1).
                                             expand(-1, group_array.shape[1], -1, -1,
                                                    group_array.shape[-1]))
        ret_group_array = ret_group_array.view(batchsize, in_dim, timestep * pts_num, -1)  
        inputs = inputs.gather(-1, ind.unsqueeze(1).expand(-1, inputs.shape[1], -1, -1))  
        return ret_group_array, inputs, ind                                                
    
    
    @staticmethod         
    def weight_select(position, topk):
        # select points with larger ranges
        weights = torch.max(torch.sum(position[:, :3] ** 2, dim=1), dim=-1)[0]  
        dists, idx = torch.topk(weights, topk, -1, largest=True, sorted=False)
        return idx


if __name__ == '__main__':
    pass

import pdb
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils

class MLPBlock(nn.Module):             #一般的mlp，使用1*1或1卷积完成，使用：MLPBlock([4, 32, 64], 2， with_bn=False)最后一个默认false
    def __init__(self, out_channel, dimension, with_bn=True):                                     #ture表示在mlp之后进行归一化和relu
        super(MLPBlock, self).__init__()
        self.layer_list = []
        if dimension == 1:
            for idx, channels in enumerate(out_channel[:-1]):
                if with_bn:
                    self.layer_list.append(
                        nn.Sequential(
                            nn.Conv1d(channels, out_channel[idx + 1], kernel_size=1),
                            nn.BatchNorm1d(out_channel[idx]),
                            nn.ReLU(inplace=True),
                        )
                    )
                else:
                    self.layer_list.append(
                        nn.Sequential(
                            nn.Conv1d(channels, out_channel[idx + 1], kernel_size=1),
                        )
                    )
        elif dimension == 2:
            for idx, channels in enumerate(out_channel[:-1]):
                if with_bn:
                    self.layer_list.append(
                        nn.Sequential(
                            nn.Conv2d(channels, out_channel[idx + 1], kernel_size=(1, 1)),
                            nn.BatchNorm2d(out_channel[idx + 1]),
                            nn.ReLU(inplace=True),
                        )
                    )
                else:
                    self.layer_list.append(
                        nn.Sequential(
                            nn.Conv2d(channels, out_channel[idx + 1], kernel_size=(1, 1)),
                        )
                    )
        self.layer_list = nn.ModuleList(self.layer_list)

    def forward(self, output):
        for layer in self.layer_list:
            output = layer(output)
        return output


class MotionBlock(nn.Module):         #一般的mlp，使用1*1或1卷积完成，使用：MotionBlock([128, 128, ], 2, 4)， mlp时候归一化和relu
    def __init__(self, out_channel, dimension, embedding_dim):                                # 4 表示对前几个维度进行额外的操作
        super(MotionBlock, self).__init__()
        self.layer_list = []
        if dimension == 1:
            self.layer_list.append(
                nn.Sequential(
                    nn.Conv1d(embedding_dim, out_channel[-1], kernel_size=1),
                    nn.BatchNorm1d(out_channel[-1]),
                    nn.ReLU(inplace=True),
                )
            )
            for idx, channels in enumerate(out_channel[:-1]):
                self.layer_list.append(
                    nn.Sequential(
                        nn.Conv1d(channels, out_channel[idx + 1], kernel_size=1),
                        nn.BatchNorm1d(out_channel[idx + 1]),
                        nn.ReLU(inplace=True),
                    )
                )
        elif dimension == 2:
            self.layer_list.append(
                nn.Sequential(
                    nn.Conv2d(embedding_dim, out_channel[-1], kernel_size=(1, 1)),
                    nn.BatchNorm2d(out_channel[-1]),
                    nn.ReLU(inplace=True),
                )
            )
            for idx, channels in enumerate(out_channel[:-1]):
                self.layer_list.append(
                    nn.Sequential(
                        nn.Conv2d(channels, out_channel[idx + 1], kernel_size=(1, 1)),
                        nn.BatchNorm2d(out_channel[idx + 1]),
                        nn.ReLU(inplace=True),
                    )
                )
        self.layer_list = nn.ModuleList(self.layer_list)

    def forward(self, output):
        position_embedding = self.layer_list[0](output[:, :4])
        feature_embedding = output[:, 4:]
        for layer in self.layer_list[1:]:
            feature_embedding = layer(feature_embedding)
        return position_embedding * feature_embedding


class GroupOperation(object):
    def __init__(self):
        pass

    def group_points(self, distance_dim, array1, array2, knn, dim):                      #聚合本帧的近邻点
        matrix, a1, a2 = self.array_distance(array1, array2, distance_dim, dim)
        dists, inputs_idx = torch.topk(matrix, knn, -1, largest=False, sorted=True)
        neighbor = a2.gather(-1, inputs_idx.unsqueeze(1).expand(dists.shape[:1] + (a2.shape[1],) + dists.shape[1:]))
        offsets = array1.unsqueeze(dim + 1) - neighbor
        offsets[:, :3] /= torch.sum(offsets[:, :3] ** 2, dim=1).unsqueeze(1) ** 0.5 + 1e-8
        return offsets

    def st_group_points(self, array, interval, distance_dim, knn, dim):     #选择每个点前，此，后帧中每帧中最近的几个点
        batchsize, channels, timestep, num_pts = array.shape
        if interval // 2 > 0:                                         #类似与卷积边缘填充  interval:前后一共多少帧
            array_padded = torch.cat((array[:, :, 0].unsqueeze(2).expand(-1, -1, interval // 2, -1),
                                      array,
                                      array[:, :, -1].unsqueeze(2).expand(-1, -1, interval // 2, -1)
                                      ), dim=2)
        else:
            array_padded = array
        neighbor_points = torch.zeros(batchsize, channels, timestep, num_pts * interval).to(array.device)
        for i in range(timestep):                                    #将每帧前后帧的点都叠加到一帧中
            neighbor_points[:, :, i] = array_padded[:, :, i:i + interval].view(batchsize, channels, -1)
        matrix, a1, a2 = self.array_distance(array, neighbor_points, distance_dim, dim)
        dists, inputs_idx = torch.topk(matrix, knn, -1, largest=False, sorted=True)  #最后一个维度做knn b * n * t * 1
        neighbor = a2.gather(-1, inputs_idx.unsqueeze(1).
                             expand(dists.shape[:1] + (a2.shape[1],) + dists.shape[1:]))
        array = array.unsqueeze(-1).expand_as(neighbor)
        ret_features = torch.cat((array[:, :4] - neighbor[:, :4], array[:, 4:], neighbor[:, 4:]), dim=1)
        # ret_features = torch.cat((array[:, :4] - neighbor[:, :4], neighbor[:, 4:]), dim=1)
        return ret_features                                            #输出结果为原始点的坐标[前四个值]与近邻点的差，拼接原始点与近邻点的剩余特征[b,c*2-4,t,n1采样,knn]

    def array_distance(self, array1, array2, dist, dim):
        # return array1.shape[-1] * array2.shape[-1] matrix
        distance_mat = array1.unsqueeze(dim + 1)[:, dist] - array2.unsqueeze(dim)[:, dist]
        mat_shape = distance_mat.shape
        mat_shape = mat_shape[:1] + (array1.shape[1],) + mat_shape[2:]
        array1 = array1.unsqueeze(dim + 1).expand(mat_shape)         #已经变成array2的形状力，被扩展
        array2 = array2.unsqueeze(dim).expand(mat_shape)
        distance_mat = torch.sqrt((distance_mat ** 2).sum(1))        #两个对应的坐标差平方（每帧，每点）
        return distance_mat, array1, array2                          #输出结果为[b,t,n1,n2]

    def fps(self, data, number):
        fps_idx = pointnet2_utils.furthest_point_sample(data, number)
        fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
        return fps_data


    def fps_dim(self, data, pts_num, dim): #dim_mun选取的是时间维度，由于fps只能处理（b,n,c）,要保证去除时间维度后，为此形状
        collect = []
        for i in range(0, data.shape[dim]):
            collect.append(self.fps(data[:,:,i].contiguous(),pts_num))
        output = torch.stack(collect, dim=dim)
        return output
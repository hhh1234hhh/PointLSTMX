import pdb
import torch
import numpy as np
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import functional as F   
import math

#input 和 output  [b,c,t,n]
class AttentionCell(nn.Module):
    def __init__(self, pts_num, in_channels, qk_dim, v_dim, bias):
        super(AttentionCell,self).__init__()
        self.bias = bias
        self.pts_num = pts_num
        self.in_channels = in_channels
        self.qk_dim = qk_dim
        self.v_dim = v_dim

        self.conv_QK = nn.Conv2d(in_channels=self.in_channels,
                              out_channels=2 * self.qk_dim,
                              kernel_size=(1, 1),
                              bias=self.bias)
        
        self.conv_V = nn.Conv2d(in_channels=self.in_channels,
                              out_channels=self.v_dim,
                              kernel_size=(1, 1),
                              bias=self.bias)
    
    def forward(self, input_tensor, qk_dim):  #[b,c,n,knn]
        combine = self.conv_QK(input_tensor)
        combine_q, combine_k = torch.split(combine, self.qk_dim,dim=1)
        energy = torch.einsum('bcnq,bcnk->bnqk', combine_q, combine_k)   #[b,n,kqn,kvn]
        attention = torch.softmax(energy / (qk_dim ** (1 / 2)), dim=3)
        combine_v = self.conv_V(input_tensor)         #[b,c,n,k]

        attention_first = attention[:,:,:,0].squeeze(-1)        #[b,n,k]
        out = torch.einsum('bnk,bcnk->bnkc', attention_first,combine_v)   #[b,]
        output = torch.sum(out,dim=2)   
             
        #out = torch.einsum('bnkq,bcnk->bnkqc', attention,combine_v)   #[b,]
        #output, indices = torch.max( torch.sum(out,dim=2), dim=2)
        return output      #[b,n,c]
    
class SelfA(nn.Module):
    def __init__(self, pts_num, in_channels, qk_dim, v_dim, num_layers, topk=16, bias=True, return_all_layers=False):
        super(SelfA, self).__init__()

        self.bias = bias
        self.topk = topk
        self.pts_num = pts_num
        self.in_channels = in_channels
        self.v_dim = v_dim
        self.qk_dim = qk_dim
        self.num_layers = num_layers
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cell_list.append(AttentionCell(pts_num=self.pts_num,
                                           in_channels=self.in_channels,
                                           qk_dim=self.qk_dim,
                                           v_dim=self.v_dim,
                                           bias=self.bias))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor):      #[8,68,32,128]
        st_group = self.st_group_points(input_tensor, 3, [0, 1, 2], self.topk, 3)    #[8,132,32,128,16]
        #input_tensor [b,c,t,n]->[b,t,n,c]
        input_tensor = input_tensor.permute(0,2,3,1)    #[8,32,128,68]
        seq_len = input_tensor.shape[1]                 #32
        position = input_tensor[:, :, :, :4]            #[8,32,128,4]
        #[b,c,t,n,k]

        layer_output = []
        for layer_idx in range(self.num_layers):
            #h = hidden_state[layer_idx]
            output_time = []
            for t in range(seq_len):
                output_one = st_group[:,:,t].squeeze(-1)   #[8,132，128，16]
                out_attention = self.cell_list[layer_idx](
                    output_one.clone(),
                    qk_dim = self.qk_dim)           #[b,n,c]
                output_time.append(out_attention)
            layer = torch.cat((position,torch.stack(output_time,dim=1)),dim=3).permute(0,3,1,2)   #stack=[b,t,n,c]->[b,c,t,n]
            layer_output.append(layer)
        
        return layer_output


    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    def group_points(self, array1, array2, dim, topk):
        dist, _, _ = self.array_distance(array1, array2, dim) 
        dists, idx = torch.topk(dist, topk, -1, largest=False, sorted=False)
        return idx

    def group_points(self, distance_dim, array1, array2, knn, dim):                      #聚合本帧的近邻点
        matrix, a1, a2 = self.array_distance(array1, array2, distance_dim, dim)
        dists, inputs_idx = torch.topk(matrix, knn, -1, largest=False, sorted=True)
        neighbor = a2.gather(-1, inputs_idx.unsqueeze(1).expand(dists.shape[:1] + (a2.shape[1],) + dists.shape[1:]))
        offsets = array1.unsqueeze(dim + 1) - neighbor
        offsets[:, :3] /= torch.sum(offsets[:, :3] ** 2, dim=1).unsqueeze(1) ** 0.5 + 1e-8
        return offsets

    def st_group_points(self, array, interval, distance_dim, knn, dim):     #选择每个点前，此，后帧中每帧中最近的几个点
        batchsize, channels, timestep, num_pts = array.shape
        if interval // 2 > 0:                                         #类似与卷积边缘填充
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

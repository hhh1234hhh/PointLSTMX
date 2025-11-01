import torch, math
from torch import nn
from einops import rearrange, repeat
#(b,c,t,n)->(b,c,t,n)
#输入维度=输出维度

class N2PAttention(nn.Module):
    def __init__(self,pts_num, in_channels, out_channels, knn):
        super(N2PAttention, self).__init__()
        self.heads = 4
        self.K = knn
        self.pts_num = pts_num
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.group_type = 'diff'
        self.q_conv = nn.Conv2d(self.in_channels, self.in_channels, 1, bias=False)
        self.k_conv = nn.Conv2d(self.in_channels, self.in_channels, 1, bias=False)
        self.v_conv = nn.Conv2d(self.in_channels, self.in_channels, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.ff = nn.Sequential(nn.Conv1d(self.in_channels, self.in_channels*2 + self.out_channels*2, 1, bias=False),
                                nn.LeakyReLU(0.2), 
                                nn.Conv1d(self.in_channels*2 + self.out_channels*2, self.out_channels, 1, bias=False))
        self.bn1 = nn.BatchNorm1d(self.out_channels)
        self.bn2 = nn.BatchNorm1d(self.out_channels)

    def forward(self, x_all):    #(b,c,t,n)
#       neighbors = ops.group(x, self.K, self.group_type)  # (B, C, N) -> (B, C, N, K)
        positoion = x_all[:,:4]
        seq_len = x_all.shape[2]
        neighbors_all = self.st_group_points(x_all, 3, [0, 1, 2], self.K, 3)  #[8,132,32,128,24]

        output_inner = []
        for t in range(seq_len):
            x = x_all[:,:,t]
            neighbors = neighbors_all[:,:,t]
            #neighbors = group(x, self.K, self.group_type)  # (B, C, N) -> (B, C, N, K)
            q = self.q_conv(rearrange(x, 'B C N -> B C N 1')).contiguous()  # (B, C, N) -> (B, C, N, 1)
            q = self.split_heads(q, self.heads)  # (B, C, N, 1) -> (B, H, N, 1, D)   C=H*D
            k = self.k_conv(neighbors)  # (B, C, N, K) -> (B, C, N, K)
            k = self.split_heads(k, self.heads)  # (B, C, N, K) -> (B, H, N, K, D)
            v = self.v_conv(neighbors)  # (B, C, N, K) -> (B, C, N, K)
            v = self.split_heads(v, self.heads)  # (B, C, N, K) -> (B, H, N, K, D)
            energy = q @ rearrange(k, 'B H N K D -> B H N D K').contiguous()  # (B, H, N, 1, D) @ (B, H, N, D, K) -> (B, H, N, 1, K)
            scale_factor = math.sqrt(q.shape[-1])
            attention = self.softmax(energy / scale_factor)  # (B, H, N, 1, K) -> (B, H, N, 1, K)
            tmp = rearrange(attention@v, 'B H N 1 D -> B (H D) N').contiguous()  # (B, H, N, 1, K) @ (B, H, N, K, D) -> (B, H, N, 1, D) -> (B, C=H*D, N)
            x = self.bn1(x + tmp)  # (B, C, N) + (B, C, N) -> (B, C, N)
            x = self.ff(x)  # (B, C, N) -> (B, C, N)
            #tmp = self.ff(x)  # (B, C, N) -> (B, C, N)
            #x = self.bn2(x + tmp)  # (B, C, N) + (B, C, N) -> (B, C, N)
            output_inner.append(x)
        output = torch.cat((positoion , torch.stack(output_inner,dim=2)),dim=1)
        return output

    @staticmethod
    def split_heads(x, heads):
        x = rearrange(x, 'B (H D) N K -> B H N K D', H=heads).contiguous()  # (B, C, N, K) -> (B, H, N, K, D)
        return x
    
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
        ret_features = torch.cat((array[:, :4] - neighbor[:, :4], array[:, 4:] - neighbor[:, 4:]), dim=1)
        #ret_features = torch.cat((array[:, :4] - neighbor[:, :4], neighbor[:, 4:]), dim=1)
        return ret_features
    
    def array_distance(self, array1, array2, dist, dim):
        # return array1.shape[-1] * array2.shape[-1] matrix
        distance_mat = array1.unsqueeze(dim + 1)[:, dist] - array2.unsqueeze(dim)[:, dist]
        mat_shape = distance_mat.shape
        mat_shape = mat_shape[:1] + (array1.shape[1],) + mat_shape[2:]
        array1 = array1.unsqueeze(dim + 1).expand(mat_shape)         #已经变成array2的形状力，被扩展
        array2 = array2.unsqueeze(dim).expand(mat_shape)
        distance_mat = torch.sqrt((distance_mat ** 2).sum(1))        #两个对应的坐标差平方（每帧，每点）
        return distance_mat, array1, array2                          #输出结果为[b,t,n1,n2]
    
def index_points(points, idx):
    """
    :param points: points.shape == (B, N, C)
    :param idx: idx.shape == (B, N, K)
    :return:indexed_points.shape == (B, N, K, C)
    """
    raw_shape = idx.shape
    idx = idx.reshape(raw_shape[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.shape[-1]))
    return res.view(*raw_shape, -1)


def knn(a, b, k):
    """
    :param a: a.shape == (B, N, C)
    :param b: b.shape == (B, M, C)
    :param k: int
    """
    inner = -2 * torch.matmul(a, b.transpose(2, 1))  # inner.shape == (B, N, M)
    aa = torch.sum(a**2, dim=2, keepdim=True)  # aa.shape == (B, N, 1)
    bb = torch.sum(b**2, dim=2, keepdim=True)  # bb.shape == (B, M, 1)
    pairwise_distance = -aa - inner - bb.transpose(2, 1)  # pairwise_distance.shape == (B, N, M)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # idx.shape == (B, N, K)
    return idx


def select_neighbors(pcd, K, neighbor_type):
    pcd = pcd.permute(0, 2, 1)  # pcd.shape == (B, N, C)
    idx = knn(pcd, pcd, K)  # idx.shape == (B, N, K)
    neighbors = index_points(pcd, idx)  # neighbors.shape == (B, N, K, C)
    if neighbor_type == 'neighbor':
        neighbors = neighbors.permute(0, 3, 1, 2)  # output.shape == (B, C, N, K)
    elif neighbor_type == 'diff':
        diff = neighbors - pcd[:, :, None, :]  # diff.shape == (B, N, K, C)
        neighbors = diff.permute(0, 3, 1, 2)  # output.shape == (B, C, N, K)
    else:
        raise ValueError(f'neighbor_type should be "neighbor" or "diff", but got {neighbor_type}')
    return neighbors


def group(pcd, K, group_type):
    if group_type == 'neighbor':
        neighbors = select_neighbors(pcd, K, 'neighbor')  # neighbors.shape == (B, C, N, K)
        output = neighbors  # output.shape == (B, C, N, K)
    elif group_type == 'diff':
        diff = select_neighbors(pcd, K, 'diff')  # diff.shape == (B, C, N, K)
        output = diff  # output.shape == (B, C, N, K)
    elif group_type == 'center_neighbor':
        neighbors = select_neighbors(pcd, K, 'neighbor')   # neighbors.shape == (B, C, N, K)
        output = torch.cat([pcd[:, :, :, None].repeat(1, 1, 1, K), neighbors], dim=1)  # output.shape == (B, 2C, N, K)
    elif group_type == 'center_diff':
        diff = select_neighbors(pcd, K, 'diff')  # diff.shape == (B, C, N, K)
        output = torch.cat([pcd[:, :, :, None].repeat(1, 1, 1, K), diff], dim=1)  # output.shape == (B, 2C, N, K)
    else:
        raise ValueError(f'group_type should be neighbor, diff, center_neighbor or center_diff, but got {group_type}')
    return output.contiguous()

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
        #ret_features = torch.cat((array[:, :4] - neighbor[:, :4], array[:, 4:], neighbor[:, 4:]), dim=1)
        ret_features = torch.cat((array[:, :4] - neighbor[:, :4], neighbor[:, 4:]), dim=1)
        return ret_features  

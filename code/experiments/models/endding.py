import torch, math
from torch import nn

#(b,c,t,n)->(b,c,t,n)
#输入维度=输出维度+4

class Endding(nn.Module):
    def __init__(self, in_channels):
        super(Endding, self).__init__()
        #self.heads = 4
        #self.K = knn
        #self.pts_num = pts_num
        self.in_channels = in_channels - 4
        #self.out_channels = out_channels

        self.maxpool_num = nn.AdaptiveMaxPool2d((None, 1))
        self.avgpool_num = nn.AdaptiveAvgPool2d((None, 1))
        self.all_conv = nn.Conv2d(self.in_channels * 2, self.in_channels, 1, bias=False)
        self.avg_conv = nn.Sequential(
                    nn.Conv2d(self.in_channels * 2, self.in_channels, 1, bias=False),
                    nn.ReLU(),
                    nn.Conv2d(self.in_channels , self.in_channels, 1, bias=False)
        )
        self.end_conv = nn.Conv2d(self.in_channels , self.in_channels, 1, bias=False)
        self.maxend = nn.AdaptiveMaxPool2d((1, 1))


    def forward(self, pointnet, pointlstm, in_channels):    #(b,c,t,n)
        #character_num = in_channels - 4
        character_num =  4

        maxpointnet = self.maxpool_num(pointnet[:, character_num:])   #b,c,t,1
        maxpointlstm = self.maxpool_num(pointlstm[:, character_num:])

        avgpointnet = self.avgpool_num(pointnet[:, character_num:])   #b,c,t,1
        avgpointlstm = self.avgpool_num(pointlstm[:, character_num:])

        connect_all = torch.cat((maxpointlstm + maxpointnet , avgpointlstm + avgpointnet) , dim=1) #b,2c,t,1
        connect_all = self.all_conv(connect_all)    #b,c,t,1
        connect_avg = torch.cat((avgpointlstm , avgpointnet) , dim=1)
        connect_avg = self.avg_conv(connect_avg)    #b,c,t,1

        connect = self.end_conv(connect_avg + connect_all)         #b,c,t,1
        weight = torch.sigmoid(connect)

        max_out = weight * maxpointlstm + (1 - weight) * maxpointnet + maxpointnet + maxpointlstm
        output = self.maxend(max_out)
        #connect_max = torch.cat((maxpointlstm , maxpointnet) , dim=1)

        return output
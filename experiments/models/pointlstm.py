import pdb
import torch
import numpy as np
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import functional as F   
import math

class PointLSTMCell(nn.Module):
    def __init__(self, pts_num, in_channels, hidden_dim, offset_dim, bias, project_factor):
        super(PointLSTMCell, self).__init__()
        self.bias = bias
        self.pts_num = pts_num
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.project_factor = project_factor
        self.offset_dim = offset_dim
        self.pool = nn.Sequential(nn.AdaptiveMaxPool2d((None, 1)))

        self.up_proj = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels,
                              out_channels=self.project_factor * self.in_channels,
                              kernel_size=(1, 1),
                              bias=self.bias), 
                nn.GELU(),
                nn.Conv2d(in_channels=self.project_factor * self.in_channels,
                              out_channels=self.in_channels,
                              kernel_size=(1, 1),
                              bias=self.bias)
                )
        self.activation = nn.GELU()
        self.conv = nn.Conv2d(in_channels=self.in_channels + self.offset_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=(1, 1),
                              bias=self.bias)
        self.conv_skip_connect_lstm = nn.Conv2d(in_channels=self.in_channels,
                              out_channels=self.hidden_dim,
                              kernel_size=(1, 1),
                              bias=self.bias)     
        self.conv_skip_connect_xlstm = nn.Conv2d(in_channels=self.hidden_dim + self.hidden_dim,
                              out_channels=self.hidden_dim,
                              kernel_size=(1, 1),
                              bias=self.bias)    
        self.conv_xs_ifog = nn.Conv2d(in_channels=self.hidden_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=(1, 1),
                              bias=self.bias)
        self.conv_xm_ifo = nn.Conv2d(in_channels=self.hidden_dim + self.hidden_dim,
                              out_channels=3 * self.hidden_dim,
                              kernel_size=(1, 1),
                              bias=self.bias)
        self.conv_xm_qkv = nn.Conv2d(in_channels=self.hidden_dim + self.hidden_dim,
                              out_channels=3 * self.hidden_dim,
                              kernel_size=(1, 1),
                              bias=self.bias)

        #mogrify
        self.conv_input_to_hidden = nn.Conv2d(in_channels=self.in_channels,
                              out_channels=self.hidden_dim + self.offset_dim,
                              kernel_size=(1, 1),
                              bias=self.bias)
        self.conv_hidden_to_input = nn.Conv2d(in_channels=self.hidden_dim + self.offset_dim,
                              out_channels=self.in_channels,
                              kernel_size=(1, 1),
                              bias=self.bias)

    def five_to_four_pool(self,input,dim):
        if dim == 1:
            tilde = input.permute(0, 2, 3, 4, 1)
        else:
            tilde = input.permute(0, 1, 3, 4, 2)
        cuda_out = []
        for i in range(0, tilde.shape[0]):
            cuda_out.append(self.pool(tilde[i]))
        next = torch.stack(cuda_out, dim=0).squeeze(-1)
        return next
    
    def match(self, match_state, forget_state, input_state):#xlstm stabilizer state
        match_stable = torch.max(torch.log(forget_state) + match_state,torch.log(input_state))
        input_stable = torch.sigmoid(torch.log(input_state) - match_stable)
        forget_stable = torch.sigmoid(torch.log(forget_state) + match_state + match_stable)
        return match_stable, forget_stable, input_stable
    
    def mogrify(self,xt,ht):
        for i in range(1,6): #for i in range(1,self.mog_iterations+1):
            if (i % 2 == 0):
                ht = (2*torch.sigmoid(torch.einsum('abcd,be->aecd', xt, self.Q))) * ht
            else:
                xt = (2*torch.sigmoid(torch.einsum('abcd,be->aecd', ht, self.R))) * xt
        return xt, ht
    
    def mogrify_conv(self,xt,ht):
        for i in range(1,6): #for i in range(1,self.mog_iterations+1):
            if (i % 2 == 0):
                ht = (2*torch.sigmoid(self.conv_input_to_hidden(xt))) * ht
            else:
                xt = (2*torch.sigmoid(self.conv_hidden_to_input(ht))) * xt
        return xt, ht
    

    #def forward(self, input_tensor, hidden_state, cell_state):
        #hidden_state[:, :4] -= input_tensor[:, :4]
        hidden_state[:, :4] =hidden_state[:, :4] - input_tensor[:, :4]
        #input_tensor, hidden_state = self.mogrify(input_tensor, hidden_state) #mogrification
        input_tensor, hidden_state = self.mogrify_conv(input_tensor, hidden_state) #mogrification
        combined = torch.cat([input_tensor, hidden_state], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * cell_state + i * g
        h_next = o * torch.tanh(c_next)
        return self.pool(h_next), self.pool(c_next)
    
    #mlstm
    #def forward(self, input_tensor, hidden_state, cell_state, normal_state, match_state):
        #input_tensor, hidden_state = self.mogrify_conv(input_tensor, hidden_state) #mogrification
        #hidden_state[:, :4] -= input_tensor[:, :4]
        hidden_state[:, :4] =hidden_state[:, :4] - input_tensor[:, :4]
        #input_tensor, hidden_state = self.mogrify(input_tensor, hidden_state) #mogrification
        #input_tensor, hidden_state = self.mogrify_conv(input_tensor, hidden_state) #mogrification
        combined = torch.cat([input_tensor, hidden_state], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1) #lstm

        i_lstm = torch.sigmoid(cc_i)
        f_lstm = torch.sigmoid(cc_f)
        o_lstm = torch.sigmoid(cc_o)
        g_lstm = torch.tanh(cc_g)

        c_lstm = f_lstm * cell_state + i_lstm * g_lstm
        h_lstm = o_lstm * torch.tanh(c_lstm)

        combined_xm = torch.cat([c_lstm, h_lstm], dim=1)
        combined_conv_ifo = self.conv_xm_ifo(combined_xm)
        combined_conv_qkv = self.conv_xm_qkv(combined_xm)

        cc_i_xm, cc_f_xm, cc_o_xm  = torch.split(combined_conv_ifo, self.hidden_dim, dim=1)  #b,256,64,16
        cc_q_xm, cc_k_xm, cc_v_xm  = torch.split(combined_conv_qkv, self.hidden_dim, dim=1)  #b,256,64,16

        i_xm = torch.sigmoid(cc_i_xm)
        f_xm = torch.sigmoid(cc_f_xm)
        o_xm = torch.sigmoid(cc_o_xm)
        cc_k_xm = (1 / math.sqrt(self.hidden_dim)) * cc_k_xm

        m_next, f_stable, i_stable = self.match(match_state, f_xm, i_xm)

        #c_next = f_xm * c_lstm + i_xm * torch.matmul(cc_q_xm.unsqueeze(2), cc_k_xm.unsqueeze(1)).squeeze(1)
        c_next = f_stable * c_lstm + i_stable * (cc_k_xm + cc_v_xm)
        n_next = f_stable * normal_state + i_stable * cc_k_xm
        max_nq = torch.max(torch.abs(n_next + cc_q_xm),torch.tensor(1.0))
        h_next = o_xm * cc_q_xm * c_next / max_nq

        return self.pool(h_next), self.pool(c_next), self.pool(n_next), self.pool(m_next)
    
    #slstm
    def forward(self, input_tensor, hidden_state, cell_state, normal_state, match_state):
        
        #lstm_skip_connect = self.conv_skip_connect_lstm(input_tensor)
        #hidden_state[:, :4] -= input_tensor[:, :4]
        hidden_state[:, :4] =hidden_state[:, :4] - input_tensor[:, :4]

        combined = torch.cat([input_tensor, hidden_state], dim=1)
        combined_conv = self.conv(combined)
        
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1) #lstm

        i_lstm = torch.sigmoid(cc_i)
        f_lstm = torch.sigmoid(cc_f)
        o_lstm = torch.sigmoid(cc_o)
        g_lstm = torch.tanh(cc_g)

        c_lstm = f_lstm * cell_state + i_lstm * g_lstm
        h_lstm = o_lstm * torch.tanh(c_lstm)

        combined_xs = torch.cat([c_lstm, h_lstm], dim=1)
        combined_conv_ifo = self.conv_xs_ifog(combined_xs)
        #xlstm_skip_connect = self.conv_skip_connect_xlstm(combined)
        cc_i_xs, cc_f_xs, cc_o_xs, cc_g_xs  = torch.split(combined_conv_ifo, self.hidden_dim, dim=1)  #b,256,64,16

        i_xs = torch.sigmoid(cc_i_xs)
        f_xs = torch.sigmoid(cc_f_xs)
        o_xs = torch.sigmoid(cc_o_xs)
        g_xs = torch.tanh(cc_g_xs)

        m_next, f_stable, i_stable = self.match(match_state, f_xs, i_xs)

        c_next = f_stable * c_lstm + i_stable * g_xs
        n_next = f_stable * normal_state + i_stable

        #h_next = o_xs * c_next / n_next + input_tensor
        #h_next = o_xs * c_next / n_next + lstm_skip_connect
        h_next = o_xs * c_next / n_next

        return self.pool(h_next), self.pool(c_next), self.pool(n_next), self.pool(m_next)
    
    
    def init_hidden(self, batch_size):
        return (torch.zeros(batch_size, self.hidden_dim, self.pts_num, 1).cuda(),
                torch.zeros(batch_size, self.hidden_dim, self.pts_num, 1).cuda(),
                torch.zeros(batch_size, self.hidden_dim, self.pts_num, 1).cuda(),
                torch.zeros(batch_size, self.hidden_dim, self.pts_num, 1).cuda())


class PointLSTM(nn.Module):
    def __init__(self, pts_num, in_channels, hidden_dim, offset_dim, num_layers, topk=16, project_factor=2,offsets=False,
                 batch_first=True, bias=True, return_all_layers=False):
        super(PointLSTM, self).__init__()
        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.bias = bias
        self.topk = topk
        self.project_factor = project_factor
        self.offsets = offsets
        self.pts_num = pts_num
        self.in_channels = in_channels
        self.offset_dim = offset_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_in_channels = self.in_channels if i == 0 else self.hidden_dim[i - 1] + 4
            cell_list.append(PointLSTMCell(pts_num=self.pts_num,
                                           in_channels=cur_in_channels,
                                           hidden_dim=self.hidden_dim[i],
                                           offset_dim=self.offset_dim,
                                           bias=self.bias,
                                           project_factor=self.project_factor))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        # batch, timestep, c, n (N points, M neighbor)
        if not self.batch_first:
            # (t, b, c, n) -> (b, t, c, n)
            input_tensor = input_tensor.permute(1, 0, 2, 3)
        #if hidden_state is not None:
            #raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list = []
        position = input_tensor[:, :, :4]
        if self.offsets:
            centroids = torch.mean(position[:, :, :3], dim=3)
            group_offsets = (centroids[:, :-1] - centroids[:, 1:])[:, :, :, None]
            group_ind = torch.cat(
                (
                    self.group_points(position[:, 0, :3], position[:, 0, :3], dim=2,
                                      topk=self.topk).unsqueeze(1),
                    self.group_points(position[:, 1:, :3] + group_offsets, position[:, :-1, :3],
                                      dim=3, topk=self.topk),
                ),
                dim=1
            )
        else:
            group_ind = torch.cat(
                (
                    self.group_points(position[:, 0, :3], position[:, 0, :3], dim=2,
                                      topk=self.topk).unsqueeze(1),
                    self.group_points(position[:, 1:, :3], position[:, :-1, :3],
                                      dim=3, topk=self.topk),
                ),
                dim=1
            )
        seq_len = input_tensor.shape[1]

        cur_layer_input = input_tensor.unsqueeze(-1)
        for layer_idx in range(self.num_layers):
            h, c ,n ,m= hidden_state[layer_idx]
            #h, c= hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                past = 0 if t == 0 else t - 1
                center_pts = cur_layer_input[:, t].expand(-1, -1, -1, self.topk)
                h_with_pos = torch.cat((position[:, past].unsqueeze(-1), h), dim=1)
                h_grouped = h_with_pos.squeeze(-1).unsqueeze(1).expand(-1, self.pts_num, -1, -1). \
                    gather(3, group_ind[:, t].unsqueeze(2).
                           expand(-1, -1, self.hidden_dim[layer_idx] + self.offset_dim, -1)) \
                    .permute(0, 2, 1, 3)
                c_grouped = c.squeeze(-1).unsqueeze(1).expand(-1, self.pts_num, -1, -1). \
                    gather(3, group_ind[:, t].unsqueeze(2).expand(-1, -1, self.hidden_dim[layer_idx], -1)) \
                    .permute(0, 2, 1, 3)
                n_grouped = n.squeeze(-1).unsqueeze(1).expand(-1, self.pts_num, -1, -1).gather(3, group_ind[:, t].unsqueeze(2).expand(-1, -1, self.hidden_dim[layer_idx], -1)).permute(0, 2, 1, 3)
                m_grouped = m.squeeze(-1).unsqueeze(1).expand(-1, self.pts_num, -1, -1).gather(3, group_ind[:, t].unsqueeze(2).expand(-1, -1, self.hidden_dim[layer_idx], -1)) .permute(0, 2, 1, 3)
                #h, c = self.cell_list[layer_idx](
                #h, c, n = self.cell_list[layer_idx](
                h, c, n, m = self.cell_list[layer_idx](
                    input_tensor=center_pts.clone(),
                    hidden_state=h_grouped.clone(),
                    cell_state=c_grouped.clone(),
                    normal_state = n_grouped.clone(),
                    match_state = m_grouped.clone()
                )
                output_inner.append(h)
            layer_output = torch.cat((position.unsqueeze(-1), torch.stack(output_inner, dim=1)), dim=2)
            cur_layer_input = layer_output
            layer_output_list.append(layer_output)
            last_state_list.append([h, c, n, m])
            #last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list, group_ind

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    def group_points(self, array1, array2, dim, topk):
        dist, _, _ = self.array_distance(array1, array2, dim)
        dists, idx = torch.topk(dist, topk, -1, largest=False, sorted=False)
        return idx

    @staticmethod
    def array_distance(array1, array2, dim):
        # return array1.shape[-1] * array2.shape[-1] matrix
        distance_mat = array1.unsqueeze(dim + 1) - array2.unsqueeze(dim)
        mat_shape = distance_mat.shape
        mat_shape = mat_shape[:1] + (array1.shape[1],) + mat_shape[2:]
        array1 = array1.unsqueeze(dim + 1).expand(mat_shape)
        array2 = array2.unsqueeze(dim).expand(mat_shape)
        distance_mat = torch.sqrt((distance_mat ** 2).sum(dim - 1))
        return distance_mat, array1, array2

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

    @staticmethod
    def tensor2numpy(tensor, name="test"):
        np.save(name, tensor.cpu().detach().numpy())

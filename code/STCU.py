import torch
import torch.nn as nn
import math

class SAM(nn.Module):
    def __init__(self, num_hidden, height, width):
        super(SAM, self).__init__()
        self.softmax = nn.Softmax(dim=2)
        self.d = num_hidden * height * width
    def forward(self, c_t, c_historys, h_historys):

        tau = c_historys.shape[2]
        weights_t = []
        for i in range(tau):
            weights_t.append(
                (c_historys[:, :, i:i + 1, :, :].squeeze(2) * c_t) / math.sqrt(self.d))
        weights_t = torch.stack(weights_t, dim=2)
        weights_t = self.softmax(weights_t)
        T_aug = h_historys * weights_t
        result = T_aug.sum(dim=2)

        return result, weights_t


class CSTIM(nn.Module):
    def __init__(self, num_hidden):
        super(CSTIM, self).__init__()
        self.conv_1 = nn.Conv3d(num_hidden, num_hidden, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_2 = nn.Sequential(
            nn.Conv3d(num_hidden, num_hidden, kernel_size=1, stride=1, padding=0, bias=False),
            nn.PReLU(),
            nn.Conv3d(num_hidden, num_hidden, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid())

    def forward(self, weights_t, f_historys):

        tem = torch.einsum('bcthw->bct', weights_t)
        tem = tem.unsqueeze(3)
        tem = tem.unsqueeze(4)
        att1 = self.conv_1(weights_t)
        att2 = self.conv_2(tem)
        weights_f = att1 * att2
        F_aug = f_historys * weights_f
        result = F_aug.sum(dim=2)
        return result


class STCU(nn.Module):
    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride):
        super(STCU, self).__init__()
        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        self.r = 4
        self.g_t = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding,
                      bias=False),
            nn.LayerNorm([num_hidden, height, width])
        )
        self.g_f = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding,
                      bias=False),
            nn.LayerNorm([num_hidden, height, width])
        )
        self.conv_z = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden * 5, kernel_size=filter_size, stride=stride, padding=self.padding,
                      bias=False),
            nn.LayerNorm([num_hidden * 5, height, width])
        )
        self.conv_h = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding,
                      bias=False),
            nn.LayerNorm([num_hidden * 3, height, width])
        )
        self.conv_f = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 2, kernel_size=filter_size, stride=stride, padding=self.padding,
                      bias=False),
            nn.LayerNorm([num_hidden * 2, height, width])
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding,
                      bias=False),
            nn.LayerNorm([num_hidden, height, width])
        )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0)
        self.sam = SAM(num_hidden, height, width)
        self.cstim = CSTIM(num_hidden)

    def forward(self, z_t, h_t, c_t, h_historys, c_historys, f_historys, f_t):
        z_concat = self.conv_z(z_t)
        h_concat = self.conv_h(h_t)
        f_concat = self.conv_f(f_t)
        i_z, g_z, i_z_prime, g_z_prime, o_z = torch.split(z_concat, self.num_hidden, dim=1)
        i_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_f, g_f = torch.split(f_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_z + i_h)
        g_t = torch.tanh(g_z + g_h)

        T_aug, weights_t = self.sam(c_t, c_historys, h_historys)
        gata_t = self.g_t(T_aug + h_t + z_t)
        c_new = torch.sigmoid(gata_t) * c_t + i_t * g_t

        F_aug = self.cstim(weights_t, f_historys)
        gata_f = self.g_f(F_aug + z_t + f_t)
        i_t_prime = torch.sigmoid(i_z_prime + i_f)
        g_t_prime = torch.tanh(g_z_prime + g_f)
        f_new = torch.sigmoid(gata_f) * f_t + i_t_prime * g_t_prime

        mem = torch.cat((c_new, f_new), 1)
        o_t = torch.sigmoid(o_z + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, f_new

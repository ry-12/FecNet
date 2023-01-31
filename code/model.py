import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from STCU import STCU

use_gpu = True
device = torch.device("cuda" if use_gpu else "cpu")

class RNN(nn.Module):
    def __init__(self, num_hidden, frame_channel, filter_size, stride, sr_size, total_length, input_length, height, width):
        super(RNN, self).__init__()

        self.frame_channel = frame_channel
        self.num_hidden = num_hidden
        self.num_layers = len(self.num_hidden)
        self.filter_size = filter_size
        self.stride = stride
        self.sr_size = sr_size
        self.total_length = total_length
        self.input_length = input_length
        self.MSE_criterion = nn.MSELoss()

        cell_list = []
        self.height = height // self.sr_size
        self.width = width // self.sr_size

        for i in range(self.num_layers):
            num_hidden_in = self.num_hidden[i - 1]
            in_channel = num_hidden_in
            cell_list.append(
                STCU(in_channel, num_hidden[i], self.height, self.width, self.filter_size, self.stride)
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[self.num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)

        n = int(math.log2(self.sr_size))

        encoders = []
        encoder = nn.Sequential()
        encoder.add_module(name='encoder_t_conv{}'.format(-1),
                           module=nn.Conv2d(in_channels=self.frame_channel,
                                            out_channels=self.num_hidden[0],
                                            stride=1,
                                            padding=0,
                                            kernel_size=1))
        encoder.add_module(name='relu_t_{}'.format(-1),
                           module=nn.LeakyReLU(0.2, inplace=True))
        encoders.append(encoder)
        for i in range(n):
            encoder = nn.Sequential()
            encoder.add_module(name='encoder_t{}'.format(i),
                               module=nn.Conv2d(in_channels=self.num_hidden[0],
                                                out_channels=self.num_hidden[0],
                                                stride=(2, 2),
                                                padding=(1, 1),
                                                kernel_size=(3, 3)
                                                ))

            encoder.add_module(name='encoder_t_relu{}'.format(i),
                               module=nn.LeakyReLU(0.2, inplace=True))
            encoders.append(encoder)
        self.encoders = nn.ModuleList(encoders)

        decoders = []

        for i in range(n - 1):
            decoder = nn.Sequential()
            decoder.add_module(name='c_decoder{}'.format(i),
                               module=nn.ConvTranspose2d(in_channels=self.num_hidden[-1],
                                                         out_channels=self.num_hidden[-1],
                                                         stride=(2, 2),
                                                         padding=(1, 1),
                                                         kernel_size=(3, 3),
                                                         output_padding=(1, 1)
                                                         ))
            decoder.add_module(name='c_decoder_relu{}'.format(i),
                               module=nn.LeakyReLU(0.2, inplace=True))
            decoders.append(decoder)

        if n > 0:
            decoder = nn.Sequential()
            decoder.add_module(name='c_decoder{}'.format(n - 1),
                               module=nn.ConvTranspose2d(in_channels=self.num_hidden[-1],
                                                         out_channels=self.num_hidden[-1],
                                                         stride=(2, 2),
                                                         padding=(1, 1),
                                                         kernel_size=(3, 3),
                                                         output_padding=(1, 1)
                                                         ))
            decoders.append(decoder)
        self.decoders = nn.ModuleList(decoders)

    def forward(self, frames, mask_true):
        # [batch, length, channel, height, width] -> [batch, channel, length, height, width]
        frames = frames.permute(0, 2, 1, 3, 4).contiguous()
        mask_true = mask_true.permute(0, 4, 1, 2, 3).contiguous()

        batch = frames.shape[0]
        height = self.height
        width = self.width

        next_frames = []
        h_t = []
        c_t = []
        h_t_history = []
        c_t_history = []
        f_t_history = []
        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).cuda()
            h_t.append(zeros)
            c_t.append(zeros)
            h_t_history.append(zeros.unsqueeze(2))
            c_t_history.append(zeros.unsqueeze(2))
            f_t_history.append(zeros.unsqueeze(2))
        f_t = torch.zeros([batch, self.num_hidden[-1], height, width]).cuda()

        for t in range(self.total_length - 1):

            if t < self.input_length:
                net = frames[:, :, t]
            else:
                net = mask_true[:, :, t - self.input_length] * frames[:, :, t] + \
                      (1 - mask_true[:, :, t - self.input_length]) * x_gen

            frames_feature = net
            frames_feature_encoded = []
            for i in range(len(self.encoders)):
                frames_feature = self.encoders[i](frames_feature)
                frames_feature_encoded.append(frames_feature)

            h_t[0], c_t[0], f_t = self.cell_list[0](frames_feature, h_t[0], c_t[0], h_t_history[0], c_t_history[0],
                                                    f_t_history[-1], f_t)
            h_t_history[0] = torch.cat([h_t_history[0], h_t[0].unsqueeze(2)], 2)
            c_t_history[0] = torch.cat([c_t_history[0], c_t[0].unsqueeze(2)], 2)
            f_t_history[0] = torch.cat([f_t_history[0], f_t.unsqueeze(2)], 2)
            for i in range(1, self.num_layers):

                h_t[i], c_t[i], f_t = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], h_t_history[i], c_t_history[i],
                                                           f_t_history[i - 1][:, :, :-1, :, :], f_t)
                h_t_history[i] = torch.cat([h_t_history[i], h_t[i].unsqueeze(2)], 2)
                c_t_history[i] = torch.cat([c_t_history[i], c_t[i].unsqueeze(2)], 2)
                f_t_history[i] = torch.cat([f_t_history[i], f_t.unsqueeze(2)], 2)
                out = h_t[self.num_layers - 1]
                for i in range(len(self.decoders)):
                    out = out + frames_feature_encoded[-1 - i]
                    out = self.decoders[i](out)

            x_gen = self.conv_last(out)
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, channel, length, height, width]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 2, 0, 3, 4).contiguous()
        labels = frames[:, :, 1:]
        loss = self.MSE_criterion(next_frames, labels)
        return next_frames, loss

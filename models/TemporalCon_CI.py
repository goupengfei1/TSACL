import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding

from layers.dilated_conv import DilatedConvEncoder

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class Decoder(nn.Module):
    def __init__(self, kernel_size, seq_len, pred_len, repr_dims, d_layers):
        super(Decoder, self).__init__()
        self.decompose = series_decomp(kernel_size + 1)
        self.depth = d_layers

        self.channel_mixer = nn.ModuleList()
        self.projection_head = nn.ModuleList()

        for _ in range(d_layers - 1):
            self.channel_mixer.append(
                nn.Sequential(
                    nn.Linear(repr_dims, repr_dims),
                    nn.GELU(),
                    nn.Dropout(0.1)
                )
            )

            self.projection_head.append(
                nn.Sequential(
                    nn.Linear(seq_len, seq_len),
                    nn.GELU(),
                    nn.Dropout(0.1)
                )
            )

        self.channel_mixer.append(
            nn.Sequential(
                nn.Linear(repr_dims, 1),
                nn.Dropout(0.1)
            )
        )

        self.projection_head.append(
            nn.Sequential(
                nn.Linear(seq_len, pred_len),
                nn.Dropout(0.1)
            )
        )

    def forward(self, x):
        for i in range(self.depth):
            x = self.projection_head[i](x.transpose(1, 2)).transpose(1, 2)
            x = self.channel_mixer[i](x)
            _, x = self.decompose(x)
        return x

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        self.channels = configs.enc_in
        self.c_out = configs.c_out
        self.hidden_dims = configs.d_model
        self.mlp_dims = configs.mlp_dims
        self.repr_dims = configs.d_ff
        self.depth = configs.e_layers


        self.TSACon = configs.TSACon
        self.TSACon_wnorm = configs.TSACon_wnorm
        enc_in = 1  # Channel Independence

        self.enc_embedding = DataEmbedding(enc_in, self.hidden_dims, configs.embed, configs.freq, dropout=0.1)
        self.feature_extractor = DilatedConvEncoder(
            self.hidden_dims,
            [self.hidden_dims] * self.depth + [self.repr_dims],
            kernel_size=configs.kernel_size,
        )
        self.norm1 = nn.BatchNorm1d(configs.seq_len)
        self.repr_dropout = nn.Dropout(p=0.1)
        self.input_decom = series_decomp(kernel_size=configs.decomp_size+1)
        self.res_linear2 = nn.Sequential(nn.Linear(configs.seq_len, configs.seq_len),
                                         nn.GELU(),
                                         nn.Linear(configs.seq_len, configs.pred_len),
                                         nn.Dropout(0.1),)
        self.decoder = Decoder(kernel_size=configs.decoder_desize, seq_len=self.seq_len,
                               pred_len=self.pred_len,repr_dims=self.repr_dims,d_layers=configs.d_layers)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, onlyrepr=False):
        # x: [Batch, Input length, Channel]

        if self.TSACon_wnorm == 'ReVIN':
            seq_mean = x_enc.mean(dim=1, keepdim=True).detach()
            seq_std = x_enc.std(dim=1, keepdim=True).detach()
            short_x = (x_enc - seq_mean) / (seq_std + 1e-9)
            long_x = short_x.clone()
        elif self.TSACon_wnorm == 'Mean':
            seq_mean = x_enc.mean(dim=1, keepdim=True).detach()
            short_x = (x_enc - seq_mean)
            long_x = short_x.clone()
        elif self.TSACon_wnorm == 'Decomp':
            _, long_x = self.input_decom(x_enc)
            short_x = long_x.clone()
        elif self.TSACon_wnorm == 'LastVal':
            seq_last = x_enc[:,-1:,:].detach()
            short_x = (x_enc - seq_last)
            long_x = short_x.clone()
        else:
            raise Exception(f'Not Supported Window Normalization:{self.TSACon_wnorm}. Use {"{ReVIN | Mean | LastVal | Decomp}"}.')

        B, T, C = long_x.shape
        long_x = long_x.permute(0, 2, 1).reshape(B * C, T, -1)
        x_mark_enc = x_mark_enc.squeeze(1).repeat(1, C, 1, 1).reshape(B * C, T, -1)
        enc_out = self.enc_embedding(long_x, x_mark_enc)
        enc_out = enc_out.permute(0, 2, 1)

        ori_repr = self.feature_extractor(enc_out).transpose(1, 2)
        ori_repr = self.repr_dropout(ori_repr)
        repr = self.norm1(ori_repr)  # (B, T, C)

        len_out = F.gelu(repr)
        trend_outs = self.decoder(len_out)
        trend_outs = trend_outs.reshape(B, C, self.pred_len).permute(0, 2, 1)
        y_temporal = self.res_linear2(short_x.permute(0,2,1)).permute(0,2,1)

        pred = trend_outs+ y_temporal

        if self.TSACon_wnorm == 'ReVIN':
            pred = pred*(seq_std+1e-9) + seq_mean
        elif self.TSACon_wnorm == 'Mean':
            pred = pred + seq_mean
        elif self.TSACon_wnorm == 'Decomp':
            pred = pred
        elif self.TSACon_wnorm == 'LastVal':
            pred = pred + seq_last
        else:
            raise Exception()

        if self.TSACon:
            return pred, repr
        else:
            return pred

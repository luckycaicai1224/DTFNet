__all__ = ['DTFNet1.py']

import math

import numpy as np
# Cell
import torch

from torch import nn, Tensor

from yacs.config import CfgNode as CN
import torch.nn.functional as F
from layers.PatchTST_layers import positional_encoding, Transpose, get_activation_fn
from layers.RevIN import RevIN
from einops import rearrange
from typing import Optional
from pytorch_wavelets import DWT1D, IDWT1D

class Model(nn.Module):
    def __init__(self, configs, d_k: Optional[int] = None, d_v: Optional[int] = None,
                 norm: str = 'BatchNorm', attn_dropout: float = 0.,
                 act: str = "gelu", key_padding_mask: bool = 'auto', padding_var: Optional[int] = None,
                 attn_mask: Optional[Tensor] = None,
                 pre_norm: bool = False, **kwargs
                 ):

        super().__init__()
        store_attn = configs.store_attn
        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len

        n_branches = configs.n_branches
        d_model = configs.d_model
        dropout = configs.dropout
        head_dropout = configs.head_dropout

        individual = configs.individual

        patch_len = configs.patch_len_ls
        stride = configs.stride_ls
        self.learning_rate = configs.learning_rate
        self.d_model = d_model
        self.batch_size = configs.batch_size

        padding_patch = configs.padding_patch

        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        if isinstance(configs.dilation_rates, str):
            dilation_rates = configs.dilation_rates.split(',')
            dilation_rates = [int(i) for i in dilation_rates]

        decomposition = configs.decomposition
        self.batch_size = configs.batch_size

        # model
        self.decomposition = decomposition
        self.tfactor = configs.tfactor
        self.e_layers = configs.e_layers
        n_heads=configs.n_heads
        d_ff = configs.d_ff

        wavelet_layers = configs.wavelet_layers
        wavelet_type = configs.wavelet_type
        wavelet_mode = configs.wavelet_mode
        wavelet_dim = configs.wavelet_dim

        cfg = CN()

        configs = vars(configs)
        for k in configs.keys():
            cfg[k] = configs[k]

        # res_attention = cfg.get('res_attn', False)
        pe = cfg.get('pe', 'zeros')
        learn_pe = cfg.get('no_learn_pe', True)

        self.model = DTFNet_backbone(c_in=c_in, context_window=context_window, target_window=target_window,
                                   patch_len=patch_len, stride=stride, padding_patch=padding_patch,dropout=dropout, head_dropout=head_dropout, n_branches=n_branches,
                                   tfactor=self.tfactor, d_model=d_model, individual=individual, revin=revin,
                                   affine=affine, subtract_last=subtract_last, cfg=cfg,pe=pe,learn_pe=learn_pe,e_layers=self.e_layers,
                                    n_heads = n_heads,norm=norm,act=act, d_k = d_k, d_v = d_v, d_ff = d_ff,store_attn=store_attn,
                                    pre_norm=pre_norm, attn_dropout=attn_dropout,wavelet_layers=wavelet_layers,wavelet_type=wavelet_type,
                                    wavelet_mode=wavelet_mode,wavelet_dim=wavelet_dim,dilation_rates=dilation_rates, batch_size=self.batch_size, **kwargs)


    def forward(self, x):  # x: [Batch, Input length, Channel]
        # 转换数据形状为 [Batch, Channel, Input length] 以适应模型输入
        z = x.permute(0, 2, 1)  # x: [Batch, Channel, Input length]

        z, draw_list, attn_ls = self.model(z)

        z = z.permute(0, 2, 1)

        return z, draw_list, attn_ls

class DTFNet_backbone(nn.Module):
    def __init__(self, c_in:int, context_window:int, target_window:int, patch_len:int, stride:int, padding_patch:bool,dilation_rates:int,batch_size:int,dropout=0, head_dropout=0,
                 n_branches:int=3, tfactor=1, d_model=128, individual = False, revin = True, affine = True, subtract_last = False,pe='zeros',
                 learn_pe=True,e_layers=1,cfg=CN(), d_k=None, d_v=None, d_ff=None,n_heads=16,norm: str = 'BatchNorm',act: str = "gelu",store_attn:bool=False,
                 pre_norm: bool = False, attn_dropout: float = 0.,wavelet_layers=5,
                 wavelet_type='haar',wavelet_mode='symmetric',wavelet_dim=64, **kwargs):

        super().__init__()
        self.e_layers = e_layers
        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        self.batch_size = batch_size
        # Patching
        if isinstance(patch_len, str):
            self.patch_len = patch_len.split(',')
            self.patch_len = [int(i) for i in self.patch_len]
        if isinstance(stride, str):
            self.stride = stride.split(',')
            self.stride = [int(i) for i in self.stride]
        # self.patch_num = [int(math.ceil((context_window - self.patch_len[j]) / self.stride[j] + 1)) for j in range(n_branches)]
        self.patch_num = [int((context_window - self.patch_len[j]) / self.stride[j] + 1) for j in range(n_branches)]
        self.padding_patch = padding_patch
        if padding_patch == 'end' or padding_patch == "'end'":
            self.patch_num =[p_n + 1 for p_n in self.patch_num]

        self.target_window = target_window
        self.context_window = context_window
        self.individual = individual
        self.channels = c_in
        self.n_branches = n_branches
        self.d_model = d_model

        self.res_attention = cfg.get('res_attn', False)
        self.deepPointConvBlock = nn.ModuleList()
        self.W_P = nn.ModuleList()
        for j in range(n_branches):
            self.deepPointConvBlock.append(DeepPointConvBlock(patch_num=self.patch_num[j], stride=self.stride[j],dilation_rates=dilation_rates))
            self.W_P.append(nn.Linear(self.patch_len[j], d_model))

        # TST encoding
        self.linearBlock = nn.ModuleList()
        for j in range(n_branches):
            self.linearBlock.append(nn.ModuleList())
            for i in range(self.e_layers):
                if i == self.e_layers-1:
                    self.linearBlock[j].append(LinearBlock(self.d_model, self.patch_num[j], self.target_window, tfactor, head_dropout,self.individual))
                else:
                    self.linearBlock[j].append(
                        LinearBlock(self.d_model, self.patch_num[j], self.context_window, tfactor, head_dropout,
                                    self.individual))


        if padding_patch == 'end' or padding_patch == "'end'":  # can be modified to general case
            self.padding_patch_layer = nn.ModuleList()
            for j in range(n_branches):
                self.padding_patch_layer.append(nn.ReplicationPad1d((0, self.stride[j])))

        # Residual dropout
        self.dropout = nn.Dropout(dropout)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Positional encoding
        self.W_pos = []
        for j in range(n_branches):
            self.W_pos.append(
                positional_encoding(pe, learn_pe, self.patch_num[j], d_model).to(self.device))

        wavelet_layers = wavelet_layers
        wavelet_type = wavelet_type
        wavelet_mode = wavelet_mode
        wavelet_dim = wavelet_dim
        self.wavelet_dim = wavelet_dim
        self.dwt = DWT1D(J=wavelet_layers, wave=wavelet_type, mode=wavelet_mode)
        self.idwt = IDWT1D(wave=wavelet_type, mode=wavelet_mode)

        temp_seq = torch.rand(1, 1, context_window)
        temp_seq_yl, temp_seq_yh = self.dwt(temp_seq)
        seq_len_J = [y.shape[-1] for y in temp_seq_yh] + [temp_seq_yl.shape[-1]]

        temp_pred = torch.rand(1, 1, context_window)
        temp_pred_yl, temp_pred_yh = self.dwt(temp_pred)
        pred_len_J = [y.shape[-1] for y in temp_pred_yh] + [temp_pred_yl.shape[-1]]

        self.in_proj_h = nn.ModuleList([
            nn.Linear(seq_len_J[i], wavelet_dim)
            for i in range(wavelet_layers)
        ])
        self.in_proj_l = nn.Linear(seq_len_J[-1], wavelet_dim)
        self.out_proj_h = nn.ModuleList([
            nn.Sequential(
                nn.Linear(wavelet_dim, wavelet_dim * 2),
                nn.ReLU if act == 'relu' else nn.GELU(),
                nn.Linear(wavelet_dim * 2, pred_len_J[i])
            ) for i in range(wavelet_layers)
        ])
        self.out_proj_l = nn.Sequential(
            nn.Linear(wavelet_dim, wavelet_dim * 2),
            nn.ReLU if act == 'relu' else nn.GELU(),
            nn.Linear(wavelet_dim * 2, pred_len_J[-1])
        )
        self.Linear_ = nn.Linear(context_window, target_window)

        self.waveletSALayers_h = nn.ModuleList()
        for i in range(wavelet_layers):
            self.waveletSALayers_h.append(TSTEncoderLayer(c_in, d_model=wavelet_dim, n_heads=n_heads, d_k=d_k,
                         d_v=d_v, d_ff=d_ff, norm=norm,
                         attn_dropout=attn_dropout, dropout=dropout,
                         activation=act, res_attention=self.res_attention,
                         pre_norm=pre_norm, store_attn=store_attn, cfg=cfg,batch_size=self.batch_size,c_in=c_in
                         ))

        self.waveletSALayer_l = TSTEncoderLayer(c_in, d_model=wavelet_dim, n_heads=n_heads, d_k=d_k,
                         d_v=d_v, d_ff=d_ff, norm=norm,
                         attn_dropout=attn_dropout, dropout=dropout,
                         activation=act, res_attention=self.res_attention,
                         pre_norm=pre_norm, store_attn=store_attn, cfg=cfg,batch_size=self.batch_size,c_in=c_in
                         )

        self.w1 = nn.Parameter(torch.tensor(0.4), requires_grad=True)
        self.w2 = nn.Parameter(torch.tensor(0.6), requires_grad=True)

        self.NFICSA = TSTEncoderLayer(wavelet_layers+1, d_model=wavelet_dim, n_heads=n_heads, d_k=d_k,
                        d_v=d_v, d_ff=d_ff, norm=norm,
                        attn_dropout=attn_dropout, dropout=dropout,
                        activation=act, res_attention=self.res_attention,
                        pre_norm=pre_norm, store_attn=store_attn, cfg=cfg, batch_size=self.batch_size, c_in=wavelet_layers+1, NFIC=0
                        )


    def forward(self, x):
        # norm
        if self.revin:
            x = x.permute(0, 2, 1)
            x = self.revin_layer(x, 'norm')
            x = x.permute(0, 2, 1)

        # yl: [B, M, L/2^J], yh: [[B, M, L/2^j]]
        yl, yh = self.dwt(x)
        # [B, M, L/2^j] -> [B, M, 1, D]
        for i in range(len(yh)):
            yh[i] = self.in_proj_h[i](yh[i]).unsqueeze(-2)
        yl = self.in_proj_l(yl).unsqueeze(-2)
        # [[B, M, 1, D]] -> [B, M, J, D]
        enc_in1 = torch.cat(yh, dim=-2)
        enc_in2 = torch.cat([yl], dim=-2)


        o_yh = []
        for i in range(len(yh)):
            result = self.waveletSALayers_h[i](enc_in1[:,:,i,:])
            o_yh.append(result[0])
            if i == 0:
                attn = result[1][0]
            else:
                attn += result[1][0]

        enc_in1 = torch.stack(o_yh, dim=-2)

        o_yl = []
        result2 = self.waveletSALayer_l(enc_in2[:,:,0,:])
        o_yl.append(result2[0])
        attn += result2[1][0]
        enc_in2 = torch.stack(o_yl, dim=-2)

        enc_in = torch.cat((enc_in1,enc_in2), dim=2)
        enc_in_NFIC_in = rearrange(enc_in, 'b n w d -> (b n) w d')
        enc_in_NFIC_out = (self.NFICSA(enc_in_NFIC_in))[0]
        enc_in = rearrange(enc_in_NFIC_out, '(b n) w d -> b n w d', n=self.channels)

        enc_in = list(torch.unbind(enc_in, dim=-2))
        for i in range(len(yh)):
            yh[i] = self.out_proj_h[i](enc_in[i])
        yl = self.out_proj_l(enc_in[-1])
        x1 = self.idwt((yl, yh)).permute(0, 2, 1)[:, :, :self.channels].permute(0, 2, 1)
        output1 = self.Linear_(x1)
        # 输出数据
        outputs = []
        for j in range(self.n_branches):
            input = x
            for i in range(self.e_layers):
                # do patching
                if self.padding_patch == 'end' or self.padding_patch == "'end'":
                    x_ = self.padding_patch_layer[j](input)

                x_ = x_.unfold(dimension=-1, size=self.patch_len[j],
                                             step=self.stride[j])  # z: [bs x nvars x patch_num x patch_len]
                x_ = x_.reshape(x_.size(0), x_.size(1), -1, self.patch_len[j])

                x_ = self.W_P[j](x_)

                x_u = rearrange(x_, 'b n p d -> (b n) p d')
                x_u = self.dropout(x_u + self.W_pos[j])  # u: [bs * nvars x patch_num x d_model]

                x_u = self.deepPointConvBlock[j](x_u)

                x_u = rearrange(x_u, '(b n) p d -> b n p d', n=self.channels)
                x_o = x_u.flatten(2)
                x_output = self.linearBlock[j][i](x_o)
                if i != self.e_layers-1:
                    input = input + x_output
            outputs.append(x_output)

        z = torch.stack(outputs, dim=-1).sum(dim=-1, keepdim=False) #[Batch, Channel, Output length]
        z = z * self.w1 + output1 * self.w2

        # denorm
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0, 2, 1)
        return z, [], []

class DeepPointConvBlock(nn.Module):
    def __init__(self, patch_num:int,stride:int,dilation_rates:int, use_channel_conv:bool=True):
        super().__init__()
        self.PointConv_1 = nn.Sequential(
            nn.Conv1d(patch_num, patch_num, kernel_size=1,padding='same'),
            nn.BatchNorm1d(patch_num),
            nn.GELU(),
        )
        self.DeepConvs_1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    patch_num, patch_num, kernel_size=stride, stride=1,
                    groups=patch_num, dilation=rate, padding='same'
                ),
                nn.BatchNorm1d(patch_num),
                nn.GELU()
            ) for rate in dilation_rates
        ])
        self.patch_num = patch_num
        self.num_rates = len(dilation_rates)
        # 可学习的融合权重
        self.dilation_weights = nn.Parameter(torch.ones(self.num_rates))

    def forward(self, x):
        # 计算 softmax 权重（shape: [num_rates]）
        weights = F.softmax(self.dilation_weights, dim=0)
        # 加权融合不同 dilation 的分支输出
        deep_outputs = [w * branch(x) + x for w, branch in zip(weights, self.DeepConvs_1)]
        x_u = self.PointConv_1(sum(deep_outputs))
        return x_u



class LinearBlock(nn.Module):
    def __init__(self, d_model, patch_num, target_window, tfactor, head_dropout, individual):
        super().__init__()

        self.individual = individual
        self.target_window = target_window
        if self.individual:
            self.Linear_ = nn.ModuleList()
            for i in range(self.channels):
                self.Linear_.append(
                    nn.Sequential(nn.Linear(d_model * patch_num, target_window * tfactor),
                                  nn.GELU(),
                                  nn.Dropout(head_dropout),
                                  nn.Linear(target_window * tfactor, target_window),
                                  nn.Dropout(head_dropout)))

        else:
            self.Linear_ = nn.Sequential(nn.Linear(d_model * patch_num, target_window * tfactor),
                               nn.GELU(),
                               nn.Dropout(head_dropout),
                               nn.Linear(target_window * tfactor, target_window),
                               nn.Dropout(head_dropout))

    def forward(self, x):
        if self.individual:
            x_output = torch.zeros([x.size(0), x.size(1), self.target_window],
                                   dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                x_output[:, i, :] = self.Linear_[i](x[:, i, :])
        else:
            x_output = self.Linear_(x)
        return x_output


class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False,
                 pre_norm=False, cfg=CN(),batch_size=256,c_in=7, NFIC = 1
                 ):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v


        self.pre_norm = pre_norm

        # Multi-Head attention
        self.res_attention = cfg.get('res_attn', False)
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention, cfg=cfg,batch_size=batch_size,c_in=c_in, NFIC=NFIC)
        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(self, src:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None,
                attn_bias:Optional[Tensor]=None) -> Tensor:
        # src : [10272, 85, 128] = [bs x nvar, patch_num, d_model]
        # Multi-Head attention sublayer
        res = src
        if self.pre_norm:
            src = self.norm_attn(src)

        ## Multi-Head attention
        if self.res_attention:
            src, attn, scores = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask,
                                       attn_bias=attn_bias)
        else:
            src, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask,
                                   attn_bias=attn_bias)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = res + self.dropout_attn(src) # Add: residual connection with residual dropout

        if not self.pre_norm:
            src = self.norm_attn(src)

        res = src
        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)

        ## Position-wise Feed-Forward
        src = self.ff(src)
        ## Add & Norm
        src = res + self.dropout_ffn(src) # Add: residual connection with residual dropout
        if not self.pre_norm: # default pre_norm = False
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, attn, scores

        else:
            return src, attn # always save attn_list

class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False,cfg=CN(),batch_size=256,c_in=7,NFIC=1):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa,batch_size=batch_size,c_in=c_in,NFIC=NFIC)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))


    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None,
                attn_bias:Optional[Tensor]=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_headsn_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask,
                                                              attn_bias=attn_bias)
        else:
            output, attn_weights = self.attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask,
                                                 attn_bias=attn_bias)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False,batch_size=256,c_in=7,NFIC=1):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa
        # 初始化一个形状为 [256, 16, 7, 7] 的可学习矩阵
        self.NFIC = NFIC
        self.matrix = nn.Parameter(torch.randn(batch_size, n_heads, c_in, c_in))
        # self.matrix = nn.Parameter(torch.ones(batch_size, n_heads, c_in, c_in))
        # self.matrix = nn.Parameter(torch.sigmoid(torch.randn(256, 16, 7, 7)))

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None,
                attn_bias:Optional[Tensor]=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
            attn_bias       : [1 x seql_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''
        save_score = []

        if self.NFIC == 0:
            attn_scores = torch.matmul(q, k) * self.scale  # attn_scores : [bs x n_heads x max_q_len x q_len]
        else:
            attn_scores = torch.matmul(q, k) * self.matrix * self.scale

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        if attn_bias is not None:
            attn_scores = attn_scores + attn_bias

        # Attention mask (optional)
        if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        save_score.append(attn_scores)
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention:
            return output, save_score, attn_scores
        # else: return output, attn_weights
        else:
            return output, save_score #instead of saving final weight, save weight_list=[qk_score, bias_score, total_score]

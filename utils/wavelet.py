import pywt
import pywt.data
import torch
import torch.nn.functional as F


def create_wavelet_filter(wt_type='db1', in_channels=1, out_channels=1, dtype=torch.float):
    # 这里我们用简化的处理方法来生成小波滤波器
    # 你可以根据需要选择不同的小波类型（例如 Daubechies, Haar, etc.）

    if wt_type == 'db1':  # Daubechies 1 小波（最简单的小波类型）
        # db1 小波（Haar小波），简单示例
        low_pass_filter = torch.tensor([0.7071, 0.7071], dtype=dtype).view(1, 1, 2)  # 低频滤波器
        high_pass_filter = torch.tensor([-0.7071, 0.7071], dtype=dtype).view(1, 1, 2)  # 高频滤波器
    else:
        raise ValueError(f"Unsupported wavelet type: {wt_type}")

    # 生成对应的滤波器
    wt_filter = torch.cat([low_pass_filter, high_pass_filter], dim=1).repeat(in_channels, 1, 1)  # 为每个输入通道生成
    iwt_filter = wt_filter  # 逆小波滤波器可以与小波滤波器相同（在简化的情况下）

    return wt_filter, iwt_filter

def wavelet_transform(x, filters):
    # x: 输入的时间序列数据，形状为 (batch_size, channels, time_steps)
    # filters: 小波滤波器，形状为 (channels, 2, filter_size)

    batch_size, channels, time_steps = x.shape
    low_pass_filter, high_pass_filter = filters[:, 0:1, :], filters[:, 1:2, :]

    # 使用1D卷积进行小波变换
    low_freq = F.conv1d(x, low_pass_filter, stride=2, padding=1)  # 低频部分
    high_freq = F.conv1d(x, high_pass_filter, stride=2, padding=1)  # 高频部分

    return torch.cat([low_freq, high_freq], dim=2)  # 连接低频和高频部分

def inverse_wavelet_transform(x, filters):
    # x: 输入的小波系数数据，形状为 (batch_size, channels, time_steps / 2)
    # filters: 小波滤波器，形状为 (channels, 2, filter_size)

    batch_size, channels, time_steps = x.shape
    low_pass_filter, high_pass_filter = filters[:, 0:1, :], filters[:, 1:2, :]

    # 分离低频和高频部分
    low_freq, high_freq = torch.split(x, [time_steps // 2, time_steps // 2], dim=2)

    # 使用1D卷积进行逆小波变换
    low_freq_reconstructed = F.conv_transpose1d(low_freq, low_pass_filter, stride=2, padding=1)
    high_freq_reconstructed = F.conv_transpose1d(high_freq, high_pass_filter, stride=2, padding=1)

    # 合并重建的信号
    return low_freq_reconstructed + high_freq_reconstructed

import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(
        self,
        content_dim: int = 256,
        hidden_dim: int = 128,
        use_pitch: bool = True,
        content_sr: int = 16000,
        hop_length: int = 320,
        target_sr: int = 48000,
    ):
        super(Generator, self).__init__()
        self.content_dim = content_dim
        self.hidden_dim = hidden_dim
        self.use_pitch = use_pitch
        self.content_sr = content_sr
        self.hop_length = hop_length
        self.target_sr = target_sr

        self.upsample_factor = int(round(target_sr * hop_length / float(content_sr)))

        self.content_conv = nn.Conv1d(content_dim, hidden_dim, kernel_size=1)
        self.pitch_conv = nn.Conv1d(1, hidden_dim, kernel_size=1) if use_pitch else None

        factors = (
            [5, 5, self.upsample_factor // 25]
            if self.upsample_factor >= 25
            else [self.upsample_factor]
        )
        self.upsample_layers = nn.ModuleList()
        in_channels = hidden_dim
        out_channels_sequence = [64, 32, 1] if len(factors) == 3 else [32, 1]

        for i, factor in enumerate(factors):
            out_ch = (
                out_channels_sequence[i]
                if i < len(out_channels_sequence)
                else max(1, in_channels // 2)
            )
            kernel_size = factor * 2
            layer = nn.ConvTranspose1d(
                in_channels, out_ch, kernel_size=kernel_size, stride=factor, padding=(factor // 2)
            )
            self.upsample_layers.append(layer)
            in_channels = out_ch

        self.activation = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, content_units: torch.Tensor, pitch: torch.Tensor = None) -> torch.Tensor:
        B, T, C = content_units.shape
        x = content_units.transpose(1, 2)
        x = self.content_conv(x)
        if self.use_pitch and pitch is not None:
            p = pitch.unsqueeze(1) if pitch.dim() == 2 else pitch.transpose(1, 2)
            p_embed = self.pitch_conv(p.to(x.dtype))
            if p_embed.shape[2] != x.shape[2]:
                min_len = min(p_embed.shape[2], x.shape[2])
                p_embed = p_embed[:, :, :min_len]
                x = x[:, :, :min_len]
            x = x + p_embed

        for layer in self.upsample_layers:
            x = layer(x)
            if layer is not self.upsample_layers[-1]:
                x = self.activation(x)

        waveform = x.squeeze(1)
        return waveform

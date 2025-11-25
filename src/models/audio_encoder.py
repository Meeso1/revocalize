from typing import Optional, Tuple, Union
import torch
import torchaudio


class AudioEncoder:
    def __init__(
        self,
        hubert_path: Optional[str] = None,
        rmvpe_path: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        if hubert_path is not None:
            try:
                self.content_model = torch.jit.load(hubert_path, map_location=device)
            except Exception:
                self.content_model = torchaudio.pipelines.HUBERT_BASE.get_model().to(device)
        else:
            self.content_model = torchaudio.pipelines.HUBERT_BASE.get_model().to(device)
        self.content_model.eval()
        self.pitch_model = None
        if rmvpe_path is not None:
            try:
                self.pitch_model = torch.jit.load(rmvpe_path, map_location=device)
                self.pitch_model.eval()
            except Exception:
                self.pitch_model = None
        self.use_pitch_model = self.pitch_model is not None
        self.content_sr = 16000
        self.hop_length = 320

    def encode(
        self, audio: Union[str, torch.Tensor], sr: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(audio, str):
            waveform, sr0 = torchaudio.load(audio)
            waveform = waveform.mean(dim=0, keepdim=True)
            sr_in = sr0
        elif isinstance(audio, torch.Tensor):
            waveform = audio.detach().clone()
            if waveform.dim() == 2:
                waveform = waveform[:1]
            else:
                waveform = waveform.unsqueeze(0)
            sr_in = sr if sr is not None else self.content_sr
        else:
            raise TypeError("Audio input must be a file path or a torch.Tensor.")

        if sr_in != self.content_sr:
            waveform = torchaudio.functional.resample(waveform, sr_in, self.content_sr)
        waveform = waveform.to(self.device)

        with torch.inference_mode():
            content_feats = self.content_model(waveform)
            if isinstance(content_feats, tuple):
                content_feats = content_feats[0]
        content_units = content_feats[0].cpu()

        if self.use_pitch_model:
            with torch.inference_mode():
                wave_cpu = waveform.squeeze(0).to("cpu")
                try:
                    f0 = self.pitch_model(wave_cpu)
                except Exception as e:
                    raise RuntimeError("RMVPE pitch model inference failed.") from e
                f0 = f0.to(torch.float32)
        else:
            frame_time = self.hop_length / float(self.content_sr)
            f0_vals = torchaudio.functional.detect_pitch_frequency(
                waveform, self.content_sr, frame_time=frame_time
            )
            f0 = f0_vals[0].cpu()

        return content_units, f0

    def encode_segment(
        self, audio_segment: torch.Tensor, sr: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encode(audio_segment, sr)

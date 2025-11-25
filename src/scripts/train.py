import os
import random
import numpy as np
import torch
import torchaudio
from pathlib import Path
import argparse
from torch.utils.data import Dataset, DataLoader

from generator import Generator
from data_models import PreprocessedSample


class VoiceDataset(Dataset):
    def __init__(
        self,
        feature_dir: str,
        data_dir: str,
        content_sr: int = 16000,
        hop_length: int = 320,
        target_sr: int = 48000,
        segment_frames: int = 64,
    ):
        self.feature_dir = Path(feature_dir)
        self.data_dir = Path(data_dir)
        self.content_sr = content_sr
        self.hop_length = hop_length
        self.target_sr = target_sr
        self.segment_frames = segment_frames
        self.samples_per_frame = int(round(target_sr * hop_length / float(content_sr)))
        unit_files = sorted(self.feature_dir.glob("*_units.npy"))
        self.file_ids = [f.name.split("_units.npy")[0] for f in unit_files]

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        content = np.load(self.feature_dir / f"{file_id}_units.npy")
        f0 = np.load(self.feature_dir / f"{file_id}_f0.npy")
        num_frames = content.shape[0]
        if num_frames >= self.segment_frames:
            start_frame = random.randint(0, num_frames - self.segment_frames)
            end_frame = start_frame + self.segment_frames
        else:
            start_frame = 0
            end_frame = num_frames
        content_seg = content[start_frame:end_frame]
        f0_seg = f0[start_frame:end_frame]

        audio_path = self.data_dir / f"{file_id}.wav"
        if not audio_path.exists():
            audio_path = self.data_dir / f"{file_id}.flac"
        if not audio_path.exists():
            raise FileNotFoundError(f"No audio file found for ID {file_id}")
        waveform, sr = torchaudio.load(str(audio_path))
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        wave_np = waveform.numpy()
        wave_arr = wave_np[0] if wave_np.ndim == 2 else wave_np
        thresh = 1e-3
        nz = np.where(np.abs(wave_arr) > thresh)[0]
        if nz.size > 0:
            waveform = waveform[:, nz[0] : nz[-1] + 1]
        if sr != self.target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.target_sr)

        if num_frames >= self.segment_frames:
            start_sample = int(start_frame * self.samples_per_frame)
            end_sample = int(end_frame * self.samples_per_frame)
            waveform_seg = waveform[:, start_sample:end_sample]
        else:
            waveform_seg = waveform

        content_t = torch.from_numpy(content_seg).float()
        f0_t = torch.from_numpy(f0_seg).float()
        audio_t = waveform_seg.to(torch.float32)

        return PreprocessedSample(
            content_vector=content_t.numpy(), pitch_feature=f0_t.numpy(), audio=audio_t.numpy()
        )

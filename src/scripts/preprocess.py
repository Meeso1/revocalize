import os
os.environ["TORCHAUDIO_USE_TORCHCODEC"] = "0"

import torchaudio
import numpy as np
import torch
from pathlib import Path
import argparse

from src.models.audio_encoder import AudioEncoder
from src.models.index_creator import IndexCreator

if hasattr(torchaudio, 'set_audio_backend'):
    torchaudio.set_audio_backend("sox_io")

def main():
    parser = argparse.ArgumentParser(description="Preprocess audio data for RVC.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with raw audio files (wav/flac).")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for features and index.")
    parser.add_argument("--hubert_path", type=str, default=None, help="Path to HuBERT/ContentVec model (optional).")
    parser.add_argument("--rmvpe_path", type=str, default=None, help="Path to RMVPE pitch model (optional).")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    encoder = AudioEncoder(hubert_path=args.hubert_path, rmvpe_path=args.rmvpe_path)

    all_content_vectors = []
    audio_paths = sorted(list(data_dir.rglob("*.wav")) + list(data_dir.rglob("*.flac")))
    if not audio_paths:
        print(f"No audio files found in {data_dir}")
        return

    for audio_path in audio_paths:
        try:
            waveform, sr = torchaudio.load(str(audio_path))
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            wave_np = waveform.numpy()
            wave_arr = wave_np[0] if wave_np.ndim == 2 else wave_np
            thresh = 1e-3
            nonzero_indices = np.where(np.abs(wave_arr) > thresh)[0]
            if nonzero_indices.shape[0] > 0:
                start_idx = nonzero_indices[0]
                end_idx = nonzero_indices[-1] + 1
                waveform = waveform[:, start_idx:end_idx]
            peak = waveform.abs().max()
            if peak > 0:
                waveform = waveform / peak

            content, f0 = encoder.encode(waveform.to(torch.float32), sr)
            content_np = content.numpy().astype(np.float32)
            f0_np = f0.numpy().astype(np.float32)

            file_id = audio_path.stem
            np.save(out_dir / f"{file_id}_units.npy", content_np)
            np.save(out_dir / f"{file_id}_f0.npy", f0_np)

            all_content_vectors.append(content_np)
            print(f"Processed {audio_path.name}: {content_np.shape[0]} frames.")
        except Exception as e:
            print(f"Warning: failed to process {audio_path.name}: {e}")

    if not all_content_vectors:
        print("No features extracted, index creation skipped.")
        return

    combined_vectors = np.concatenate(all_content_vectors, axis=0).astype("float32")
    dim = combined_vectors.shape[1]
    index_creator = IndexCreator(dimension=dim)
    index_creator.add(combined_vectors)
    index_creator.save(str(out_dir / "faiss_index.index"))
    np.save(out_dir / "content_vectors.npy", combined_vectors)
    print(f"FAISS index saved. Indexed {combined_vectors.shape[0]} vectors.")

if __name__ == "__main__":
    main()
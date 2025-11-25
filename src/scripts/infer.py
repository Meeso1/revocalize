import argparse
import torch
import torchaudio
import numpy as np
from pathlib import Path

from ..models.audio_encoder import AudioEncoder
from ..models.generator import Generator


def main():
    parser = argparse.ArgumentParser(
        description="Run inference to convert voice using trained generator."
    )
    parser.add_argument("--input", type=str, required=True, help="Path to input audio file.")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to trained generator .pth file."
    )
    parser.add_argument("--output", type=str, required=True, help="Path to save converted audio.")
    parser.add_argument(
        "--hubert_path", type=str, default=None, help="Path to HuBERT model (optional)."
    )
    parser.add_argument(
        "--rmvpe_path", type=str, default=None, help="Path to RMVPE model (optional)."
    )
    parser.add_argument("--content_dim", type=int, default=256, help="Dimension of content vector.")
    parser.add_argument(
        "--use_pitch", action="store_true", help="Whether to use pitch (f0) features."
    )
    parser.add_argument(
        "--target_sr", type=int, default=48000, help="Target sample rate for output audio."
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = AudioEncoder(hubert_path=args.hubert_path, rmvpe_path=args.rmvpe_path)
    content, f0 = encoder.encode(args.input)

    content = content.unsqueeze(0).to(device)
    f0 = f0.unsqueeze(0).to(device) if args.use_pitch else None

    model = Generator(
        content_dim=args.content_dim, use_pitch=args.use_pitch, target_sr=args.target_sr
    ).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    with torch.inference_mode():
        wav = model(content, f0).cpu().numpy()[0]

    torchaudio.save(args.output, torch.tensor(wav).unsqueeze(0), args.target_sr)
    print(f"Inference complete. Saved to {args.output}")


if __name__ == "__main__":
    main()

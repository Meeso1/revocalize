import argparse
import torch
import torchaudio
from pathlib import Path

from src.models.audio_encoder import AudioEncoder
from src.models.generator import Generator


def main():
    parser = argparse.ArgumentParser(description="Run inference to convert voice.")
    parser.add_argument("--input", type=str, required=True, help="Path to input audio file (.wav or .mp3).")
    parser.add_argument("--model", type=str, required=True, help="Path to trained generator .pth file.")
    parser.add_argument("--output", type=str, required=True, help="Path to save output audio.")
    parser.add_argument("--hubert_path", type=str, help="Path to HuBERT model.")
    parser.add_argument("--rmvpe_path", type=str, help="Path to RMVPE model.")
    parser.add_argument("--content_dim", type=int, default=256)
    parser.add_argument("--use_pitch", action="store_true")
    parser.add_argument("--target_sr", type=int, default=48000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load audio
    waveform, sr = torchaudio.load(args.input)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak
    encoder = AudioEncoder(hubert_path=args.hubert_path, rmvpe_path=args.rmvpe_path)
    content, f0 = encoder.encode(waveform.to(torch.float32), sr)

    content = content.unsqueeze(0).to(device)
    f0 = f0.unsqueeze(0).to(device) if args.use_pitch else None

    model = Generator(
        content_dim=769, use_pitch=args.use_pitch, target_sr=args.target_sr
    ).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    # 🔧 SMALL FIX: add a zero channel if needed so content matches conv input channels
    expected_in_ch = model.content_conv.in_channels  # e.g. 769
    cur_ch = content.shape[-1]                       # e.g. 768

    if cur_ch < expected_in_ch:
        pad_ch = expected_in_ch - cur_ch
        pad = torch.zeros(
            content.shape[0],
            content.shape[1],
            pad_ch,
            device=content.device,
            dtype=content.dtype,
        )
        content = torch.cat([content, pad], dim=-1)

    with torch.inference_mode():
        wav = model(content, f0 if args.use_pitch else None).cpu()

    # Make sure we end up with [samples]
    if wav.dim() == 3:
        wav = wav[0, 0]
    elif wav.dim() == 2:
        wav = wav[0]
    wav = wav.clamp(-1.0, 1.0)

    with torch.inference_mode():
        output = model(content)
        output = output.squeeze(0).clamp(-1, 1).cpu()

    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    torchaudio.save(args.output, output.unsqueeze(0), args.target_sr)
    print(f"Inference complete. Saved to {args.output}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import torch
import torchaudio

from src.train_deepspeed2 import (
    CONFIG_FILENAME,
    TrainingConfig,
    DeepSpeech2Small,
    LogMelExtractor,
    greedy_decode,
    indices_to_text,
    training_config_from_json,
    resolve_device,
)

# Update these paths for your environment.
MODEL_PATH = Path("/home/metythorn/konai/services/asr-service/logs/deepspeed-100eps-clean-text/ds2_small_clean.pt")
AUDIO_PATH = Path("/home/metythorn/konai/services/asr-service/data/samples/openslr/km_openslr_undefined_khm_0308_0038959268_KH_20250909_173102_8fe10f24.wav")


def load_model_artifacts(
    model_path: Path, device: torch.device
) -> tuple[TrainingConfig, list[str], dict[str, torch.Tensor]]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint does not contain a valid state_dict mapping.")

    metadata_path = model_path.parent / CONFIG_FILENAME
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    if not {"config", "vocab"} <= metadata.keys():
        missing = {"config", "vocab"} - metadata.keys()
        raise KeyError(f"Metadata file is missing required keys: {sorted(missing)}")

    cfg = training_config_from_json(metadata["config"])
    cfg.output_path = model_path  # keep config aligned with supplied checkpoint
    vocab = list(metadata["vocab"])
    return cfg, vocab, state_dict


def prepare_waveform(path: Path, target_sr: int) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found at {path}")

    waveform, sr = torchaudio.load(path)
    waveform = waveform.float()

    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0)
    else:
        waveform = waveform.squeeze(0)

    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)

    return waveform.clamp(-1.0, 1.0)


def transcribe(model_path: Path = MODEL_PATH, audio_path: Path = AUDIO_PATH) -> str:
    device = resolve_device(None)

    cfg, vocab, state_dict = load_model_artifacts(model_path, device)
    blank_idx = vocab.index("<blank>")

    model = DeepSpeech2Small(cfg, vocab_size=len(vocab)).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    feature_extractor = LogMelExtractor(cfg).to(device)
    feature_extractor.eval()

    waveform = prepare_waveform(audio_path, cfg.sample_rate).to(device)

    with torch.no_grad():
        features = feature_extractor(waveform.unsqueeze(0))
        log_probs = model(features)
        tokens = greedy_decode(log_probs, blank_idx)[0]

    transcript = indices_to_text(tokens, vocab)
    return transcript.strip()


if __name__ == "__main__":
    start_time = perf_counter()
    transcript = transcribe()
    elapsed = perf_counter() - start_time
    print(f"Transcript: {transcript}")
    print(f"Inference time: {elapsed:.2f}s")

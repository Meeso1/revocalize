from dataclasses import dataclass
import numpy as np


@dataclass
class UnprocessedTrainingData:
    audio_dir_path: str


@dataclass
class SavedPreprocessedData:
    content_dir_path: str


@dataclass
class PreprocessedData:
    content_vectors: np.ndarray # [n_segments, n_content_features]
    pitch_features: np.ndarray # [n_segments, n_pitch_features]
    audios: list[np.ndarray] # n_segments entries, each of shape [freq * sample_length,]


@dataclass
class FaissIndex:
    # TODO: Add representation
    pass


@dataclass
class InputData:
    # TODO: Verify
    content_vectors: np.ndarray # [n_samples, n_content_features]
    pitch_features: np.ndarray # [n_samples, n_pitch_features]


@dataclass
class OutputData:
    # TODO: Verify
    wav_data: list[np.ndarray] # n_samples entries, each of shape [freq * sample_length,]
from dataclasses import dataclass
import numpy as np


@dataclass
class UnprocessedTrainingData:
    audio_dir_path: str


@dataclass
class SavedPreprocessedData:
    content_dir_path: str


@dataclass
class AudioSegment:
    audio: np.ndarray
    sample_rate: int


@dataclass
class SegmentedAudio:
    segments: list[AudioSegment]


@dataclass
class PreprocessedSample:
    content_vector: np.ndarray
    pitch_feature: np.ndarray
    audio: np.ndarray


@dataclass
class PreprocessedData:
    content_vectors: np.ndarray
    pitch_features: np.ndarray
    audios: list[np.ndarray]

    @staticmethod
    def from_samples(samples: list[PreprocessedSample]) -> "PreprocessedData":
        return PreprocessedData(
            content_vectors=np.stack([s.content_vector for s in samples]),
            pitch_features=np.stack([s.pitch_feature for s in samples]),
            audios=[s.audio for s in samples],
        )


@dataclass
class FaissIndex:
    pass


@dataclass
class InputData:
    content_vectors: np.ndarray
    pitch_features: np.ndarray


@dataclass
class OutputData:
    wav_data: list[np.ndarray]

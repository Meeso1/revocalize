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
    audio: np.ndarray  # [n_samples] - float32, normalized [-1, 1]
    sample_rate: int  # Should be 16000 for ContentVec compatibility


@dataclass
class SegmentedAudio:
    segments: list[AudioSegment]


@dataclass
class PreprocessedSample:
    content_vector: np.ndarray  # [n_frames, n_content_features]
    pitch_feature: np.ndarray  # [n_frames, n_pitch_features]
    audio: np.ndarray  # [n_samples]


@dataclass
class PreprocessedData:
    content_vectors: np.ndarray  # [n_segments, n_frames, n_content_features]
    pitch_features: np.ndarray  # [n_segments, n_frames, n_pitch_features]
    audios: list[np.ndarray]  # [n_segments] each with shape [n_samples]
    
    @staticmethod
    def from_samples(samples: list[PreprocessedSample]) -> 'PreprocessedData':
        return PreprocessedData(
            content_vectors=np.stack([sample.content_vector for sample in samples]),
            pitch_features=np.stack([sample.pitch_feature for sample in samples]),
            audios=[sample.audio for sample in samples]
        )


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
    wav_data: list[np.ndarray]  # [n_samples] each with shape [n_audio_samples]
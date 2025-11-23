from data_models.data_models import AudioSegment, PreprocessedData, PreprocessedSample, SegmentedAudio


class AudioEncoder:
    def encode(self, segmented_audio: SegmentedAudio) -> PreprocessedData:
        # TODO: Implement
        pass
    
    def encode_segment(self, audio_segment: AudioSegment) -> PreprocessedSample:
        # TODO: Implement
        pass
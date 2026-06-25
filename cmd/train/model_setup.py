from dataset import ID_TO_LABEL, LABEL_TO_ID
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForAudioFrameClassification

MODEL_ID = "facebook/wav2vec2-xls-r-300m"
NUM_LABELS = len(LABEL_TO_ID)


def load_model_and_feature_extractor() -> tuple[Wav2Vec2ForAudioFrameClassification, Wav2Vec2FeatureExtractor]:
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForAudioFrameClassification.from_pretrained(
        MODEL_ID,
        num_labels=NUM_LABELS,
        label2id=LABEL_TO_ID,
        id2label=ID_TO_LABEL,
        ignore_mismatched_sizes=True,
    )

    # Freeze the CNN feature encoder — standard practice, big speedup, no quality loss
    model.freeze_feature_encoder()
    return model, feature_extractor

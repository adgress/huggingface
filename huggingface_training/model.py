
from transformers import AutoModelForImageClassification, AutoImageProcessor


def load_huggingface_pretrained_model(checkpoint, num_labels, id2label, label2id):
    """Load a pretrained model from Hugging Face."""
    try:
        model = AutoModelForImageClassification.from_pretrained(
            checkpoint,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id
        )
        print(f"Model '{checkpoint}' loaded successfully.")
        processor = AutoImageProcessor.from_pretrained(checkpoint, use_fast=True)
        return model, processor
    except Exception as e:
        print(f"Error loading model '{checkpoint}': {e}")
        raise ValueError(f"Failed to load model '{checkpoint}': {e}")
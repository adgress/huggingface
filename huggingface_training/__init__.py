"""
Hugging Face Training Package

A Python package for training Hugging Face models on Google Cloud Vertex AI.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import main components
from .data import (
    load_huggingface_dataset,
    load_beans_dataset,
    get_label_mappings,
    create_image_processor,
    create_transform_function,
    prepare_datasets,
)

from .model import (
    load_huggingface_pretrained_model,
)

from .trainer import (
    setup_training,
)

__all__ = [
    "load_huggingface_dataset",
    "load_beans_dataset", 
    "get_label_mappings",
    "create_image_processor",
    "create_transform_function",
    "prepare_datasets",
    "load_huggingface_pretrained_model",
    "setup_training",
]
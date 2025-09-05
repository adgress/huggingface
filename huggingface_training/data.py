from datasets import load_dataset, DatasetDict, Image
import ml_collections
from pydantic import ConfigDict
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor
from ml_collections import config_dict



def _load_huggingface_dataset(name) -> DatasetDict:
    """Load a dataset from the Hugging Face Hub."""
    try:
        dataset = load_dataset(name)
        assert isinstance(dataset, DatasetDict), "Loaded dataset is not a DatasetDict"
        print(f"Dataset '{name}' loaded successfully.")
        return dataset
    except Exception as e:
        print(f"Error loading dataset '{name}': {e}")
        raise ValueError(f"Failed to load dataset '{name}': {e}")


def load_beans_dataset() -> ml_collections.ConfigDict:
    """Load the beans dataset from Hugging Face"""
    cfg = config_dict.ConfigDict()
    cfg.dataset_dict = _load_huggingface_dataset("AI-Lab-Makerere/beans");
    cfg.id2label = {0: "angular_leaf_spot", 1: "bean_rust", 2: "healthy"}
    cfg.label2id = {"angular_leaf_spot": 0, "bean_rust": 1, "healthy": 2}
    return cfg

def create_image_processor(checkpoint="google/vit-base-patch16-224-in21k"):
    """Create and return an image processor"""
    processor = AutoImageProcessor.from_pretrained(checkpoint, use_fast=True)
    return processor


def create_transform_function(processor):
    """Create a transform function with the given processor"""
    def transform(example):
        example["pixel_values"] = processor(example["image"], return_tensors="pt")["pixel_values"][0]
        return example
    return transform


def prepare_datasets(dataset, processor):
    """Prepare train, validation, and test datasets"""
    transform = create_transform_function(processor)
    
    train_ds = dataset["train"].map(transform)
    val_ds = dataset["validation"].map(transform)
    test_ds = dataset["test"].map(transform)
    
    return train_ds, val_ds, test_ds


def create_dataloaders(train_ds, val_ds, test_ds, batch_size=16):
    """Create DataLoaders for training, validation, and test sets"""
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

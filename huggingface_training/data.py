from datasets import load_dataset, Image
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor


def load_huggingface_dataset(name):
    """Load a dataset from the Hugging Face Hub."""
    try:
        dataset = load_dataset(name)
        print(f"Dataset '{name}' loaded successfully.")
        return dataset
    except Exception as e:
        print(f"Error loading dataset '{name}': {e}")
        raise ValueError(f"Failed to load dataset '{name}': {e}")


def load_beans_dataset():
    """Load the beans dataset from Hugging Face"""
    return load_huggingface_dataset("AI-Lab-Makerere/beans")

def get_label_mappings():
    """Get the label mappings for the beans dataset"""
    id2label = {0: "angular_leaf_spot", 1: "bean_rust", 2: "healthy"}
    label2id = {"angular_leaf_spot": 0, "bean_rust": 1, "healthy": 2}
    return id2label, label2id

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

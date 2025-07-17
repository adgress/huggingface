
from typing import Tuple
from transformers import AutoImageProcessor
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from transformers.modeling_utils import PreTrainedModel
from torch.utils.data import DataLoader
from datasets import Dataset
from huggingface_training.data import prepare_datasets, create_dataloaders


def setup_training(
    dataset: Dataset, 
    model: PreTrainedModel, 
    processor: AutoImageProcessor, 
    batch_size: int = 16
) -> Tuple[Trainer, DataLoader, DataLoader, DataLoader]:
    """
    Set up the training pipeline including dataset loading, model initialization, and trainer setup.
    
    Args:
        dataset (Dataset): The Hugging Face dataset containing train/validation/test splits
        model (PreTrainedModel): The pre-trained model for image classification
        processor (AutoImageProcessor): The image processor for preprocessing
        batch_size (int): Batch size for training and evaluation
    
    Returns:
        Tuple[Trainer, DataLoader, DataLoader, DataLoader]: A tuple containing (trainer, train_loader, val_loader, test_loader)
    """
    # Load dataset and get label mappings
    
    
    # Prepare datasets
    train_ds, val_ds, test_ds = prepare_datasets(dataset, processor)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(train_ds, val_ds, test_ds, batch_size=batch_size)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_steps=100,
        num_train_epochs=3,
        eval_strategy="epoch",  # Changed from evaluation_strategy
        save_strategy="epoch",
        logging_dir="./logs"
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=None,  # Will use default collator
    )
    
    return trainer, train_loader, val_loader, test_loader
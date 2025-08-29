from collections import defaultdict
import logging
import os
from typing import Tuple
import numpy as np
from evaluate import load
from sklearn.metrics import accuracy_score
from transformers import AutoImageProcessor
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from transformers.modeling_utils import PreTrainedModel
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict
from huggingface_training.data import prepare_datasets, create_dataloaders
from transformers.integrations.integration_utils import TensorBoardCallback
from torch.utils.tensorboard import SummaryWriter


accuracy_metric = load("accuracy")
logger = logging.getLogger(__name__)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    overall_accuracy = accuracy_metric.compute(predictions=preds, references=labels)
    if overall_accuracy is None:
        raise ValueError("Accuracy metric computation failed.")
    # Per-class accuracy
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    for label, pred in zip(labels, preds):
        per_class_total[label] += 1
        if label == pred:
            per_class_correct[label] += 1
    per_class_accuracy = {
        f"accuracy_class_{label}": per_class_correct[label] / per_class_total[label]
        for label in per_class_total
    }
    metrics = overall_accuracy | per_class_accuracy
    logger.info("Logging metrics:", metrics)
    return metrics


def setup_training(
    dataset: DatasetDict, 
    model: PreTrainedModel, 
    processor: AutoImageProcessor, 
    batch_size: int = 16
) -> Tuple[Trainer, Dataset]:
    """
    Set up the training pipeline including dataset loading, model initialization, and trainer setup.
    
    Args:
        dataset (Dataset): The Hugging Face dataset containing train/validation/test splits
        model (PreTrainedModel): The pre-trained model for image classification
        processor (AutoImageProcessor): The image processor for preprocessing
        batch_size (int): Batch size for training and evaluation
    
    Returns:
        Tuple[Trainer, Dataset]: A tuple containing (trainer, test_dataset)
    """
    # Load dataset and get label mappings
    
    
    # Prepare datasets
    train_ds, val_ds, test_ds = prepare_datasets(dataset, processor)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(train_ds, val_ds, test_ds, batch_size=batch_size)

    log_path = f"gs://huggingface-vertex-artifacts/tensorboard/vit-experiment/{os.environ.get('VERTEX_RUN_NAME')}"
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_steps=10,
        num_train_epochs=2,
        eval_strategy="steps",
        eval_steps=1,
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=1,
        
        # logging_dir="./logs",
        logging_dir=log_path,
        disable_tqdm=True,
        report_to=["tensorboard"]  # Optional: report to TensorBoard
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=None,  # Will use default collator
        compute_metrics=compute_metrics,  # Add metrics computation
        callbacks=[TensorBoardCallback()]
    )
        
    writer = SummaryWriter(log_path)
    writer.add_scalar("debug/scalar", 1.0, 0)
    writer.close()
    
    return trainer, test_ds
from collections import defaultdict
import logging
import os
import subprocess
from typing import Dict, Optional, Tuple
import numpy as np
from evaluate import load
from sklearn.metrics import accuracy_score
from transformers import AutoImageProcessor
from transformers.trainer import Trainer
from transformers.trainer_callback import TrainerCallback
from transformers.training_args import TrainingArguments
from transformers.modeling_utils import PreTrainedModel
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict
from huggingface_training.data import prepare_datasets, create_dataloaders
from transformers.integrations.integration_utils import TensorBoardCallback
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path


accuracy_metric = load("accuracy")
logger = logging.getLogger(__name__)

class TrainMetricsCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics: Optional[Dict] = None, **kwargs):
        """
        Called after the regular evaluation on the validation set.
        Used here to also evaluate and log metrics for the training set.
        """
        # Get the trainer instance from the kwargs
        trainer = kwargs["trainer"]
        
        # We need to manually trigger an evaluation on the training dataset
        # and specify the metric prefix
        train_metrics = trainer.evaluate(
            eval_dataset=trainer.train_dataset,
            metric_key_prefix="train"
        )
        
        # Log the training metrics
        trainer.log(train_metrics)



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
    batch_size: int = 64
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
        
    local_output = Path("/tmp/hf_outputs")
    local_tb  = Path("/tmp/tb")
    local_output.mkdir(parents=True, exist_ok=True)
    local_tb.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(__name__)
    tb_log_path = os.environ.get("AIP_TENSORBOARD_LOG_DIR", "./logs")
    logger.info("AIP_TENSORBOARD_LOG_DIR path: " + tb_log_path)
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=str(local_output),
        logging_dir=str(local_tb),

        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_steps=500,
        num_train_epochs=20,
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="steps",
        logging_strategy="steps",
        logging_steps=20,
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
        callbacks=[TrainMetricsCallback()]
    )
        
    writer = SummaryWriter(tb_log_path)
    writer.add_scalar("debug/scalar", 1.0, 0)
    writer.close()
    
    return trainer, test_ds


def upload_artifacts_to_gcs(local_tb_dir="/tmp/tb", local_out_dir="/tmp/hf_outputs"):
    """Call this after trainer.train() finishes."""
    gcs_tb = os.getenv("AIP_TENSORBOARD_LOG_DIR")
    gcs_model_dir = os.getenv("AIP_MODEL_DIR")  # also gs://...

    def _rsync(local, gcs):
        if gcs and gcs.startswith("gs://"):
            try:
                subprocess.run(["gsutil", "-m", "rsync", "-r", local, gcs], check=False)
            except FileNotFoundError:
                # fallback to python client if gsutil isn't available
                from google.cloud import storage
                client = storage.Client()
                bucket_name, prefix = gcs[5:].split("/", 1)
                bucket = client.bucket(bucket_name)
                for p in Path(local).rglob("*"):
                    if p.is_file():
                        blob = bucket.blob(f"{prefix}/{p.relative_to(local)}")
                        blob.upload_from_filename(str(p))

    _rsync(local_tb_dir, gcs_tb)        # push TB logs
    _rsync(local_out_dir, gcs_model_dir)  # push checkpoints/artifacts
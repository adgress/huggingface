import os
from pdb import run

from huggingface_hub import login
from .data import load_beans_dataset, create_image_processor, get_label_mappings
from .model import load_huggingface_pretrained_model
from .trainer import setup_training, upload_artifacts_to_gcs

import google.auth
from google.cloud import aiplatform

from google.cloud.logging import Client
from google.cloud.logging_v2.handlers import CloudLoggingHandler
from google.cloud.logging_v2.handlers.transports import BackgroundThreadTransport

import logging
import sys



def main():
    """Main training function that can be called as a console script"""
    print("Training script started...")
    
    logging_level = logging.INFO
    # 1. Integrate the Python logging module with Google Cloud Logging.
    # This ensures that all logs are captured and sent to Cloud Logging.
    # Setting the root logger to DEBUG here will catch most logs.
    client = Client()
    client.setup_logging(log_level=logging_level)

    # 2. Add a StreamHandler to ensure logs are also printed to the console
    # during development, even with the Cloud Logging integration.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)

    # 3. Explicitly set the logging level for the Vertex AI SDK.
    # This is a good practice to control logging from specific libraries.
    aiplatform_logger = logging.getLogger("google.cloud.aiplatform")
    aiplatform_logger.setLevel(logging_level)

    # Example usage with your code and the SDK
    logger = logging.getLogger(__name__)
    logger.info("Initializing Vertex AI SDK...")
    logger.info("This message will appear in Cloud Logging!")
    
    # Ensure the Hugging Face Hub is logged in
    logger.info("Logging in to Hugging Face Hub...")
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    if hf_token:
        login(token=hf_token)
    else:
        raise ValueError("HUGGINGFACE_TOKEN environment variable is not set. Please set it to your Hugging Face token.")
    logger.info("Hugging Face Hub login successful.")

    # Initialize Vertex AI experiment if running on Vertex AI
    experiment_name = os.environ.get("VERTEX_EXPERIMENT_NAME")
    logger.info(f"Starting Vertex AI experiment: {experiment_name}")
    aiplatform.init(
        experiment=experiment_name,
        project=os.environ.get("PROJECT_ID"),
        location=os.environ.get("REGION"),
        staging_bucket=os.environ.get("STAGING_BUCKET"),
        )
    aiplatform.start_run(run=os.environ.get("VERTEX_RUN_NAME"))

    dataset = load_beans_dataset()
    print(dataset)
    id2label, label2id = get_label_mappings()
    checkpoint="google/vit-base-patch16-224-in21k"
    processor = create_image_processor(checkpoint)
    
    # Load model
    model, _ = load_huggingface_pretrained_model(
        checkpoint,
        num_labels=3,  # for the beans dataset
        id2label=id2label,
        label2id=label2id
    )
    # Set up training pipeline
    trainer, test_data = setup_training(dataset, model, processor, batch_size=64)
    
    # Start training - the Trainer already has the datasets and training configuration
    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    validation_metrics = trainer.evaluate(metric_key_prefix="validation")
    test_metrics = trainer.predict(test_data, metric_key_prefix="test").metrics
    logger.info("Logging Metrics:")
    logger.info(f"Validation Metrics: {validation_metrics}")
    logger.info(f"Test Metrics: {test_metrics}")
    # some nice to haves:
    trainer.log_metrics("validation", validation_metrics)
    trainer.save_metrics("validation", validation_metrics)
    trainer.log_metrics("test", test_metrics)
    trainer.save_metrics("test", test_metrics)
    
    aiplatform.log_metrics(train_results.metrics)
    aiplatform.log_metrics(validation_metrics)
    aiplatform.log_metrics(test_metrics)
    
    upload_artifacts_to_gcs()
    aiplatform.end_run()
    logger.info("Done training and evaluation.")


if __name__ == "__main__":
    main()

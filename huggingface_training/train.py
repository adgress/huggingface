import os

from huggingface_hub import login
from .data import load_beans_dataset, create_image_processor, get_label_mappings
from .model import load_huggingface_pretrained_model
from .trainer import setup_training



def main():
    """Main training function that can be called as a console script"""
    # Ensure the Hugging Face Hub is logged in
    print("Logging in to Hugging Face Hub...")
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    if hf_token:
        login(token=hf_token)
    else:
        raise ValueError("HUGGINGFACE_TOKEN environment variable is not set. Please set it to your Hugging Face token.")
    print("Hugging Face Hub login successful.")


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
    trainer, train_loader, val_loader, test_loader = setup_training(dataset, model, processor, batch_size=16)

    # trainer.train()


if __name__ == "__main__":
    main()

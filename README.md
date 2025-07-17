# Hugging Face Training Package

A Python package for training Hugging Face models on Google Cloud Vertex AI.

## Features

- Easy loading and preprocessing of Hugging Face datasets
- Pre-configured models for image classification tasks
- Seamless integration with Google Cloud Vertex AI
- Command-line tools for training and deployment

## Installation

### From Source

```bash
git clone https://github.com/adgress/huggingface.git
cd huggingface
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/adgress/huggingface.git
cd huggingface
pip install -e ".[dev]"
```

## Usage

### Environment Setup

Before using the package, set up your environment variables:

```bash
export HUGGINGFACE_TOKEN="your_huggingface_token_here"
```

### Command Line Usage

#### Training Locally

```bash
hf-train
```

#### Submitting to Vertex AI

```bash
hf-vertex-submit
```

### Python API Usage

```python
from huggingface_training import (
    load_beans_dataset,
    create_image_processor,
    get_label_mappings,
    load_huggingface_pretrained_model,
    setup_training
)

# Load dataset
dataset = load_beans_dataset()
id2label, label2id = get_label_mappings()

# Create processor
processor = create_image_processor("google/vit-base-patch16-224-in21k")

# Load model
model, _ = load_huggingface_pretrained_model(
    "google/vit-base-patch16-224-in21k",
    num_labels=3,
    id2label=id2label,
    label2id=label2id
)

# Set up training
trainer, train_loader, val_loader, test_loader = setup_training(
    dataset, model, processor, batch_size=16
)
```

## Configuration

### Google Cloud Setup

1. Set up your Google Cloud project and enable the Vertex AI API
2. Configure authentication (service account key or gcloud auth)
3. Update the project settings in `vertex_submit.py`

### Vertex AI Configuration

Update the following in your code:
- `project`: Your Google Cloud project ID
- `location`: Your preferred region
- `staging_bucket`: Your GCS bucket for artifacts

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black huggingface_training/
isort huggingface_training/
```

### Type Checking

```bash
mypy huggingface_training/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
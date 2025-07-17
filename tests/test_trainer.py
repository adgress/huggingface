import pytest

from huggingface_training.trainer import setup_training
from transformers import AutoImageProcessor
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from transformers.modeling_utils import PreTrainedModel
from torch.utils.data import DataLoader
from datasets import Dataset


class TestSetupTraining:
    """Test cases for the setup_training function."""
    
    @pytest.fixture(autouse=True)
    def setup_method(self, mocker):
        """Set up test fixtures before each test method."""
        # Mock dataset
        self.mock_dataset = mocker.Mock(spec=Dataset)
        
        # Mock model
        self.mock_model = mocker.Mock(spec=PreTrainedModel)
        
        # Mock processor
        self.mock_processor = mocker.Mock(spec=AutoImageProcessor)
        
        # Mock datasets returned by prepare_datasets
        self.mock_train_ds = mocker.Mock()
        self.mock_val_ds = mocker.Mock()
        self.mock_test_ds = mocker.Mock()
        
        # Mock dataloaders
        self.mock_train_loader = mocker.Mock(spec=DataLoader)
        self.mock_val_loader = mocker.Mock(spec=DataLoader)
        self.mock_test_loader = mocker.Mock(spec=DataLoader)
        
        # Mock trainer
        self.mock_trainer = mocker.Mock(spec=Trainer)
    
    @pytest.fixture
    def mock_training_components(self, mocker):
        """Fixture to create commonly used mocks for training components."""
        mocks = {
            'prepare_datasets': mocker.patch('huggingface_training.trainer.prepare_datasets'),
            'create_dataloaders': mocker.patch('huggingface_training.trainer.create_dataloaders'),
            'trainer_class': mocker.patch('huggingface_training.trainer.Trainer'),
            'training_args': mocker.patch('huggingface_training.trainer.TrainingArguments'),
        }
        
        # Configure default returns
        mocks['prepare_datasets'].return_value = (self.mock_train_ds, self.mock_val_ds, self.mock_test_ds)
        mocks['create_dataloaders'].return_value = (self.mock_train_loader, self.mock_val_loader, self.mock_test_loader)
        mocks['trainer_class'].return_value = self.mock_trainer
        mocks['training_args'].return_value = mocker.Mock()
        
        return mocks
    
    def test_setup_training_default_params(self, mock_training_components):
        """Test setup_training with default parameters."""
        # Call the function
        result = setup_training(self.mock_dataset, self.mock_model, self.mock_processor)
        
        # Verify the result
        assert len(result) == 4
        trainer, train_loader, val_loader, test_loader = result
        
        assert trainer == self.mock_trainer
        assert train_loader == self.mock_train_loader
        assert val_loader == self.mock_val_loader
        assert test_loader == self.mock_test_loader
    
    def test_setup_training_custom_batch_size(self, mock_training_components):
        """Test setup_training with custom batch size."""
        # Call the function with custom batch size
        batch_size = 32
        result = setup_training(self.mock_dataset, self.mock_model, self.mock_processor, batch_size=batch_size)
        
        # Verify the result
        assert len(result) == 4
        
        # Verify create_dataloaders was called with custom batch size
        mock_training_components['create_dataloaders'].assert_called_once_with(
            self.mock_train_ds, self.mock_val_ds, self.mock_test_ds, batch_size=batch_size
        )
        
        # Verify TrainingArguments was called with custom batch size
        mock_training_components['training_args'].assert_called_once()
    
    def test_setup_training_function_call_order(self, mock_training_components):
        """Test that functions are called in the correct order."""
        # Call the function
        setup_training(self.mock_dataset, self.mock_model, self.mock_processor)
        
        # Verify call order using mock_calls
        # prepare_datasets should be called before create_dataloaders
        prepare_call_index = None
        dataloaders_call_index = None
        
        all_calls = (mock_training_components['prepare_datasets'].mock_calls + 
                    mock_training_components['create_dataloaders'].mock_calls)
        
        for i, call in enumerate(all_calls):
            if 'prepare_datasets' in str(call):
                prepare_call_index = i
            elif 'create_dataloaders' in str(call):
                dataloaders_call_index = i
        
        # Verify that prepare_datasets was called
        mock_training_components['prepare_datasets'].assert_called_once()
        mock_training_components['create_dataloaders'].assert_called_once()
    
    def test_setup_training_with_exception(self, mocker):
        """Test setup_training behavior when an exception occurs."""
        # Create mocks using mocker
        mock_prepare_datasets = mocker.patch('huggingface_training.trainer.prepare_datasets')
        mock_create_dataloaders = mocker.patch('huggingface_training.trainer.create_dataloaders')
        
        # Configure mock to raise an exception
        mock_prepare_datasets.side_effect = Exception("Dataset preparation failed")
        
        # Verify that the exception is raised
        with pytest.raises(Exception) as context:
            setup_training(self.mock_dataset, self.mock_model, self.mock_processor)
        
        assert str(context.value) == "Dataset preparation failed"
    
    def test_setup_training_return_type(self, mock_training_components):
        """Test that setup_training returns the correct types."""
        # Call the function
        result = setup_training(self.mock_dataset, self.mock_model, self.mock_processor)
        
        # Verify return type
        assert isinstance(result, tuple)
        assert len(result) == 4
        
        trainer, train_loader, val_loader, test_loader = result
        
        # Verify individual return types
        assert trainer == self.mock_trainer
        assert train_loader == self.mock_train_loader
        assert val_loader == self.mock_val_loader
        assert test_loader == self.mock_test_loader



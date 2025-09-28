import torch
import pandas as pd
import pytest
from PIL import Image
from pathlib import Path

from src import config, utils
from src.dataset import MultimodalHateSpeechDataset
from src.fusion_model import FusionModel

# Create a dummy data file for testing
@pytest.fixture(scope="session")
def dummy_data(tmpdir_factory):
    data_dir = tmpdir_factory.mktemp("data")
    feature_dir = tmpdir_factory.mktemp("features")
    img_dir = data_dir.mkdir("images")

    # Create a dummy image file
    dummy_img_path = img_dir.join("img1.jpg")
    Image.new('RGB', (10, 10), color = 'red').save(str(dummy_img_path))

    dummy_csv = data_dir.join("test.csv")
    df = pd.DataFrame({
        "id": ["sample1", "sample2", "sample3"],
        "text": ["some text", "another text", None],
        "img_path": ["images/img1.jpg", None, "images/img3.jpg"],
        "label": [0, 1, 0]
    })
    df.to_csv(dummy_csv, index=False)
    
    return {"data_dir": Path(str(data_dir)), "feature_dir": Path(str(feature_dir)), "df": df}

def test_dataset_loading(dummy_data):
    """Tests if the dataset can be initialized and an item can be fetched."""
    dataset = MultimodalHateSpeechDataset(dummy_data["df"], dummy_data["data_dir"], dummy_data["feature_dir"], use_precomputed=False)
    assert len(dataset) == 3
    
    # Test a sample with text and image (even if image file doesn't exist)
    sample1 = dataset[0]
    assert sample1['has_text'] == torch.tensor(1.0)
    assert sample1['has_image'] == torch.tensor(1.0) # Image file exists
    assert sample1['text_input_ids'].shape == (config.MODEL_CONFIG['text']['max_length'],)

    # Test a sample with only text
    sample2 = dataset[1]
    assert sample2['has_text'] == torch.tensor(1.0)
    assert sample2['has_image'] == torch.tensor(0.0)

def test_fusion_model_forward_pass():
    """Tests if the fusion model can perform a forward pass with dummy data."""
    device = utils.get_device()
    model = FusionModel(
        text_dim=config.MODEL_CONFIG['text']['embedding_dim'],
        image_dim=config.MODEL_CONFIG['image']['embedding_dim'],
        aux_dim=config.MODEL_CONFIG['aux']['embedding_dim'],
        hidden_dim=512,
        num_classes=1,
        dropout=0.1
    ).to(device)
    
    batch_size = 4
    dummy_text = torch.randn(batch_size, config.MODEL_CONFIG['text']['embedding_dim']).to(device)
    dummy_image = torch.randn(batch_size, config.MODEL_CONFIG['image']['embedding_dim']).to(device)
    dummy_aux = torch.randn(batch_size, config.MODEL_CONFIG['aux']['embedding_dim']).to(device)
    has_modality = torch.ones(batch_size, dtype=torch.float).to(device)
    
    logits, _ = model(dummy_text, dummy_image, dummy_aux, has_modality, has_modality, has_modality)
    assert logits.shape == (batch_size, 1)
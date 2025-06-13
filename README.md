# TRACE: Tracking Representation Abstraction and Compositional Emergence
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**TRACE** is a comprehensive Python package for analyzing transformer models during training. It provides modular tools for understanding model behavior through linguistic probes, intrinsic dimension analysis, Hessian landscape exploration, and more.

## Features

- **Linguistic Probes**: Monitor syntactic and semantic presence across layers
- **Intrinsic Dimensions**: Track representation complexity using TwoNN and other methods  
- **Hessian Analysis**: Explore loss landscapes, sharpness, and training dynamics
- **Real-time Monitoring**: Comprehensive visualization during training
- **Modular Design**: Use individual components or full training integration
- **Easy Integration**: Drop-in replacement for existing training loops

## Installation

```bash
# Clone the repository
git https://github.com/nura-j/trace_package.git
cd trace

# [Optional] create and activate the conda environment
conda env create -f environment.yml
conda activate TRACE

# Install
pip install .
```


### Basic Usage

```python
# Create tokenizer 
VOCAB_SIZE = 1000  # Example vocabulary size
CORPUS_PATH = "path/to/corpus.json"  # Path to your corpus file
CORPUS_PATH = "/Users/user/Desktop/Year_3/trace_package/data/corpus_full.json"
MODEL_TYPE = "decoder_only"  # or "encoder-decoder" or "encoder-only"
MAX_SEQ_LENGTH = 64  # Example max sequence length
from trace.tokenizer import create_tokenizer_from_data
tokenizer = create_tokenizer_from_data(vocab_file=CORPUS_PATH)
VOCAB_SIZE = tokenizer.get_vocab_size()

# Create a Transformer model
from trace.transformer import Transformer, TransformerConfig
model_config = TransformerConfig(
    model_type=MODEL_TYPE,
    vocab_size=VOCAB_SIZE,
    d_model=768,
    num_heads=12,
    num_decoder_layers=1,
    d_ff=3072,
    max_seq_length=MAX_SEQ_LENGTH,
    dropout=0.1,
    device="cpu"  
)

model = Transformer.from_config(model_config)

# Create Training config 
from trace.training import Trainer, TrainingConfig
training_config = TrainingConfig(
    epochs=1,
    learning_rate=1e-3,
    batch_size=128,
    track_linguistic_probes=True,
    track_intrinsic_dimensions=True,
    track_hessian=True,
    track_component_hessian=True,
    track_pos_performance=True,
    track_semantic_roles_performance=True,
    track_interval=100,  # Analyze every 100 steps
    plots_path="./analysis_results"
)

# Create trainer
trainer = Trainer(training_config, tokenizer, model)

# Create dataloader
from trace.dataloader import get_dataloader
train_loader, val_loader, test_loader = get_dataloader(CORPUS_PATH, tokenizer=tokenizer, batch_size=32, max_length=MAX_SEQ_LENGTH, model_type=MODEL_TYPE, val_split= 0.1,test_split= 0.1)
best_loss, results = trainer.train(train_loader, val_loader, val_loader, model)


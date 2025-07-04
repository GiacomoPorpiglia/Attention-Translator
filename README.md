# English-to-French Neural Machine Translation

A PyTorch implementation of a Transformer-based neural machine translation model for English-to-French translation. This project implements the classic encoder-decoder architecture with attention mechanisms from scratch, inspired form the famous paper "Attention Is All You Need".

## 📋 Requirements

```
torch
pandas
kagglehub
tokenizers
tqdm
regex
```

## 🏗️ Architecture

### Encoder
- Multi-layer Transformer encoder with non-causal self-attention
- Positional encoding for sequence modeling
- Layer normalization and residual connections
- Configurable number of layers and attention heads

### Decoder
- Multi-layer Transformer decoder with:
  - Causal self-attention (masked)
  - Cross-attention to encoder outputs
- Teacher forcing during training
- Autoregressive generation during inference

### Key Components
- **Attention Blocks**: Custom implementation with Flash Attention support
- **Positional Encoding**: Sinusoidal positional embeddings
- **MLP Layers**: Feed-forward networks with GELU activation
- **Custom Tokenizer**: BPE tokenizer trained on the dataset

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/GiacomoPorpiglia/Attention-Translator.git
cd Attention-Translator
```

### 2. Install Dependencies
```bash
pip install torch pandas kagglehub tokenizers tqdm regex
```

### 3. Configure Training Parameters
Edit `config.py` to adjust hyperparameters:
```python
max_seq_len = 128
embd_dim = 512
mini_batch_size = 64
start_lr = 1e-4
# ... other parameters
```

### 4. Train the Model
```bash
python main.py --mode train
```

The script will:
- Download the EN-FR translation dataset from Kaggle
- Train a custom BPE tokenizer (if not exists)
- Initialize the model architecture
- Start training with validation

### 5. Test Translation
```bash
python main.py --mode test --model_path your_cehckpoint_path.pth
```
You can find my trained checkpoint under <i>Releases</i> on the Github repo.
## 📁 Project Structure

```
├── main.py               # Main script, training and test logic
├── model.py              # Model architecture definitions
├── my_tokenizer.py       # BPE tokenizer training and loading
├── PhrasesDataset.py     # Dataset class for handling translation pairs
├── BatchSampler.py       # Custom batch sampler for efficient training
├── positional_encoder.py # Positional encoding implementation
├── config.py             # Training configuration and hyperparameters
└── README.md             # This file
```

## ⚙️ Configuration

Key hyperparameters in `config.py`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_seq_len` | Maximum sequence length (in tokens) | 128 |
| `embd_dim` | Model embedding dimension | 512 |
| `mini_batch_size` | Mini Batch size | 64 |
| `batch_size` | Batch size | 1024 |
| `start_lr` | Initial learning rate | 1e-4 |
| `min_lr` | Final learning rate | 1e-5 |
| `warmup_iters` | Warmup iterations | 2000 |
| `lr_decay_iters` | Learning rate decay iters | 100000 |
| `weight_decay` | Weight decay for regularization | 1e-5 |

## 🔧 Model Architecture Details

### Encoder
- **Layers**: 5 Transformer blocks
- **Attention Heads**: 8 per block
- **Hidden Dimension**: 512
- **Vocabulary Size**: 20,000 tokens
- **Dropout**: 0.15

### Decoder
- **Layers**: 5 Transformer blocks
- **Attention Heads**: 8 per block
- **Self-Attention**: Causal masking
- **Cross-Attention**: Attends to encoder outputs
- **Output**: Vocabulary distribution via linear projection

## 📊 Training Features

### Learning Rate Scheduling
- **Warmup Phase**: Linear increase to `start_lr`
- **Decay Phase**: Cosine decay to `min_lr`
- **Schedule Function**: Custom implementation in `get_lr()`

### Data Processing
- **Tokenization**: Custom BPE tokenizer with 20K vocabulary
- **Filtering**: Latin script filtering for clean data, avoiding non-latin characters
- **Batching**: Bucket batching by sequence length
- **Padding**: Dynamic padding with attention masks

### Training Loop
- **Gradient Accumulation**: Effective batch size scaling
- **Gradient Clipping**: Prevents exploding gradients
- **Validation**: Regular evaluation on held-out data
- **Checkpointing**: Automatic model state saving

## 📈 Dataset

The model is trained on the [EN-FR Translation Dataset](https://www.kaggle.com/datasets/dhruvildave/en-fr-translation-dataset) from Kaggle, containing parallel English-French sentence pairs. The dataset was cleaned to have only Latin characters, and was not used etirely for training cost reasons. Approzimately 4M sentence pairs were used for training.


## 📝 Example Output

```
Input english text: One of the most widely recognised animal symbols in human culture, the lion has been extensively depicted in sculptures and paintings.

(Version 1) -->  Output french text:  l'un des symboles les plus largement reconnus de la culture humaine est celui de l'espèce qui est largement caractérisée dans les sculptures et les peintures.

(Version 2) -->  Output french text:  lâun des symboles animaux les plus largement reconnus dans le domaine de la culture humaine, le lon a été largement décrit dans les sculptures et les peintures.

(Version 3) -->  Output french text:  un des symboles animaux le plus largement reconnus dans la culture humaine, lâor est largement caractérisé par des sculptures et des peintures.
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original Transformer architecture from "Attention Is All You Need" (Vaswani et al., 2017)
- Dataset from Kaggle user dhruvildave

---

⭐ **If you find this project helpful, please consider giving it a star!** ⭐
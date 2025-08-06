# English-to-French Neural Machine Translation

A PyTorch implementation of a Transformer-based translation model for English-to-French translation. This project implements the classic encoder-decoder architecture with attention mechanisms from scratch, inspired form the famous paper "Attention Is All You Need".<br>
#### Disclaimer: The pre-trained model you will find under releases has been trained for only 11 epochs due to training costs, so don't expect it to be the perfect translator! This project is intended as a way to familiarize with the seq2seq transformer architecture for now. Maybe in the future I will make a more serious training to have better performance!

## 📋 Requirements

```
torch
pandas==2.2.3
kagglehub==0.3.12
tokenizers==0.21.0
tqdm
```

## 🏗️ Architecture

### Encoder (21M parameters)
- Multi-layer Transformer encoder with non-causal self-attention
- Positional encoding for sequence modeling
- Layer normalization and residual connections
- Configurable number of layers and attention heads

### Decoder (46M parameters)
- Multi-layer Transformer decoder with:
  - Causal self-attention (masked)
  - Cross-attention to encoder outputs
- Teacher forcing during training
- Autoregressive generation during inference

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
| `max_seq_len` | Maximum sequence length (in tokens) | 192 |
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

## 📈 Dataset

The model is trained on the [EN-FR Translation Dataset](https://www.kaggle.com/datasets/dhruvildave/en-fr-translation-dataset) from Kaggle, containing parallel English-French sentence pairs. Approzimately 6M sentence pairs were used for training.


## 📝 Example Output

```
Input english text: Technology has changed the way we communicate, making it easier to stay connected but harder to truly disconnect.

(Version 1) -->  Output french text:  les technologies ont modifié notre façon de communiquer, de rendre plus faciles à demeurer branchées, et plus faciles à dissiper vraiment.

(Version 2) -->  Output french text:  la technologie a modifié notre façon de communiquer, ce qui rend plus facile la connexion de rester en contact, mais plus difficile de s’adapter à un problème véritable.

(Version 3) -->  Output french text:  la technologie a changé la façon dont nous communiquerons, de façon à le faire plus facile de demeurer branchée mais de s’adapter aux problèmes qui s’y rattachent.
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original Transformer architecture from "Attention Is All You Need" (Vaswani et al., 2017)
- Dataset from Kaggle user dhruvildave

---

⭐ **If you find this project helpful, please consider giving it a star!** ⭐
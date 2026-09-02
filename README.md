# Deep Learning Term Project — Image Captioning

**Team:** Mind Mavericks 

This project explores image caption generation using two different encoder-decoder architectures. The first approach combines a pretrained CNN image encoder with an LSTM-based text decoder, while the second uses a Vision Transformer (ViT) as the image encoder and GPT-2 as the text decoder.

## Project Overview

The project is divided into two parts:

### Part A — CNN Encoder + RNN Decoder

A pretrained **VGG16** network is used as the image encoder and a single-cell **LSTM** is used as the text decoder.

**Pipeline:**
- Images are resized to **224×224×3** and normalized.
- Captions are tokenized with `nltk.tokenize.word_tokenize()` and converted to lowercase.
- Start and padding/end tokens are added and a vocabulary is constructed.
- VGG16 pretrained on **ImageNet** extracts image features.
- All CNN layers except the final linear layer are frozen.
- The final linear layer maps the image representation to the required embedding size.
- Word tokens are converted through an embedding layer.
- Image features and word embeddings are concatenated and passed to the LSTM.
- The decoder predicts the next word until the end token is generated or the maximum caption length is reached.

**Training:**
- Optimizer: Adam
- Learning rate: `1e-4`
- Loss: Cross Entropy
- Epochs: `5`
- Batch size: `1`

Adam performed better than SGD in the experiments. A learning rate of `1e-3` led to similar losses after a few epochs, while `1e-5` made training considerably slower.

**Results:**
- BLEU: **0.08**
- ROUGE-L: **0.15**
- Captions were reported to be reasonably accurate on more than **50%** of test images.

Generated captions were stored in `result.txt`.

## Part B — Vision Transformer + GPT-2

The second approach uses a pretrained **Vision Transformer (ViT)** as the image encoder and **GPT-2** as the text decoder through the `VisionEncoderDecoderModel` framework.

**Models:**
- Encoder: `google/vit-base-patch16-224`
- Decoder: `gpt2`

**Preprocessing:**
- Training, validation, and test image paths and caption CSVs are loaded.
- Images are checked for valid dimensions using PIL.
- Invalid images are filtered out.
- Rows containing missing/NaN values are removed.
- `ViTImageProcessor` prepares images for the vision encoder.
- `GPT2TokenizerFast` handles text tokenization.

**Training:**
- Trainer: `Seq2SeqTrainer`
- Optimizer: AdamW
- Learning rate: `1e-5`
- Epochs: `10`
- Batch size: `16`
- Evaluation metrics: BLEU and ROUGE-L

**Results:**
- BLEU: **0.1750**
- ROUGE-L: **0.463**
- Captions were reported to be reasonably accurate on more than **60%** of test images.

The transformer-based approach produced substantially better BLEU and ROUGE-L scores than the CNN-LSTM approach.

## Comparison

| Approach | Image Encoder | Text Decoder | BLEU | ROUGE-L |
|---|---|---|---:|---:|
| Part A | VGG16 | LSTM | 0.08 | 0.15 |
| Part B | ViT | GPT-2 | 0.1750 | 0.463 |

## Key Takeaway

The project demonstrates two generations of image-captioning architectures:

**CNN + LSTM** → extracts visual features with VGG16 and generates captions sequentially with an LSTM.

**ViT + GPT-2** → combines transformer-based visual representations with a transformer language model for caption generation.

The second architecture achieved considerably stronger evaluation scores on the reported test set.

## Team

**Mind Mavericks — Team G-37**

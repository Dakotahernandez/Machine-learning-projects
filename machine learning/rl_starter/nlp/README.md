# Natural Language Processing

PyTorch NLP from scratch: LSTM sentiment analysis, text classification, and word embeddings.

## Projects

### `sentiment_lstm.py` — IMDB Sentiment Analysis
- Bidirectional LSTM with learned word embeddings
- Packed sequences for variable-length input
- Achieves **85%+** accuracy on IMDB movie reviews
- Attention-weighted visualisation of important words

### `text_classifier.py` — Multi-class News Classification
- TextCNN (Kim 2014) — parallel convolutions over word embeddings
- Uses AG News dataset (4 classes: World, Sports, Business, Sci/Tech)
- Fast training with 1D convolutions

### `word_embeddings.py` — Word2Vec from Scratch
- Skip-gram with negative sampling
- Train embeddings on any text file
- Visualise word clusters with t-SNE
- Analogy solving (king - man + woman = queen)

## Quick Start

```powershell
pip install -r requirements.txt

# Sentiment analysis (downloads IMDB automatically)
python sentiment_lstm.py --epochs 5 --device auto

# News classification
python text_classifier.py --epochs 10 --device auto

# Word embeddings
python word_embeddings.py --epochs 5
```

# Gemma — Google's Open-Source LLM

Hands-on projects with **Google Gemma** — the open-weight family of lightweight, state-of-the-art language models. Covers text generation, fine-tuning, RAG, structured output, embeddings, and more.

## Projects

| Script | What It Does |
|--------|-------------|
| `text_generation.py` | Load Gemma and generate text with sampling strategies (top-k, top-p, temperature, beam search) |
| `chat.py` | Interactive multi-turn chat with Gemma-IT (instruction-tuned) in the terminal |
| `finetune_lora.py` | LoRA fine-tuning on a custom dataset — train Gemma on your own data with minimal VRAM |
| `rag_pipeline.py` | Retrieval-Augmented Generation: embed documents, retrieve relevant chunks, generate grounded answers |
| `summarizer.py` | Document summarisation with chunked long-text support and extractive/abstractive modes |
| `structured_output.py` | Force Gemma to output valid JSON, lists, or specific schemas for data extraction |
| `embeddings.py` | Extract sentence/document embeddings from Gemma hidden states for similarity search & clustering |
| `benchmark.py` | Benchmark Gemma inference: tokens/sec, memory usage, quantisation comparison (FP16 vs INT8 vs INT4) |

## Quick Start

```powershell
pip install -r requirements.txt

# Basic text generation
python text_generation.py --prompt "Explain quantum computing in simple terms"

# Interactive chat
python chat.py --model google/gemma-2-2b-it

# Fine-tune on your data
python finetune_lora.py --dataset data/my_dataset.jsonl --epochs 3

# RAG pipeline
python rag_pipeline.py --docs data/documents/ --query "What is the main finding?"

# Benchmark inference speed
python benchmark.py --model google/gemma-2-2b --quantize int8
```

## Supported Models

| Model | Params | VRAM (FP16) | Best For |
|-------|--------|-------------|----------|
| `google/gemma-2-2b` | 2B | ~5 GB | Fast experiments, embeddings |
| `google/gemma-2-2b-it` | 2B | ~5 GB | Chat, instruction following |
| `google/gemma-2-9b` | 9B | ~19 GB | Higher quality generation |
| `google/gemma-2-9b-it` | 9B | ~19 GB | Best chat/instruction quality |
| `google/gemma-2-27b` | 27B | ~55 GB | Maximum quality (multi-GPU) |

## Notes

- Gemma models require accepting Google's license on [Hugging Face](https://huggingface.co/google/gemma-2-2b) and setting a `HF_TOKEN` environment variable
- All scripts default to `gemma-2-2b-it` (2B instruction-tuned) — runs on a single consumer GPU
- INT8/INT4 quantisation via bitsandbytes dramatically reduces memory requirements
- LoRA fine-tuning works on GPUs with as little as 8 GB VRAM

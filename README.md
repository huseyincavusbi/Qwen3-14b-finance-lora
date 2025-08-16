# Qwen3-14B Finance LoRA Fine-tuning

A specialized financial AI model created by fine-tuning Qwen3-14B with LoRA adapters on comprehensive finance datasets.

## What Makes This Unique

- **Multi-Dataset Fusion**: Combines two complementary finance datasets for comprehensive coverage
  - `gbharti/wealth-alpaca_lora` (wealth management focus)
  - `Josephgflowers/Finance-Instruct-500k` (broad financial instruction dataset)

- **Memory-Efficient Training**: Uses 4-bit quantization + LoRA for training on T4 GPU
- **ChatML Format**: Properly formatted for conversational finance AI interactions
- **Comprehensive Monitoring**: Full W&B integration for training metrics and system resource tracking

## Quick Start

### Option 1: Run in Google Colab
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1iy5fnEYcCtjbQwttdr0Mwrd6wtWHrMzM?usp=sharing)

### Option 2: Use Pre-trained LoRA Adapters
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="huseyincavus/qwen3-30b-finance-lora",
    max_seq_length=2048,
    load_in_4bit=True,
)
```

**🤗 Model Hub**: [huseyincavus/qwen3-14b-finance-lora](https://huggingface.co/huseyincavus/qwen3-14b-finance-lora)

## Training Results

The fine-tuning process achieved stable convergence with comprehensive W&B monitoring:

- **Loss Convergence**: Smooth decline from 3.0 to ~1.2 over 300 steps
- **GPU Utilization**: Consistent 95-100% usage with optimal memory allocation
- **System Efficiency**: Maintained 95% CPU utilization with stable 75°C GPU temperature
- **Resource Management**: Efficient 14.6GB GPU memory usage and optimized I/O patterns

## Training Configuration

- **Base Model**: Qwen3-14B (4-bit quantized)
- **LoRA Rank**: 16
- **Max Steps**: 300
- **Batch Size**: 8 (effective)
- **Learning Rate**: 2e-4
- **Monitoring**: W&B integration for real-time metrics tracking

## Technical Applications

- Quantitative risk modeling and portfolio optimization
- Financial NLP tasks (sentiment analysis, document classification)
- Automated financial report generation and analysis
- Real-time market data interpretation and forecasting
- Regulatory compliance text processing

## Key Features

- **Efficient**: 4-bit quantization reduces memory usage by ~75%
- **Scalable**: LoRA adapters are lightweight and shareable
- **Conversational**: ChatML formatting for natural interactions
- **Comprehensive**: Trained on 500k+ financial instructions

## Model on [Hugging Face](https://huggingface.co/huseyincavus/qwen3-14b-finance-lora)
---

**Built with**: Unsloth, Transformers, TRL, LoRA fine-tuning, and Weights & Biases monitoring

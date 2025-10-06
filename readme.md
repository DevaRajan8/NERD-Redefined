# CycleNER for Materials Science

## Quick Start

1. Install dependencies:
```bash
pip install torch transformers scikit-learn numpy tqdm
#and
pip install graphviz streamlit langraph langraph-groq
```

2. Train the model:
```bash
python main.py --data_path your_data.json
```

## Key Arguments

- `--model_name`: Choose `t5-small`, `t5-base`, `facebook/bart-base`, or `facebook/bart-large`
- `--epochs`: Number of training epochs (default: 2)
- `--batch_size`: Training batch size (default: 32)
- `--checkpoint_path`: Directory to save checkpoints
- `--resume_from_checkpoint`: Resume from saved checkpoint

## How CycleNER Works

CycleNER uses two models working in tandem:

1. **S2E Model (Sentence-to-Entity)**: Extracts entities from sentences
2. **E2S Model (Entity-to-Sentence)**: Generates sentences from entity sequences

### Training Process

The model trains through cycle-consistency:

1. **S-Cycle**: 
   - Take sentence → S2E model → predicted entities
   - Train E2S model to reconstruct original sentence from predicted entities

2. **E-Cycle**:
   - Take entity sequence → E2S model → predicted sentence  
   - Train S2E model to reconstruct original entities from predicted sentence

This dual training ensures both models learn complementary representations without requiring labeled data.

### Data Processing

The system automatically extracts these entity types from materials science JSON:
- Precursors, targets, solvents
- Temperatures, times, operations
- Quantities and conditions

Entity sequences use format: `entity <sep> type <sep> entity <sep> type`

## Model Comparison

**T5 Models**: Better for text-to-text generation tasks, explicit task prefixes
**BART Models**: Optimized for text generation and reconstruction tasks

Both architectures work with the same training procedure and data format.

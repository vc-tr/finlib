# Model Capacity Analysis: Parameter Counts for RNN Architectures

## Overview

This document provides theoretical parameter count analysis for different recurrent neural network architectures commonly used in time series forecasting.

## Parameter Count Formulas

### 1. LSTM (Uni-directional)

**Formula**: `#params = 4 × [H(H + D) + H] × L`

**Variables**:
- `D = input_dim`
- `H = hidden_dim` 
- `L = num_layers`

**Explanation**: 
- Each LSTM cell has 4 gates (input, forget, cell, output)
- Each gate requires weights for input (`H × D`) and hidden state (`H × H`), plus bias (`H`)
- Total per layer: `4 × [H(H + D) + H]`
- Multiplied by number of layers `L`

### 2. BiLSTM (Bidirectional LSTM)

**Formula**: `2 × (4[H(H + D) + H]) × L + head(2H × 1)`

**Explanation**:
- Two LSTM layers (forward + backward)
- Each direction has same parameters as uni-directional LSTM
- Additional head layer to combine bidirectional outputs: `2H × 1`

### 3. GRU (Gated Recurrent Unit)

**Formula**: `3 × [H(H + D) + H] × L + head(H × 1)`

**Explanation**:
- Each GRU cell has 3 gates (reset, update, candidate)
- Each gate requires weights for input (`H × D`) and hidden state (`H × H`), plus bias (`H`)
- Total per layer: `3 × [H(H + D) + H]`
- Multiplied by number of layers `L`
- Additional head layer: `H × 1`

## Parameter Count Calculations

### Given Parameters
- `D = input_dim = 1`
- `H = hidden_dim = 64`
- `L = num_layers = 2`

### Results Table

| Model | Formula | Calculation | Parameter Count |
|-------|---------|-------------|-----------------|
| **LSTM** | `4 × [H(H + D) + H] × L` | `4 × [64(64 + 1) + 64] × 2` | `4 × [4,160 + 64] × 2 = 4 × 4,224 × 2 = 33,792` |
| **BiLSTM** | `2 × (4[H(H + D) + H]) × L + head(2H × 1)` | `2 × 4,224 × 2 + 128 × 1` | `16,896 + 128 = 17,024` |
| **GRU** | `3 × [H(H + D) + H] × L + head(H × 1)` | `3 × [64(64 + 1) + 64] × 2 + 64 × 1` | `3 × 4,224 × 2 + 64 = 25,344 + 64 = 25,408` |

### Detailed Calculations

#### LSTM (Uni-directional)
```
H(H + D) + H = 64(64 + 1) + 64 = 64 × 65 + 64 = 4,160 + 64 = 4,224
4 × 4,224 × 2 = 33,792 parameters
```

#### BiLSTM (Bidirectional)
```
Forward/Backward: 4 × 4,224 × 2 = 33,792
Head layer: 2H × 1 = 128 × 1 = 128
Total: 33,792 + 128 = 33,920 parameters
```

#### GRU
```
H(H + D) + H = 64(64 + 1) + 64 = 4,224
3 × 4,224 × 2 = 25,344
Head layer: H × 1 = 64 × 1 = 64
Total: 25,344 + 64 = 25,408 parameters
```

## Summary

For the given configuration (D=1, H=64, L=2):

| Model | Parameter Count | Relative Size |
|-------|----------------|---------------|
| **LSTM** | 33,792 | 100% (baseline) |
| **BiLSTM** | 33,920 | 100.4% |
| **GRU** | 25,408 | 75.2% |

**Key Observations**:
1. **GRU is most parameter-efficient** (25% fewer parameters than LSTM)
2. **BiLSTM has minimal overhead** over LSTM (only 128 additional parameters for head layer)
3. **LSTM has highest capacity** but requires more computational resources
4. **All models scale quadratically** with hidden dimension H

## Implementation Notes

These parameter counts assume:
- Standard implementations without additional regularization layers
- No embedding layers (direct input to RNN)
- Single output head for regression/classification
- No attention mechanisms (which would add additional parameters)

For production models, consider:
- Adding dropout layers (no additional parameters)
- Attention mechanisms (adds `H × H` parameters per attention head)
- Multiple output heads (adds `H × num_outputs` per head) 
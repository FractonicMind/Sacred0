# Model Architecture

SacredPause-AI uses a minimal feedforward neural network with a custom `TernaryNeuron` as the final output layer.

### Key Components:
- **Input Layer**: 3 synthetic moral features
- **Hidden Layer**: 10 neurons with ReLU
- **Output Layer**: Custom logic returning a ternary value based on thresholds

### TernaryNeuron:
```python
if x > threshold_high:
    return +1
elif x < threshold_low:
    return -1
else:
    return 0
```


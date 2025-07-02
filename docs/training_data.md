# Training Data

Synthetic dataset representing simplified moral scenarios.

Each input has 3 values:
- `urgency`: float [-1, 1]
- `confidence`: float [-1, 1]
- `risk`: float [-1, 1]

### Labeling Logic
The labels were derived using a rule-based simulator to generate ternary outcomes.

### Sample
```python
Input: [0.8, 0.9, -0.1] → Label: +1 (act)
Input: [0.1, 0.0, 0.0] → Label: 0 (pause)
Input: [-0.6, -0.7, 0.8] → Label: -1 (refuse)
```


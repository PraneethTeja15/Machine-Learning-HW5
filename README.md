# Machine-Learning-HW5
This project implements scaled dot-product attention in NumPy and a simplified Transformer encoder block in PyTorch, including multi-head self-attention, feed-forward layers, residual connections, and layer normalization. The code verifies correct dimensions for typical input batches.
Transformer-Encoder
Name : Praneeth Teja Jilkapally 
ID : 700781935

This notebook contains solutions and implementations related to transformer-based architectures, focusing on core encoder mechanisms.

📌 Contents
HW5_Transformer_Encoder.ipynb – Main notebook containing all code implementations
Transformer Encoder Concepts
Self-Attention Mechanism
Feed-Forward Network
Layer Normalization & Residual Connections
🚀 Part 1 – Multi-Head Self-Attention
Implements the multi-head self-attention mechanism used inside transformer encoder blocks.

Features
Computes query (Q), key (K), and value (V) projections
Splits into multiple heads
Applies scaled dot-product attention
Concatenates attention outputs from all heads
Final linear projection back to d_model
Output
Attention scores
Multi-head context representations
🚀 Part 2 – Transformer Encoder Block
Implements a simplified transformer encoder layer consisting of:

Components
Multi-head self-attention
Add & Norm (residual + LayerNorm)
Feed-forward network
Linear → ReLU → Linear
Add & Norm (again after FFN)
Default Dimensions
d_model = 128
num_heads = 8
ffn_dim = 512
🧪 Output Verification
For an input batch of size:

Batch size: 32
Sequence length: 10 tokens
Embedding dimension: 128

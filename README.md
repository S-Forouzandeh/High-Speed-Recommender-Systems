# High-Speed-Recommender-Systems
Hypergraph Neural Networks with Knowledge Distillation and Language Models for High-Speed Recommender Systems
## Overview

DistillHGNN-LR is a novel knowledge distillation framework that combines Hypergraph Neural Networks (HGNNs) with language models to create efficient and accurate recommender systems. Our approach addresses key challenges in recommendation systems including sparse interaction data, group interaction modeling, and inference speed while maintaining high accuracy.

## Key Features

- Dual Knowledge Integration: Combines structural data from user-item interactions and semantic information from reviews
- Fast Inference: Significantly reduced computation time through efficient knowledge distillation
- Memory Efficient: Optimized resource usage with lightweight architecture
- High Accuracy: Maintains or exceeds HGNN accuracy
- Review Integration: Leverages textual reviews through language models
- Group Interaction Modeling: Effectively captures high-order and group relationships

## Architecture

### Teacher Model
- HGNN for capturing high-order interactions
- BERT for processing textual reviews
- Contrastive learning for embedding alignment
- Hybrid loss function combining supervised and self-supervised learning

### Student Model
- TinyGCN (lightweight single-layer GCN)
- Simplified architecture without non-linear activations
- MLP for final predictions
- Efficient knowledge transfer mechanism

## Technical Highlights

1. Knowledge Integration
   - HGNN-based structural learning
   - BERT-based semantic learning
   - Contrastive learning alignment

2. Efficient Architecture
   - Single-layer TinyGCN
   - Simplified activation functions
   - Optimized for speed
   - Low memory footprint

3. Knowledge Transfer
   - Structural knowledge preservation
   - Semantic information transfer
   - High-order relationship maintenance


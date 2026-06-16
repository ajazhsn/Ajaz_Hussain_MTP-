# Summary: From RankNet to LambdaRank to LambdaMART

**Paper:** Burges, C. (2010). From RankNet to LambdaRank to LambdaMART: An Overview. MSR-TR-2010-82.

## Core Insight
Three algorithms solving the same problem iteratively — each addressing the previous model's central shortcoming.

## RankNet
- Pairwise model: learns to rank by comparing document pairs
- Uses cross-entropy loss on predicted probability that doc i ranks above doc j
- Problem: optimizes number of pairwise errors, not the actual IR metric (NDCG)

## LambdaRank
- Same architecture as RankNet, but gradients (lambdas) are scaled by |ΔNDCG|
- Key insight: we don't need a loss function — we only need its gradient
- Problem: gradient boosted trees cannot use backprop directly

## LambdaMART
- Replaces neural network with gradient boosted regression trees (MART)
- Lambda gradients are used as pseudo-responses for tree fitting
- Best of both worlds: power of GBDT + NDCG-aware gradient signal

## Open Questions
- Why does scaling by |ΔNDCG| work so well without an explicit loss?
- How sensitive is LambdaMART to the number of trees and learning rate on LETOR 4.0?

# Intro
Project containing multiple training scripts for LLMs of various sizes.

# Projects
## llmro-mini - 400M parameter model on 10B tokens
Simplest LLM. Single-device training on 10B tokens. Architecture:
- RoPE
- SwiGLU
- RMSNorm
- GQA
- QK-norm + reordered-norm 
- 24 layers on 1024 dimensions

## llmro-test - 100M parameter model on 5B tokens
Scoped-down local run on an owned RTX 5090, ~15h. Pipeline-first: a ~11M debug config runs every stage end to end in ~2 minutes. Architecture:
- RoPE
- SwiGLU
- RMSNorm
- MHA
- QK-norm + reordered-norm
- 12 layers on 768 dimensions

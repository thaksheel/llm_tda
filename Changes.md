# Significant changes

### RepT feature construction corrected

- Preserved the response-token gradient matrix before selecting the first token.
  Previously, flattening the matrix first meant `[0]` selected a single scalar.
  RepT now concatenates two full hidden-size vectors: the last prompt-token
  representation and the first response-token gradient. The feature dimension
  changes from `d + 1` to `2d`, matching the
  [paper's sample-level signature](https://arxiv.org/html/2510.02334v1#S3.SS2).
- Representation-gradient extraction uses evaluation mode with autograd enabled,
  disabling dropout while retaining differentiation.
- CPU regression checks cover feature dimensions, the causal token offset,
  repeatability with dropout disabled, and finite-difference gradient agreement.

### Completed epoch checkpoints protected

- Completed epochs retain their adapter, tokenizer, optimizer state, parameter
  mapping, training metadata, and completion marker for later attribution.
- Recovery saves occur around quarter-epoch boundaries. The latest intermediate
  save survives interruptions; successful training and final artifact export
  remove intermediates while preserving completed epochs.
- Resume skips incomplete or invalid newer checkpoint states and tries the
  previous usable checkpoint instead of starting training from scratch.
- Attribution selects verified completed epochs and separates outputs by epoch,
  preventing an earlier epoch's results from overwriting a later epoch's results.
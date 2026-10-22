# Attribution method changes

### LESS uses asymmetric Adam features and projected cosine similarity

Implements the [LESS paper, Definition 3.1 and section 4.1](https://arxiv.org/html/2402.04333v3)
with the authors' [feature calculation](https://github.com/princeton-nlp/LESS/blob/main/less/data_selection/collect_grad_reps.py)
and [checkpoint aggregation](https://github.com/princeton-nlp/LESS/blob/main/less/data_selection/matching.py):

```text
source_direction = (beta1 * saved_m + (1-beta1) * source_gradient)
                   / sqrt(beta2 * saved_v + (1-beta2) * source_gradient**2 + eps)
source_feature = normalize(project(source_direction))
query_feature  = normalize(project(query_gradient))
score = sum_c normalized_mean_epoch_learning_rate[c]
              * dot(query_feature[c], source_feature[c])
```

- Only sources receive the Adam transformation. Query features use raw loss
  gradients, as in the authors' validation-gradient script. Moments stay fixed
  while evaluating each example; these are hypothetical independent updates.
- Moments follow the saved optimizer parameter mapping, including nonconsecutive
  IDs and empty groups. Betas and epsilon come from each saved Adam/AdamW group.
  AMSGrad, maximization, missing moments, and mismatched shapes/order are rejected.
- Follows the reference code's uncorrected moments and epsilon inside the square
  root. The paper's displayed Adam equations include bias correction; this is an
  explicit reference-code convention, not an exact AdamW optimizer step. Weight
  decay and gradient clipping are not added to the attribution direction.
- Uses one deterministic Rademacher projection (seed 0, model ID 0, 8,192 columns,
  128-column blocks) for source and query features across all checkpoints.
  Projection and normalization use float32; the reference collection script
  casts gradient batches to float16. This preserves the existing raw cache's
  precision but does not reproduce its random draws or results bit for bit.
- Normalizes after projection. Zero features remain zero and give zero cosine
  contribution. Checkpoint weights are epoch-average learning rates divided by
  their sum, following the reference matcher; this rescales the paper's sum
  without changing rankings. Dot products use IEEE float32 and accumulation
  uses float64.
- Uses this pipeline's existing LoRA checkpoints, response-only mean cross entropy,
  512-token truncation, and matching 4-bit model loading. It does not reproduce
  the paper's 5% warmup training recipe or retrain on a selected subset.
  Evaluation retains separate query scores and existing retrieval metrics,
  rather than the reference matcher's task averaging and maximum over subtasks.

### TracInCP corrected to use checkpoint-weighted gradient dot products

Scoring follows Equation 1 of the
[original TracIn paper, section 3.3](https://proceedings.neurips.cc/paper/2020/file/e6385d39ec9394f2f3a354d9d2b88eec-Paper.pdf):

```text
score(source, query) = sum over checkpoints c:
    learning_rate[c] * dot(grad_loss(source, c), grad_loss(query, c))
```
Positive scores indicate gradient alignment toward reducing query loss;
negative scores indicate opposition. Scores are influence estimates, not
probabilities or exact counterfactual loss changes.

### RepT feature construction corrected

- Preserved the response-token gradient matrix before selecting the first token.
  Previously, flattening first meant `[0]` selected a single scalar. RepT now
  concatenates the last prompt-token representation and the full first-response-
  token gradient. The feature dimension changes from `d + 1` to `2d`, matching
  the [paper's sample-level signature](https://arxiv.org/html/2510.02334v1#S3.SS2).
- Representation-gradient extraction uses evaluation mode with autograd enabled,
  disabling dropout while retaining differentiation.
- Regression checks verify feature dimensions, the causal token offset,
  repeatability with dropout disabled, and finite-difference gradient agreement.

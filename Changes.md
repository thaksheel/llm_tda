# Attribution method changes

## Retrieval relevance supports multiple facts per source

- Sources may use a JSON array of complete fact IDs in `label`. Relevance is
  nonempty intersection with the query's fact set, rather than equality between
  a scalar query label and an entire source list. Each source counts once per
  query even if several fact IDs match. Existing scalar-label results are
  unchanged.
- Recall denominators count matching selected source examples, not the number
  of annotations. FTRACE preparation normalizes subject/relation/object direction
  and restricts source facts to the masked example's annotated entity pairs.
  Matching labels indicate annotated fact relevance, not measured causal influence.

## TrackStar adapts optimizer correction and task-mixed whitening to LoRA checkpoints

Reimplements the estimator from
[Scalable Influence and Fact Tracing, Equations 1–4 and Appendix A.2](https://proceedings.iclr.cc/paper_files/paper/2025/file/65798a76cc176c29b6bfefe84b0a03ff-Paper-Conference.pdf):

```text
phi(z) = two_sided_Gaussian_projection(gradient_sum_loss(z) / (sqrt(V) + eps))
R(block) = 0.1 * phi_source.T @ phi_source + 0.9 * phi_query.T @ phi_query
u(z) = concatenate_blocks(R(block)^(-1/2) @ phi(z, block))
score(source, query) = dot(u_source / norm(u_source), u_query / norm(u_query))
```

- Uses the selected checkpoint's trainable LoRA parameters and saved AdamW
  `exp_avg_sq` as V, applying the same correction to sources and queries before
  projection. This is an explicit adaptation: the paper uses full-model gradients
  excluding embeddings and Adafactor states. Frozen base-weight optimizer states
  are unavailable in these checkpoints. No first-moment update, source confidence
  weighting, or checkpoint learning-rate weighting is applied.
- Computes missing response-token-mean gradients in memory and immediately
  projects each batch, saving no new raw-gradient files. Existing compatible raw
  rows are reused read-only, including completed prefixes of partial caches.
  Completed projection batches resume without recomputing their gradients, and
  existing TrackStar feature caches retain the same identity.
- Restores response-token-sum gradients from either computed or cached mean
  gradients by multiplying each row by its number of valid, causally shifted labels. Counts
  use the existing tokenizer, prompt masking, and 512-token truncation. This
  algebraic conversion retains the cached gradients' floating-point precision.
- Pools consecutive transformer layers into up to eight groups, splitting
  attention and MLP branches. LoRA A matrices are transposed before concatenation
  so A/B matrices share their rank axis. Two-sided independent Gaussian projections
  produce 64×64 coordinates per group/family, totaling 65,536 for TinyLlama.
  This is an adapter-space pooling layout, not the paper's dense-weight layout.
- Uses the closed-set mixing value 0.9. Curvature is an uncentered sum of outer
  products, without division by example counts; query selection therefore affects
  whitening for both sides. This fixed value is not claimed to be optimal for the
  harmful-tuning dataset. Only gradients, never relevance labels, fit curvature.
- Whitens separately per block, then normalizes the entire concatenated feature
  once. Per-block normalization would change the estimator. Zero vectors remain
  zero. A compact inverse-root eigenbasis gives the same inner products and norms
  as a symmetric inverse root; discarded coordinates are padded with zeros.
- Uses float32 projection, whitening, and scoring with float64 score storage.
  No ridge is added. Rank-deficient blocks use a pseudoinverse, retaining covariance
  eigenvalues above `max(total_examples, block_dimension) * float32_eps * max_eigenvalue`.
  This numerical null-space convention is additional to the paper's equations.
  The inverse root is fitted by thin SVD of the weighted feature matrix,
  avoiding the precision loss from explicitly forming its Gram matrix.

## LoGRA-random uses projected weight gradients and damped layer curvature

Follows the Kronecker projection and influence expressions in
[LoGRA, Equations 3 and 6](https://arxiv.org/html/2405.13954v1), using
64 input and output directions per layer as in its LLM experiment:

```text
g(example, layer) = flatten(sum_response_loss_gradient projected on input/output axes)
H(layer) = G_source.T @ G_source / number_of_sources
lambda(layer) = 0.1 * trace(H) / projected_dimension
score(query, source) = sum_layers g_query.T @ inverse(H + lambda I) @ g_source
```

- Captures forward activations and backward output derivatives, projecting both
  before multiplying and summing across tokens. Keeps the entire 64×64 projected
  gradient, without LoRIF's additional rank-one factorization or truncated
  curvature. Smaller layers use `min(64, input_width, output_width)` on both axes.
- Uses the random variant with the same uniform initialization distributions as
  [LogIX's linear projection layers](https://github.com/logix-project/logix/blob/main/logix/lora/modules.py).
  Module-specific seeded generators make projections repeatable without changing
  the global RNG. This implements the equivalent projection through hooks rather
  than inserting zero-output bottleneck layers. PCA initialization is not used.
- Uses the uncentered empirical gradient covariance and per-layer trace damping,
  matching [LogIX's covariance normalization and inverse](https://github.com/logix-project/logix/blob/main/logix/state.py).
  This is an empirical-Fisher approximation from observed response labels, not
  an exact loss Hessian or Fisher estimated from model-sampled targets. Source
  limits also restrict the curvature-fitting data.
- Evaluates one selected completed checkpoint with the pipeline's quantized
  adapter model, response-only token-sum cross entropy, and 512-token truncation.
  Selects attention and MLP weight branches, excluding embeddings, output head,
  and adapter matrices. These choices adapt the estimator to this fine-tuning
  setup; they do not reproduce the paper's full pretraining-data experiment.
- Uses the reference's basic `dot` influence mode, without source normalization,
  confidence weighting, or checkpoint learning-rate weights. Positive scores
  estimate support for reducing query loss. The paper's qualitative example uses
  an optional normalized mode; this implementation keeps the base influence score.
- Uses float32 projections, storage, and Cholesky solves, with float64 layer-score
  accumulation. Solves in source space when it is smaller than feature space;
  this is algebraically the same damped inverse, not an extra approximation.
  A zero training-gradient layer contributes zero.

## TRAK uses model-output gradients, a training kernel, and source confidence weights

Implements the single-model case of the
[TRAK estimator](https://proceedings.mlr.press/v202/park23c.html), with the
[DsDm language-model adaptation, Appendix B.3](https://arxiv.org/html/2401.12926v1#A2.SS3):

```text
f(example) = sum_response_tokens log(p(correct_token) / (1 - p(correct_token)))
Q(example) = mean_response_tokens (1 - p(correct_token))
Phi = project(gradient_of_f) / sqrt(number_of_selected_parameters)
scores = Phi_query @ pinv(Phi_source) @ diag(Q_source)
```

- Uses causally shifted response positions, excluding prompt and padding labels,
  with the existing chat formatting and 512-token truncation. Log-odds are computed
  as the true-token logit minus logsumexp of alternative logits, avoiding unstable
  subtraction of probabilities near one. The LM token-sum output and token-mean
  confidence follow the published DsDm equations; this is an LM adaptation, not
  the BERT sequence-classification experiment or an exact scalar loss-chain identity.
- Differentiates the model output, not cross-entropy loss. Existing loss gradients
  generally cannot reconstruct token-specific log-odds gradients, so no loss-gradient
  cache is relabeled as TRAK features. The saved optimizer mapping fixes LoRA
  parameter selection/order, using the matching quantized adapter model in eval mode.
- Uses the bundled BasicProjector with 2,048 Rademacher columns, seed 0, model ID 0,
  128-column blocks, and float32 computation/storage. The common parameter-count
  scale follows the bundled TRAKer; there is no row/cosine normalization.
- Computes the unregularized kernel through a thin SVD of source features, rather
  than explicitly inverting the squared-condition Gram matrix. Keeps singular
  values above `max(source_count, projection_dim) * float32_eps * largest_value`.
  This is the Moore–Penrose extension for small/rank-deficient source selections;
  it equals the paper's inverse expression when source features have full column
  rank and all singular directions are retained. No implicit ridge penalty is
  added. A zero source matrix scores zero.
- Applies Q only to source columns. Scores use float32 matrix products and are
  stored as float64. The positive inverse rescaling used by the bundled score
  computer for half precision is omitted; it changes only the overall scale for
  a single model, not its source ranking.
- Evaluates one selected completed checkpoint, without learning-rate weighting,
  epoch aggregation, or a multi-seed model ensemble. Reports the existing retrieval
  metrics. This does not reproduce the paper's BERT training recipe or ensemble results.

## LoRIF uses projected weight-gradient factors and spectral curvature

Uses the bundled [official LoRIF library](https://github.com/doub7e/LoRIF)
for the [paper's factor and Woodbury constructions](https://arxiv.org/html/2601.21929v1):

```text
layer_gradient = projected_input.T @ projected_output_gradient
layer_gradient ~= left_factor @ right_factor.T
layer_curvature ~= V @ diag(sigma**2) @ V.T
score = sum_layers query_gradient.T
                   @ inverse(layer_curvature + damping * I) @ source_gradient
```

- Collects gradients of the effective linear weight branches in all supported
  attention/MLP projections (`q/k/v/o/gate/up/down_proj`), including PEFT base
  branches. LoRA A/B matrices, embeddings, and the language-model head are excluded.
  Existing adapter gradients cannot reconstruct these weight gradients and are
  not substituted. Model weights remain frozen; input gradients enable collection
  through the existing quantized adapter model without changing its weights.
  Scores probe linear-weight directions, rather than the LoRA optimizer's
  adapter-parameter directions used by TracIn, LESS, RapidIn, TRAK, and TrackStar.
- Uses projection factor 16, rank-1 factors with 8 power iterations, seed 0, and
  the release's legacy two-sided Rademacher projection scaling. Collection and
  factorization use FP32; factors are stored in BF16 and converted to FP32 for
  curvature/scoring, matching the release's storage convention.
- Uses token-summed response cross entropy (prompt/padding masked, truncated to
  512 tokens). A batch sum preserves per-example scale; the shared raw adapter-gradient
  cache instead stores response-token means. The paper's GPT-2 workflow instead
  supervises all non-padding next tokens in fixed WikiText blocks.
- Fits each layer's training-only, unaveraged gradient covariance with the release's
  streamed randomized SVD: rank up to 2,048, oversampling 10, three power iterations
  plus the initial multiply. The complementary subspace retains inverse weight
  `1 / damping`; tail correction and gradient normalization are not applied.
- Uses the release default `damping = 0.5 * trace(G.T @ G) / D`. This explicitly
  differs from the paper experiments' historical `0.1 * mean(sigma[:r+p]**2)`.
  A layer with identically zero training gradients contributes zero scores.
- Sums proponent-positive scores across layers at the selected completed epoch.
  There is no trajectory aggregation or learning-rate weighting. Scores accumulate
  in float64; per-layer calculations use float32. The existing per-query retrieval
  metrics are reported, not the paper's subset-retraining LDS/tail-patch experiments.

## RapidIn uses shared shuffle/sign compression and per-parameter L2 normalization

Implements the sample-level calculation in the
[RapidIn paper, sections 3.2–3.3](https://aclanthology.org/2024.acl-long.48/)
and the authors' [RapidGrad](https://github.com/huawei-lin/RapidIn/blob/master/RapidIn/RapidGrad.py)
and [gradient normalization](https://github.com/huawei-lin/RapidIn/blob/master/RapidIn/calc_inner.py) code:

```text
v = concatenate(L2_normalize(each parameter's gradient))
v = zero_pad(v, next multiple of 2**24)
RapidGrad = signed_block_sum(shared_shuffle(v), K=2**16)
score(source, query) = dot(RapidGrad(source), RapidGrad(query))
```

- Replaces global `layer_norm` with L2 normalization of each flattened parameter
  tensor, matching the reference code's interpretation of layer-wise normalization.
  Zero tensors stay zero; normalization uses epsilon 1e-12.
- Replaces the dense Rademacher matrix with a shared factor-based shuffle,
  element-wise Rademacher signs, and contiguous block sums. A local NumPy
  RandomState uses seed 42, draws signs before permutations, and follows the
  reference's 20 row/column shuffle pairs. The same transform applies to every
  source and query. Composing its index mapping once preserves those operations.
  The old helper drew different shuffles for different examples.
- Pads to a multiple of 2**24 before compression so the default K=65,536 divides
  the vector length. No cosine normalization, Adam transformation, learning-rate
  weighting, or aggregation across checkpoints is applied; `--epochs` selects
  one completed checkpoint, as in the reference's sample-level retrieval.
- Uses saved LoRA parameter boundaries/order and response-only mean cross entropy
  with the pipeline's 512-token truncation and matching 4-bit model loading.
  Compression and saved features use float32, versus the reference's half-precision
  path; scoring uses IEEE float32 dot products and float64 accumulation. These
  precision and parameter-order conventions prevent bit-for-bit reproduction.
- Produces sample-level query/source rankings and the existing retrieval metrics.
  The paper's additional token-wise refinement is not implemented by this mode.

## LESS uses asymmetric Adam features and projected cosine similarity

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

## TracInCP corrected to use checkpoint-weighted gradient dot products

Scoring follows Equation 1 of the
[original TracIn paper, section 3.3](https://proceedings.neurips.cc/paper/2020/file/e6385d39ec9394f2f3a354d9d2b88eec-Paper.pdf):

```text
score(source, query) = sum over checkpoints c:
    learning_rate[c] * dot(grad_loss(source, c), grad_loss(query, c))
```
Positive scores indicate gradient alignment toward reducing query loss;
negative scores indicate opposition. Scores are influence estimates, not
probabilities or exact counterfactual loss changes.

## RepT feature construction corrected

- Preserved the response-token gradient matrix before selecting the first token.
  Previously, flattening first meant `[0]` selected a single scalar. RepT now
  concatenates the last prompt-token representation and the full first-response-
  token gradient. The feature dimension changes from `d + 1` to `2d`, matching
  the [paper's sample-level signature](https://arxiv.org/html/2510.02334v1#S3.SS2).
- Representation-gradient extraction uses evaluation mode with autograd enabled,
  disabling dropout while retaining differentiation.

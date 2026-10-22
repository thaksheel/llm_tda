# Attribution method changes

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

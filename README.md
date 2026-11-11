# LLM Training Data Attribution

Fine-tune a causal language model with LoRA, then rank training examples by their
attribution scores for query responses.

Supported methods: RepT, TracIn, LESS, RapidIn, LoRIF, TRAK, LoGRA, and TrackStar.
BM25 provides a CPU lexical baseline with the same evaluation metrics.
Evaluation reports auPRC, precision@k, recall@k, and MRR using matching source and
query labels.

- [Dataset preparation](DATA_PREPARATION.md): prepare FTRACE and T-REx subsets.
- [Pipeline guide](RUN_PIPELINE.md): setup, inputs, commands, flags, and saved outputs.
- [Method changes](Changes.md): mathematical corrections, numerical choices, and
  adaptations from the papers and reference implementations.

`run_pipeline.py` is the command-line entry point. Supporting code lives in `src/`,
regression tests in `tests/`, and bundled method libraries in `modules/`.

Run the regression suite from the repository root with the environment activated:

```bash
python -m unittest discover -s tests
```

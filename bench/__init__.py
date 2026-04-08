"""
bench — Lightweight benchmark runner for routed inference evaluation.

Runs standard NLP benchmarks (COPA, StoryCloze, GSM8K) with and without
the intervention router, using the existing signal_lab Agent pipeline.

No dependency on lm-evaluation-harness — these benchmarks are small
enough that a direct implementation is cleaner.
"""

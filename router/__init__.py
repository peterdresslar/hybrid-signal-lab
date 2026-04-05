"""
router — Inference-time intervention routing for hybrid linear attention models.

Public API:
    InterventionRouter  — loads trained classifier, predicts best gain profile
    routed_pass         — two-pass pipeline: baseline → classify → intervene
    routed_score_target — two-pass pipeline with teacher-forced scoring

Note: routed_pass and routed_score_target depend on signal_lab's model
package (PyTorch, transformers, Python 3.11+). Import them explicitly
from router.pipeline when running on the inference host.
"""

from router.router import InterventionRouter

__all__ = ["InterventionRouter"]

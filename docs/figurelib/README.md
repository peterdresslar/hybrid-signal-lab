# Figure Library

Reusable helpers for generating paper figures from analyzed sweep outputs.

## Layout

- `common.py`
  Shared paths, typography helpers, and small formatting utilities.
- `constant_gain.py`
  Reusable line-plot generator for constant-gain dose-response figures.
- `head_entropy_correlations.py`
  Head-by-head baseline-entropy correlation heatmap generator.
- `oracle_headroom.py`
  Oracle-headroom distribution figure generator.
- `pca_baseline_entropy.py`
  Two-panel PCA scatter generator for baseline attention-entropy structure.
- `task_dependent_gain.py`
  Representative-profile sparkline + heatmap figure generator.
- `type_responsiveness.py`
  Intra-family type-responsiveness figure generator.

The top-level scripts in `docs/` are thin wrappers that call these helpers:

- `make_constant_gain_dose_response_figure.py`
- `make_constant_gain_dose_response_block_figure.py`
- `make_constant_gain_dose_response_balanced_attention_figure.py`
- `make_constant_gain_dose_response_balanced_block_figure.py`
- `make_head_entropy_correlation_figure.py`
- `make_head_entropy_shared_vs_specific_figure.py`
- `make_intra_family_type_responsiveness_figure.py`
- `make_intra_family_type_responsiveness_balanced_attention_figure.py`
- `make_intra_family_type_responsiveness_balanced_attention_top8_figure.py`
- `make_intra_family_type_responsiveness_balanced_block_figure.py`
- `make_intra_family_type_responsiveness_balanced_block_top8_figure.py`
- `make_oracle_headroom_distribution_figure.py`
- `make_oracle_headroom_distribution_balanced_block_hybrid_figure.py`
- `make_oracle_headroom_distribution_balanced_hybrid_figure.py`
- `make_pca_baseline_entropy_figure.py`
- `make_representative_profiles_figure.py`

## Regenerating current figures

From the repo root:

```bash
MPLCONFIGDIR=/tmp/mpl python docs/make_constant_gain_dose_response_figure.py
MPLCONFIGDIR=/tmp/mpl python docs/make_constant_gain_dose_response_block_figure.py
MPLCONFIGDIR=/tmp/mpl python docs/make_constant_gain_dose_response_balanced_attention_figure.py
MPLCONFIGDIR=/tmp/mpl python docs/make_constant_gain_dose_response_balanced_block_figure.py
MPLCONFIGDIR=/tmp/mpl python docs/make_head_entropy_correlation_figure.py
MPLCONFIGDIR=/tmp/mpl python docs/make_head_entropy_shared_vs_specific_figure.py
MPLCONFIGDIR=/tmp/mpl python docs/make_intra_family_type_responsiveness_figure.py
MPLCONFIGDIR=/tmp/mpl python docs/make_intra_family_type_responsiveness_balanced_attention_figure.py
MPLCONFIGDIR=/tmp/mpl python docs/make_intra_family_type_responsiveness_balanced_attention_top8_figure.py
MPLCONFIGDIR=/tmp/mpl python docs/make_intra_family_type_responsiveness_balanced_block_figure.py
MPLCONFIGDIR=/tmp/mpl python docs/make_intra_family_type_responsiveness_balanced_block_top8_figure.py
MPLCONFIGDIR=/tmp/mpl python docs/make_oracle_headroom_distribution_figure.py
MPLCONFIGDIR=/tmp/mpl python docs/make_oracle_headroom_distribution_balanced_block_hybrid_figure.py
MPLCONFIGDIR=/tmp/mpl python docs/make_oracle_headroom_distribution_balanced_hybrid_figure.py
MPLCONFIGDIR=/tmp/mpl python docs/make_pca_baseline_entropy_figure.py
MPLCONFIGDIR=/tmp/mpl python docs/make_representative_profiles_figure.py
```

`MPLCONFIGDIR=/tmp/mpl` avoids cache-path issues on sandboxed or cluster-like environments.

## Reusing the constant-gain generator for new bench data

The most reusable component here is `constant_gain.py`.

Use `plot_constant_gain_dose_response(...)` with one or more `ConstantGainPanel`
definitions that point at analyzed `analysis_type_gain_summary.csv` files.

Minimal example:

```python
from pathlib import Path

from figurelib.constant_gain import ConstantGainPanel, plot_constant_gain_dose_response

plot_constant_gain_dose_response(
    panels=[
        ConstantGainPanel(
            path=Path("path/to/model_a/analysis_type_gain_summary.csv"),
            title="Model A",
        ),
        ConstantGainPanel(
            path=Path("path/to/model_b/analysis_type_gain_summary.csv"),
            title="Model B",
        ),
    ],
    output_path=Path("docs/my_new_constant_gain_figure.png"),
    xlim=(0.4, 3.1),
    ylim=(-0.45, 0.40),
)
```

Behavior built into the generator:

- filters to `g_profile` names starting with `constant_`
- parses the gain value from the profile name
- injects the true baseline point `(g=1.0, Δp=0)` if the CSV does not contain it
- orders prompt types by the first panel's peak response
- keeps colors consistent across panels

## Notes

- These helpers assume the sweep has already been analyzed into the standard
  `analysis/*.csv` outputs.
- The figure wrappers intentionally keep hard-coded paper-specific titles,
  captions, and axis ranges. For new figure families, prefer adding a new thin
  wrapper rather than editing the reusable helper directly unless the change is
  truly generic.

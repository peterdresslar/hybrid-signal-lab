uv run -m battery.src.annotate_battery \
  --analysis-dir ~/workspace/data/dropzone/calibration/b4 \
  --candidates battery/data/battery_4/all_candidates.json \
  --output ~/battery/data/battery_4/annotation_manifest.json \
  --hard-threshold 0.01 \
  --easy-threshold 0.85

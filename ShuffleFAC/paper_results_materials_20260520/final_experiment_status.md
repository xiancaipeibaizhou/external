# Final Experiment Status

## Completed and Locked

- Table 1: clip-level front-end diagnostic and parameter efficiency for ShuffleFAC, ResNet18, and MobileNetV2. Statistics in this paper material are recomputed as mean ± sample std over seeds 42/43/44.
- Table 2: formal non-SN recording-level heads for ShuffleFAC, ResNet18, MobileNetV2, and PANNs-CNN14 frozen.
- Table 3: formal `sn_expd_warmup5` cross-front-end results for ShuffleFAC, ResNet18, MobileNetV2, and PANNs-CNN14 frozen.

## Final Best Result

- ShuffleFAC + SN-ExpD-Warmup5: Macro-F1 = 0.800943 ± 0.021182, ACC = 0.792350 ± 0.009465.
- This reproduces the original DeepShip SN ExpD Warmup5 main result.

## Exclusions

- Smoke outputs are excluded.
- `sn_decoupled` outputs are excluded.
- `--head all` outputs are excluded.
- Earlier incomplete PANNs non-sharedcache directories are excluded; only PANNs `*_formal_sharedcache` completed runs are used for Table 3.
- PANNs fine-tune training-stage mean-prob rows are not used as final recording-level aggregation results.

## Locked Source Files

- Table 1 source: `external/ShuffleFAC/FRONTEND_CLIP_ACCURACY_COMPARISON.md` plus underlying formal front-end metrics.
- Table 2 source: `results/FRONTEND_ROBUSTNESS/formal_non_sn_summary_20260520/table2_lock_20260520/`.
- Table 3 source: `results/FRONTEND_ROBUSTNESS/formal_sn_expd_warmup5_cross_frontend_20260520/table3_lock_20260520/`.

## Paper Material Files

- `paper_results_tables.md`
- `paper_results_analysis.md`
- `paper_discussion_points.md`
- `final_experiment_status.md`

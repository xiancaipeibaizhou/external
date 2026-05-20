# ShipsEar Feasibility Audit

Date: 2026-05-20

Scope: read-only audit of existing ShipsEar split/cache and result files. No training code was modified and no new experiment was run.

## Recommendation

**Recommendation: appendix.**

ShipsEar is not strong enough for the main paper because the strict test set has only 19 recordings, each class has only 3 to 6 test recordings, seed variance is large, and several SN variants produce exactly tied metrics. It is still useful as an appendix-only supplementary result because the existing three-seed runs are complete and show an important limitation: SN-ExpD-Warmup5 is not uniformly beneficial across datasets.

Do not run new ShipsEar experiments just to support the current main claim. If included, frame ShipsEar as a small-scale supplementary robustness check with cautious language.

## Sources Read

- `shipsear_shufflefac_recording_split_3s_7_1_2.json`
- `shufflefac_feature_cache_shipsear_seed{42,43,44}/recording_level_{train,val,test}_*.pt`
- `results/ShuffleFAC/0502_External_ShuffleFAC_ShipsEar_gamma16_multiseed_3s_7_1_2/seed_*/metrics.txt`
- `results/ShuffleFAC/0502_External_ShuffleFAC_ShipsEar_gamma16_multiseed_3s_7_1_2_recording_eval/seed_*/metrics.txt`
- `results/ShuffleFAC_GRAPHHEAD/0502_ShipsEar_seed*_attention_S8_ms5/metrics.txt`
- `results/ShuffleFAC_GRAPHHEAD/0502_ShipsEar_seed*_graph_S8_ms5/metrics.txt`
- `results/ShuffleFAC_GRAPHHEAD/0502_ShipsEar_seed*_graph_aware_attention_S8_ms5/metrics.txt`
- `results/ShuffleFAC_SIGNAL_NOISE/ShipsEar_*/metrics.json`
- `external/ShuffleFAC/EXPERIMENT_RESULTS_SUMMARY.md`

All aggregate statistics below use **mean ± sample std over seeds 42/43/44**. The existing `EXPERIMENT_RESULTS_SUMMARY.md` appears to use population std in some rows, so this audit recalculates sample std for decision making.

## Strict Recording-Level Split

Split metadata reports `dataset=ShipsEar`, `protocol=recording_level`, `total_recordings=90`, `total_segments=3738`, 3-second segments, and 16 kHz target sample rate.

| split | recordings | segments | class recording counts |
|---|---:|---:|---|
| train | 62 | 2571 | A=11, B=13, C=21, D=8, E=9 |
| val | 9 | 371 | A=2, B=2, C=3, D=1, E=1 |
| test | 19 | 796 | A=3, B=4, C=6, D=3, E=3 |

The test set is small. One recording changes test ACC by about 5.26 percentage points, so apparent gains below that scale should be treated cautiously.

## Cache Check

All nine ShipsEar ShuffleFAC feature caches for seeds 42/43/44 and train/val/test exist. Direct cache loading confirmed `metadata.dataset=ShipsEar` and `metadata.protocol=recording_level` for every cache.

| seed | train entries | val entries | test entries | cache protocol |
|---:|---:|---:|---:|---|
| 42 | 2571 | 371 | 796 | recording_level |
| 43 | 2571 | 371 | 796 | recording_level |
| 44 | 2571 | 371 | 796 | recording_level |

The per-seed caches share the same strict recording-level split metadata; the seeds correspond to model training randomness, not different recording splits.

## ShuffleFAC Stage-1 Metrics

Stage-1 clip-level metrics come from the main seed directories. Recording-level metrics come from the `_recording_eval` directories.

| seed | clip ACC | clip Macro-F1 | recording ACC | recording Macro-F1 | best val Macro-F1 | best epoch |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.707286 | 0.581571 | 0.631579 | 0.550476 | 0.690724 | 12 |
| 43 | 0.795226 | 0.705874 | 0.789474 | 0.739377 | 0.619848 | 53 |
| 44 | 0.741206 | 0.631438 | 0.684211 | 0.586667 | 0.771417 | 28 |
| mean ± sample std | 0.747906 ± 0.044351 | 0.639628 ± 0.062555 | 0.701755 ± 0.080396 | 0.625507 ± 0.100261 | - | - |

Stage-1 is highly seed-sensitive on ShipsEar. Seed 43 is much stronger than seeds 42 and 44 at recording level.

## Existing Non-SN Results

Existing ShipsEar non-SN recording-level heads are complete for seeds 42/43/44.

| head | ACC mean ± sample std | Macro-F1 mean ± sample std | status |
|---|---:|---:|---|
| attention | 0.736842 ± 0.052632 | 0.687619 ± 0.084068 | complete |
| graph | 0.684211 ± 0.052632 | 0.635628 ± 0.078012 | complete |
| graph_aware_attention | 0.719298 ± 0.080396 | 0.676508 ± 0.102862 | complete |

Best existing non-SN result is **attention**, with Macro-F1 **0.687619 ± 0.084068**.

## Existing SN Results

Existing ShipsEar SN variants are complete for seeds 42/43/44.

| SN variant | ACC mean ± sample std | Macro-F1 mean ± sample std | note |
|---|---:|---:|---|
| SN baseline | 0.719298 ± 0.080396 | 0.680159 ± 0.110411 | temporal similarity, full soft attention |
| SN threshold graph | 0.719298 ± 0.080396 | 0.680159 ± 0.110411 | threshold_similarity, sim_threshold=0.8 |
| SN temperature | 0.719298 ± 0.080396 | 0.680159 ± 0.110411 | learnable temperature |
| SN threshold + temperature | 0.719298 ± 0.080396 | 0.680159 ± 0.110411 | threshold + temperature |
| SN ExpD Top4 Warmup5 | 0.701754 ± 0.060774 | 0.649841 ± 0.086137 | threshold_similarity, signal_top_k=4, warmup=5 |
| SN ExpE Top4 NoWarmup | 0.701754 ± 0.030387 | 0.644615 ± 0.053969 | temporal_similarity, signal_top_k=4, warmup=0 |

Best existing SN group is tied between SN baseline, threshold graph, temperature, and threshold + temperature, all at Macro-F1 **0.680159 ± 0.110411**. This is still below the best non-SN attention head.

## Multiple Variant Ties

Several variant groups have exactly tied metrics, which weakens the interpretability of ShipsEar as a main-table result.

- Three-seed SN baseline, SN threshold graph, SN temperature, and SN threshold + temperature have identical per-seed ACC, Macro-F1, and Weighted-F1.
- The fixed seed43 ablations `ThresholdOnly`, `ETA Complete`, and `ExpD Warmup5` all have ACC 0.736842, Macro-F1 0.703333, and Weighted-F1 0.716667.
- The fixed seed43 threshold sweep at sim_threshold 0.80, 0.90, 0.95, and 0.98 also keeps ACC 0.736842, Macro-F1 0.703333, and Weighted-F1 0.716667 while only graph degree changes.
- Some non-SN per-seed rows also tie, especially on seed44. With only 19 test recordings, this is expected and makes small differences unreliable.

## Best Non-SN vs SN-ExpD Gap

| comparison | Macro-F1 mean ± sample std | ACC mean ± sample std |
|---|---:|---:|
| best non-SN attention | 0.687619 ± 0.084068 | 0.736842 ± 0.052632 |
| SN-ExpD Top4 Warmup5 | 0.649841 ± 0.086137 | 0.701754 ± 0.060774 |
| SN-ExpD minus best non-SN | -0.037778 | -0.035088 |

SN-ExpD-Warmup5 is worse than the best non-SN ShipsEar head by **0.037778 Macro-F1**. Even the best tied SN baseline group is slightly below attention by **0.007460 Macro-F1**.

## Decision

Use ShipsEar only as an appendix result if the paper needs cross-dataset transparency. The defensible message is:

- ShipsEar confirms the strict recording-level protocol and complete existing three-seed artifacts.
- Ordinary non-SN attention is the best ShipsEar head among current complete runs.
- SN-ExpD-Warmup5 does not improve ShipsEar and should not be used as supporting evidence for a universal gain claim.
- The result is too underpowered for the main paper because the test split has 19 recordings and only 3 to 6 recordings per class.

Final recommendation: **appendix**, not main. If manuscript space is tight or the paper wants to avoid underpowered secondary datasets, omit ShipsEar entirely rather than adding new runs.

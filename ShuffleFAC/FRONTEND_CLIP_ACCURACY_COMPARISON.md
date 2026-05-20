# DeepShip Front-End Clip-Level Accuracy Comparison

更新时间：2026-05-20

本文件专门记录 DeepShip strict recording-level protocol 下不同 clip-level front-end 的分片预测指标，用于和 ShuffleFAC 等前端进行对比。这里的主指标是 `test_clip_acc` / clip-level ACC，不是论文主线的 recording-level aggregation 指标；PANNs smoke 结果不写入本表。

## Protocol Notes

- 数据协议：DeepShip strict recording-level split，train/val/test recording 不重叠。
- ResNet18 / MobileNetV2：使用 `train_clip_frontend.py`，`--pretrained none`，从 DeepShip train recordings 训练，不使用 ImageNet；checkpoint 选择基于 validation recording-level Macro-F1。
- ShuffleFAC gamma=16：使用既有 strict recording-level split 的 Stage-1 ShuffleFAC 训练输出；`test_predictions.csv` 每个 seed 有 11,628 条 test clip 预测。
- mean/std 使用 seed 42/43/44，std 为 population std，与 `EXPERIMENT_RESULTS_SUMMARY.md` 的统计口径一致。
- `macs=N/A` 表示当前正式产物未记录 MACs 字段，不代表未计算模型复杂度。

## Clip-Level Summary

| Front-End | Seeds | Test Clip ACC mean +/- std | Test Clip Macro-F1 mean +/- std | Params | Trainable Params | MACs | Source |
|---|---:|---:|---:|---:|---:|---:|---|
| ShuffleFAC gamma=16 | 42/43/44 | 0.684325 +/- 0.001866 | 0.683512 +/- 0.002374 | 39,031 | 39,031 | 2.585M | `results/ShuffleFAC/0502_External_ShuffleFAC_gamma16_multiseed_3s_7_1_2/shufflefac_summary.csv` |
| ResNet18 | 42/43/44 | 0.675582 +/- 0.010762 | 0.675608 +/- 0.009378 | 11,172,292 | 11,172,292 | N/A | `results/FRONTENDS/DeepShip_resnet18_seed*_formal/metrics.json` |
| MobileNetV2 | 42/43/44 | 0.667670 +/- 0.004039 | 0.667747 +/- 0.004370 | 2,228,420 | 2,228,420 | N/A | `results/FRONTENDS/DeepShip_mobilenet_v2_seed*_formal/metrics.json` |

按分片准确率均值排序：ShuffleFAC gamma=16 > ResNet18 > MobileNetV2。

## Per-Seed Clip-Level Details

| Front-End | Seed | Test Clip ACC | Test Clip Macro-F1 | Recording ACC Reference | Recording Macro-F1 Reference | Output / Source |
|---|---:|---:|---:|---:|---:|---|
| ShuffleFAC gamma=16 | 42 | 0.682921 | 0.681208 | 0.770492 | 0.762786 | `results/ShuffleFAC/0502_External_ShuffleFAC_gamma16_multiseed_3s_7_1_2/seed_42` |
| ShuffleFAC gamma=16 | 43 | 0.686963 | 0.686778 | 0.786885 | 0.780370 | `results/ShuffleFAC/0502_External_ShuffleFAC_gamma16_multiseed_3s_7_1_2/seed_43` |
| ShuffleFAC gamma=16 | 44 | 0.683093 | 0.682549 | 0.770492 | 0.775663 | `results/ShuffleFAC/0502_External_ShuffleFAC_gamma16_multiseed_3s_7_1_2/seed_44_parallel` |
| ResNet18 | 42 | 0.660647 | 0.663196 | 0.754098 | 0.774421 | `results/FRONTENDS/DeepShip_resnet18_seed42_formal` |
| ResNet18 | 43 | 0.685587 | 0.685860 | 0.803279 | 0.780563 | `results/FRONTENDS/DeepShip_resnet18_seed43_formal` |
| ResNet18 | 44 | 0.680513 | 0.677768 | 0.778689 | 0.761874 | `results/FRONTENDS/DeepShip_resnet18_seed44_formal` |
| MobileNetV2 | 42 | 0.672687 | 0.672665 | 0.795082 | 0.792460 | `results/FRONTENDS/DeepShip_mobilenet_v2_seed42_formal` |
| MobileNetV2 | 43 | 0.667527 | 0.668531 | 0.729508 | 0.726216 | `results/FRONTENDS/DeepShip_mobilenet_v2_seed43_formal` |
| MobileNetV2 | 44 | 0.662797 | 0.662046 | 0.778689 | 0.768681 | `results/FRONTENDS/DeepShip_mobilenet_v2_seed44_formal` |

## Recording-Level Reference Summary

本节只作为参考，避免把分片预测能力和 recording-level 聚合能力混在一起解读。

| Front-End | Seeds | Test Recording ACC mean +/- std | Test Recording Macro-F1 mean +/- std |
|---|---:|---:|---:|
| ShuffleFAC gamma=16 | 42/43/44 | 0.775956 +/- 0.007728 | 0.772940 +/- 0.007432 |
| ResNet18 | 42/43/44 | 0.778689 +/- 0.020078 | 0.772286 +/- 0.007778 |
| MobileNetV2 | 42/43/44 | 0.767760 +/- 0.027863 | 0.762452 +/- 0.027400 |

## Excluded Rows

- PANNs Cnn14 formal classifier-only / fc1-only / block6-fc1 baselines are not included here until formal non-smoke runs are available.
- `results/FRONTENDS/*_smoke` outputs are excluded by design and must not be used as performance conclusions.

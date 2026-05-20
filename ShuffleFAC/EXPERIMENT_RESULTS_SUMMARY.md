# ShuffleFAC 实验结果总表

更新时间：2026-05-20

本文件汇总当前项目中 ShuffleFAC 相关实验的主要结果。除特别说明外，主指标均为 strict recording-level test split 上的 recording-level 指标，不混入 clip-level 或 segment-level 指标。多 seed 统计使用 seed 42/43/44。

## 表 1：多 Seed 汇总指标

| 序号 | 实验组 | 数据集 | 实验/模型 | 阶段 | Seeds | ACC mean +/- std | Macro-F1 mean +/- std | Weighted-F1 mean +/- std | Params | Trainable Params | MACs | 备注 |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | ShuffleFAC encoder | DeepShip | recording_eval | Stage-1 frozen encoder eval | 3 | 0.775956 +/- 0.007728 | 0.772940 +/- 0.007432 | 0.776559 +/- 0.007413 | 39,031 | 39,031 | 2.585M | strict recording-level split |
| 2 | ShuffleFAC encoder | DeepShip | aggregation_eval | Stage-1 frozen encoder eval | 3 | 0.775956 +/- 0.007728 | 0.772940 +/- 0.007432 | 0.776559 +/- 0.007413 | 39,031 | 39,031 | 2.585M | strict recording-level split |
| 3 | ShuffleFAC encoder | ShipsEar | recording_eval | Stage-1 frozen encoder eval | 3 | 0.701755 +/- 0.065643 | 0.625507 +/- 0.081863 | 0.655520 +/- 0.079463 | 39,160 | 39,160 | 2.585M | strict recording-level split |
| 4 | ShuffleFAC encoder | ShipsEar | aggregation_eval | Stage-1 frozen encoder eval | 3 | 0.701755 +/- 0.065643 | 0.625507 +/- 0.081863 | 0.655520 +/- 0.079463 | 39,160 | 39,160 | 2.585M | strict recording-level split |
| 5 | GraphHead | DeepShip | attention | Stage-2 trained head | 3 | 0.775956 +/- 0.036860 | 0.771075 +/- 0.043832 | 0.775977 +/- 0.037857 | N/A | 9,093 | N/A | S8, eval_samples=5, deterministic multi-bag |
| 6 | GraphHead | DeepShip | graph | Stage-2 trained head | 3 | 0.773224 +/- 0.020446 | 0.770719 +/- 0.023670 | 0.773592 +/- 0.021351 | N/A | 58,758 | N/A | S8, eval_samples=5, deterministic multi-bag |
| 7 | GraphHead | DeepShip | graph_aware_attention | Stage-2 trained head | 3 | 0.781421 +/- 0.010223 | 0.772453 +/- 0.014504 | 0.780935 +/- 0.011098 | N/A | 83,462 | N/A | S8, eval_samples=5, deterministic multi-bag |
| 8 | GraphHead | ShipsEar | attention | Stage-2 trained head | 3 | 0.736842 +/- 0.042974 | 0.687619 +/- 0.068641 | 0.706391 +/- 0.054167 | N/A | 9,222 | N/A | S8, eval_samples=5, deterministic multi-bag |
| 9 | GraphHead | ShipsEar | graph | Stage-2 trained head | 3 | 0.684211 +/- 0.042974 | 0.635628 +/- 0.063697 | 0.652954 +/- 0.053170 | N/A | 58,887 | N/A | S8, eval_samples=5, deterministic multi-bag |
| 10 | GraphHead | ShipsEar | graph_aware_attention | Stage-2 trained head | 3 | 0.719298 +/- 0.065643 | 0.676508 +/- 0.083986 | 0.694695 +/- 0.069825 | N/A | 83,591 | N/A | S8, eval_samples=5, deterministic multi-bag |
| 11 | Signal-Noise decoupled | ShipsEar | SN baseline | Stage-2 trained downstream | 3 | 0.719298 +/- 0.065643 | 0.680159 +/- 0.090150 | 0.692690 +/- 0.081989 | 55,488 | 16,328 | 20.794M | temporal_similarity, full soft attention |
| 12 | Signal-Noise decoupled | ShipsEar | SN threshold graph | Stage-2 trained downstream | 3 | 0.719298 +/- 0.065643 | 0.680159 +/- 0.090150 | 0.692690 +/- 0.081989 | 55,488 | 16,328 | 20.794M | threshold_similarity sim_threshold=0.8; avg graph degree 6.212 |
| 13 | Signal-Noise decoupled | ShipsEar | SN temperature | Stage-2 trained downstream | 3 | 0.719298 +/- 0.065643 | 0.680159 +/- 0.090150 | 0.692690 +/- 0.081989 | 55,490 | 16,330 | 20.794M | learnable temperature; avg graph degree 3.072 |
| 14 | Signal-Noise decoupled | ShipsEar | SN threshold + temperature | Stage-2 trained downstream | 3 | 0.719298 +/- 0.065643 | 0.680159 +/- 0.090150 | 0.692690 +/- 0.081989 | 55,490 | 16,330 | 20.794M | threshold_similarity + temperature; avg graph degree 6.211 |
| 15 | Signal-Noise decoupled | ShipsEar | SN ExpD Top4 Warmup5 | Stage-2 trained downstream | 3 | 0.701754 +/- 0.049622 | 0.649841 +/- 0.070331 | 0.669591 +/- 0.065163 | 55,488 | 16,328 | 20.794M | threshold_similarity, signal_top_k=4, warmup=5; avg graph degree 6.249; avg signal top-k 4.0 |
| 16 | Signal-Noise decoupled | ShipsEar | SN ExpE Top4 NoWarmup | Stage-2 trained downstream | 3 | 0.701754 +/- 0.024811 | 0.644615 +/- 0.044065 | 0.667521 +/- 0.035805 | 55,488 | 16,328 | 20.794M | temporal_similarity, signal_top_k=4, warmup=0; avg graph degree 3.071; avg signal top-k 4.0 |
| 17 | Signal-Noise decoupled | DeepShip | SN baseline | Stage-2 trained downstream | 3 | 0.775956 +/- 0.007728 | 0.779129 +/- 0.002623 | 0.776148 +/- 0.006917 | 55,326 | 16,295 | 20.794M | temporal_similarity, full soft attention; avg graph degree 2.980; avg signal top-k 8.0 |
| 18 | Signal-Noise decoupled | DeepShip | SN ExpE Top4 NoWarmup | Stage-2 trained downstream | 3 | 0.784153 +/- 0.010223 | 0.786350 +/- 0.017095 | 0.784107 +/- 0.010699 | 55,326 | 16,295 | 20.794M | temporal_similarity, signal_top_k=4, warmup=0; avg graph degree 2.981; avg signal top-k 4.0 |
| 19 | Signal-Noise decoupled | DeepShip | SN ThresholdOnly | Stage-2 trained downstream | 3 | 0.775956 +/- 0.007728 | 0.779129 +/- 0.002623 | 0.776148 +/- 0.006917 | 55,326 | 16,295 | 20.794M | threshold_similarity sim_threshold=0.8, full soft attention; avg graph degree 6.490 |
| 20 | Signal-Noise decoupled | DeepShip | SN ETA Complete | Stage-2 trained downstream | 3 | 0.784153 +/- 0.010223 | 0.786350 +/- 0.017095 | 0.784107 +/- 0.010699 | 55,326 | 16,295 | 20.794M | threshold_similarity sim_threshold=0.8, signal_top_k=4, warmup=0; avg graph degree 6.134 |
| 21 | Signal-Noise decoupled | DeepShip | SN ExpD Warmup5 | Stage-2 trained downstream | 3 | 0.792350 +/- 0.007728 | 0.800943 +/- 0.017295 | 0.792980 +/- 0.008654 | 55,326 | 16,295 | 20.794M | threshold_similarity sim_threshold=0.8, signal_top_k=4, warmup=5; avg graph degree 6.310 |

## 表 2：不同 Seed 明细指标

该表保留每个实验在不同 seed 上的 ACC、Macro-F1、Weighted-F1，方便观察随机种子波动。

| 序号 | 实验组 | 数据集 | 实验/模型 | Seed | ACC | Macro-F1 | Weighted-F1 | Params | Trainable Params | MACs | 备注 |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | ShuffleFAC encoder | DeepShip | recording_eval | 42 | 0.770492 | 0.762786 | 0.770803 | 39,031 | 39,031 | 2.585M | strict recording-level split |
| 2 | ShuffleFAC encoder | DeepShip | recording_eval | 43 | 0.786885 | 0.780370 | 0.787026 | 39,031 | 39,031 | 2.585M | strict recording-level split |
| 3 | ShuffleFAC encoder | DeepShip | recording_eval | 44 | 0.770492 | 0.775663 | 0.771849 | 39,031 | 39,031 | 2.585M | strict recording-level split |
| 4 | ShuffleFAC encoder | DeepShip | aggregation_eval | 42 | 0.770492 | 0.762786 | 0.770803 | 39,031 | 39,031 | 2.585M | strict recording-level split |
| 5 | ShuffleFAC encoder | DeepShip | aggregation_eval | 43 | 0.786885 | 0.780370 | 0.787026 | 39,031 | 39,031 | 2.585M | strict recording-level split |
| 6 | ShuffleFAC encoder | DeepShip | aggregation_eval | 44 | 0.770492 | 0.775663 | 0.771849 | 39,031 | 39,031 | 2.585M | strict recording-level split |
| 7 | ShuffleFAC encoder | ShipsEar | recording_eval | 42 | 0.631579 | 0.550476 | 0.577444 | 39,160 | 39,160 | 2.585M | strict recording-level split |
| 8 | ShuffleFAC encoder | ShipsEar | recording_eval | 43 | 0.789474 | 0.739377 | 0.764556 | 39,160 | 39,160 | 2.585M | strict recording-level split |
| 9 | ShuffleFAC encoder | ShipsEar | recording_eval | 44 | 0.684211 | 0.586667 | 0.624561 | 39,160 | 39,160 | 2.585M | strict recording-level split |
| 10 | ShuffleFAC encoder | ShipsEar | aggregation_eval | 42 | 0.631579 | 0.550476 | 0.577444 | 39,160 | 39,160 | 2.585M | strict recording-level split |
| 11 | ShuffleFAC encoder | ShipsEar | aggregation_eval | 43 | 0.789474 | 0.739377 | 0.764556 | 39,160 | 39,160 | 2.585M | strict recording-level split |
| 12 | ShuffleFAC encoder | ShipsEar | aggregation_eval | 44 | 0.684211 | 0.586667 | 0.624561 | 39,160 | 39,160 | 2.585M | strict recording-level split |
| 13 | GraphHead | DeepShip | attention | 42 | 0.827869 | 0.832826 | 0.829202 | N/A | 9,093 | N/A | S8, eval_samples=5, deterministic multi-bag |
| 14 | GraphHead | DeepShip | attention | 43 | 0.754098 | 0.744891 | 0.754367 | N/A | 9,093 | N/A | S8, eval_samples=5, deterministic multi-bag |
| 15 | GraphHead | DeepShip | attention | 44 | 0.745902 | 0.735507 | 0.744360 | N/A | 9,093 | N/A | S8, eval_samples=5, deterministic multi-bag |
| 16 | GraphHead | DeepShip | graph | 42 | 0.795082 | 0.801320 | 0.797261 | N/A | 58,758 | N/A | S8, eval_samples=5, deterministic multi-bag |
| 17 | GraphHead | DeepShip | graph | 43 | 0.778689 | 0.767170 | 0.777994 | N/A | 58,758 | N/A | S8, eval_samples=5, deterministic multi-bag |
| 18 | GraphHead | DeepShip | graph | 44 | 0.745902 | 0.743667 | 0.745521 | N/A | 58,758 | N/A | S8, eval_samples=5, deterministic multi-bag |
| 19 | GraphHead | DeepShip | graph_aware_attention | 42 | 0.795082 | 0.789561 | 0.795714 | N/A | 83,462 | N/A | S8, eval_samples=5, deterministic multi-bag |
| 20 | GraphHead | DeepShip | graph_aware_attention | 43 | 0.770492 | 0.754099 | 0.768971 | N/A | 83,462 | N/A | S8, eval_samples=5, deterministic multi-bag |
| 21 | GraphHead | DeepShip | graph_aware_attention | 44 | 0.778689 | 0.773699 | 0.778120 | N/A | 83,462 | N/A | S8, eval_samples=5, deterministic multi-bag |
| 22 | GraphHead | ShipsEar | attention | 42 | 0.684211 | 0.593333 | 0.635088 | N/A | 9,222 | N/A | S8, eval_samples=5, deterministic multi-bag |
| 23 | GraphHead | ShipsEar | attention | 43 | 0.789474 | 0.754762 | 0.766291 | N/A | 9,222 | N/A | S8, eval_samples=5, deterministic multi-bag |
| 24 | GraphHead | ShipsEar | attention | 44 | 0.736842 | 0.714762 | 0.717794 | N/A | 9,222 | N/A | S8, eval_samples=5, deterministic multi-bag |
| 25 | GraphHead | ShipsEar | graph | 42 | 0.631579 | 0.558788 | 0.587560 | N/A | 58,887 | N/A | S8, eval_samples=5, deterministic multi-bag |
| 26 | GraphHead | ShipsEar | graph | 43 | 0.684211 | 0.633333 | 0.653509 | N/A | 58,887 | N/A | S8, eval_samples=5, deterministic multi-bag |
| 27 | GraphHead | ShipsEar | graph | 44 | 0.736842 | 0.714762 | 0.717794 | N/A | 58,887 | N/A | S8, eval_samples=5, deterministic multi-bag |
| 28 | GraphHead | ShipsEar | graph_aware_attention | 42 | 0.631579 | 0.560000 | 0.600000 | N/A | 83,591 | N/A | S8, eval_samples=5, deterministic multi-bag |
| 29 | GraphHead | ShipsEar | graph_aware_attention | 43 | 0.789474 | 0.754762 | 0.766291 | N/A | 83,591 | N/A | S8, eval_samples=5, deterministic multi-bag |
| 30 | GraphHead | ShipsEar | graph_aware_attention | 44 | 0.736842 | 0.714762 | 0.717794 | N/A | 83,591 | N/A | S8, eval_samples=5, deterministic multi-bag |
| 31 | Signal-Noise decoupled | ShipsEar | SN baseline | 42 | 0.631579 | 0.560000 | 0.582456 | 55,488 | 16,328 | 20.794M | temporal_similarity, full soft attention |
| 32 | Signal-Noise decoupled | ShipsEar | SN baseline | 43 | 0.736842 | 0.703333 | 0.716667 | 55,488 | 16,328 | 20.794M | temporal_similarity, full soft attention |
| 33 | Signal-Noise decoupled | ShipsEar | SN baseline | 44 | 0.789474 | 0.777143 | 0.778947 | 55,488 | 16,328 | 20.794M | temporal_similarity, full soft attention |
| 34 | Signal-Noise decoupled | ShipsEar | SN threshold graph | 42 | 0.631579 | 0.560000 | 0.582456 | 55,488 | 16,328 | 20.794M | threshold_similarity sim_threshold=0.8; avg graph degree 6.212 |
| 35 | Signal-Noise decoupled | ShipsEar | SN threshold graph | 43 | 0.736842 | 0.703333 | 0.716667 | 55,488 | 16,328 | 20.794M | threshold_similarity sim_threshold=0.8; avg graph degree 6.212 |
| 36 | Signal-Noise decoupled | ShipsEar | SN threshold graph | 44 | 0.789474 | 0.777143 | 0.778947 | 55,488 | 16,328 | 20.794M | threshold_similarity sim_threshold=0.8; avg graph degree 6.212 |
| 37 | Signal-Noise decoupled | ShipsEar | SN temperature | 42 | 0.631579 | 0.560000 | 0.582456 | 55,490 | 16,330 | 20.794M | learnable temperature; avg graph degree 3.072 |
| 38 | Signal-Noise decoupled | ShipsEar | SN temperature | 43 | 0.736842 | 0.703333 | 0.716667 | 55,490 | 16,330 | 20.794M | learnable temperature; avg graph degree 3.072 |
| 39 | Signal-Noise decoupled | ShipsEar | SN temperature | 44 | 0.789474 | 0.777143 | 0.778947 | 55,490 | 16,330 | 20.794M | learnable temperature; avg graph degree 3.072 |
| 40 | Signal-Noise decoupled | ShipsEar | SN threshold + temperature | 42 | 0.631579 | 0.560000 | 0.582456 | 55,490 | 16,330 | 20.794M | threshold_similarity + temperature; avg graph degree 6.211 |
| 41 | Signal-Noise decoupled | ShipsEar | SN threshold + temperature | 43 | 0.736842 | 0.703333 | 0.716667 | 55,490 | 16,330 | 20.794M | threshold_similarity + temperature; avg graph degree 6.211 |
| 42 | Signal-Noise decoupled | ShipsEar | SN threshold + temperature | 44 | 0.789474 | 0.777143 | 0.778947 | 55,490 | 16,330 | 20.794M | threshold_similarity + temperature; avg graph degree 6.211 |
| 43 | Signal-Noise decoupled | ShipsEar | SN ExpD Top4 Warmup5 | 42 | 0.631579 | 0.550476 | 0.577444 | 55,488 | 16,328 | 20.794M | threshold_similarity, signal_top_k=4, warmup=5; avg graph degree 6.249; avg signal top-k 4.0 |
| 44 | Signal-Noise decoupled | ShipsEar | SN ExpD Top4 Warmup5 | 43 | 0.736842 | 0.703333 | 0.716667 | 55,488 | 16,328 | 20.794M | threshold_similarity, signal_top_k=4, warmup=5; avg graph degree 6.249; avg signal top-k 4.0 |
| 45 | Signal-Noise decoupled | ShipsEar | SN ExpD Top4 Warmup5 | 44 | 0.736842 | 0.695714 | 0.714662 | 55,488 | 16,328 | 20.794M | threshold_similarity, signal_top_k=4, warmup=5; avg graph degree 6.249; avg signal top-k 4.0 |
| 46 | Signal-Noise decoupled | ShipsEar | SN ExpE Top4 NoWarmup | 42 | 0.684211 | 0.597179 | 0.632389 | 55,488 | 16,328 | 20.794M | temporal_similarity, signal_top_k=4, warmup=0; avg graph degree 3.071; avg signal top-k 4.0 |
| 47 | Signal-Noise decoupled | ShipsEar | SN ExpE Top4 NoWarmup | 43 | 0.736842 | 0.703333 | 0.716667 | 55,488 | 16,328 | 20.794M | temporal_similarity, signal_top_k=4, warmup=0; avg graph degree 3.071; avg signal top-k 4.0 |
| 48 | Signal-Noise decoupled | ShipsEar | SN ExpE Top4 NoWarmup | 44 | 0.684211 | 0.633333 | 0.653509 | 55,488 | 16,328 | 20.794M | temporal_similarity, signal_top_k=4, warmup=0; avg graph degree 3.071; avg signal top-k 4.0 |
| 49 | Signal-Noise decoupled | DeepShip | SN baseline | 42 | 0.770492 | 0.782781 | 0.772036 | 55,326 | 16,295 | 20.794M | temporal_similarity, full soft attention; avg signal top-k 8.0 |
| 50 | Signal-Noise decoupled | DeepShip | SN baseline | 43 | 0.786885 | 0.777864 | 0.785891 | 55,326 | 16,295 | 20.794M | temporal_similarity, full soft attention; avg signal top-k 8.0 |
| 51 | Signal-Noise decoupled | DeepShip | SN baseline | 44 | 0.770492 | 0.776741 | 0.770518 | 55,326 | 16,295 | 20.794M | temporal_similarity, full soft attention; avg signal top-k 8.0 |
| 52 | Signal-Noise decoupled | DeepShip | SN ExpE Top4 NoWarmup | 42 | 0.786885 | 0.803874 | 0.787415 | 55,326 | 16,295 | 20.794M | temporal_similarity, signal_top_k=4, warmup=0; avg signal top-k 4.0 |
| 53 | Signal-Noise decoupled | DeepShip | SN ExpE Top4 NoWarmup | 43 | 0.795082 | 0.792011 | 0.795239 | 55,326 | 16,295 | 20.794M | temporal_similarity, signal_top_k=4, warmup=0; avg signal top-k 4.0 |
| 54 | Signal-Noise decoupled | DeepShip | SN ExpE Top4 NoWarmup | 44 | 0.770492 | 0.763165 | 0.769665 | 55,326 | 16,295 | 20.794M | temporal_similarity, signal_top_k=4, warmup=0; avg signal top-k 4.0 |
| 55 | Signal-Noise decoupled | DeepShip | SN ThresholdOnly | 42 | 0.770492 | 0.782781 | 0.772036 | 55,326 | 16,295 | 20.794M | threshold_similarity sim_threshold=0.8; avg graph degree 7.000 |
| 56 | Signal-Noise decoupled | DeepShip | SN ThresholdOnly | 43 | 0.786885 | 0.777864 | 0.785891 | 55,326 | 16,295 | 20.794M | threshold_similarity sim_threshold=0.8; avg graph degree 5.720 |
| 57 | Signal-Noise decoupled | DeepShip | SN ThresholdOnly | 44 | 0.770492 | 0.776741 | 0.770518 | 55,326 | 16,295 | 20.794M | threshold_similarity sim_threshold=0.8; avg graph degree 6.750 |
| 58 | Signal-Noise decoupled | DeepShip | SN ETA Complete | 42 | 0.786885 | 0.803874 | 0.787415 | 55,326 | 16,295 | 20.794M | threshold_similarity sim_threshold=0.8, signal_top_k=4; avg graph degree 6.666 |
| 59 | Signal-Noise decoupled | DeepShip | SN ETA Complete | 43 | 0.795082 | 0.792011 | 0.795239 | 55,326 | 16,295 | 20.794M | threshold_similarity sim_threshold=0.8, signal_top_k=4; avg graph degree 5.350 |
| 60 | Signal-Noise decoupled | DeepShip | SN ETA Complete | 44 | 0.770492 | 0.763165 | 0.769665 | 55,326 | 16,295 | 20.794M | threshold_similarity sim_threshold=0.8, signal_top_k=4; avg graph degree 6.386 |
| 61 | Signal-Noise decoupled | DeepShip | SN ExpD Warmup5 | 42 | 0.803279 | 0.823044 | 0.805217 | 55,326 | 16,295 | 20.794M | threshold_similarity sim_threshold=0.8, signal_top_k=4, warmup=5; avg graph degree 6.536 |
| 62 | Signal-Noise decoupled | DeepShip | SN ExpD Warmup5 | 43 | 0.786885 | 0.780818 | 0.786910 | 55,326 | 16,295 | 20.794M | threshold_similarity sim_threshold=0.8, signal_top_k=4, warmup=5; avg graph degree 5.395 |
| 63 | Signal-Noise decoupled | DeepShip | SN ExpD Warmup5 | 44 | 0.786885 | 0.798967 | 0.786811 | 55,326 | 16,295 | 20.794M | threshold_similarity sim_threshold=0.8, signal_top_k=4, warmup=5; avg graph degree 6.999 |


## 表 3：ShipsEar Best Foundation Checkpoint 单 Backbone 消融

本节固定使用 Stage-1 ShipsEar seed 43 checkpoint：`results/ShuffleFAC/0502_External_ShuffleFAC_ShipsEar_gamma16_multiseed_3s_7_1_2/seed_43/best.pt`。这些结果不使用 `seed_*` wildcard，目的是让所有 downstream ETA 消融共享同一个高质量 frozen encoder。

| 实验 | Edge mode | Top-K | Warmup | Test Loss | ACC | Macro-F1 | Weighted-F1 | Avg Graph Degree | Avg Signal Top-K Count | Best Val Macro-F1 | Best Epoch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Existing seed43 baseline | temporal_similarity | 0 | 0 | 1.087619 | 0.736842 | 0.703333 | 0.716667 | N/A | N/A | 0.904762 | 4 |
| BestSeed43 ThresholdOnly | threshold_similarity | 0 | 0 | 1.087682 | 0.736842 | 0.703333 | 0.716667 | 6.148958 | 8.000000 | 0.904762 | 4 |
| BestSeed43 ETA Complete | threshold_similarity | 4 | 0 | 1.110502 | 0.736842 | 0.703333 | 0.716667 | 6.142708 | 4.000000 | 0.893333 | 4 |
| BestSeed43 ExpD Warmup5 | threshold_similarity | 4 | 5 | 1.041543 | 0.736842 | 0.703333 | 0.716667 | 6.184375 | 4.000000 | 0.893333 | 5 |

观察：固定使用 ShipsEar seed 43 foundation 后，三组 ETA 消融与已有 seed43 baseline 在 ACC、Macro-F1、Weighted-F1 上完全持平。`sim_threshold=0.8` 的图度数约为 6.14-6.18，仍然是高密度图，不是健康稀疏图；Warmup5 的 test loss 最低。


## 表 4：ShipsEar Best Foundation Checkpoint 阈值稀疏性 Sweep

本节固定使用 Stage-1 ShipsEar seed 43 checkpoint，并固定 ETA strongest 配置：`edge_mode=threshold_similarity`、`signal_top_k=4`、`topk_warmup_epochs=5`。目标是通过提高 `sim_threshold` 将平均图度数压到健康稀疏区间 1.5-3.5。

| sim_threshold | Test Loss | ACC | Macro-F1 | Weighted-F1 | Avg Graph Degree | In 1.5-3.5? | Best Val Macro-F1 | Best Epoch |
|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 0.80 | 1.041543 | 0.736842 | 0.703333 | 0.716667 | 6.184375 | No | 0.893333 | 5 |
| 0.90 | 1.041598 | 0.736842 | 0.703333 | 0.716667 | 5.352083 | No | 0.893333 | 5 |
| 0.95 | 1.041650 | 0.736842 | 0.703333 | 0.716667 | 4.258333 | No | 0.893333 | 5 |
| 0.98 | 1.041630 | 0.736842 | 0.703333 | 0.716667 | 3.195833 | Yes | 0.893333 | 5 |

观察：提高阈值能够单调降低图密度，`sim_threshold=0.98` 首次进入目标健康稀疏区间，同时保持 ACC、Macro-F1、Weighted-F1 不变。当前最适合论文叙事的稀疏图设置是 `0.98`。

## 关键观察

- DeepShip 上，GraphHead 的 graph_aware_attention 在三 seed 平均 Macro-F1 为 0.772453，和 Stage-1 recording eval 的 0.772940 非常接近，提升不明显。
- ShipsEar 上，GraphHead graph_aware_attention 三 seed 平均 Macro-F1 为 0.676508，高于 ShuffleFAC Stage-1 recording eval 的 0.625507。
- ShipsEar 上，Signal-Noise baseline 三 seed 平均 Macro-F1 为 0.680159，略高于 GraphHead graph_aware_attention 的 0.676508，但差距很小。
- threshold graph、temperature、threshold+temperature 三个消融在 ShipsEar 上与 Signal-Noise baseline 指标完全持平。
- ExpD Top4 Warmup5 在 ShipsEar 上平均 Macro-F1 为 0.649841，低于 Signal-Noise baseline 的 0.680159，说明 Top-K sparse signal pooling 当前设置没有带来收益。
- ExpE Top4 NoWarmup 在 ShipsEar 上平均 Macro-F1 为 0.644615，也低于 Signal-Noise baseline，并且略低于 ExpD；纯 hard Top-K 从第 1 个 epoch 开始启用后没有带来收益。
- DeepShip 上，Signal-Noise baseline 三 seed 平均 Macro-F1 为 0.779129，略高于 Stage-1 recording eval 的 0.772940；ExpE Top4 NoWarmup 进一步到 0.786350，但 seed 间方差也更大。
- DeepShip 新增阈值图消融中，`sim_threshold=0.8` 的平均图度数为 6.13-6.49，明显高于目标健康稀疏区间 1.5-3.5，接近 8 clips 下的全连接图；其中 ExpD Warmup5 的 Macro-F1 最高，为 0.800943。
- 固定 ShipsEar best foundation checkpoint seed43 后，ThresholdOnly、ETA Complete、ExpD Warmup5 三组 downstream 消融的 Macro-F1 都为 0.703333，与已有 seed43 baseline 持平；`sim_threshold=0.8` 的图度数约 6.15，仍然过密。
- ShipsEar seed43 threshold sweep 表明，将 `sim_threshold` 提高到 0.98 可把 avg graph degree 降到 3.195833，进入目标 1.5-3.5 稀疏区间，且 Macro-F1 保持 0.703333。

## 文件来源

- Stage-1 ShuffleFAC：`results/ShuffleFAC/*_recording_eval/seed_*/metrics.txt` 与 `results/ShuffleFAC/*_aggregation_eval/seed_*/metrics.txt`。
- GraphHead：`results/ShuffleFAC_GRAPHHEAD/0502_*_seed*_S8_ms5/metrics.txt`。
- Signal-Noise：`results/ShuffleFAC_SIGNAL_NOISE/*/seed*/metrics.json`。
- ExpD：`results/ShuffleFAC_SIGNAL_NOISE/ShipsEar_ExpD_Top4_Warmup5_Sim08/seed*/metrics.json`。
- ExpE：`results/ShuffleFAC_SIGNAL_NOISE/ShipsEar_ExpE_Top4_NoWarmup_TempSim/seed*/metrics.json`。
- DeepShip Signal-Noise：`results/ShuffleFAC_SIGNAL_NOISE/DeepShip_Baseline_TempSim/seed*/metrics.json` 与 `results/ShuffleFAC_SIGNAL_NOISE/DeepShip_ExpE_Top4_NoWarmup/seed*/metrics.json`。
- DeepShip threshold ablations：`results/ShuffleFAC_SIGNAL_NOISE/DeepShip_ThresholdOnly/seed*/metrics.json`、`results/ShuffleFAC_SIGNAL_NOISE/DeepShip_ETA_Complete/seed*/metrics.json`、`results/ShuffleFAC_SIGNAL_NOISE/DeepShip_ExpD_Warmup5/seed*/metrics.json`。
- ShipsEar best seed43 ablations：`results/ShuffleFAC_SIGNAL_NOISE/ShipsEar_BestSeed43_ThresholdOnly/metrics.json`、`results/ShuffleFAC_SIGNAL_NOISE/ShipsEar_BestSeed43_ETA_Complete/metrics.json`、`results/ShuffleFAC_SIGNAL_NOISE/ShipsEar_BestSeed43_ExpD_Warmup5/metrics.json`。
- ShipsEar seed43 threshold sweep：`results/ShuffleFAC_SIGNAL_NOISE/ShipsEar_BestSeed43_ExpD_Warmup5_Thr090/metrics.json`、`results/ShuffleFAC_SIGNAL_NOISE/ShipsEar_BestSeed43_ExpD_Warmup5_Thr095/metrics.json`、`results/ShuffleFAC_SIGNAL_NOISE/ShipsEar_BestSeed43_ExpD_Warmup5_Thr098/metrics.json`。

## 表 5：DeepShip 前端鲁棒性与 SN-ExpD 论文结果锁定（2026-05-20）

本节补充 `paper_results_materials_20260520` 中已经锁定的论文结果材料。统计口径统一为 **mean ± sample std over seeds 42/43/44**。不包含 smoke rows、不包含旧 `sn_decoupled` rows、不包含 `--head all` 输出、不包含未完成的 PANNs 非 sharedcache 目录；PANNs fine-tune 训练阶段 mean-prob test result 不作为最终 recording-level aggregation result。

### 表 5.1：DeepShip clip-level front-end diagnostic 与参数效率

该表只用于前端切片级能力和参数效率诊断，不作为最终 recording-level aggregation 排名依据。

| front-end | Test clip ACC | Test clip Macro-F1 | Params | Trainable Params | MACs | Protocol note |
|---|---:|---:|---:|---:|---:|---|
| ShuffleFAC gamma=16 | 0.684325 ± 0.002286 | 0.683512 ± 0.002907 | 39,031 | 39,031 | 2.585M | DeepShip-trained ShuffleFAC, strict recording-level split |
| ResNet18 | 0.675582 ± 0.013181 | 0.675608 ± 0.011486 | 11,172,292 | 11,172,292 | N/A | `train_clip_frontend.py`, `pretrained=none`, best checkpoint by val recording Macro-F1 |
| MobileNetV2 | 0.667670 ± 0.004947 | 0.667747 ± 0.005353 | 2,228,420 | 2,228,420 | N/A | `train_clip_frontend.py`, `pretrained=none`, best checkpoint by val recording Macro-F1 |

PANNs-CNN14 frozen 不放入该表，因为锁定的 PANNs 行使用 official AudioSet frozen embedding front-end，没有与 DeepShip-trained clip classifier 完全可比的 clip-level diagnostic。PANNs 只进入后续 frozen recording-level front-end 表。

### 表 5.2：DeepShip formal non-SN recording-level aggregation

该表评估 frozen front-end embeddings 上的普通 recording-level heads。

| front-end | mean | attention | BiGRU | MIL linear softmax | best non-SN head | best non-SN Macro-F1 | seeds |
|---|---:|---:|---:|---:|---|---:|---|
| ShuffleFAC | 0.774015 ± 0.011171 | 0.774338 ± 0.011997 | 0.790255 ± 0.001098 | 0.692614 ± 0.024915 | BiGRU | 0.790255 ± 0.001098 | 42/43/44 |
| ResNet18 | 0.756192 ± 0.013252 | 0.740216 ± 0.036356 | 0.730710 ± 0.027281 | 0.647293 ± 0.030482 | mean | 0.756192 ± 0.013252 | 42/43/44 |
| MobileNetV2 | 0.741241 ± 0.031799 | 0.733914 ± 0.050809 | 0.741725 ± 0.035584 | 0.651732 ± 0.061479 | BiGRU | 0.741725 ± 0.035584 | 42/43/44 |
| PANNs-CNN14 frozen | 0.575547 ± 0.098200 | 0.517527 ± 0.026984 | 0.555630 ± 0.007820 | 0.583431 ± 0.030770 | MIL linear softmax | 0.583431 ± 0.030770 | 42/43/44 |

观察：ShuffleFAC + BiGRU 是当前最强 non-SN baseline，Macro-F1 = 0.790255 ± 0.001098。

### 表 5.3：DeepShip best non-SN vs formal SN-ExpD-Warmup5

正式 SN head 是 `sn_expd_warmup5`，不是旧的简化 `sn_decoupled`。该 head 使用 threshold-similarity noise graph、temporal edge、`signal_top_k=4` 和 `topk_warmup_epochs=5`。

| front-end | best non-SN head | best non-SN Macro-F1 | SN-ExpD-Warmup5 Macro-F1 | Macro-F1 gain | SN-ExpD-Warmup5 ACC | seeds |
|---|---|---:|---:|---:|---:|---|
| ShuffleFAC | BiGRU | 0.790255 ± 0.001098 | 0.800943 ± 0.021182 | 0.010688 | 0.792350 ± 0.009465 | 42/43/44 |
| ResNet18 | mean | 0.756192 ± 0.013252 | 0.705805 ± 0.013812 | -0.050387 | 0.743169 ± 0.012521 | 42/43/44 |
| MobileNetV2 | BiGRU | 0.741725 ± 0.035584 | 0.769124 ± 0.029399 | 0.027399 | 0.775956 ± 0.017063 | 42/43/44 |
| PANNs-CNN14 frozen | MIL linear softmax | 0.583431 ± 0.030770 | 0.651811 ± 0.033977 | 0.068380 | 0.677596 ± 0.034126 | 42/43/44 |

最终最佳结果是 **ShuffleFAC + SN-ExpD-Warmup5**，Macro-F1 = **0.800943 ± 0.021182**，ACC = **0.792350 ± 0.009465**。该结果复现原 DeepShip SN ExpD Warmup5 主结果，并比 ShuffleFAC + BiGRU non-SN baseline 高 0.010688 Macro-F1。

### 结果解释与论文边界

- 论文主线应表述为：在可替换 frozen clip-level front-ends 之上研究 strict recording-level signal-noise decoupled aggregation。
- ShuffleFAC 仍是参数效率最高且 clip-level diagnostic 最强的可比前端，但不要把论文主贡献写成新的 clip-level ShuffleFAC front-end。
- 普通 recording-level heads 已经很强，尤其 ShuffleFAC + BiGRU 已达到 0.790255 ± 0.001098 Macro-F1。
- SN-ExpD-Warmup5 在 ShuffleFAC、MobileNetV2、PANNs-CNN14 frozen 上提升，但在 ResNet18 上下降 0.050387 Macro-F1，因此不能宣称 SN 对所有前端一致提升。
- PANNs-CNN14 frozen 在 SN-ExpD 下有明显相对提升，但绝对 Macro-F1 仍低于 ShuffleFAC 和 MobileNetV2。
- 正式 SN 结果来自 `sn_expd_warmup5` / 原完整 ExpD Warmup5 机制，不来自旧 `sn_decoupled`。

### 锁定来源

- Table 1 source: `external/ShuffleFAC/FRONTEND_CLIP_ACCURACY_COMPARISON.md` 及对应 formal front-end metrics。
- Table 2 source: `results/FRONTEND_ROBUSTNESS/formal_non_sn_summary_20260520/table2_lock_20260520/`。
- Table 3 source: `results/FRONTEND_ROBUSTNESS/formal_sn_expd_warmup5_cross_frontend_20260520/table3_lock_20260520/`。
- 论文材料文件：`external/ShuffleFAC/paper_results_materials_20260520/paper_results_tables.md`、`paper_results_analysis.md`、`paper_discussion_points.md`、`final_experiment_status.md`。

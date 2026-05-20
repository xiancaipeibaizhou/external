# Paper Results Tables

All statistics are reported as mean ± sample std over seeds 42/43/44 unless noted otherwise. Smoke rows, `sn_decoupled` rows, `--head all` outputs, and incomplete PANNs non-sharedcache directories are excluded.

## Table 1: Clip-Level Front-End Diagnostic and Parameter Efficiency

This diagnostic table measures clip-level front-end ability. It is not the main recording-level aggregation result.

| front-end | test clip ACC | test clip Macro-F1 | params | trainable params | MACs | protocol note |
|---|---:|---:|---:|---:|---:|---|
| ShuffleFAC gamma=16 | 0.684325 ± 0.002286 | 0.683512 ± 0.002907 | 39,031 | 39,031 | 2.585M | DeepShip-trained ShuffleFAC, strict recording-level split |
| ResNet18 | 0.675582 ± 0.013181 | 0.675608 ± 0.011486 | 11,172,292 | 11,172,292 | N/A | train_clip_frontend.py, pretrained=none, best checkpoint by val recording Macro-F1 |
| MobileNetV2 | 0.667670 ± 0.004947 | 0.667747 ± 0.005353 | 2,228,420 | 2,228,420 | N/A | train_clip_frontend.py, pretrained=none, best checkpoint by val recording Macro-F1 |

PANNs-CNN14 frozen is not included in Table 1 because the locked PANNs rows use the official AudioSet frozen embedding front-end without a comparable DeepShip-trained clip classifier diagnostic. PANNs appears in Tables 2 and 3 as a frozen recording-level front-end.

## Table 2: Formal Non-SN Recording-Level Aggregation

This table evaluates ordinary recording-level heads on frozen front-end embeddings.

| front-end | mean | attention | BiGRU | MIL linear softmax | best non-SN head | best non-SN Macro-F1 | seeds |
|---|---:|---:|---:|---:|---|---:|---|
| shufflefac | 0.774015 ± 0.011171 | 0.774338 ± 0.011997 | 0.790255 ± 0.001098 | 0.692614 ± 0.024915 | bigru | 0.790255 ± 0.001098 | 42/43/44 |
| resnet18 | 0.756192 ± 0.013252 | 0.740216 ± 0.036356 | 0.730710 ± 0.027281 | 0.647293 ± 0.030482 | mean | 0.756192 ± 0.013252 | 42/43/44 |
| mobilenet_v2 | 0.741241 ± 0.031799 | 0.733914 ± 0.050809 | 0.741725 ± 0.035584 | 0.651732 ± 0.061479 | bigru | 0.741725 ± 0.035584 | 42/43/44 |
| panns_cnn14_frozen | 0.575547 ± 0.098200 | 0.517527 ± 0.026984 | 0.555630 ± 0.007820 | 0.583431 ± 0.030770 | mil_linear_softmax | 0.583431 ± 0.030770 | 42/43/44 |

## Table 3: Best Non-SN vs SN-ExpD-Warmup5

This table evaluates strict recording-level signal-noise decoupled aggregation with the formal `sn_expd_warmup5` head.

| front-end | best non-SN head | best non-SN Macro-F1 | SN-ExpD-Warmup5 Macro-F1 | Macro-F1 gain | SN-ExpD-Warmup5 ACC | seeds |
|---|---|---:|---:|---:|---:|---|
| shufflefac | bigru | 0.790255 ± 0.001098 | 0.800943 ± 0.021182 | 0.010688 | 0.792350 ± 0.009465 | 42/43/44 |
| resnet18 | mean | 0.756192 ± 0.013252 | 0.705805 ± 0.013812 | -0.050387 | 0.743169 ± 0.012521 | 42/43/44 |
| mobilenet_v2 | bigru | 0.741725 ± 0.035584 | 0.769124 ± 0.029399 | 0.027399 | 0.775956 ± 0.017063 | 42/43/44 |
| panns_cnn14 | mil_linear_softmax | 0.583431 ± 0.030770 | 0.651811 ± 0.033977 | 0.068380 | 0.677596 ± 0.034126 | 42/43/44 |

The final best result is ShuffleFAC + SN-ExpD-Warmup5 with Macro-F1 = 0.800943 ± 0.021182.

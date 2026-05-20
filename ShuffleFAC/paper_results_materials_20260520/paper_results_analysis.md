# Paper Results Analysis

## Scope and Protocol

The experiments use the DeepShip strict recording-level split with seeds 42/43/44. Results are separated into three levels: clip-level front-end diagnostics, non-SN recording-level aggregation, and formal SN-ExpD recording-level aggregation. All aggregate statistics use sample standard deviation over the three seeds.

## Table 1: Clip-Level Front-End Diagnostic

ShuffleFAC gamma=16 gives the strongest clip-level diagnostic among the comparable DeepShip-trained front-ends, with test clip ACC 0.684325 ± 0.002286 and clip Macro-F1 0.683512 ± 0.002907. This is achieved with only 39,031 parameters and 2.585M MACs, far fewer parameters than ResNet18 and MobileNetV2.
ResNet18 reaches clip ACC 0.675582 ± 0.013181 and clip Macro-F1 0.675608 ± 0.011486 with 11,172,292 parameters.
MobileNetV2 reaches clip ACC 0.667670 ± 0.004947 and clip Macro-F1 0.667747 ± 0.005353 with 2,228,420 parameters.

Table 1 should be interpreted only as a front-end diagnostic. It does not determine the final recording-level ranking, because later aggregation heads can change the ordering.

## Table 2: Non-SN Recording-Level Aggregation

For shufflefac, the best non-SN head is bigru with Macro-F1 0.790255 ± 0.001098.
For resnet18, the best non-SN head is mean with Macro-F1 0.756192 ± 0.013252.
For mobilenet_v2, the best non-SN head is bigru with Macro-F1 0.741725 ± 0.035584.
For panns_cnn14_frozen, the best non-SN head is mil_linear_softmax with Macro-F1 0.583431 ± 0.030770.

Across non-SN heads, ShuffleFAC + BiGRU is the strongest non-SN baseline with Macro-F1 0.790255 ± 0.001098. This establishes a high ordinary recording-level aggregation baseline before adding signal-noise decoupling.

## Table 3: SN-ExpD-Warmup5 Recording-Level Aggregation

The formal SN head is `sn_expd_warmup5`, which uses threshold-similarity noise graph construction, temporal edges, signal Top-K=4, and five warmup epochs. It is not the older simplified `sn_decoupled` head.
For shufflefac, SN-ExpD-Warmup5 improves over the best non-SN baseline: 0.800943 ± 0.021182 vs 0.790255 ± 0.001098, gain 0.010688.
For resnet18, SN-ExpD-Warmup5 drops below the best non-SN baseline: 0.705805 ± 0.013812 vs 0.756192 ± 0.013252, gain -0.050387.
For mobilenet_v2, SN-ExpD-Warmup5 improves over the best non-SN baseline: 0.769124 ± 0.029399 vs 0.741725 ± 0.035584, gain 0.027399.
For panns_cnn14, SN-ExpD-Warmup5 improves over the best non-SN baseline: 0.651811 ± 0.033977 vs 0.583431 ± 0.030770, gain 0.068380.

The final best result is ShuffleFAC + SN-ExpD-Warmup5, with Macro-F1 = 0.800943 ± 0.021182 and ACC = 0.792350 ± 0.009465. This reproduces the original DeepShip SN ExpD Warmup5 main result and improves over the best ShuffleFAC non-SN head, BiGRU, by +0.010688 Macro-F1.

The improvement is not universal across front-ends. MobileNetV2 and PANNs-CNN14 frozen improve with SN-ExpD, but ResNet18 declines by -0.050387 Macro-F1 relative to its best non-SN mean head. Therefore the paper should claim that strict signal-noise decoupled aggregation can improve selected front-ends under this protocol, not that it consistently improves every front-end.

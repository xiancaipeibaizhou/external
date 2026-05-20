# Paper Discussion Points

## Main Takeaways

- The main contribution should be framed as strict recording-level signal-noise decoupled aggregation on top of interchangeable frozen clip-level front-ends.
- ShuffleFAC remains a strong and parameter-efficient front-end: it has the best comparable clip-level diagnostic while using far fewer parameters than ResNet18 and MobileNetV2.
- Ordinary recording-level heads are competitive: ShuffleFAC + BiGRU already reaches 0.790255 ± 0.001098 Macro-F1.
- SN-ExpD-Warmup5 gives the best final DeepShip result with ShuffleFAC, reaching 0.800943 ± 0.021182 Macro-F1.

## Cross-Front-End Interpretation

- SN-ExpD-Warmup5 improves ShuffleFAC, MobileNetV2, and PANNs-CNN14 frozen relative to each front-end's best non-SN baseline.
- ResNet18 declines under the same SN-ExpD settings, so the method should not be described as uniformly beneficial across all front-ends.
- The ResNet18 drop suggests that the SN head may interact with front-end embedding geometry, feature smoothness, or class evidence distribution. This is a discussion point, not a confirmed mechanism.
- PANNs-CNN14 frozen benefits substantially in relative terms, but its final absolute Macro-F1 remains below ShuffleFAC and MobileNetV2 in this setup.

## Claim Boundaries

- Do not claim a new clip-level ShuffleFAC front-end as the paper's main contribution.
- Do not claim SN improves every front-end; Table 3 explicitly shows a ResNet18 decrease.
- Do not use the older simplified `sn_decoupled` implementation as the formal SN result.
- Do not mix clip-level diagnostics with recording-level aggregation results when ranking methods.
- Do not include smoke rows or incomplete PANNs non-sharedcache runs.

## Suggested Writing Angle

A defensible Results/Discussion narrative is: Table 1 establishes front-end diagnostic capacity and parameter efficiency; Table 2 shows that ordinary recording-level heads already provide strong baselines under frozen front-ends; Table 3 shows that the full SN-ExpD-Warmup5 aggregation can further improve the strongest ShuffleFAC baseline and also benefits MobileNetV2 and PANNs-CNN14 frozen, while exposing front-end dependence through the ResNet18 drop.

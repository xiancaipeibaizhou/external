# ShuffleFAC 代码简要注释

本文档用于快速阅读 `external/ShuffleFAC` 下的代码结构。它是代码注释索引，不替代训练日志、实验结果表或 README。

## 核心文件

| 文件 | 作用 | 关键逻辑 |
|---|---|---|
| `model/shuffleFAC.py` | ShuffleFAC 模型定义 | 轻量 CNN 主干，组合 FAC 频率感知注入、grouped pointwise conv、depthwise conv 和 channel shuffle。 |
| `main.py` | 上游原始训练入口 | 简化的 class-folder 音频训练/验证流程；本项目正式实验主要使用 `run_deepship.py`。 |
| `run_deepship.py` | Stage-1 适配训练/评估入口 | 扫描录音、构造 frame-level 或 strict recording-level split、缓存 log-mel 特征、训练 ShuffleFAC 并保存完整指标。 |
| `run_graphhead.py` | Stage-2 冻结编码器聚合头 | 读取 Stage-1 checkpoint，冻结 ShuffleFAC CNN，把 clip cache 组织为 recording bag，训练 attention/graph 聚合头。 |
| `train_eval_signal_noise_decoupled.py` | Stage-2 Signal-Noise / ETA 实验 | 冻结 ShuffleFAC，把 embedding 分解为 signal/noise 子空间，在 noise 空间建图，在 signal 空间执行 attention 或 Top-K sparse pooling。 |
| `eval_graphaware_artifacts.py` | Graph-aware head 评估脚本 | 只做评估：加载已有 graph-aware head checkpoint，重新生成 metrics、预测 CSV、混淆矩阵和复杂度统计。 |
| `run_frontend_robustness.py` | Cross-front-end robustness 新入口 | 将 strict recording-level clip cache 编码为 `[R,S,D]` embedding cache，并统一训练 mean/attention/SN/BiGRU/MIL heads。 |
| `utils/data_preprocessing.py` | 原始音频 Dataset 辅助 | 扫描类别目录、加载音频、转 log-mel tensor。 |
| `utils/utils.py` | 模型统计辅助 | 根据特征配置构造 dummy input，统计参数量与 MACs。 |

## 主要执行流

1. Stage-1 ShuffleFAC 训练从 `run_deepship.py` 开始。
2. `run_deepship.py` 保存 split metadata，并生成可复用的 log-mel feature cache。
3. Stage-2 脚本复用这些 cache，保证下游 head 都在同一 recording-level split 上评估。
4. `run_graphhead.py` 训练标准 attention、graph、graph-aware attention 聚合头。
5. `train_eval_signal_noise_decoupled.py` 训练 Signal-Noise / ETA head，并保存 checkpoint、per-seed metrics、混淆矩阵、预测文件和复杂度统计。

## 兼容性要点

- 下游 Stage-2 脚本默认冻结 ShuffleFAC encoder，只训练聚合头。
- 需要 recording-level 评估的脚本会严格校验 cache 协议，避免误用 frame-level cache。
- `train_eval_signal_noise_decoupled.py` 的新增消融参数保持向后兼容：不传新参数时仍走原始 full soft attention。
- `threshold_similarity` 会产生 variable-degree graph；代码先记录真实平均度数，再为了 `gather()` 对邻居索引做 padding。
- `signal_top_k=0` 表示 full soft attention；`signal_top_k>0` 才启用 Top-K sparse signal evidence pooling，并受 warmup epoch 控制。

## Cross-Front-End Robustness

- 新脚本：`run_frontend_robustness.py`。
- 支持 front-end：`shufflefac`, `resnet18`, `mobilenet_v2`, `panns_cnn14`。
- 支持 heads：`mean`, `attention`, `sn_decoupled`, `bigru`, `mil_linear_softmax`。
- 所有 heads 只读取 `[B,S,D]` 的 recording-level embedding bag，不直接读取原始 waveform。
- 输出：每个 head 目录下的 `metrics.json/csv`、`epoch_metrics.csv`，以及顶层 `summary.json/csv`。

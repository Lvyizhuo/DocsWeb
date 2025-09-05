### 项目指南：动态影响力最大化（精简版）

本指南汇总项目背景、当前核心框架、主要不足，以及面向动态影响力最大化的可行优化清单与路线图，便于快速理解与实施。

### 背景与目标

- **目标**：在演化网络中选择规模为 \(k\) 的种子集合，最大化独立级联（IC）等扩散过程的期望覆盖。
- **当前核心**：
  - 快照：自适应时间窗口聚合事件形成动态图快照（支持时间衰减边权）；
  - 标签：基于动态IFC分数生成"是否为种子"的二分类标签；
  - 模型：GraphSAGE（结构）+ 增强BiLSTM（时间注意力）预测候选种子；
  - 选种：支持greedy/CELF/CELF++算法完成影响力最大化。

### 现有框架（概览）

- 数据处理：自适应窗口聚合 → 时间衰减边权 → 动态IFC标签（9个特征，含4个动态特征）；
- 模型与训练：`models/gnn_bilstm_model.py`（GNN+增强BiLSTM+时间注意力），支持排序损失训练；
- 推理与选种：候选打分 → 多算法选取 \(k\) 个节点；
- 主要脚本：`main.py`（训练/IM）、`trainer/model_trainer.py`、`data/`（快照与标签）。

### 主要不足

- 动态IFC启发式标签噪声较大，偏离真实边际增益；
- 仅二分类，未直接优化排序/覆盖；
- 推理端贪心计算成本高，未利用学习信号加速。

### 可行优化清单

- ~~**时间与快照构建**✅~~
  - ~~自适应/重叠窗口（按事件密度调整；stride < window）；~~
  - ~~时间衰减边权（\(w=\exp(-(t_{now}-t)/\tau)\)）与加权结构特征；~~
  - ~~时间结构特征：时间模体、活动节律、演化社区特征。~~
- ~~**标签与监督** ✅~~
  - ~~动态IFC：跨快照平滑 + 新邻居率 + IFC波动特征；~~
  - ~~排序学习：pairwise/listwise 损失，优化 NDCG@k/Recall@k；~~
  - 稀疏正类：Focal Loss/PU 学习；近似 Shapley 构造高置信样本。

- **模型结构** ✅
  - ~~时间注意力：多头自注意力 + 位置编码 + 特征融合；~~
  - ~~增强BiLSTM：集成注意力机制，适配9个特征（含4个动态特征）；~~
  - 多任务：二分类 + 边际增益回归 + 社区覆盖，动态损失加权；
  - 自监督预训练：Node2Vec/GraphSAGE 预训练与时间对比。

- **目标与正则** ✅
  - ~~排序损失（pairwise logistic，top-k正负对）+ 分类损失；~~
  - 多样性正则：社区覆盖、冗余惩罚；
  - 置信度校准：温度缩放/深度集成/MC Dropout，并联动仿真触发阈值。

- **影响力最大化算法层** ✅
  - ~~贪心加速：CELF/CELF++ 懒惰评估、并行与增量缓存；~~
  - ~~边权传播：支持边权缩放传播概率；~~
  - ~~两阶段：模型筛选 top-p% → 在候选上运行贪心/IMM；~~
  - 增量 IM：跨时间窗复用候选与边际增益缓存。

- **在线/自适应**
  - 经验回放与小步微调，漂移检测触发再训练；
  - Bandit/RL 逐步选种，以近似覆盖为奖励，结合模型打分先验。

- **工程与可扩展性**
  - 仿真/采样并行与分布式，必要时 C++/CUDA 加速；
  - 训练推理优化：邻居采样、AMP/bfloat16、预计算与缓存；
  - 统一配置与复现：Hydra/OMEGACONF，固定随机种与完善日志。

- **评估协议**
  - 滚动评测（rolling-origin，训练区间 [t−W, t) → t 时刻评测）；
  - 指标：spread@k、NDCG@k、regret@k、运行时、加速比、校准误差。

### 路线图（里程碑）

- 阶段一（1–2 周）✅
  - 自适应/重叠窗口 + 时间衰减边；
  - 排序损失 + CELF++ 贪心；
  - 时间注意力 + 动态特征；
  - 规范评估与日志。

- 阶段二（2–3 周）
  - 集成 TGN/TGAT；
  - 两阶段选种（筛选 → 贪心/IMM）；
  - 蒸馏教师软标签（少量高质量仿真）。

- 阶段三（2–3 周）
  - RR 集（IMM/TIM+）与偏置采样；
  - 多任务（二分类 + 边际增益回归）；
  - 不确定性驱动主动仿真。

- 阶段四（2 周）
  - 在线增量与漂移应对；
  - 工程化加速（分布式 RR、AMP、缓存）；
  - 完成消融与报告。

### 关键目录（简版）

- `main.py`：入口（训练/影响力最大化）；
- `models/gnn_bilstm_model.py`：GNN+增强BiLSTM+时间注意力模型；
- `trainer/model_trainer.py`：训练与评估（支持排序损失）；
- `data/`：动态图生成、快照与标签（9个特征）；
- `results/`：结果与可视化。

### 使用示例

- 快照生成（自适应+时间衰减）
```bash
python data/snapshots_create-tw.py --data ./data/CollegeMsg.txt --adaptive --target_events 500 --min_days 1 --max_days 30 --decay_tau_days 7 --aggregation_mode weighted
```

- 训练（时间注意力+排序损失+优化Adam）
```bash
python main.py --mode train \
  --data_dir data/processed_data/CollegeMsg_snapshots_lables \
  --batch_size 4 --temporal_window 4 \
  --use_temporal_attention --num_attention_heads 4 \
  --ranking_loss_weight 0.5 --ranking_topk 100 \
  --optimizer adamw --learning_rate 0.001 \
  --use_lr_scheduler --scheduler_patience 10 \
  --num_epochs 300 --patience 30
```

- 影响力最大化（CELF++ + 边权传播）
```bash
python main.py --mode influence_max \
  --k 5 --data_dir data/processed_data/CollegeMsg_snapshots_lables \
  --model_path results/.../best_model.pth --temporal_window 4 \
  --im_algo celfpp --use_edge_weight \
  --propagation_prob 0.5 --monte_carlo_sims 100
```

- 传统影响力最大化实验
```bash
python results/traditional_im_experiment.py \
  --data_dir data/processed_data/CollegeMsg_snapshots_lables \
  --algorithm celfpp --k 5 \
  --propagation_prob 0.5 --monte_carlo_sims 100 \
  --use_edge_weight --output_dir results
```

### 新增功能亮点

- **增强优化器**: 支持Adam/AdamW/SGD，可调参数，学习率调度
- **传统IM实验**: 独立脚本，支持多算法对比，完整结果保存
- **训练可视化**: 6图布局，包含学习率曲线和训练总结
- **实验脚本**: `run_experiments.sh` 提供完整实验流程示例
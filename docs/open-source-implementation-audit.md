# 开源实现审计：从不确定度到动作原语

审计日期：2026-08-04

审计对象：CNABU / Dengler uncertainty-informed action selection、Act-Sense-Act、`pomdp_py`、OpenVLA、Octo

本项目对应版本：Interaction-Uncertainty v0.2

本文件回答一个工程问题：现有开源实现中，哪些模块真正把不确定度、候选动作、动作后果和机器人动作连接起来；哪些只是论文主张、未发布部分或不适用于本项目；v0.2 应当在哪些接口上借鉴、适配或进行 clean-room 独立实现。

这是一份工程与研究归属记录，不是法律意见。更完整的依赖级许可证总表见 [`code-reuse-license-audit.md`](code-reuse-license-audit.md)。

## 1. 审计口径

本文严格区分四类陈述：

- **代码事实**：在下表固定 commit 的公开源码中可定位到的类、函数、默认参数或调用路径；
- **上游声明**：论文或官方 README 声称的能力，但未必存在完整公开部署路径；
- **审计判断**：根据源码调用关系得出的限制或可复现性判断；
- **项目决定**：Interaction-Uncertainty v0.2 的接口与复用策略，不代表上游实现。

上游源码路径仅作为核查定位器，不表示本项目复制或 vendor 了这些文件。除本地研究 clone 外，`pomdp_py`、OpenVLA 和 Octo 的核查基于下列固定 commit、官方文档和官方源码；本项目当前没有把这些大型仓库加入核心依赖。

### 1.1 固定版本

| 项目 | 官方仓库 | 审计 commit（默认分支 `main`） | 审计材料 |
|---|---|---:|---|
| CNABU / manipulation-enhanced mapping | [NilsDengler/manipulation_enhanced_map_prediction](https://github.com/NilsDengler/manipulation_enhanced_map_prediction) | [`8ffa53f0c32e31f81ac9c1a75612b2e5a9443456`](https://github.com/NilsDengler/manipulation_enhanced_map_prediction/commit/8ffa53f0c32e31f81ac9c1a75612b2e5a9443456) | 本地研究 clone + 官方 README/论文 |
| Act-Sense-Act | [Joringell/ASA](https://github.com/Joringell/ASA) | [`786d59cd84af08470eafc68445dddd32af7153fa`](https://github.com/Joringell/ASA/commit/786d59cd84af08470eafc68445dddd32af7153fa) | 本地研究 clone + 官方 README/论文 |
| `pomdp_py` | [h2r/pomdp-py](https://github.com/h2r/pomdp-py) | [`bd0e4392247aebfe9a95b449275237dcc25e7737`](https://github.com/h2r/pomdp-py/commit/bd0e4392247aebfe9a95b449275237dcc25e7737) | 官方源码与 API 文档 |
| OpenVLA | [openvla/openvla](https://github.com/openvla/openvla) | [`c8f03f48af692657d3060c19588038c7220e9af9`](https://github.com/openvla/openvla/commit/c8f03f48af692657d3060c19588038c7220e9af9) | 官方源码与 README |
| Octo | [octo-models/octo](https://github.com/octo-models/octo) | [`241fb3514b7c40957a86d869fecb7c7fc353f540`](https://github.com/octo-models/octo/commit/241fb3514b7c40957a86d869fecb7c7fc353f540) | 官方源码与 README |

对应论文：

- [Map Space Belief Prediction for Manipulation-Enhanced Mapping](https://arxiv.org/abs/2502.20606)；
- [Efficient Manipulation-Enhanced Semantic Mapping With Uncertainty-Informed Action Selection](https://arxiv.org/abs/2506.02286)；
- [Act, Sense, Act](https://arxiv.org/abs/2602.04600)；
- [OpenVLA](https://arxiv.org/abs/2406.09246)；
- [Octo](https://arxiv.org/abs/2405.12213)。

## 2. 总结：公开实现没有现成的完整桥

我们需要的完整路径是：

```text
final-goal prompt + public observation
    -> prompt-conditioned belief / uncertainty
    -> localized information need
    -> multiple grounded primitive candidates
    -> action-conditioned transition and observation forecast
    -> counterfactual posterior and task-risk comparison
    -> selected primitive
    -> VLA / skill backend realizes continuous actions
    -> actual re-observation updates the real belief
```

五个上游项目分别覆盖其中一部分：

| 项目 | 不确定度/信念 | 原语候选 | 动作后果模型 | 候选比较 | 连续动作实现 | 对本项目的主要价值 |
|---|---|---|---|---|---|---|
| CNABU / Dengler | Beta/Dirichlet map belief | push；论文另含 NBV | push 后地图预测 | push+view 与 view+view 的信息增益 | push motion planning | 最接近的 `proposal -> effect -> value` 原型 |
| Act-Sense-Act | 监督式任务完成概率，不是显式环境不确定度 | 无离散多原语池 | 无显式 `T/Z` 或 post-belief | 无 open/remove/rotate/direct/stop 比较 | joint view/manipulation flow action chunks | temporal cognition 与 action backend 参考 |
| `pomdp_py` | 通用 belief 容器 | 用户提供 enumerable actions / options | 用户实现 transition/observation model | POUCT/POMCP 等 | 无机器人 VLA | 规划抽象与 reference baseline |
| OpenVLA | 无显式环境不确定度 | 无 planner-level 原语池 | 无 | 无 | autoregressive action-token 解码 | 选定原语后的可选 action backend |
| Octo | 无显式环境不确定度 | 无 planner-level 原语池 | 无 | 无 | diffusion/discrete/continuous action heads | 历史、多相机、chunked backend 接口参考 |

因此，v0.2 的原创工程边界不是“再计算一个 uncertainty scalar”，而是显式拥有 `proposal -> effect -> value -> routing`。VLA 位于最终连续动作实现层；仅把 uncertainty map 写进自然语言 prompt，不等于打通上述桥。

## 3. CNABU 与 Dengler uncertainty-informed action selection

### 3.1 公开代码中实际存在的模块

本节定位均相对于 commit `8ffa53f...`。

| 上游路径 | 实际功能 | v0.2 对应概念 |
|---|---|---|
| `shelf_gym/utils/learning_utils/data_preprocessing.py` | 初始化全 1 的 Beta/Dirichlet prior；把 occupied/free 和 semantic evidence 写入 map | `EvidencePacket`、`EvidentialBeliefFilter` 的方法学参考 |
| `shelf_gym/utils/models/UNet.py` | 通过 `ReLU(output) + 1 + epsilon` 产生正 evidence/concentration | evidence 参数化参考 |
| `shelf_gym/utils/learning_utils/losses.py` | evidential classification/regression loss 与到 uniform prior 的 KL；push 预测包含 focused variants | 未来 evidence/effect critic 的训练参考 |
| `shelf_gym/scripts/run_cnabu_pipeline.py` | 组装 belief、view IG、push candidates、post-push prediction 和第一步动作选择 | v0.2 controller/critic/router 的上游原型 |
| `shelf_gym/utils/uncertainty_informed_push_utils.py` | uncertain region、ray/corridor、occluder、push direction 与边界采样 | `InformationNeed -> PrimitiveCall` 的几何 baseline 参考 |
| `shelf_gym/utils/information_gain_utils.py` | 对候选相机计算 map information gain | `CounterfactualRollout` 的信息价值参照 |
| `shelf_gym/utils/map_calibration_utils.py` | mECE、Brier、reliability 等评估 | calibration metrics 参考；不是一个训练出的 calibrator |
| `shelf_gym/utils/scaling_utils.py` | 固定 power scaling 后重新归一化语义概率 | 后处理参考；不能声称为 learned calibration |

### 3.2 Belief 与 uncertainty 的实际计算

`run_cnabu_pipeline.py::occupancy_mean_variance` 把交错通道解释为 Beta 参数，并计算：

```text
mean       = alpha / (alpha + beta)
epistemic  = alpha*beta / ((alpha+beta)^2 * (alpha+beta+1))
aleatoric  = alpha*beta / ((alpha+beta) * (alpha+beta+1))
```

语义 total uncertainty 在 `prepare_uncertainty_informed_push_sampling` 中写成 `K / sum(alpha_k)`。代码据此对阈值后的不确定区域计算距离，并选择少量相互分离的目标点。

这里有两个必须保留在复现实验记录中的 caveat：

1. `prepare_uncertainty_informed_push_sampling` 的默认参数是 `use_no_uncertainty=True`；当前 `get_possible_maps_push(... use_semantic_distance=True)` 调用没有将它关闭。因此，目标区域确实可以由语义 total uncertainty 选出，但传给后续 corridor scoring 的 `u_map` 在这条默认调用路径中被替换为 occupancy mean。不能把整个候选生成过程概括为“全程由 uncertainty map 驱动”。
2. 不同辅助函数对交错 Beta 通道的命名并不统一。v0.2 不应继承 raw channel convention，而应使用带语义名称的参数结构，并通过手算样例验证 `alpha_occupied`、`beta_free` 与均值方向。

### 3.3 uncertainty region 到 push candidate

`uncertainty_informed_push_utils.py` 的公开路径大致为：

1. `generate_points` 对阈值区域做 Euclidean distance transform，或对 weighted uncertainty 做 Dijkstra distance；最多选择 4 个、间隔至少 20 pixel 的点；
2. `find_push_corridor` 从 uncertain point 向固定 shelf-front row `74` 的所有横向位置做 inverse raycast；
3. `build_corridors` 按相邻射线上穿过的实例标签序列分组；
4. `score_ray_group` 用手工权重组合穿过的对象数、平均路径长度、空域置信与 corridor 宽度，低分优先；
5. `first_object_hit` / `object_to_push` 把选中射线上的第一项实例视为需要推动的 occluder；
6. 从 occluder 中心在 `60°–300°` 内产生 144 个方向，按 corridor 再筛选；
7. 沿物体边界采 push 起点，push 长度随机取 `10–50` pixel；
8. `ps.get_samples` 再用运动规划器筛选可执行的 joint-space path。

这是一种有效但高度场景化的 proposal heuristic：固定 shelf geometry、固定 front row、push-only、多个手工阈值。v0.2 可以 clean-room 实现一个几何 proposer baseline，但不能把这段代码直接改名后作为通用多原语生成器。

### 3.4 candidate 到 predicted effect

`run_cnabu_pipeline.py::get_possible_maps_push` 是最值得保留的方法学抓手：

1. 为每条可行 push path 计算 swept volume 与 motion parameterization；
2. 用 push-CNABU 预测该候选执行后的 occupancy Beta map、semantic Dirichlet map 和 change evidence；
3. 用预测的 change probability 在 prior map 与 predicted changed map 之间融合；
4. 得到每个 push candidate 的假想 post-action belief。

这一步建立了“动作是一个测量干预”的接口：动作价值不是由动作名称或 LLM 解释直接给出，而是先预测执行该动作后世界和观测会如何变化，再计算任务价值。

v0.2 对应的是：

- `ActionOutcomeCritic.forecast(...)`；
- `CandidateEffectForecast.transition_matrix`，表示 `T_a(s'|s)`；
- `ObservationOutcomeModel.likelihood_by_post_state`，表示 `Z_a(y|s')`；
- `rollout_forecast(...)` 依据 `T/Z` 推导 branch probability 与 posterior，而不是允许 critic 直接写 post-belief。

这是接口层面的独立概括，不表示复制 CNABU 网络、数据结构或源码。

### 3.5 effect 到动作选择

公开的 `run()` 主路径进行如下比较：

- 当前 belief 下最佳 view，再加下一最佳 view 的 cumulative information gain；
- 每个 candidate push 的 predicted post-belief 下，最佳后续 fixed-camera view 的 information gain；
- 若最佳 push path 的分数高于纯 observation path，则先执行 push；否则先 observation。

因此，CNABU 中最有价值的 bridge 不是“uncertainty 超阈值就推”，而是：

```text
candidate proposal
-> predicted post-action belief
-> common-horizon utility comparison
-> execute the first action
```

v0.2 把这个结构扩展为 prompt-conditioned task risk，并将 `DIRECT_ACT`、`STOP_NOT_FOUND` 和多种 information primitive 放进同一 `CandidateSet`。对应实现为 `rollout_forecast(...)` 和 `rank_primitives(...)`。

### 3.6 不能声称的内容与默认路径 caveat

- 官方 README 明确称仓库仍是 work in progress，论文使用的 network training/evaluation code 尚未全部更新；
- `ManipulationEnhancedMapping` 构造器与 `__main__` 都默认 `use_uncertainty_informed_sampling=False`，所以开箱 demo 的 push proposal 是随机采样，不是 uncertainty-informed sampling；
- `eval_push_igs_rl` 调用 `self.rl_model.predict` 和 `self.minimal_step`，但在审计的主类与默认主路径中没有找到相应 RL model 初始化，主 `run()` 也不调用该函数；因此不能声称该仓库完整复现了 2506.02286 中 continuous RL NBV 部分；
- 主路径 `get_igs_for_map(... use_alternative=True)` 使用 occupancy mean 上的 Bernoulli entropy；不能把它误写成端到端使用 Beta differential entropy；
- 公开完整路径仍以 global semantic mapping、push 与 active/fixed viewpoint 为目标；它没有 prompt-conditioned task loss，也没有 open/remove/rotate/pick-to-inspect 等多原语集合。

### 3.7 许可证与项目决定

审计 commit 的根目录没有 `LICENSE` 或 `COPYING`。嵌套子模块、URDF 或资产的许可证不自动授予根项目代码。

项目决定：

- 不复制、不修改、不分发其 Python 源码、checkpoint、mesh、图像或 demo dataset；
- 只根据论文公式和上述可观察接口关系进行 clean-room 独立实现；
- 文档使用“method/interface reference”或“clean-room implementation informed by the paper”，不使用“forked code”或“reused implementation”；
- 如果将来确需代码级复用，先获得作者明确许可或上游增加适用许可证，并重新审计资产和模型条款。

## 4. Act-Sense-Act

### 4.1 公开代码中的实际模块

本节定位均相对于 commit `786d59c...`。

| 上游路径 | 实际功能 | v0.2 可借鉴位置 |
|---|---|---|
| `model/vla_flow_memory.py::TaskStateHead` | label token 对 VLM hidden states 做 cross-attention，MLP 输出一个 binary logit | 未来可选 `TaskProgressEstimator`，不是 uncertainty router |
| `model/vla_flow_memory.py::MemoryBufferEncoder` | proprio projection + temporal Transformer；定义了 completion-label embedding 与交叉注意力 | 历史接口参考 |
| `model/vla_flow_arch_memory.py` | Qwen3-VL prefix、task-state loss、joint view/manipulation flow matching 与 `sample_actions` | `PolicyBackend` action-chunk adapter 参考 |
| `model/vla_flow_decoder_memory.py` | view/manip input projection、self/cross attention、memory cross-attention、AdaRMSNorm time conditioning | 连续 action decoder 参考 |
| `data/data_processor.py` | 组织历史图像、proprio、`history/complete` 与 `current/complete` 标签、未来动作监督 | 训练数据合同参考 |
| `scripts/flow_human_train_two_stage.sh` / `flow_robot_finetune.sh` | human pretraining 与 robot fine-tuning 的发布入口 | 算力/训练流程参考，不进入 v0.2 核心 |

### 4.2 实际机制

`TaskStateHead` 并不计算环境 uncertainty。它用一个 label token 作为 query，对当前 VLM token memory 做 cross-attention，随后输出一个二分类 logit。监督标签来自 `data_processor.py` 中的 `current/complete`，训练使用 focal loss。因此它估计的是数据定义下的任务完成状态，而不是：

- Beta/Dirichlet evidence strength；
- epistemic/aleatoric uncertainty；
- action-conditioned expected information gain；
- 某个 occluder 被操作后的 post-belief；
- 多个探索原语相对 direct/stop 的净价值。

连续动作部分用 flow matching 同时训练 view action stream 与 bimanual manipulation stream：

```text
x_t = t * noise + (1 - t) * action
velocity target = noise - action
```

`sample_actions` 默认做 10 个 Euler 更新步，返回 view action chunk、manipulation action chunk 和 `sigmoid(task_state_logits)`。这个结构说明 ASA 已把感知动作和操作动作放在同一个生成模型里，但不等于它显式解决了本项目的 primitive proposal/effect/value bridge。

### 4.3 代码级 caveat

- `MemoryBufferEncoder.forward` 虽然计算 `label_embedding`，也定义了 `proprio_to_label_attn` 与 `label_to_proprio_attn`，但审计 commit 的实际 forward 设置 `fused = proprio_embedding`；这两个 label/cross-attention 分支没有进入返回的 memory embedding。不能根据类成员存在就声称 completion label 已在该 encoder 中完成双向融合；
- 官方 README 明确说不发布完整 inference/deployment code，因为其依赖 proprietary robot SDK 和 hardware-specific interfaces；公开 training code/checkpoints 不能等同于完整机器人复现；
- `sample_actions` 把 `view_proprio`、`ee_proprio`、`gripper_proprio` 传给 `build_prefix`，但项目侧 `build_prefix` 没有 proprio projector，而是通过 `**kwargs` 继续传到基础 VLM。公开 helper 至少不是一个可以不加适配就部署的、明确消费 proprio 的完整路径；
- view stream 是 active viewpoint change，超出 Interaction-Uncertainty 当前实验边界；
- 没有显式 `PrimitiveCall` 候选表、动作效果 `T/Z`、风险/代价和 `DIRECT_ACT`/`STOP_NOT_FOUND` 比较。

### 4.4 许可证与项目决定

ASA 根项目是 Apache-2.0。许可证允许代码级使用和修改，但直接复用时需要保留适用的 copyright、LICENSE、NOTICE（若有）并标明修改；专利与商标条款也应遵守。Qwen、MANO、CaptainCook4D、EgoExo4D、Monte02 数据/SDK 和 checkpoint 不因根仓库 Apache-2.0 而自动获得相同授权。

项目决定：

- v0.2 核心不复制 ASA 模型代码，也不声称复现其真实机器人部署；
- 当前只在接口层把它视为可选 `PolicyBackend`，接收已经选定的 `VLAExecutionRequest.selected_primitive`，输出 `ActionChunk`；
- 如果未来使用 task-state head，应单独命名为 task progress/completion estimator，并做 calibration；其输出只可辅助 direct/stop 价值，不能冒充 prompt-conditioned environment uncertainty；
- 若未来复制 Apache-2.0 源文件，必须在提交中显式记录源文件、commit、修改和 notice，不以“参考后重写”掩盖实际复制。

## 5. `pomdp_py`

### 5.1 实际模块与机制

本节定位相对于 commit `bd0e439...`。官方 API 见 [`pomdp_py.framework`](https://h2r.github.io/pomdp-py/html/api/pomdp_py.framework.html)。

| 上游路径 | 实际功能 | v0.2 对应关系 |
|---|---|---|
| `pomdp_py/framework/basics.pyx` | `State`、`Action`、`Observation`、`TransitionModel`、`ObservationModel`、`RewardModel`、`PolicyModel`、`BlackboxModel`、`Agent`、`Environment`、`Option` | 规划器概念对齐；不作为核心父类 |
| `pomdp_py/representations/distribution/histogram.pyx` | 离散 histogram belief | 小型 exact baseline |
| `pomdp_py/representations/distribution/particles.pyx` | `Particles` / `WeightedParticles` | generative baseline |
| `pomdp_py/algorithms/po_uct.pyx` | `POUCT`、`ActionPrior` | 候选优先级与 tree-search baseline |
| `pomdp_py/algorithms/pomcp.pyx` | particle belief 下的 `POMCP` | 可选 planning baseline |
| `pomdp_py/algorithms/po_rollout.pyx` | rollout planner | one-step router 对照 |

其接口要求领域代码自行提供 `T(s'|s,a)`、`Z(o|s',a)`、reward/utility 与可用动作。`BlackboxModel.sample(state, action)` 可以直接生成 `(s', o, r)`；`Option` 用 initiation、内部 policy 和 termination 表示 temporally extended action；`ActionPrior.get_preferred_actions` 可以加入合法或启发式候选。

### 5.2 能借鉴与不能替代的部分

可借鉴：

- 把 grounded primitive 视为 hashable `Action`，或把带 stop condition 的 skill 视为 `Option`；
- 把 `CandidateEffectForecast` 封装成 generative transition/observation model；
- 用 `ActionPrior` 表示 proposer 的优先候选而不是最终价值；
- 在小型离散场景中用 POUCT/POMCP 作为长时域 planning baseline。

不能替代：

- `pomdp_py` 不会从 RGB/prompt 自动产生 `BeliefState`、`InformationNeed`、occluder 或 primitive grounding；
- 它不提供本项目的 action effect model，仍需要我们定义可采样的 `T/Z`；
- POUCT/POMCP 需要可枚举或可采样的候选与可信 generative model。若 effect critic 尚未校准，增加 tree depth 不会自动提高真实性；
- 当前 v0.2 的 one-step exact rollout 更适合作为可解释主路径；`pomdp_py` 应是 optional baseline，不是核心运行时依赖。

### 5.3 许可证与项目决定

`pomdp_py` 为 MIT。可以直接使用、修改或分发，但复制源码或 substantial portions 时必须保留 copyright 与完整许可文本。

当前项目决定是不 vendor 源码：未来通过 `.[pomdp]` extra 或独立 adapter 接入，保持以下方向映射：

```text
PrimitiveCall             -> pomdp_py.Action / Option
CandidateEffectForecast   -> TransitionModel + ObservationModel / BlackboxModel
Candidate proposer        -> PolicyModel / ActionPrior
BeliefState summary       -> Histogram / Particles adapter
```

## 6. OpenVLA

### 6.1 公开代码中的动作实现

本节定位相对于 commit `c8f03f4...`。

| 上游路径 | 实际功能 | v0.2 可借鉴位置 |
|---|---|---|
| `prismatic/extern/hf/modeling_prismatic.py::OpenVLAForActionPrediction` | HF 模型封装与 `predict_action` | `PolicyBackend.generate` 的服务端实现 |
| `prismatic/vla/action_tokenizer.py::ActionTokenizer` | 连续动作到离散 action token 的量化/反量化 | backend 内部 action representation |
| `prismatic/vla/datasets/datasets.py::RLDSBatchTransform` | image + language + action token 监督 prompt | 数据导出/微调 adapter 参考 |
| `prismatic/vla/datasets/datasets.py::RLDSDataset` | RLDS/Open-X mixture、action statistics 与 iterable loader | 未来服务器训练管线参考 |
| `vla-scripts/deploy.py` | REST 服务部署入口 | 独立进程 adapter 参考 |

`predict_action` 的实际路径是：

1. 根据 `unnorm_key` 获取目标数据集的 action dimension 与 normalization statistics；
2. autoregressive 生成相应数量的 action tokens；
3. 把 token ID 映射回预定义的均匀 bin center；
4. 使用 `q01/q99` 和可选 action mask 反归一化为连续动作。

审计 commit 中的 vanilla `RLDSDataset` 默认 `window_size=1`、`future_action_window_size=0`，加载 primary camera 与 language，不加载 proprio/depth；这描述的是该路径的默认值，不代表所有 OpenVLA 衍生实现只能单帧或不能 action chunk。

### 6.2 不能声称的内容

- vanilla `predict_action` 返回动作向量，不返回环境 Beta/Dirichlet uncertainty、localized information need 或 action-conditioned post-belief；
- action-token logits/entropy 最多是 policy-output uncertainty 的候选信号，不能未经验证地等同于 environment uncertainty；
- OpenVLA 不会替 v0.2 比较 `OPEN_CONTAINER`、`ROTATE_TO_LABEL`、`DIRECT_ACT` 和 `STOP_NOT_FOUND` 的 expected task risk；
- 把 uncertainty summary 拼入 OpenVLA prompt 是一个 prompt-only baseline，不是显式 effect/value bridge。

### 6.3 v0.2 adapter 对齐

v0.2 的 `VLAExecutionRequest` 已固定：

- exact final-goal prompt；
- 已选定的 typed `PrimitiveCall` 与 public anchor；
- belief summary；
- step budget、timeout 和 nonterminal 后必须 reobserve 的 contract。

当前核心已提供模型中立的 `RemotePolicyBackend` JSON adapter；OpenVLA 服务端只负责把请求翻译成模型 prompt/action representation，并返回 `ActionChunk`。它不得静默替换 primitive family。adapter 会校验 echo、动作维数、action space、normalization ID、backend/checkpoint SHA-256 和随机种子；服务端还必须记录：

- `unnorm_key`；
- action dimension/action space；
- normalization statistics ID；
- backend/checkpoint ID；
- selected primitive 与执行报告 echo 是否一致。

### 6.4 许可证与项目决定

OpenVLA 仓库代码为 MIT；如果复制源码，须保留 MIT copyright 与许可文本。官方 README 明确指出 pretrained models 继承底层 Llama-2 等模型的独立限制，因此 MIT code license 不覆盖权重。Open-X/RLDS 数据集也有各自条款。

项目决定：核心包不 import 或 vendor OpenVLA，不分发其权重；通过 `RemotePolicyBackend` 连接可选远程/独立进程服务。任何服务器实验都需记录 code commit、checkpoint hash、base-model license、数据来源和 normalization key。

## 7. Octo

### 7.1 公开代码中的动作实现

本节定位相对于 commit `241fb35...`。

| 上游路径 | 实际功能 | v0.2 可借鉴位置 |
|---|---|---|
| `octo/model/octo_model.py::OctoModel.create_tasks` | 从 text 或 goal image 构建 task dict 与 pad masks | `VLAExecutionRequest` 到 backend task 的转换 |
| `octo/model/octo_model.py::OctoModel.sample_actions` | 校验 observation/task shape，运行 transformer 和 action head，反归一化 action chunk | `PolicyBackend.generate` |
| `octo/model/components/action_heads.py::ActionHead` | 统一 `loss` / `predict_action` contract | backend capability interface 参考 |
| `octo/model/components/action_heads.py::DiffusionActionHead` | transformer readout 条件下的 diffusion action chunk | 可选连续动作 backend |
| 同一文件的 continuous/discrete heads | MSE/L1 与 256-bin 离散 action heads | backend ablation 参考 |
| `octo/model/octo_module.py` | observation/task tokenizer、blockwise transformer 与 readout heads | 未来独立 head 研究参考 |

Octo 的实际调用要求 observation 形状为 `(batch, window, ...)`，task 为 `(batch, ...)`，并使用 `timestep_pad_mask`。`sample_actions` 返回 `(*sample_shape, batch, action_horizon, action_dim)`；可按 mean/std 或 `p01/p99` 反归一化。官方 README 说明现有预训练模型使用 history window 2、action chunk 4，并支持 primary/wrist 等多 RGB 输入以及 language/goal-image conditioning。

`DiffusionActionHead` 先用 transformer 产生 action conditioning，再按 diffusion schedule 对 action chunk 去噪；`sample_shape` 可以产生多个动作样本。但多个 action samples 的分歧首先是 policy stochasticity/disagreement，不能自动解释为 prompt-conditioned environment uncertainty。

### 7.2 不能声称的内容

- 预训练 Octo 不直接输出本项目的 `EvidencePacket`、`InformationNeed`、`CandidateEffectForecast` 或 task-weighted expected information value；
- modular readout 结构意味着未来可以训练额外 uncertainty/value head，但这只是扩展建议，不是现有 checkpoint 的能力；
- `sample_actions` 生成的是选定 task 下的动作，不负责在本项目多原语候选池中做可审计的 Bayes-risk routing；
- Octo 的 JAX/Flax/TensorFlow/RLDS 运行栈不适合作为 v0.2 CPU core 的硬依赖。

### 7.3 v0.2 adapter 对齐与许可证

Octo adapter 应显式映射：

- current/history primary RGB 与 wrist RGB；
- `timestep_pad_mask` / missing-modality pad mask；
- selected primitive 转换后的 text task；
- embodiment action dimension；
- normalization mode 与 statistics；
- action horizon 与是否使用 receding-horizon execution。

Octo 代码为 MIT；直接复用须保留 copyright 与许可文本，checkpoint 和数据集另行审计。项目当前只保留独立服务 adapter 设计，不复制 Octo 源码或分发权重。

## 8. 上游机制到 v0.2 模块的精确映射

### 8.1 v0.2 已有 first-party contract

本仓库当前的主路径位于 `src/interaction_uncertainty/v2/`：

| v0.2 模块 | 责任 | 上游只提供了什么参考 |
|---|---|---|
| `PromptEvidenceModel` / `EvidencePacket` | 从 final-goal prompt 与 public observation 产生 evidence、sufficiency 和 localized deficits | CNABU 提供 evidential map 思想；没有 prompt-conditioned task evidence contract |
| `EvidentialBeliefFilter` / `BeliefState` | 只用实际 observation 更新真实 belief，处理重复证据与历史 digest | CNABU 提供 Beta/Dirichlet map belief 参考 |
| `BayesRiskNeedExtractor` / `InformationNeed` | 把 task uncertainty 转成不含动作名的证据需求 | 五个项目均无同构接口；这是 v0.2 的显式桥接层 |
| `PrimitiveProposer` / `NeedDrivenPrimitiveProposer` | 基于 needs 与 public anchors 产生 typed `CandidateSet` | Dengler 提供 push-only corridor heuristic；`pomdp_py.ActionPrior` 提供 proposal 概念 |
| `ActionOutcomeCritic` | 为每个物理候选（包括有 public `task_target` anchor 的 `DIRECT_ACT`）预测 `T/Z`、风险、代价与 sufficiency outcomes；只有 `STOP_NOT_FOUND` 无 forecast | CNABU 的 post-push belief predictor 是最接近原型 |
| `CandidateEffectForecast` / `rollout_forecast` | 由共同 `T/Z` 推导 predictive belief、branch posterior 和信息价值 | CNABU 进行 post-map IG；`pomdp_py` 提供 transition/observation 抽象 |
| `rank_primitives` / `PrimitiveDecision` | 将 direct、stop 与 information candidates 放在同一 task-risk objective 下排序；显式分解 decision commitment、physical progress 与 conditional information value | CNABU 比较 push/view；本项目扩展到 prompt loss、多原语、direct/not-found |
| `VLAExecutionRequest` / `PolicyBackend` / `ActionChunk` | 已选原语的连续实现和严格 echo/reobserve contract | ASA、OpenVLA、Octo 提供 action generation 参考 |
| `EpisodeController` | actual observation 与 predicted branch 分离；非终止动作后必须重新观测 | 通用 receding-horizon/POMDP 原则 |

“first-party contract”仅表示这些接口存在于当前 Interaction-Uncertainty 源树；本审计不以此主张任何上游代码被复制，也不对历史开发来源作超出仓库记录的推断。

### 8.2 推荐的可选 adapter 边界

```text
core (NumPy, typed contracts)
├── clean-room geometric proposer baseline
├── clean-room action-outcome baseline / learned remote critic
├── exact one-step rollout and risk router
├── optional PomdpPyAdapter
├── optional ASA task-progress/action service
├── optional OpenVLA PolicyBackend
└── optional Octo PolicyBackend
```

`PolicyBackend` 的能力描述应至少包含：

```text
supports_history
supports_multi_camera
supports_goal_image
action_horizon
action_dimension
action_space
normalization_modes
checkpoint_id
```

这些能力元数据不参与伪造环境 uncertainty；它们用于验证 selected primitive 能否被目标 embodiment 正确实现。

## 9. v0.2 的复用/clean-room 决策矩阵

| 上游 | 根许可证 | 允许的当前用途 | 当前禁止/不采用 | 若未来改变决定 |
|---|---|---|---|---|
| CNABU / Dengler repo | 根目录无许可证 | 阅读论文/源码行为；clean-room 实现数学与接口；引用论文 | 复制、修改、分发源码/checkpoint/asset；声称 fork/复现完整 RL path | 取得明确许可证/书面许可，逐文件重审 |
| ASA | Apache-2.0 | optional adapter；按条款直接复用小模块也法律上可行 | 当前不复制进 core；不声称完整部署复现 | 记录 source commit/file、LICENSE/NOTICE、修改与外部模型/数据条款 |
| `pomdp_py` | MIT | optional dependency / planning baseline | 不 vendor；不把它描述为 domain effect model | 若复制源码，保留 copyright 与 MIT 文本 |
| OpenVLA | code MIT；weights 另有 Llama 条款 | 独立服务 adapter | 不 vendor core；不分发 weights；不把 action entropy 当 environment U | 记录代码与权重两套许可证、checkpoint/action stats |
| Octo | MIT；weights/data 另审 | 独立服务 adapter | 不 vendor core；不把 action samples 当环境后验 | 记录 source/weight/data provenance 和 normalization stats |

## 10. 研究与实现中必须避免的过度陈述

以下表述在当前证据下不应出现：

- “我们复用了 CNABU/Dengler 的代码。”当前决定是 clean-room 方法参考；根项目无许可证；
- “CNABU 官方仓库已经完整复现 uncertainty-informed RL NBV + push。”审计 commit 的默认主路径不支持这一表述；
- “ASA 根据 uncertainty 判断何时探索。”公开实现是监督式 task completion head 与 joint action generation；
- “ASA 的 memory encoder 融合了 completion labels。”审计 commit 的实际 forward 返回 proprio-only fused path；
- “OpenVLA/Octo 自带环境 uncertainty 或信息增益规划。”它们是通用 action policies，不提供本项目所需的显式 effect/value bridge；
- “多个 VLA action samples 的分歧就是空间/语义 uncertainty。”除非有单独 calibration 与因果验证，这最多是 policy-output disagreement；
- “`pomdp_py` 自动解决了我们的 POMDP。”它只提供框架与 planner，领域 state/action/transition/observation/reward 仍由我们定义；
- “privileged simulator semantic ID 只用于 grounding，所以可以进 policy。”不可以；v0.2 firewall 只允许 privileged state 出现在 offline target/evaluator 管线。

更准确的主张是：

> v0.2 independently defines a prompt-conditioned, multi-primitive proposal–effect–value bridge. It is methodologically informed by evidential belief prediction and action-conditioned information gain, while VLA systems are connected only as optional continuous-action backends.

## 11. 发布前复核清单

- [ ] 每个第三方 adapter 固定 repository、commit/tag、checkpoint hash 和访问日期；
- [ ] 区分 code license、weight license、dataset license 与 asset/mesh license；
- [ ] CNABU/Dengler 相关实现没有复制无许可证源码、权重、图像或资产；
- [ ] 论文只声称我们实际实现和评测的路径，不把 README/论文中的未发布组件算作已复现；
- [ ] `TaskSpec`、`EvidencePacket`、`CandidateSet`、`CandidateEffectForecast` 与 execution trace 中不存在 evaluator-private ID/pose/goal location；
- [ ] direct、stop-not-found 与所有 nonterminal primitives 使用同一 candidate table 和 objective；`DIRECT_ACT` 必须有 public `task_target` anchor 与 forecast，只有 stop-not-found 无 forecast；
- [ ] effect critic 输出归一化的 `T/Z`，失败是显式 branch，预测 branch 不写入真实 belief；
- [ ] OpenVLA/Octo adapter 验证 action dimension、normalization statistics、primitive echo 与 re-observation contract；
- [ ] 如果复制 Apache-2.0/MIT 源文件，提交中保留并记录所需 notices；如果只是接口参考，不用模糊措辞暗示复制；
- [ ] 正式 release 前重新检查所有上游仓库许可证是否变化。

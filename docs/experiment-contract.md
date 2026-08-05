# 实验合同

本合同用于约束 BenchV0、toy experiment、主实验和消融。其目的不是提前规定某个方法一定获胜，而是确保“prompt-conditioned uncertainty 是否改善动作选择”可以被独立、可复现地检验。

## 1. 研究问题

主问题：

> 在固定相机、不向 policy 暴露 simulator privileged semantics 的条件下，显式的 prompt-conditioned task belief 和 action-conditioned posterior decision value，是否能帮助机器人在直接执行、停止以及多类 interactive manipulation / information-enrichment primitives 之间做出更可靠、更高效的选择？

子问题：

1. uncertainty 是否随真实信息改善而降低，并且经过校准？
2. 同一观测下，改变最终目标 prompt 是否会改变 task belief 和动作价值，而不是改变无关物理预测？
3. effect model 能否预测不同候选动作的 (T_a(s'\mid s))、(Z_a(y\mid s')) 与 realized Bayes-risk reduction？
4. 明确比较候选动作是否优于 uncertainty threshold + prompt heuristic？
5. `DIRECT_ACT` 与 `STOP_NOT_FOUND` 是否能抑制过度探索和无效重复动作？
6. 改善来自 belief、candidate proposal、effect prediction、router，还是低层执行器？

## 2. 范围

### 2.1 允许的动作

```text
DIRECT_ACT
OPEN_CONTAINER
PULL_DRAWER
UNCOVER
CLEAR_OCCLUDER
PUSH_ASIDE
BRING_CLOSER
ROTATE_TO_LABEL
PICK_AND_INSPECT
STOP_NOT_FOUND
```

### 2.2 禁止的动作

- 主动移动第三人称或腕部相机来获得更好视角；
- NBV、导航、walk-around、teleport、oracle look-at；
- simulator 直接设置物体位姿来模拟 policy action；
- 通过读取容器内部 object list、semantic ID 或 target pose 决定动作；
- 利用 BDDL/asset filename 解码目标类别。

相机可以记录操作前后画面，但相机运动不能成为被奖励的信息原语。`BRING_CLOSER` 和 `PICK_AND_INSPECT` 移动的是物体，而不是以主动视角规划为目标移动相机。

## 3. 场景覆盖

测试集至少覆盖以下互补条件：

| 场景族 | 正确行为类别 | 关键失败模式 |
|---|---|---|
| 目标清晰可见 | `DIRECT_ACT` | 过度探索 |
| 关闭冰箱/橱柜 | `OPEN_CONTAINER` | 直接抓取、oracle 知道内部物体 |
| 关闭抽屉 | `PULL_DRAWER` | 把 drawer joint state 泄漏给 policy |
| 碗/盆覆盖目标 | `UNCOVER` | 把目标 mask 泄漏给 proposer |
| 前景物体局部遮挡 | `CLEAR_OCCLUDER` 或受约束的 `PUSH_ASIDE` | 只按 global clutter 推物、无关物体位移过大 |
| 标签朝向错误 | `ROTATE_TO_LABEL` | 选择对象正确但标签仍不可见 |
| 标签分辨率不足 | `BRING_CLOSER` 或 `PICK_AND_INSPECT` | digital crop 代替物理 enrichment |
| 完成有界搜索仍无目标 | `STOP_NOT_FOUND` | 无限探索或虚假抓取 |

不要求把同一个任务人为划分多个 difficulty label；应通过对象、布局、遮挡、纹理和随机种子获得自然变化，并报告每个场景族结果。

## 4. Policy 与 evaluator 数据边界

### 4.1 Policy 可见

- RGB / wrist RGB；
- 经部署可用方法得到的深度或 proprioception，如果该 baseline 明确使用；
- 当前最终目标 prompt；
- action/observation history；
- deployable detector/VLA 产生的视觉 anchors、公开 region 与 affordance；
- policy 自己维护的 belief、uncertainty 和 memory；
- 由公开输入预测的 feasibility、cost、risk 和 effects。

### 4.2 Evaluator 私有

- simulator semantic/instance IDs；
- ground-truth target presence、identity、pose、container membership；
- semantic/instance segmentation；
- MuJoCo body/geom/joint IDs、完整 qpos/qvel；
- oracle action 和所有候选动作的真实 outcome；
- hidden paired post-action frame；
- success predicate 和 collision/drop/disturbance ground truth。

Evaluator-private 数据只允许：

- 构造 episode；
- 验证原语执行结果；
- 计算 labels、指标和 oracle upper bound；
- 生成离线训练 target。

它不能作为 policy observation、candidate feature、effect-model input 或 prompt 内容。

### 4.3 强制泄漏检查

每个运行必须：

1. 对 observation、belief、candidate、effect 和 VLA packet 调用 `PolicyFirewall`；
2. 保存 policy-facing trace；
3. 保存独立 evaluator trace；
4. 在测试中注入一个禁止字段并确认 firewall 失败；
5. 检查 image ref、anchor token、provenance 和字符串是否编码 asset/target ID；
6. 人工审计每类场景至少一个 trace。

## 5. Prompt 合同

Prompt 只描述最终目标。例如：

```text
把可乐放到桌面指定区域。
```

不能写：

```text
先打开冰箱寻找可乐。
如果看不清就把盒子拿近。
移开遮挡物后再抓目标。
```

每个场景应通过 `ensure_final_goal_prompt` 的词法检查，并额外人工检查语义泄漏。

必须加入 prompt-conditioned 对照：保持 RGB、proprioception 和 history 完全相同，只替换合法最终目标 prompt。期望：

- world-visible anchors 不应凭空变化；
- prompt-independent physical feasibility 不应因 prompt 任意变化；
- task hypotheses/relevance、loss matrix、information value 和最终动作可以变化；
- 无关 prompt 下，原目标相关探索动作的净价值应下降。

## 6. Paired before/after effect 数据合同

BenchV0 可以为同一初始状态提供动作前和动作后画面，例如：

```text
fridge_closed -> fridge_open
label_away -> label_facing_camera
object_far -> object_held_close
occluder_present -> occluder_removed
```

使用规则：

1. 动作前评分时，policy 只能读取 before observation；
2. after observation 可以作为 effect fixture、训练 target 或 evaluator truth；
3. test episode 的 after observation 不能出现在训练/提示样例或 candidate generation 中；
4. 模型预测和真实 after-frame 重推断的 belief 必须分别记录；
5. paired state 必须由同一合法原语或可复现控制器产生；
6. 若使用直接 state reset 生成图像，必须标记为 counterfactual fixture，不能报告为 policy execution success；
7. 多结果动作必须保留 outcome branches 或用重复 rollout 估计 outcome distribution，不能只保存成功结果。

## 7. 数据划分

最低要求：

- 按 episode/scene seed 分割，相关连续帧不能跨 train/test；
- 对象实例或包装纹理至少有一个 held-out split；
- 容器、遮挡布局和目标位置至少有组合泛化 split；
- prompt paraphrase 不应跨 split 复制到同一物理 episode；
- effect-model test action outcomes 不得用于拟合 scorer weight；
- 公开列出每个 split 的 episode IDs、scenario family、seed 和 asset license/provenance。

若数据量不足，应明确把结果称为 toy validation，不使用“generalization benchmark”措辞。

## 8. 一次 episode 的强制记录

每一步至少记录：

```text
schema_version
episode_id / scenario_family / seed / step_index
final_goal_prompt
public observation references and hashes
task belief and uncertainty report
candidate set and proposer provenance
effect predictions and effect-model provenance
Bayes-consistency residual
full action-value decomposition and ranking
selected primitive and JSON VLA packet
executor status
actual next public observation
realized task belief
evaluator-private success/failure metrics in a separate file
```

不能只保存最终成功率；否则无法判断失败来自 belief、proposal、effect、selection 还是 execution。

`full action-value decomposition` 至少包含 `current_bayes_risk`、`current_decision_risk`、`decision_commitment_penalty`、`physical_progress_value`、`conditional_information_value` 和 `total_task_risk_reduction`。对 `DIRECT_ACT`，physical progress 是固定 direct-decision row 下的物理转移价值，不得记为纯 information gain。对无 forecast 的 `STOP_NOT_FOUND`，`CandidateValue` 仍必须记录固定 stop decision 的 commitment penalty。

## 9. 主指标

### 9.1 端到端

- task success rate；
- correct terminal decision rate；
- mean interaction actions before termination；
- excessive-exploration rate：初始信息已充分仍执行探索动作；
- premature-commit rate：信息不足时直接执行；
- false-not-found 与 missed-stop rate；
- collision、drop、irreversible failure、non-target displacement；
- wall-clock latency，分别报告 perception、proposal、effect、routing 和 execution。

### 9.2 Belief 与校准

- hypothesis NLL；
- Brier score；
- ECE 或 adaptive ECE，并公开 binning；
- information sufficiency Brier/ECE；
- risk–coverage / AURC；
- OOD/遮挡/分辨率条件下的 calibration shift；
- predictive entropy、MI、vacuity、dissonance 与真实错误/信息充分度的相关性。

这些 uncertainty scalar 不能只用“远图高、近图低”的单例证明有效；必须与 held-out correctness 或 decision loss 对齐。

### 9.3 Candidate proposer

- beneficial primitive recall@K；
- oracle-valid primitive recall@K；
- invalid/ungrounded candidate rate；
- duplicate candidate rate；
- terminal candidate coverage：`STOP_NOT_FOUND` 始终存在，`DIRECT_ACT` 只在存在当前 public `task_target` anchor 时存在；
- grounding correctness，由 evaluator-private annotation 计算。

### 9.4 Effect model

- 每个物理 candidate（包括 `DIRECT_ACT`）的 forecast coverage，以及 `STOP_NOT_FOUND` 无 forecast 合同的违反率；
- predicted vs realized post-action Bayes risk MAE/RMSE；
- predicted vs realized risk-reduction calibration；
- outcome probability NLL/Brier；
- candidate pairwise ranking accuracy；
- Spearman correlation between predicted and realized candidate value；
- Bayes-consistency L1；
- feasibility Brier/ECE；
- action cost/risk/disturbance prediction error。

### 9.5 Selector

对同一初始状态实际评估全部安全候选或使用 evaluator replay，定义：

\[
\operatorname{Regret}
=L_{\mathrm{realized}}(a_{\mathrm{selected}})
-\min_{a\in\mathcal C_t}L_{\mathrm{realized}}(a).
\]

报告 mean/median regret、zero-regret rate、top-1 action accuracy 和按 scenario family 的 confusion matrix。

## 10. 必要 baselines

1. **Direct-only**：不探索，直接执行当前最优终止决策；
2. **Prompt-VLA**：只给 RGB、history 和最终目标，由 VLA/LLM 自行决定；
3. **Prompt + uncertainty text**：把 uncertainty report 写进 prompt，不显式比较 effect；
4. **Threshold gate**：当前 uncertainty 超阈值则用固定 heuristic 原语；
5. **Action-independent uncertainty**：根据当前最高 uncertainty region 选动作，不预测动作后 outcome；
6. **Global uncertainty objective**：优化全局 entropy/vacuity，而非 prompt task risk；
7. **Full bridge**：prompt-conditioned belief + candidate-conditioned effect + Bayes-risk router；
8. **Oracle-effect upper bound**：使用 evaluator-private 真实 outcomes 排序，仅作上界，必须清楚标为 oracle；
9. **Optional discrete POMDP**：在小型离散任务上使用 `pomdp_py`，用于验证一步近似和长时域 planner 的差距。

所有方法尽可能共享同一个 candidate set 和低层 executor。否则必须分别报告 proposal recall 与 execution success，避免不公平比较。

## 11. 必要消融

- 去掉 prompt conditioning；
- prompt-conditioned task risk 改成 global uncertainty；
- Bayes risk 改成 predictive entropy；
- 去掉 outcome branches，只保留平均 predictive belief；
- 去掉 action effect，只按当前 deficit–primitive mapping；
- 去掉显式 execution-failure outcome，改成只建模成功结果的错误 effect model；
- 分别去掉 cost、physical risk、non-target disturbance；
- 去掉 `DIRECT_ACT`；
- 去掉 `STOP_NOT_FOUND`；
- 每次去掉一种 primitive family；
- 在 partial-occlusion 场景比较 `PUSH_ASIDE` 与其他可行原语，并单独报告位移和扰动；
- rule-based proposer vs learned/VLA proposer；
- frozen recorded-output regression vs learned effect critic（前者只验证软件，不作为方法 baseline）；
- structured JSON adapter vs prompt-only adapter；
- Beta/Dirichlet strength 校准前后；
- 不积累相邻帧 evidence vs 直接 pseudo-count accumulation。

## 12. 软件验证、toy sanity check 与科学验证分开

### 12.1 软件合同

无 simulator、无网络、无预训练权重的 CI 必须验证：

- Beta/Dirichlet 合法域与 exact hard-subset projection；
- 同像素换 frame ID 不改变 evidence，换像素内容必须改变；
- 重复 evidence event 不重复累计；
- prompt/task/ontology swap 不能沿用旧 belief；
- candidate grounding 必须来自当前公开 anchor；
- `DIRECT_ACT` 必须有当前 public `task_target` anchor；
- 每个物理 candidate（包括 `DIRECT_ACT`）恰有一个 effect forecast，只有 `STOP_NOT_FOUND` 没有；
- 信息动作使用 `BAYES_AFTER_OBSERVATION`，`DIRECT_ACT` 使用 `FIXED_DIRECT_ACT`；
- 风险分解满足 `total = physical + information - commitment`，stop 无 rollout 但 commitment penalty 不丢失；
- 候选输入顺序不改变 ranking；
- 每个 transition row 和 observation post-state column 都归一化；
- posterior mixture 等于 transition-predictive belief；
- executor 不能偷换 candidate 或 primitive family；
- 预测 posterior 不写入真实 episode state；
- hash-linked integrity trace 的 parent chain、event identity 和 content digest 可验证，但不把它当作防篡改存储或完整 replay 证明；
- 相同 public record、不同 evaluator-private state 时 policy trace 完全一致。

解析 fake 可以用于这些测试，但必须依赖输入内容或手算公式，不能按 frame/prompt/candidate ID 查表。

### 12.2 Toy sanity check

far/near、label-back/front、fridge-closed/open 等 paired observations 只能证明数据流能处理信息改善、指标方向合理、effect/trace 合同可生成。它们不能证明模型校准、泛化或学会了正确 primitive。

### 12.3 科学验证

只有 held-out episode 上的 evidence calibration、candidate recall、effect forecast calibration、selection regret 与闭环 task efficiency 才能支撑方法主张。旧代码是 feasibility-only prototype；v0.2 是无兼容承诺的 clean rewrite，旧 YAML 中预填 posterior 与正确动作的 demo 已从生产仓库删除。

## 13. 统计报告

- 在看 test result 前冻结 split、主要指标、scorer 权重和停止规则；
- 至少报告每个 scenario family 的 episode 数与随机种子；
- 置信区间以 episode 为 cluster 做 bootstrap，不把相邻帧当独立样本；
- 同初始状态/seed 的方法比较使用 paired analysis；
- 报告 effect-model checkpoint、VLA 版本、temperature、sampling count、硬件与延迟；
- 同时报告均值、置信区间和 failure counts，不只报告最佳 seed；
- 所有 oracle upper bounds、人工修正和失败 episode 必须明确标记。

## 14. 结果解释边界

以下证据不足以单独支撑论文主张：

- 两张图上 uncertainty 一高一低；
- 一个 scripted demo 选择了预期动作；
- 使用 ground-truth after frame 得到高信息增益；
- prompt 中直接写探索建议后 VLA 成功；
- 只比较不同 VLM 输出分数，没有动作后果；
- 只报告 task success，未隔离执行器差异；
- simulator semantic ID 虽未直接传给 VLA，却用于生成候选或 uncertainty map。

完整证据链应是：belief 校准 → candidate coverage → effect prediction → action ranking/regret → 闭环 task efficiency，并在 prompt swap、无探索正例、目标不存在和 held-out objects 上成立。

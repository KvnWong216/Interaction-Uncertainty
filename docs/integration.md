# v0.2 集成指南

本指南描述真实模型、LIBERO 和 VLA/skill executor 如何接入当前技术主线。它不提供按场景名查答案的 fallback：没有 evidence model 或 action-outcome critic 时，闭环应立即 fail closed，而不是等待服务就绪或静默切换为 oracle fixture。v0.2 是相对旧 feasibility-only prototype 的 clean rewrite，不提供旧 API、配置、trace 或行为兼容承诺。

## 1. 进程边界

推荐把系统拆成三个环境：

```text
CPU bridge
  TaskSpec / filter / needs / candidates / Bayes rollout / planner / trace

GPU evidence service
  public wrist RGB + final-goal prompt -> calibrated evidence + localized deficits

GPU effect/VLA services
  candidate-conditioned T/Z forecast
  selected typed primitive -> bounded continuous action chunk
```

CPU bridge 只依赖 NumPy，要求 Python >=3.10。项目 CI 覆盖 Python 3.10–3.12，更新的 Python 版本尚未保证兼容。OpenVLA、Octo、ASA/CoMe-VLA、Torch、JAX 和 LIBERO 不进入核心依赖树，这样各自可以使用不同的 CUDA/Python 环境。

## 2. TaskSpec：prompt 到决策问题

`TaskSpec` 固定：

- 最终目标 prompt 的 SHA-256；
- task hypotheses 及顺序；
- `DIRECT_ACT`、`NOT_FOUND` 两个终止决策；
- prompt-specific loss matrix；
- Dirichlet base rate；
- 需要辨认的属性。

先验证示例：

```bash
interaction-uncertainty validate-task \
  --task examples/v2/task_orange_juice.json
```

当前 `TaskSpec` 由实验配置显式编译，而不是让 LLM 在运行时任意改变 ontology 或 loss。未来的 prompt compiler 必须输出同一 schema，并在 episode 开始前冻结。prompt 或 ontology 变化时必须启动新的 task state，不能沿用旧 posterior。

`DIRECT_ACT` 在决策类型上是终止决策，但在执行上是物理动作，因此必须有 effect forecast 和具有 `task_target` affordance 的 public anchor。`STOP_NOT_FOUND` 不执行物理动作，是唯一没有 forecast 的候选。

## 3. Public observation 与 LIBERO

### 3.1 允许输入

`LiberoPublicObservationAdapter` 默认只读取：

```python
image_keys = ("robot0_eye_in_hand_image",)
proprioception_keys = (
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
)
```

RGB 引用由 dtype、shape 与像素内容共同计算 SHA-256，不使用 BDDL task name、asset filename 或 frame ID 作为视觉内容代理。默认 `image_publisher` 将原数组无损编码成 `data:application/x-npy;sha256=...;base64,...`，所以独立 GPU service 能取得真实像素，并可用 `decode_embedded_public_image` 验证内容哈希。单张嵌入图像限制为 16 MiB。第三方 detector/tracker 可以生成临时 `VisualAnchor`，但来源必须是 deployable/public source。

```python
from interaction_uncertainty.v2.libero import LiberoPublicObservationAdapter

adapter = LiberoPublicObservationAdapter(anchor_detector=my_public_detector)
public = adapter.adapt(
    raw_observation=libero_observation,
    frame_id="episode-17-step-0",
    prompt="Place the orange juice in the wicker basket.",
)
```

生产数据若不适合内嵌，可给 adapter 传入自定义 `image_publisher(key, image, digest)`，先把图像发布到双方可访问的内容存储再返回 HTTPS/S3 URI。返回引用必须携带给定的期望 `digest`；纯 `frame_id`、场景名或不随像素变化的 URL 会被拒绝。但 URL 里包含 digest 本身不证明远程内容正确：模型服务必须重新获取实际字节，按相同 dtype/shape 合同重算 digest，并在不匹配时拒绝。默认内嵌引用的解码方式为：

```python
from interaction_uncertainty.v2.libero import decode_embedded_public_image

pixels = decode_embedded_public_image(public.image_refs[0])
```

自定义 image-ref fetch 是一个 SSRF 边界。服务端应使用 scheme/host/port allowlist，每次 redirect 后重新验证目标，拒绝云 metadata、link-local、loopback 与未授权内网地址，并限制下载大小、超时、content type 和解码资源。不应让不可信用户直接决定模型服务获取的 URI。

### 3.2 明确禁止

以下字段即使由 simulator 提供，也不得进入 policy packet、anchor、模型输入或候选参数：

- semantic/instance segmentation 与 target mask；
- object/body/geom/joint ID；
- object pose、完整 qpos/qvel；
- container membership/object list；
- ground-truth target identity/presence；
- oracle action、oracle utility；
- BDDL path 与 asset filename。

它们只能出现在物理隔离的 evaluator/training-target 表。`PolicyFirewall` 是第二道检查，不替代 allowlist adapter。

## 4. Evidence service

### 4.1 请求

`RemoteEvidenceModel` 发送：

```json
{
  "schema_version": "interaction-uncertainty.evidence-request.v2",
  "task": {"...": "TaskSpec"},
  "context": {"...": "public PolicyContext"},
  "observation_digest": "64-hex-content-digest"
}
```

服务端模型的建议结构为：

```text
public RGB / public tracks
          + final-goal prompt embedding
                    |
             shared visual encoder
                    |
        +-----------+--------------+
        |           |              |
 hypothesis head  sufficiency head  localized-deficit head
 Dirichlet e_k    Beta e+/e-        kind/anchor/probability
```

模型输出 raw logits 后，应通过非负 link（例如 softplus）得到 evidence，并在独立 calibration split 上确定 evidence strength/temperature。服务端不得输出动作标签或“最佳 primitive”。

### 4.2 响应

顶层响应：

```json
{
  "schema_version": "interaction-uncertainty.evidence-response.v2",
  "evidence": {"...": "EvidencePacket"}
}
```

`EvidencePacket` 必须回显同一 `TaskKey`、hypothesis 顺序与 observation content digest。`event_id` 必须由：

```text
TaskKey + observation_digest + ModelStamp
```

内容寻址生成。这样给相同像素更换 `frame_id` 不能伪造一份新的独立 pseudo-evidence。

默认 `EvidentialBeliefFilter(mode=REPLACE)` 用当前模型证据替换高度相关视频帧的 pseudo-evidence。`DISCOUNTED_EVIDENCE` 是显式启用的时序启发式，必须报告 retention 与 same-group discount；不能称为严格共轭 Bayes 更新。

## 5. InformationNeed 与 candidate proposal

`BayesRiskNeedExtractor` 将 prompt-relevant deficit 转成 `InformationNeed`。该对象只描述：

- 哪个命题缺证据；
- 对应哪个当前 public anchor；
- 缺的是 identity、presence、occlusion、unobserved surface、resolution、coverage 还是 access；
- 当前 deficit probability、prompt relevance、sufficiency shortfall；
- 当前任务最多还可能降低多少 Bayes risk。

它不能包含 `OPEN_CONTAINER`、`ROTATE_TO_LABEL` 等动作答案。

当前 `NeedDrivenPrimitiveProposer` 是可审计的 recall-first baseline：根据 deficit 与公开 affordance 的兼容关系产生多个 typed candidates，并始终加入 stop；只有当前 public anchors 中存在 `task_target` affordance 时才能加入 `DIRECT_ACT`。它不是论文最终 learned proposer，也不应被描述为“由 uncertainty 端到端学会了动作生成”。后续 learned/VLM proposer 应与 registry proposer 取并集，经过：

```text
schema validation
-> current-frame anchor validation
-> affordance/precondition validation
-> semantic deduplication
-> recall-oriented top-M pruning
```

最佳 candidate 由 effect/planner 决定，而不是 proposer 自己决定。

## 6. Action-outcome critic

### 6.1 请求

`RemoteActionOutcomeCritic` 发送：

```json
{
  "schema_version": "interaction-uncertainty.effect-request.v2",
  "task": {"...": "TaskSpec"},
  "context": {"...": "public PolicyContext"},
  "belief": {"...": "BeliefState"},
  "needs": [{"...": "InformationNeed"}],
  "candidates": {"...": "CandidateSet"},
  "observation_digest": "64-hex-content-digest"
}
```

服务端对每个物理 candidate 返回一个 `CandidateEffectForecast`。这包括所有信息动作和 `DIRECT_ACT`；只有 `STOP_NOT_FOUND` 不发生物理动作、不请求 forecast。forecast 预测有限 latent task-state transition：

\[
T_a(s'\mid s),
\]

以及穷尽的 post-action observation outcomes：

\[
Z_a(y\mid s').
\]

每个 post-state 列上的 outcome likelihood 必须和为 1。失败、部分执行、没有获得新证据都要成为显式 branch，不能同时再乘一个外层 feasibility 概率而重复计算失败。

### 6.2 响应

```json
{
  "schema_version": "interaction-uncertainty.effect-response.v2",
  "task_key": {"...": "same TaskKey"},
  "observation_digest": "same digest",
  "forecasts": [{"...": "CandidateEffectForecast"}]
}
```

bridge 严格检查：

- 每个物理 candidate（包括 `DIRECT_ACT`）恰有一个 forecast；
- `STOP_NOT_FOUND` 是唯一没有 forecast 的 candidate；
- `DIRECT_ACT` 必须绑定当前、未修改、具有 `task_target` affordance 的 public anchor；
- ontology、TaskKey、observation digest 完全一致；
- transition rows 与 observation columns 均归一化；
- outcome IDs 唯一；
- outcome 不能声称解决 candidate 未声明处理的 InformationNeed；
- critic model/checkpoint/calibrator、RNG seed 和自身 uncertainty 均被记录。

critic 不返回任意 post-belief。`rollout_forecast` 使用同一个 (T/Z) 合同按 Bayes rule 推导 predictive belief、branch probability 和 posterior，因而自动满足 posterior martingale。信息动作按 `BAYES_AFTER_OBSERVATION` 在每个 branch 重新选择最优终止决策；`DIRECT_ACT` 按 `FIXED_DIRECT_ACT` 始终计算已承诺的 direct loss，不能借预测观测后改变决策来获得虚假信息价值。预测 branch 只用于排序。

rollout 和 planner 的风险分解必须保留候选的固定决策承诺：

```text
current_bayes_risk = rho(current belief)
current_decision_risk = current Bayes risk                 # information action
current_decision_risk = fixed decision-row risk            # DIRECT_ACT / STOP
decision_commitment_penalty = current_decision_risk - current_bayes_risk >= 0
physical_progress_value = current_decision_risk - transition_predictive_risk
conditional_information_value = transition_predictive_risk - expected_posterior_risk
total_task_risk_reduction = physical + information - commitment
```

`DIRECT_ACT` 的 physical progress 是固定 direct row 下的物理转移价值，不是纯 information value。`STOP_NOT_FOUND` 没有 rollout，但其 `CandidateValue` 仍显式记录 stop row 的 `current_decision_risk` 和 `decision_commitment_penalty`；它的 physical/information value 为 0，total reduction 等于负 commitment penalty。

## 7. 一步规划

evidence 与 effect 两个规划服务启动后：

```bash
interaction-uncertainty plan-remote \
  --task examples/v2/task_orange_juice.json \
  --observation examples/v2/public_observation.json \
  --episode-id libero-episode-0001 \
  --evidence-endpoint http://127.0.0.1:8101/v2/evidence \
  --effect-endpoint http://127.0.0.1:8102/v2/effects \
  --output /tmp/plan.json \
  --trace /tmp/policy-trace.jsonl
```

`plan.json` 保存完整 belief、needs、candidates、forecast、rollout、value decomposition、ranking 和 `VLAExecutionRequest`。不启动真实服务时，该命令预期失败；这是有意的 fail-closed 行为。

remote endpoints 必须由受信任配置提供，不能直接当作不可信用户可控 URL。生产环境应限制 scheme/host/port、验证 TLS/服务身份、对 redirect 重新做地址策略检查，并拒绝 metadata、link-local、loopback 和未授权内网目标，防止 SSRF 与数据外发。命令行中的 `127.0.0.1` 只用于本机开发。

## 8. VLA / skill executor

### 8.1 职责边界

router 已经选择 `PrimitiveCall`。VLA/skill backend 可以：

- 根据 public RGB/proprioception 细化 grasp/contact grounding；
- 在 primitive 参数和步数预算内输出连续 action chunk；
- 根据 typed stop condition提前结束；
- 报告成功、失败、拒绝或超时。

它不能：

- 把 `OPEN_CONTAINER` 静默改成 `ROTATE_TO_LABEL`；
- 自行选择另一个 candidate；
- 使用 evaluator-private state；
- 把自身“感觉看清了”直接写入 task belief；
- 在非终止动作后跳过真实再观测。

`ExecutionReport` 必须原样回显 `execution_id`、`candidate_id` 与 primitive family。controller 对任何偷换都 fail closed。

核心包提供模型中立的 `RemotePolicyBackend`，服务合同为：

```text
interaction-uncertainty.policy-request.v2
  VLAExecutionRequest
      ↓
OpenVLA / Octo / ASA / skill service
      ↓
interaction-uncertainty.policy-response.v2
  echoed execution_id + candidate_id + primitive_kind
  ActionChunk(actions, action_space, normalization_stats_id,
              backend_id, backend_sha256, rng_seed)
```

客户端必须在部署配置中固定 action space、动作维数、最大 chunk horizon、normalization statistics ID、backend ID 和 checkpoint SHA-256；响应逐项不一致即拒绝。`STOP_NOT_FOUND` 不调用 policy service，而是产生零步终止 report；`DIRECT_ACT` 虽然是终止决策，但包含真实物理执行，因此可以调用 policy service。

### 8.2 OpenVLA

OpenVLA backend 应在独立 CUDA 服务中：

1. 将 final-goal prompt 与已选 primitive 转换为该 checkpoint 的语言表示；
2. 使用 checkpoint 对应的 `unnorm_key` 与 action normalization statistics；
3. 验证动作维数、gripper convention 和控制频率；
4. 把生成的动作限制在 `maximum_skill_steps` 与 `timeout_s`；
5. 在 `ExecutionReport` 中记录 code commit、weight hash 与 normalization key。

OpenVLA 的 action token entropy 不是环境 uncertainty，也不能替代 effect critic。

### 8.3 Octo

Octo backend 还需正确构造 observation history 与 pad mask，并固定 diffusion/continuous/discrete action head、sampling seed 与 chunk horizon。多 action samples 可作为 policy disagreement 诊断，但未经 calibration 不能当作 target identity、occlusion 或 information sufficiency posterior。

### 8.4 Act-Sense-Act

ASA/CoMe-VLA 可作为长时序执行研究的后端参考，但其公开 `TaskStateHead` 是监督式完成状态分类器，不是本项目的 calibrated environment belief 或 VoI router。其 view-action stream也不能直接加入本项目主动作集合，因为当前范围排除 active viewpoint change。

## 9. 实际再观测

非终止 primitive 的正确闭环是：

```text
selected PrimitiveCall
-> bounded VLA/skill execution
-> ExecutionReport
-> actual new wrist RGB/proprioception
-> new EvidencePacket
-> new BeliefState
```

规划时预测的“成功后很确定”branch 永远不能变成在线 posterior。即使 critic 预测开门后目标可见，而真实画面仍然模糊，下一步必须服从真实画面重推断的 belief。

## 10. Trace、重试与复现

`EpisodeController` 写入 hash-linked integrity trace：

```text
ObservationReceived
-> EvidenceProduced
-> BeliefUpdated
-> InformationNeedsExtracted
-> CandidatesProposed
-> EffectsForecast
-> RankingComputed
-> CommandIssued
-> ExecutionFinished
```

使用 Python API `verify_trace_chain(events)` 或 CLI `interaction-uncertainty verify-trace --trace TRACE.jsonl` 检查 parent link、event identity 和包含 timestamp/schema 的 content digest。该链能检测未同时重建 digest 的内容/链接变化，但不是防篡改存储：攻击者可重写整条链，未外部固定的尾部截断或整体删除也不能单靠链检出。trace 也没有封装模型权重、服务、executor 与环境状态，所以不提供完整物理 replay。外部执行使用由 episode、step、candidate 与 belief history 生成的幂等 `execution_id`，避免网络重试重复执行物理动作。

每个正式实验还应记录：repository commit、dirty flag、config hash、model/checkpoint/calibrator hash、Python/依赖版本、设备/dtype、随机种子与所有 public image content hashes。live API 结果必须保存响应；CI replay 不重新调用网络。

## 11. 训练边界

### Evidence model 数据

输入 feature：public RGB、final-goal prompt、public tracks/history。训练 target 可以来自独立 evaluator：task hypothesis、identity/presence、readability、occlusion/coverage/access 标签。privileged labels 不能拼回模型输入或 policy trace。

### Effect critic 数据

每条记录至少包含：

```text
public pre-action history
+ pre-action belief
+ grounded candidate
+ actual execution report
+ actual next public observation
+ next prompt-conditioned evidence
+ separately stored evaluator targets
```

同一个克隆初始 simulator state 应执行多个安全 candidates 和多个 seeds，才能监督 candidate-conditioned outcome distribution。只保存被当前 policy 选择的动作会产生严重 selection bias。

### Planner

v0.2 planner 是解析 Bayes-risk objective，不训练。scorer weights 必须在 validation split 冻结。未来 end-to-end 训练可以把 need/candidate/effect token 融入 VLA，但应作为独立扩展与当前可审计 bridge 对照。

## 12. 故障处理

- schema 缺字段、出现未知字段或 privileged key：拒绝；
- TaskKey、prompt、ontology、observation digest 不一致：拒绝；
- NaN/Inf/负 evidence、非 stochastic (T/Z)：拒绝；
- 任一物理 candidate（含 `DIRECT_ACT`）缺 forecast，或 `STOP_NOT_FOUND` 意外携带 forecast：拒绝；
- stale anchor 或修改过的 public anchor：拒绝；
- VLA 偷换 candidate/family：拒绝且 controller 不推进；
- 非终止动作没有实际新观测：停在等待状态；
- 远程模型超时/响应过大：失败并记录，不执行未经比较的 fallback。

这些失败行为是论文可审计性的组成部分，不应为了“demo 总能跑”而隐藏。

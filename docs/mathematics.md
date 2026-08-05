# v0.2 数学实现规范：从 prompt-conditioned uncertainty 到交互动作

本文档描述当前代码真实实现的数学合同。旧代码是 feasibility-only prototype；v0.2 是 clean rewrite，不承诺旧 API、配置、trace 或行为兼容，也不把旧版静态 belief、手写 post-belief 或“正确动作”查表当作方法。

主链是：

\[
\text{TaskSpec}
\rightarrow
\text{prompt-conditioned evidence}
\rightarrow
\text{stateful belief}
\rightarrow
\text{InformationNeed}
\rightarrow
\text{typed candidates}
\rightarrow
T/Z\text{ forecast}
\rightarrow
\text{Bayes rollout}
\rightarrow
\text{task-risk ranking}.
\]

这些单项理论都不是新提出的。可能的研究贡献只能来自它们在 fixed-wrist、无 privileged semantics、prompt-conditioned、多物理原语 benchmark 中的组合与实证结果。

## 1. TaskSpec：prompt 定义决策问题

用户只给最终目标 prompt \(q\)，例如“把橙汁放入篮子”。当前 TaskSpec 显式冻结：

\[
\mathcal T_q=(k_q,\mathcal H_q,\mathcal D_q,L_q,\mathbf a_q),
\]

其中：

- \(k_q\) 是 TaskKey，绑定 prompt digest、ontology ID 与版本；
- \(\mathcal H_q=\{h_1,\ldots,h_K\}\) 是 task hypotheses；
- \(\mathcal D_q=\{\mathrm{DIRECT\_ACT},\mathrm{NOT\_FOUND}\}\)；
- \(L_q(d,h)\ge 0\) 是终止决策损失；
- \(\mathbf a_q\) 是 Dirichlet base rate，满足 \(a_{q,k}>0\)、\(\sum_k a_{q,k}=1\)。

当前代码不包含运行时自由生成 ontology/loss 的 LLM prompt compiler；TaskSpec 由实验配置显式编译并通过 strict schema 验证。prompt 或 ontology 一旦改变，必须创建新 task/controller，不能继承旧 posterior。

prompt 只描述最终目标，不能泄漏“先开门”“先旋转”“先移开障碍物”等过程答案。

## 2. Policy-visible history 与 task belief

令 public observation/action history 为：

\[
H_t^{\mathrm{pub}}
=(O_0^{\mathrm{pub}},A_0,\ldots,A_{t-1},O_t^{\mathrm{pub}}).
\]

任务 belief 为：

\[
b_t^q(h)
=P(H_q=h\mid H_t^{\mathrm{pub}},q).
\]

这里的 \(H_q\) 是 task-relevant latent hypothesis，不是完整 simulator state。policy 不读取 semantic/instance ID、target mask、object pose、container membership、MuJoCo ID 或 oracle action。

同一 public world observation 可以在不同 prompt 下投影为不同 task belief；但 prompt 不应任意改变与任务语义无关的物理可行性。

## 3. Prompt-conditioned evidence

真实 evidence model 的合同为：

\[
E_\theta(O_t^{\mathrm{pub}},q,M_{t-1})
\mapsto
\mathcal E_t,
\]

其中 EvidencePacket 包含：

- task-hypothesis non-negative pseudo-evidence \(\mathbf e_t^q\)；
- prompt-specific sufficiency evidence \((e_{s,t}^{+},e_{s,t}^{-})\)；
- localized evidence deficits；
- public observation digest；
- model/checkpoint/calibrator stamp；
- temporal correlation group。

模型不能输出 primitive label 或 oracle action。每个 deficit 只描述缺少什么证据以及对应哪个当前 public anchor。

### 3.1 内容寻址与去重

令：

\[
d_t=\operatorname{SHA256}
(\text{public image refs, proprioception, anchors, public history}).
\]

frame ID 不参与 \(d_t\)，所以给相同像素改名不能制造新证据。evidence event ID 为：

\[
\operatorname{event\_id}
=\operatorname{SHA256}(k_q,d_t,\text{ModelStamp}).
\]

filter 拒绝重复 event ID。相同 frame ID 换了图像内容时，digest 与 evidence 必须改变。

## 4. Dirichlet hypothesis belief

令：

\[
\boldsymbol\pi_t^q
\sim\operatorname{Dir}(\boldsymbol\alpha_t^q),
\qquad
\alpha_{t,k}^q=e_{t,k}^q+W_q a_{q,k},
\qquad e_{t,k}^q\ge 0.
\]

Dirichlet strength 与 predictive mean 为：

\[
S_t^q=\sum_k\alpha_{t,k}^q,
\qquad
\bar p_{t,k}^q=\frac{\alpha_{t,k}^q}{S_t^q}.
\]

代码逐类要求：

\[
\alpha_{t,k}^q\ge W_q a_{q,k}.
\]

只检查 \(S_t^q\ge W_q\) 不足以保证 subjective-logic evidence 非负。

### 4.1 Hard subset 到 Beta 的严格投影

若闭集类别概率满足：

\[
\boldsymbol\pi\sim\operatorname{Dir}(\boldsymbol\alpha),
\]

且 prompt target 对应非空 proper subset \(\mathcal K_q\)，则：

\[
\sum_{k\in\mathcal K_q}\pi_k
\sim
\operatorname{Beta}
\left(
\sum_{k\in\mathcal K_q}\alpha_k,\,
\sum_{k\notin\mathcal K_q}\alpha_k
\right).
\]

这是严格分布性质。对 CLIP/VLM soft similarity 加权和，一般不能声称仍为 exact Beta。

## 5. Beta information sufficiency

sufficiency 是 prompt-specific proposition：

> 当前公开证据是否足以低风险地提交一个终止决策？

表示为：

\[
P_{\mathrm{suff},t}^q
\sim\operatorname{Beta}(\alpha_{s,t}^q,\beta_{s,t}^q).
\]

其 mean、variance 和 vacuity 为：

\[
\mu_{s,t}^q
=\frac{\alpha_{s,t}^q}{\alpha_{s,t}^q+\beta_{s,t}^q},
\]

\[
\operatorname{Var}(P_{\mathrm{suff},t}^q)
=
\frac{\alpha_{s,t}^q\beta_{s,t}^q}
{(\alpha_{s,t}^q+\beta_{s,t}^q)^2
(\alpha_{s,t}^q+\beta_{s,t}^q+1)},
\]

\[
u_{s,t}^q
=
\frac{W_s}{\alpha_{s,t}^q+\beta_{s,t}^q}.
\]

Beta parameter variance 不是 Bernoulli predictive variance；vacuity 也不是天然校准的真实错误概率。

## 6. 不确定度诊断量

对 predictive categorical mean \(\bar{\mathbf p}\)，Shannon entropy 为：

\[
H(\bar{\mathbf p})
=-\sum_k\bar p_k\log\bar p_k.
\]

归一化预测熵为：

\[
H_{\mathrm{norm}}
=\frac{H(\bar{\mathbf p})}{\log K}.
\]

Dirichlet second-order mutual information 为：

\[
I(Y;\boldsymbol\pi)
=H(\mathbb E[\boldsymbol\pi])
-\mathbb E_{\boldsymbol\pi}[H(Y\mid\boldsymbol\pi)].
\]

Subjective Logic vacuity \(u=W/S\) 描述在指定 evidential parameterization 下缺少多少 evidence。dissonance 描述已支持 hypotheses 之间的冲突。

这些量不能互换：

- entropy 高可能来自多个类别均有大量冲突证据；
- vacuity 高表示证据总量少；
- MI 依赖 learned second-order distribution；
- sufficiency 是 prompt-conditioned 任务命题；
- localized deficit 指出不确定性在哪里、缺什么信息。

它们都需要在 held-out correctness/decision loss 上做 NLL、Brier、ECE、risk–coverage 等校准验证。

## 7. 时序 evidence filter

### 7.1 REPLACE

默认模式直接用当前 observation 的 neural pseudo-evidence 构造 posterior：

\[
\boldsymbol\alpha_t
=\mathbf e_t+W\mathbf a.
\]

这避免把高度相关视频帧当作独立 counts。它是保守的在线表示，不是完整动态 Bayesian filter。

### 7.2 DISCOUNTED_EVIDENCE

显式启用时：

\[
\widetilde{\mathbf e}_t
=
\lambda\widetilde{\mathbf e}_{t-1}
+\nu_t\mathbf e_t,
\]

其中 \(\lambda\in[0,1]\) 是 retention；若当前 packet 与上一 packet 属于同一 correlation group，则 \(\nu_t\le 1\)。

该式是 neural pseudo-evidence heuristic，不是 exact conjugate update。配置和 trace 必须记录 \(\lambda,\nu_t\)。

### 7.3 Exact count update

只有真正独立的 Bernoulli/categorical counts 才能严格使用：

\[
\alpha'=\alpha+n^+,\qquad
\beta'=\beta+n^-,
\]

或：

\[
\boldsymbol\alpha'=\boldsymbol\alpha+\mathbf n.
\]

旧 evidence 不能因为同一图像被 pipeline 重试而再次增加。

## 8. InformationNeed

InformationNeed 是 uncertainty 到 action 之间的一等中间对象。它包含 proposition/deficit ID、当前 public anchor、deficit type/probability、prompt relevance、sufficiency shortfall 和用于 proposal budget 的 priority；它不包含动作标签。

当前 BayesRiskNeedExtractor 首先计算：

\[
\rho_q(b)
=\min_{d\in\mathcal D_q}
\sum_hL_q(d,h)b(h),
\]

\[
\rho_q^{\mathrm{PI}}(b)
=
\sum_h b(h)\min_d L_q(d,h),
\]

\[
\Delta\rho_q^{\max}
=\max(0,\rho_q(b)-\rho_q^{\mathrm{PI}}(b)).
\]

对 deficit \(i\)，当前实现的 priority 是：

\[
\boxed{
w_i
=p_i^{\mathrm{def}}\,
r_i^q\,
\left[
\Delta\rho_q^{\max}
+\lambda_s(1-\mu_s)
\right].
}
\]

这只是可审计的 localized proposal priority：

- \(\Delta\rho_q^{\max}\) 是整体 task-level upper bound，不是命题 \(i\) 的精确 EVPI；
- 它没有考虑具体动作效果、成本或风险；
- 多个 deficits 可能重叠；
- 真正的 action value 要等 candidate-conditioned \(T/Z\) rollout。

因此不能把 InformationNeed.priority 称为 EIG、EVSI 或 proposition-level EVPI。

## 9. Typed candidate proposal

每个信息动作 candidate 写为：

\[
a=(\kappa,g,\xi,\mathcal I_a),
\]

其中 \(\kappa\) 是 primitive family，\(g\) 是当前 public VisualAnchor，\(\xi\) 是 typed/bounded parameters，\(\mathcal I_a\) 是 candidate 声明处理的 need IDs。

动作集合只含开容器、拉抽屉、揭开覆盖物、移开/推开遮挡、拿近观察、旋转标签、拿起检查，以及 direct/stop；不含 NBV、camera move、navigation、walk-around 或主动视角原语。

`DIRECT_ACT` 是特殊的物理终止候选：它不声称处理 InformationNeed，但必须绑定当前、未修改、具有 `task_target` affordance 的 public source anchor。`STOP_NOT_FOUND` 不是物理动作，不携带 anchor。

当前 NeedDrivenPrimitiveProposer 是 recall-first registry baseline：一个 need 可以产生多个 affordance-compatible candidates。它不是 learned proposer，也不决定胜者。未来 learned/VLM proposer 应与 registry candidates 取并集，再由同一 effect model/reranker 比较。

## 10. Action-conditioned \(T/Z\) forecast

令需要 forecast 的物理候选集合为

\[
\mathcal A_t^{\mathrm{phys}}
=\mathcal A_t^{\mathrm{info}}\cup\{\mathrm{DIRECT\_ACT}\}.
\]

对每个 \(a\in\mathcal A_t^{\mathrm{phys}}\)，critic 返回：

\[
T_a(h'\mid h),
\qquad
Z_a(y\mid h').
\]

代码要求：

\[
\forall h:\quad\sum_{h'}T_a(h'\mid h)=1,
\]

\[
\forall h':\quad\sum_{y\in\mathcal Y_a}Z_a(y\mid h')=1.
\]

\(\mathcal Y_a\) 是互斥、穷尽、无条件 outcome 集。成功、部分执行、执行失败和“执行但未获得新证据”都必须显式表示为 branch。当前实现不在 branches 外再乘一个独立 feasibility；否则会重复计算失败概率。

每个 branch 还包含 execution status、predicted sufficiency evidence、resolved need IDs、action cost、physical risk 与 non-target disturbance。critic 自身的 epistemic/OOD uncertainty \(U_\phi(a)\in[0,1]\) 单独进入 planner penalty。

critic 不能直接返回任意 post-belief。它预测 \(T/Z\)，由 bridge 推导 posterior。`STOP_NOT_FOUND` 不执行物理动作，是唯一不使用 \(T/Z\) forecast 的候选。

## 11. Counterfactual Bayes rollout

当前 belief mean 为 \(\mathbf b_t\)。transition-predictive belief：

\[
\mathbf b_a^-
=T_a^\top\mathbf b_t.
\]

branch probability：

\[
p(y\mid \mathbf b_t,a)
=
\sum_{h'}Z_a(y\mid h')b_a^-(h').
\]

若该 branch 概率非零：

\[
b_{a,y}^+(h')
=
\frac{Z_a(y\mid h')b_a^-(h')}
{p(y\mid \mathbf b_t,a)}.
\]

由于 \(Z\) 对每个 post-state 穷尽归一化：

\[
\sum_y p(y\mid \mathbf b_t,a)\mathbf b_{a,y}^+
=\mathbf b_a^-.
\]

代码记录 posterior martingale residual：

\[
\epsilon_{\mathrm{mart}}(a)
=
\left\|
\sum_y p(y\mid a)\mathbf b_{a,y}^+
-\mathbf b_a^-
\right\|_1.
\]

规范化 \(T/Z\) 下它应接近浮点误差。这个一致性只证明内部概率代数正确，不证明 learned critic 准确。

branch sufficiency mean 由同一 prior/evidence 参数化计算。当前是 replacement semantics；没有声称对 counterfactual neural evidence 做精确 Bayesian filtering。

## 12. Bayes risk 与价值分解

对任意 task belief \(b\)：

\[
\rho_q(b)
=\min_{d\in\mathcal D_q}
\sum_hL_q(d,h)b(h).
\]

信息动作的决策规则是 `BAYES_AFTER_OBSERVATION`：对每个 branch posterior 重新选择最优终止决策。`DIRECT_ACT` 的决策规则是 `FIXED_DIRECT_ACT`：动作一旦被当作 direct candidate 评估，所有 branch 都必须沿用 `DIRECT_ACT` loss row，不得借预测观测后改变决策。定义规则化风险：

\[
r_a^-(b)=
\begin{cases}
\rho_q(b), & a\in\mathcal A_t^{\mathrm{info}},\\
\sum_hL_q(\mathrm{DIRECT\_ACT},h)b(h),
& a=\mathrm{DIRECT\_ACT},
\end{cases}
\]

\[
r_{a,y}^+(b)=
\begin{cases}
\rho_q(b), & a\in\mathcal A_t^{\mathrm{info}},\\
\sum_hL_q(\mathrm{DIRECT\_ACT},h)b(h),
& a=\mathrm{DIRECT\_ACT}.
\end{cases}
\]

定义当前 Bayes 风险与当前候选决策风险：

\[
\rho_t^{\mathrm{Bayes}}
=\rho_q(\mathbf b_t),
\]

\[
\rho_{t,a}^{\mathrm{decision}}
=
\begin{cases}
\rho_q(\mathbf b_t), & a\in\mathcal A_t^{\mathrm{info}},\\
\sum_hL_q(\mathrm{DIRECT\_ACT},h)b_t(h),
& a=\mathrm{DIRECT\_ACT},\\
\sum_hL_q(\mathrm{NOT\_FOUND},h)b_t(h),
& a=\mathrm{STOP\_NOT\_FOUND}.
\end{cases}
\]

`current_bayes_risk` 等于 \(\rho_t^{\mathrm{Bayes}}\)；`current_decision_risk` 等于 \(\rho_{t,a}^{\mathrm{decision}}\)。当前 Bayes 决策是所有 terminal rows 中的最小值，因此固定一个 terminal row 的承诺惩罚必为非负：

\[
\operatorname{CP}(a)
=\rho_{t,a}^{\mathrm{decision}}-\rho_t^{\mathrm{Bayes}}
\ge 0.
\]

对有 forecast 的物理 candidate，令 \(\rho_a^-=r_a^-(\mathbf b_a^-)\)，观测后的期望规则化风险为：

\[
\bar\rho_a^+
=\sum_y p(y\mid a)r_{a,y}^+(\mathbf b_{a,y}^+).
\]

当前 rollout 与 `CandidateValue` 报告：

\[
\operatorname{PV}(a)
=\rho_{t,a}^{\mathrm{decision}}-\rho_a^-,
\]

\[
\operatorname{IV}(a)
=\rho_a^--\bar\rho_a^+,
\]

\[
\Delta\rho(a)
=\rho_t^{\mathrm{Bayes}}-\bar\rho_a^+
=\operatorname{PV}(a)+\operatorname{IV}(a)-\operatorname{CP}(a).
\]

PV 是从当前候选决策风险到 transition-predictive risk 的变化；IV 是在该 candidate 的决策规则下，observation branches 额外带来的 conditional decision value；CP 单独显示从当前最优 Bayes 决策切换为固定 direct/stop row 所付出的承诺代价。三者是内部预测量，不是 realized improvement。`DIRECT_ACT` 的 PV 是已承诺 direct 决策下的物理转移价值，不能称为纯 information value。

对信息动作，规范 Bayes branches、相同 decision set/loss 下，expected information value 不应显著为负；明显负值通常提示模型、数值或 ontology 不一致。对 `DIRECT_ACT`，固定 loss row 对 belief 是线性的，结合 posterior martingale 可得 \(\operatorname{IV}(\mathrm{DIRECT\_ACT})=0\)（数值误差除外）；它的 forecast 用于衡量物理转移、执行失败、成本、风险与扰动，不是为已承诺动作虚构观测选择权。

## 13. 统一动作目标

对每个有 forecast 的物理 candidate \(a\in\mathcal A_t^{\mathrm{phys}}\)：

\[
\begin{aligned}
J_q(a)
={}&
\sum_y p(y\mid \mathbf b_t,a)
\Big[
r_{a,y}^+(\mathbf b_{a,y}^+)
+\lambda_c C_{a,y}
+\lambda_r R_{a,y}
+\lambda_d D_{a,y}\\
&\qquad\qquad
+\lambda_s(1-\mu_{s,a,y})
\Big]
+\lambda_u U_\phi(a).
\end{aligned}
\]

sufficiency 项是可审计 surrogate，不是建立在完整 joint belief \(P(H_q,S_{\mathrm{suff}})\) 上的严格 Bayes risk。论文必须单独报告纯 task risk 与 sufficiency penalty，不能把二者混称。

对唯一没有 forecast 的 `STOP_NOT_FOUND`：

\[
J_q(\mathrm{STOP\_NOT\_FOUND})
=
\sum_hL_q(\mathrm{NOT\_FOUND},h)b_t(h)
+\lambda_s(1-\mu_{s,t}).
\]

`STOP_NOT_FOUND` 没有 rollout，但它的 `CandidateValue` 仍显式记录：

\[
\operatorname{CP}(\mathrm{STOP\_NOT\_FOUND})
=
\sum_hL_q(\mathrm{NOT\_FOUND},h)b_t(h)
-\rho_q(\mathbf b_t),
\]

\[
\operatorname{PV}=0,\qquad
\operatorname{IV}=0,\qquad
\Delta\rho=-\operatorname{CP}.
\]

因此 stop 与 direct 的固定决策承诺都不会被隐藏在 physical/information value 中。

DIRECT_ACT、STOP_NOT_FOUND 与所有信息动作进入同一个候选池：

\[
a_t^*
=
\arg\min_{a\in\mathcal A_t^{\mathrm{phys}}\cup
\{\mathrm{STOP\_NOT\_FOUND}\}}J_q(a).
\]

这不同于“uncertainty 超阈值就探索，否则直接做”的两阶段 gate。

当前 cost/risk/disturbance 都是可校准 penalty，不是形式化 safety guarantee。hard physical precondition 与 stale/privileged grounding 必须在候选验证阶段拒绝。

## 14. 预测与真实更新的边界

counterfactual posterior 只用于动作前排序：

\[
\{\mathbf b_{a,y}^+\}_{a,y}
\not\rightarrow
\text{live episode belief}.
\]

非终止 primitive 执行后，controller 必须等待真实 ExecutionReport 和真实新 public observation：

\[
O_{t+1}^{\mathrm{actual}}
\xrightarrow{E_\theta}
\mathcal E_{t+1}^{\mathrm{actual}}
\xrightarrow{F}
b_{t+1}^{\mathrm{actual}}.
\]

即使 critic 乐观预测“开门后目标可见”，真实画面仍模糊时，也只能使用真实画面的 evidence。VLA 自述“已经看清”不能更新 belief。

## 15. 不是同一个概念的量

| 量 | 计算对象 | 回答的问题 | 不能替代 |
|---|---|---|---|
| predictive entropy | 当前 hypothesis mean | 预测分布是否分散 | evidence 缺失、动作价值 |
| Dirichlet MI | second-order belief | 参数/分布不确定性的代理 | task loss、空间定位 |
| vacuity | evidence strength | evidence 是否少 | calibrated error |
| dissonance | supported masses | 证据是否冲突 | lack of evidence |
| sufficiency Beta | prompt-conditioned proposition | 当前信息是否足以提交 | 具体动作效果 |
| deficit Beta | localized proposition | 哪类信息在何处缺失 | 最佳 primitive |
| Bayes risk | belief + prompt loss | 当前提交决策代价 | 物理可行性 |
| conditional IV | \(T/Z\) branches + loss | 某动作后的观测是否值得 | realized gain |
| critic uncertainty | effect model | forecast 是否可能 OOD | environment uncertainty |

## 16. 必须成立的软件不变量

1. 所有对象使用同一 TaskKey 与 hypothesis 顺序；
2. prompt digest 必须与 exact final-goal prompt 一致；
3. prompt/ontology 变化不能沿用旧 belief；
4. policy packet 不含 evaluator-private semantics；
5. Dirichlet/Beta evidence 非负、参数有限且在合法域；
6. 相同 observation content 与 model stamp 只能产生同一 event ID；
7. 重复 event 不重复累计；
8. correlated pseudo-evidence 不能当独立 counts；
9. InformationNeed 不含 action label，priority 明确是近似；
10. candidate 使用当前、未修改的 public anchor；
11. primitive 参数类型、范围与 stop condition 有界；
12. primitive registry 不含 active viewpoint action；
13. `DIRECT_ACT`、`STOP_NOT_FOUND` 与 information candidates 同池；
14. `DIRECT_ACT` 必须有当前 public `task_target` anchor；
15. 每个物理 candidate（含 `DIRECT_ACT`）恰有一个 effect forecast，只有 `STOP_NOT_FOUND` 没有；
16. 信息动作使用 `BAYES_AFTER_OBSERVATION`，`DIRECT_ACT` 使用 `FIXED_DIRECT_ACT`；
17. `decision_commitment_penalty = current_decision_risk - current_bayes_risk >= 0`，且 `total_task_risk_reduction = physical_progress_value + conditional_information_value - decision_commitment_penalty`；
18. `STOP_NOT_FOUND` 虽无 rollout，`CandidateValue` 仍显式记录其固定 decision risk 和 commitment penalty；
19. \(T\) rows 与 \(Z\) columns 归一化；
20. outcomes 互斥、穷尽，失败只计算一次；
21. forecast 不能声称解决 candidate 未处理的 need；
22. posterior martingale residual 接近数值误差；
23. candidate 输入顺序不改变 ranking；
24. VLA/executor 不得偷换 candidate 或 primitive family；
25. 非终止动作后必须真实再观测；
26. predicted posterior 永不提交为 live posterior；
27. hash-linked integrity trace 的 parent、event identity 与 content digest 可验证，但不声称防篡改或完整 replay。

## 17. 当前不能声称

- 不能声称首次 uncertainty \(\rightarrow\) action；
- 不能声称 Beta/Dirichlet、POMDP、VoI、Bayes risk 是新公式；
- 不能把 arbitrary VLM confidence 当 calibrated probability；
- 不能把 \(W/S\) 直接称为真实 epistemic error；
- 不能把 repeated-frame pseudo-count 说成 exact Bayes filtering；
- 不能把 InformationNeed.priority 说成 exact EVPI/EIG/EVSI；
- 不能把 rule-based need-to-family registry 说成 learned action generation；
- 不能只预测一个平均 posterior 后声称保留 conditional information value；
- 不能把 normalized \(T/Z\) 的内部一致性当 effect accuracy；
- 不能把 predicted risk reduction 当 realized risk reduction；
- 不能把 sufficiency penalty 说成严格 joint Bayes risk；
- 不能把 action-sample disagreement 直接等同于环境不确定性；
- 不能把 generic penalty 说成 safety guarantee；
- 不能把 modular VLA handoff 说成 end-to-end VLA training；
- 不能把单元测试或 paired toy image 当方法有效性的完整证据。

## 18. 主要理论与方法来源

- POMDP belief planning：[Smallwood & Sondik, 1973](https://doi.org/10.1287/opre.21.5.1071)，[Kaelbling, Littman & Cassandra, 1998](https://doi.org/10.1016/S0004-3702(98)00023-X)。
- Active perception：[Bajcsy, 1988](https://doi.org/10.1109/5.5968)。
- Entropy 与 KL：[Shannon, 1948](https://doi.org/10.1002/j.1538-7305.1948.tb00917.x)，[Kullback & Leibler, 1951](https://doi.org/10.1214/aoms/1177729694)。
- Bayesian experiment/VoI：[Lindley, 1956](https://doi.org/10.1214/aoms/1177728069)，[Howard, 1966](https://doi.org/10.1109/TSSC.1966.300074)。
- Task-driven information gathering：[Hsiao, Kaelbling & Lozano-Pérez, RSS 2010](https://www.roboticsproceedings.org/rss06/p29.html)。
- Subjective Logic 与 EDL：[Jøsang, 2016](https://doi.org/10.1007/978-3-319-42337-1)，[Sensoy et al., NeurIPS 2018](https://papers.nips.cc/paper/2018/hash/a981f2b708044d6fb4a71a1463242520-Abstract.html)。
- Action-conditioned belief/effect：[CNABU, RSS 2025](https://www.roboticsproceedings.org/rss21/p039.html)，[Dengler et al., 2025](https://arxiv.org/abs/2506.02286)。
- Foundation-model belief-space planning：[Seeing is Believing, 2025](https://arxiv.org/abs/2504.03245)。
- VLA/long-horizon sensing-action 对照：[Act-Sense-Act, 2026](https://arxiv.org/abs/2602.04600)，[UAOR, 2026](https://arxiv.org/abs/2602.18020)。

更完整的“单个量是谁提出、哪些 novelty claim 已被占据”见 [research-symbols-and-prior-art.md](research-symbols-and-prior-art.md)。

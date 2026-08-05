# 数学符号、技术路线与前人工作溯源

本页回答：当前系统里的数学量是谁的、哪些 claim 已被前人占据、我们还能把贡献放在哪里。

结论是：POMDP belief、Beta/Dirichlet evidence、entropy、mutual information、vacuity、Bayes risk、value of information、information-gathering manipulation 与 action-conditioned belief prediction都已有明确前例。不能把任何单一公式包装成 novelty。

目前值得检验的交集是：

> final-goal prompt 定义 task belief/loss；公开 RGB 产生可校准 evidence；localized InformationNeed 不携带动作答案；多类 fixed-wrist 物理信息原语由候选 effect 决定；所有物理候选（包括有 public `task_target` anchor 的 direct）都有 effect forecast，只有 stop 无 forecast；direct/stop 与信息动作在同一 Bayes-risk objective 下比较；VLA 只实现已选原语。

这仍是待实验验证的系统性组合，不是已经成立的“首次”结论。

## 1. 符号与归属

| 符号/对象 | 当前含义 | 主要理论传统 | 是否为新量 |
|---|---|---|---|
| \(s_t,a_t,o_t,h_t\) | 状态、动作、观测、历史 | MDP/POMDP | 否 |
| \(b_t(s)\) | 部分可观测状态 belief | POMDP | 否 |
| \(T(s'\mid s,a)\) | transition model | POMDP | 否 |
| \(Z(o\mid s',a)\) | observation model | POMDP | 否 |
| \(\tau(b,a,o)\) | belief update operator | POMDP | 否 |
| \(q\) | 用户最终目标 prompt | language-conditioned models | 记号选择非贡献 |
| \(H_q\) | prompt-relevant task latent | task projection/latent state | 具体 ontology 可设计，概念非首创 |
| \(b_t^q\) | prompt-conditioned task belief | language-conditioned belief | 组合对象 |
| \(L_q(d,h)\) | prompt-specific terminal loss | Bayesian decision theory | 否 |
| \(\rho_q(b)\) | posterior minimum Bayes risk | Bayesian decision theory、task-driven exploration | 否 |
| \(H(p)\) | Shannon entropy | Shannon 1948 | 否 |
| \(D_{\mathrm{KL}}(p\|q)\) | KL divergence | Kullback–Leibler 1951 | 否 |
| \(I(X;Y)\) | mutual information | information theory | 否 |
| EIG | expected information gain | Bayesian experimental design | 否 |
| EVSI/VoI | 信息对决策价值 | Lindley、Howard | 否 |
| \(\boldsymbol\alpha\) | Dirichlet concentrations | Dirichlet、EDL | 否 |
| \(S=\sum_k\alpha_k\) | Dirichlet strength | Dirichlet/Subjective Logic | 否 |
| \(e_k\) | non-negative evidence | Subjective Logic、EDL | 否 |
| \(a_k\) | base rate | Subjective Logic | 否；代码避免与 action \(a\) 混淆 |
| \(W\) | prior weight | Subjective Logic | 否 |
| \(u=W/S\) | vacuity | Subjective Logic | 否 |
| dissonance | supported hypotheses 的冲突 | Subjective Logic、multidimensional evidential UQ | 否 |
| \(\operatorname{Beta}(\alpha,\beta)\) | binary proposition belief | Beta–Bernoulli/Subjective Logic | 否 |
| \(\epsilon_{\mathrm{mart}}\) | posterior mixture 一致性 residual | law of total expectation | 名称自定义，性质非新 |
| PV | action transition 带来的 task-risk change | 本项目记账名 | 代数非新 |
| IV | action 后 observation 的 conditional decision value | EVSI/VoI | 名称自定义，思想非新 |
| \(\mathcal N_t\) | localized InformationNeed 集合 | information requirement/task-driven exploration | typed 中间层可能是系统贡献 |
| \(a=(\kappa,g,\xi)\) | primitive family、public grounding、参数 | parameterized skill | 接口记法非理论贡献 |
| \(U_\phi(a)\) | effect critic OOD/epistemic penalty | model-based RL/risk-aware planning | 否 |
| \(J_q(a)\) | task risk 与动作代价的统一目标 | stochastic control/decision theory | 具体组合需消融 |

## 2. POMDP 与 task-driven risk

[Smallwood & Sondik (1973)](https://doi.org/10.1287/opre.21.5.1071) 和 [Kaelbling, Littman & Cassandra (1998)](https://doi.org/10.1016/S0004-3702(98)00023-X) 奠定 belief-state planning：belief 不是单一 uncertainty scalar；动作价值必须结合 transition、observation 与 reward/loss。

[Task-Driven Tactile Exploration](https://www.roboticsproceedings.org/rss06/p29.html) 已明确围绕任务损失决定是否继续收集信息，而不是最小化全局 uncertainty。这占据了“task uncertainty 到 information action”的核心抽象：

- terminal 与 information action 可以同树比较；
- entropy 降低若不改变任务决策，价值可能很小；
- stop 条件要对应 task loss；
- action cost 必须进入选择。

我们加入的是 natural-language task specification、视觉 evidence、多类 fixed-wrist manipulation primitive、现代 VLA execution 和 benchmark，而不是重新发明 task-directed exploration。

## 3. Beta/Dirichlet 与 evidential uncertainty

[Subjective Logic](https://doi.org/10.1007/978-3-319-42337-1) 给出 evidence、base rate、prior mass 与 vacuity；[Evidential Deep Learning](https://papers.nips.cc/paper/2018/hash/a981f2b708044d6fb4a71a1463242520-Abstract.html) 让网络输出非负 evidence 并形成 Dirichlet second-order distribution。

可借鉴：

- task hypotheses 用 Dirichlet；
- binary sufficiency/deficit 用 Beta；
- predictive entropy、expected entropy、MI、vacuity、dissonance 分开；
- 用 held-out NLL、Brier、ECE、risk–coverage 验证。

不能直接推出：

- evidence strength 天然校准；
- \(W/S\) 就是真实 epistemic error；
- 相邻视频帧 pseudo-count 可独立相加；
- 任意 soft VLM score aggregation 仍是 exact Beta。

只有 hard category subset aggregation 严格保持 Beta。

## 4. CNABU 与 Dengler uncertainty-informed action selection

[CNABU / Map Space Belief Prediction](https://www.roboticsproceedings.org/rss21/p039.html) 学习 action-conditioned map-space belief update，并强调 calibration 对 information-gain planning 的重要性。它是 learned belief dynamics 的直接前例。

[Dengler et al. (2025)](https://arxiv.org/abs/2506.02286) 进一步把 map uncertainty 接到 NBV 与 push：定位不确定区域、构造 corridor、选择 occluder、预测 push 后 belief，再比较候选序列的信息收益。这说明：

- uncertainty 可以驱动“为了发现未知”的物理动作；
- action effect prediction 比只看当前 uncertainty 更接近完整决策；
- NBV 和 manipulation 可在一个 objective 下比较。

它仍与本项目存在明确边界：

- 主要目标是全局 semantic mapping/completeness，不是 final-goal prompt loss；
- candidate generation 高度依赖 shelf geometry、ray/corridor 与 push；
- action family有限；
- VLA 不负责高层 uncertainty-to-primitive bridge；
- 当前公开代码默认路径与论文完整 RL-NBV 叙述之间存在复现 caveat，详见 [开源实现审计](open-source-implementation-audit.md)。

我们应把它作为最重要的近邻和 baseline 灵感，而不是声称它没有做 uncertainty-to-action。

## 5. 部分可观测 TAMP 与 foundation models

[TAMPURA](https://www.roboticsproceedings.org/rss20/p118.html) 处理初始状态与 action outcome uncertainty、信息收集和不可逆风险，说明 information action、task planning 与 physical risk 的结合已有明确前例。

[Seeing is Believing](https://arxiv.org/abs/2504.03245) 用 foundation model 估计 symbolic grounding uncertainty，并在 belief-space planner 中加入 information-gathering skills。它已接近“语言目标 + foundation model uncertainty + skill planning”。

因此我们的差异必须落到可验证的更具体维度：

- 连续、校准的 prompt-conditioned evidence，而非只做 yes/no/unknown judgment；
- localized deficits 与多类 fixed-wrist physical primitives；
- candidate-conditioned stochastic \(T/Z\)；
- candidate recall、effect calibration、selection regret 与 direct/stop boundary 的 benchmark。

如果最终只是让 VLM 输出“看不清”再调用固定 skill，新颖性很弱。

## 6. Mechanical search

[Mechanical Search](https://arxiv.org/abs/1903.01588)、[Mechanical Search on Shelves](https://arxiv.org/abs/2207.02347) 与 [Semantic Mechanical Search](https://arxiv.org/abs/2302.12915) 已用 push、pick、remove、stack/destack 和语言/视觉语义发现遮挡目标。

它们补足了我们的动作库、clutter dynamics 与 multi-step retrieval 维度，也限制了 novelty：

- 不能把“通过移开物体寻找目标”称为新任务；
- 应比较 retrieval-centric/global declutter baseline；
- 应报告 non-target displacement、drop、collision 与 search efficiency。

我们的重点是 prompt-conditioned information sufficiency、rotate/bring-close/open/remove 等 heterogeneous primitives 的 effect/value comparison，以及何时 direct/stop。

## 7. VLA 中的不确定度与 sensing-action

[Act-Sense-Act](https://arxiv.org/abs/2602.04600) 学习 view/manipulation streams、temporal memory 和 task-state completion head。其公开 completion head 是监督式完成分类，不是 calibrated environment belief、EVSI 或多 primitive effect comparison；view stream也超出当前 no-active-viewpoint 范围。它适合作为已选 primitive 的执行/长时序表征参考。

[UAOR](https://arxiv.org/abs/2602.18020) 以 action entropy 决定何时重新注入 observation features，解决当前 VLA 解码中的观察利用问题，不构造 task environment belief，也不比较 open/remove/rotate/direct/stop。

[Shifting Uncertainty to Critical Moments](https://arxiv.org/abs/2603.18342) 研究 VLA action uncertainty 的时序聚合与 failure prediction，支持“平均 token entropy 会掩盖关键风险”，但没有解决 environment belief 到探索原语。

[AtVLA](https://arxiv.org/abs/2608.02197) 于 2026-08-03 发布，超出原定截至 2026-07 的主检索窗口。它用 action-chunk disagreement 触发局部 crop/re-encoding，是 uncertainty-gated digital visual refinement，不是物理 information enrichment primitive comparison。

## 8. 一个必须纠正的材料问题

arXiv:2602.23574 实际是 [Evidential Neural Radiance Fields](https://arxiv.org/abs/2602.23574)，不是 SCALE/VLA action-selection 论文。它可作为 3D scene uncertainty 的跨领域参考，但不能支撑 VLA uncertainty-to-action 的直接 related-work claim。若课件把该编号与 SCALE 图对应，必须更正。

## 9. Novelty collision audit

以下声明不可使用：

- “首次把 uncertainty 用于机器人动作选择”；
- “首次做 action-conditioned belief prediction”；
- “首次为了获得信息执行 manipulation”；
- “首次把语言目标与 belief-space planning 结合”；
- “首次比较 sensing 与 acting”；
- “首次用 Beta/Dirichlet 表示环境不确定度”；
- “首次让 VLA 在不确定时利用更多视觉信息”。

较可辩护但仍需实验支撑的表述是：

> 一个面向 fixed-wrist interactive perception 的 prompt-conditioned proposal–effect–value bridge：从部署可得 RGB 产生校准 task evidence，将其转成不携带动作答案的 localized InformationNeeds，生成多类 typed physical information primitives，对所有物理候选（包括有 public `task_target` anchor 的 direct）用 action-conditioned stochastic \(T/Z\) 推演 task post-beliefs，只有 stop 无 forecast；三者在同一 Bayes-risk objective 下比较，最后约束 VLA只实现已选 primitive。

## 10. 证据链与否证条件

完整论文证据需要：

1. far/near、closed/open、label-back/front 上的 evidence sensitivity；
2. held-out episode 上 hypothesis 与 sufficiency calibration；
3. prompt swap 下 task relevance/value 变化，同时物理预测保持合理不变；
4. proposer beneficial-candidate recall@K；
5. effect outcome NLL/Brier、risk-reduction error 与 branch consistency；
6. selected-action regret 与 oracle-effect upper bound；
7. direct-visible 正例上不过度探索；
8. target-absent 有界搜索后正确 stop；
9. 与 prompt-only VLA、uncertainty text、threshold gate、global entropy、action-independent uncertainty、push-only/mechanical-search baseline 的对照；
10. primitive、belief、effect、cost/risk/sufficiency 的消融。

若 evidence 不能校准、best action 不进入 candidate set、effect model 无法优于 action-independent heuristic，或者 full router 不降低 regret，那么 method claim 应收缩为 benchmark/interface contribution。

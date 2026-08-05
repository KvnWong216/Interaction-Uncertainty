# 文档索引

- [系统架构](architecture.md)：v0.2 有状态闭环、模块边界、VLA 接入和不变量。
- [数学与归属](mathematics.md)：符号、Beta/Dirichlet、Bayes risk、信息价值、动作评分以及哪些内容来自既有理论。
- [数学符号与前人溯源](research-symbols-and-prior-art.md)：哪些公式与 claim 已有先例、novelty collision 和材料纠错。
- [实验合同](experiment-contract.md)：benchmark 数据边界、主实验、指标、消融和结果报告要求。
- [代码复用与许可证审计](code-reuse-license-audit.md)：CNABU/Dengler、ASA、OpenVLA、Octo、SayCan、pomdp_py、EDL、RLDS 的复用结论。
- [开源实现审计](open-source-implementation-audit.md)：固定 commit 下的实际代码路径、默认调用链、可复用接口和不能过度陈述的 caveat。
- [集成指南](integration.md)：接入 LIBERO、evidence service、action-outcome critic 和远程 VLA/skill executor。

这些文档描述 `0.2.0` technical alpha。旧代码是 feasibility-only prototype；v0.2 是 clean rewrite，旧静态查表路径已从生产包删除，不承诺旧 API、配置、trace 或行为兼容。若代码和文档不一致，应先把差异作为 issue 记录，不得静默改变概率语义、policy/evaluator 数据边界或 benchmark 口径。

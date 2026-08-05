# Contributing

感谢对 Interaction-Uncertainty 的贡献。本项目是研究基础设施；一个看似小的字段、公式或 fixture 修改可能改变实验问题，因此可复现性和数据边界优先于 API 便利性。

## 1. 开发环境

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
ruff check .
pytest
interaction-uncertainty validate-task --task examples/v2/task_orange_juice.json
```

包声明支持 Python >=3.10；CI 当前覆盖 Python 3.10、3.11 和 3.12。更新的 Python 版本在加入 CI 前不作兼容保证。

## 2. 不可静默改变的研究合同

以下改动必须在 issue/PR 中明确说明，并同步更新测试、schema 和文档：

- 添加或移除 primitive type；
- 修改 task hypotheses、loss matrix 或 value units；
- 修改 Beta/Dirichlet evidence semantics；
- 把新的字段加入 policy observation、belief、candidate、effect 或 VLA packet；
- 修改 effect outcome、termination 或 re-observation 语义；
- 修改 prompt leakage 检查；
- 修改 benchmark split、oracle/evaluator 数据边界或成功条件；
- 加入 active viewpoint、导航或 camera-motion action。

当前 scope 明确排除 active viewpoint。若研究需要探索这一方向，应建立独立 benchmark/config，不应让结果与本项目 fixed-camera protocol 混报。

## 3. Privileged-information policy

禁止向任何 policy-facing object 添加：

- simulator semantic/instance ID；
- ground-truth segmentation、target mask/bbox、target instance；
- MuJoCo/robosuite `qpos`、`qvel`、body/geom/joint ID；
- oracle target location、oracle action、oracle utility；
- BDDL path、asset filename 或可解码目标身份的 token。

Ground truth 只能进入独立 evaluator channel。新增 adapter 时必须：

1. 对输入输出调用 `PolicyFirewall`；
2. 添加一个正向 public-packet 测试；
3. 添加一个注入禁止字段后失败的测试；
4. 确认 provenance 不包含 simulator-private 名称。

## 4. 概率与数学改动

- 写明变量的随机对象；例如 Beta variance 是概率参数 variance 还是 label predictive variance；
- 区分 predictive belief、observation-conditioned posterior 和实际新观测后的 posterior；
- 不把平均 posterior 用作 EIG；
- 区分 realized gain 与 expected value；
- 新 uncertainty scalar 必须给出 primary reference、单位、范围和 calibration protocol；
- heuristic 加权和必须称为 heuristic/scalarization，不标成 mutual information 或定理；
- 更新 [数学与归属](docs/mathematics.md) 和 `references.bib`。

## 5. 新 candidate proposer

Proposer 应返回多个候选并接受后续统一排序。必须验证：

- grounded anchor 来自当前公开 frame；
- candidate IDs 唯一；
- 每个物理候选都有当前 public frame 中的 grounding；`DIRECT_ACT` 还必须绑定带 `task_target` affordance 的 public anchor；
- 参数 JSON-safe 且有范围；
- candidate set 包含终止决策；
- 没有把 uncertainty deficit 直接伪装成已选 action；
- 报告 proposal recall@K、invalid rate 和 provenance。

## 6. 新 effect model

必须返回每个 `requires_effect_forecast` 物理候选的 matching `CandidateEffectForecast`，并用归一化 (T/Z) 表示互斥、穷尽的 outcomes。这包括 `DIRECT_ACT`；只有不执行物理动作的 `STOP_NOT_FOUND` 没有 forecast：

- 每个 transition row 和 observation post-state column 归一化；
- failure 作为一个 branch，只计算一次；
- ontology、TaskKey 和 observation digest 一致；
- 报告 Bayes-consistency residual；
- 分别记录预测值与真实动作后值。

若只预测 expected post-action risk，应把它作为独立 baseline，不声称是完整 observation model，也不能用它替代下一步真实 belief update。

## 7. 测试 fixture 与 benchmark episode

Demo prompt 只能表达最终目标，不能包含探索提示。Fixture 必须标注：

- public observation；
- evaluator-private truth 的独立位置；
- scene/seed；
- paired action 和生成方式；
- asset/data license；
- 合法候选与 evaluator-private 最优动作标注；
- 是否为真实执行、replay 或 simulator reset counterfactual。

不要把 frame/prompt/candidate keyed lookup 放入 production，也不要手工修改 test output 以匹配方法。解析 fake 只能验证数学/接口，不得被当作模型结果。

## 8. 外部代码、模型和数据

添加依赖前完成：

```text
official repository URL
exact tag/commit
root license
copied/adapted files, if any
weight license
dataset/asset license
dependency and compute cost
reason an interface-only integration is insufficient
```

CNABU/Dengler 的相关根仓库在 2026-08-04 审计时没有许可证，因此禁止复制其代码、权重和素材。只允许依据论文 clean-room 实现并引用。

优先使用 optional dependency 或 HTTP/JSON adapter，不 vendor OpenVLA、Octo、ASA、RLDS 或完整仿真仓库。

## 9. 提交前检查

```bash
ruff check .
pytest
interaction-uncertainty validate-task --task examples/v2/task_orange_juice.json
```

另外检查：

- JSON schemas 可解析；
- README/docs 中所有相对路径存在；
- 没有残留本地绝对路径、token 或私有数据；
- 新文献使用 DOI、出版社页面、arXiv 或官方项目页；
- `NOTICE`、`CITATION.cff`、`references.bib` 和 license audit 已同步；
- 运行日志不包含 evaluator-private state。

## 10. PR 描述模板

```text
Research question affected:
Policy-visible inputs changed:
Evaluator-private inputs changed:
Probability/action semantics changed:
New third-party code/weights/data:
Tests and demos run:
Expected metric impact:
Known limitations:
```

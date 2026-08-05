# 代码复用与许可证审计

审计日期：2026-08-04。

本文件是研究工程的依赖与归属记录，不是法律意见。许可证可能在上游更新；每次正式 release 前必须重新检查上游仓库、所用 commit、模型权重、数据集和资产文件。

## 1. 结论

当前最稳妥的工程策略是：

- 核心包只依赖 NumPy；严格 JSON schemas 和 pytest/ruff/jsonschema 仅属于开发环境；
- Beta/Dirichlet、Bayes risk 和桥接协议独立实现；
- CNABU/Dengler 只根据论文 clean-room 参考，不复制代码或素材；
- POMDP、VLA、RLDS 等通过 optional dependency、独立进程或 JSON adapter 接入；
- 不 vendor 大型上游仓库；
- 代码许可证、模型权重许可证、数据许可证和 mesh/图像资产许可证分别审计。

## 2. 审计总表

| 上游 | 根许可证/状态 | 可参考的接口 | 依赖重量 | 本项目决定 |
|---|---|---|---|---|
| CNABU / Map Space Belief Prediction | **根仓库未提供许可证** | candidate manipulation → predicted map belief → information value | 极重 | 仅论文与接口 clean-room 参考；不复制/不 vendor |
| Dengler view-point-pushing | **根仓库未提供许可证** | uncertainty region → corridor/occluder → push candidate | 重 | 仅算法思想参考；不复制 |
| `pomdp_py` | MIT | State/Action/Observation、belief、POMCP/POUCT | 中等，含 Cython | 可选 baseline dependency，不 vendor |
| EDL PyTorch community implementation | MIT；非原作者官方实现 | evidence transform、Dirichlet loss | 代码小但版本很旧 | 依据论文独立实现，不安装旧 requirements |
| `subjective-logic` | MIT，小型第三方包 | opinion/vacuity 映射 | 轻 | 仅可选测试 oracle，不作为核心依赖 |
| SayCan official example | Google Research Apache-2.0 | language usefulness × affordance scoring | demo notebook 较杂 | 只参考 score-composition 接口 |
| Act-Sense-Act / ASA | Apache-2.0 | cognition head、memory、subtask transition、训练流程 | 极重 | 只参考训练/接口；不复制到核心 |
| Octo | MIT | task/observation API、action sampling/head | 极重，JAX+TF | 独立服务 adapter |
| OpenVLA | MIT code；weights 有独立限制 | `predict_action`、action tokenizer、REST deploy | 7B，GPU 较重 | 独立服务 adapter；不分发权重 |
| OpenVLA-OFT | MIT code；weights 另审计 | continuous action head、action chunk、proprio adapter | GPU 较重 | 未来 latent/head 研究参考 |
| RLDS | Apache-2.0，仓库已归档 | episode/step schema | 实际使用 TF-heavy | 采用 schema 语义；可选 exporter |

## 3. CNABU / Marques et al.

论文：

- [Map Space Belief Prediction for Manipulation-Enhanced Mapping, RSS 2025](https://arxiv.org/abs/2502.20606)

官方补充仓库：

- [NilsDengler/manipulation_enhanced_map_prediction](https://github.com/NilsDengler/manipulation_enhanced_map_prediction)
- [HumanoidsBonn organization fork](https://github.com/HumanoidsBonn/manipulation_enhanced_map_prediction)

本地审计过的研究 clone commit：

```text
8ffa53f0c32e31f81ac9c1a75612b2e5a9443456
```

### 3.1 许可证结论

上述根仓库文件列表没有 `LICENSE` 或 `COPYING`。本地树只发现嵌套组件/资产的许可证，例如 LGPL-3.0 的 scikit-geometry 和 ROS-Industrial BSD 风格 URDF 许可。这些嵌套许可证不构成对整个根仓库的授权。

因此：

- 公开可读不等于可复制、修改或再分发；
- 本项目不复制 Python 源码、预训练模型、mesh、图像或 demo dataset；
- 本项目只依据论文定义独立实现数据类型、公式和测试；
- README 与代码注释使用“clean-room reimplementation inspired by the published method”，不使用“forked from”或“based on their code”；
- 若未来需要直接复用，必须先获得明确许可证或作者书面许可。

### 3.2 值得参考但不复制的模块

上游 README 指向：

```text
shelf_gym/scripts/run_cnabu_pipeline.py
shelf_gym/utils/information_gain_utils.py
shelf_gym/utils/uncertainty_informed_push_utils.py
shelf_gym/utils/learning_utils/losses.py
shelf_gym/utils/models/UNet.py
```

可抽象思想是：

```text
current map belief
    -> candidate pushes/views
    -> action-conditioned predicted beliefs
    -> information-gain comparison
    -> execute first action
```

本项目的 `CandidateEffectForecast(T/Z)` 是更一般、独立设计的接口：支持 Open/Remove/Rotate/Bring-Close 等 fixed-wrist 交互原语和穷尽 observation branches，不保留上游文件结构或实现代码。

### 3.3 依赖原因

上游研究环境涉及 PyBullet、Open3D、CuPy/CUDA、Torch、Stable-Baselines3、Klampt、Toppra、Voxelmap、Lightning、Hydra 等。即使未来获得许可，也不适合成为本项目核心 dependency。

## 4. Dengler uncertainty-informed action selection

论文：

- [Efficient Manipulation-Enhanced Semantic Mapping With Uncertainty-Informed Action Selection](https://arxiv.org/abs/2506.02286)

相关仓库：

- [NilsDengler/view-point-pushing](https://github.com/NilsDengler/view-point-pushing)
- [manipulation_enhanced_map_prediction](https://github.com/NilsDengler/manipulation_enhanced_map_prediction)

`view-point-pushing` 根目录同样未提供可见许可证。因此仅参考“uncertainty region → ray/corridor → occluder → push candidate”这一算法关系，不复制 corridor、push planner 或仿真环境实现。

论文目标是全局 semantic mapping，包含 NBV 并主要生成 push。Interaction-Uncertainty 的范围差异为：

- 明确排除 active viewpoint；
- task belief 和损失由最终 prompt 定义；
- 比较多个 interactive manipulation / information-enrichment primitive families；
- `DIRECT_ACT` 与 `STOP_NOT_FOUND` 进入同一决策池；
- 通过中立 JSON 接口连接 VLA。

这些差异是研究问题差异，不构成对无许可证源代码的复制许可。

## 5. `pomdp_py`

- 仓库：[h2r/pomdp-py](https://github.com/h2r/pomdp-py)
- 许可证：[MIT](https://github.com/h2r/pomdp-py/blob/main/LICENSE)
- 文档：[framework API](https://h2r.github.io/pomdp-py/html/api/pomdp_py.framework.html)

关键结构：

```text
pomdp_py/framework
pomdp_py/algorithms
```

它提供 State、Action、Observation、Transition/Observation/Reward/Policy Model、Agent、Environment，以及 POMCP/POUCT 等算法。

决定：仅作为 `.[pomdp]` optional dependency，用于小型离散 reference planner。核心代码不 import 它，也不 vendor 其 MIT 源码。高维 RGB、开放语义和连续 manipulation parameters 需要另外的抽象，不把 `pomdp_py` 描述为本项目主 planner。

若未来复制/修改 MIT 源文件，必须保留原 copyright 和完整 MIT license；当前无此需求。

## 6. Evidential Deep Learning 与 Subjective Logic

原论文：

- [Sensoy, Kaplan & Kandemir, Evidential Deep Learning, NeurIPS 2018](https://papers.nips.cc/paper/2018/hash/a981f2b708044d6fb4a71a1463242520-Abstract.html)
- [Jøsang, Subjective Logic, 2016](https://doi.org/10.1007/978-3-319-42337-1)

常用 community implementation：

- [dougbrion/pytorch-classification-uncertainty](https://github.com/dougbrion/pytorch-classification-uncertainty)，MIT；
- [waleedqk/subjective-logic](https://github.com/waleedqk/subjective-logic)，MIT。

第一个仓库不是论文作者官方代码，且固定在很旧的 Python/PyTorch 版本。第二个包规模小、成熟度有限，不是 Jøsang 的权威 reference implementation。

决定：依据论文公式独立实现所需的 evidence→Dirichlet/Beta、entropy、MI、vacuity 和 dissonance。没有复制这些仓库的源文件，也不安装其环境。若未来把第三方包用作测试 oracle，应 pin tag/commit 并在 `THIRD_PARTY_NOTICES`/SBOM 记录。

## 7. SayCan

- 论文：[Do As I Can, Not As I Say: Grounding Language in Robotic Affordances](https://arxiv.org/abs/2204.01691)
- 官方示例：[google-research/saycan](https://github.com/google-research/google-research/tree/master/saycan)
- Google Research 根许可证：[Apache-2.0](https://github.com/google-research/google-research/blob/master/LICENSE)

官方发布主要是 Colab/demo，而不是稳定的可安装 library。最值得参考的是候选 skill 的 language usefulness 与 affordance feasibility 组合评分。

决定：本项目独立定义 Bayes-risk router，并加入 action-conditioned task information value、cost 和 risk；不复制 notebook、GPT-3/CLIP/ViLD/CLIPort demo 代码。若未来复制 Apache-2.0 代码，需附 LICENSE、保留 notices/headers，并显著标注修改。

## 8. Act-Sense-Act / CoMe-VLA

- 论文：[Act, Sense, Act, arXiv:2602.04600](https://arxiv.org/abs/2602.04600)
- 项目页：[jern-li.github.io/asa](https://jern-li.github.io/asa/)
- 官方代码：[Joringell/ASA](https://github.com/Joringell/ASA)
- 许可证：Apache-2.0。

仓库公开三阶段训练、数据预处理、模型和 checkpoints，但 README 说明完整 inference/deployment 依赖 proprietary robot SDK，未完全开放。

决定：只参考 cognition head、subtask transition、dual-track memory 和训练流程的公开接口。当前 bridge 不复制 ASA 代码、不加载其 checkpoint、不声称复现其机器人实验。未来若接 ASA-derived service，作为独立 optional adapter，并分别核查 MANO、CaptainCook4D、EgoExo4D、Monte02 和 checkpoint 的许可证。

## 9. Octo

- 论文：[Octo: An Open-Source Generalist Robot Policy](https://arxiv.org/abs/2405.12213)
- 官方代码：[octo-models/octo](https://github.com/octo-models/octo)
- 许可证：[MIT](https://github.com/octo-models/octo/blob/main/LICENSE)

关键接口位于 `octo/model/octo_model.py`、`octo/model/components/action_heads.py`、tokenizers 和 dataset pipeline。`sample_actions` 的多样本接口可用于生成候选或 action disagreement，但 action disagreement 不能自动解释为 environment uncertainty。

Octo 依赖 JAX、Flax、TensorFlow、TFDS、RLDS、Optax 等；不适合核心环境。

决定：使用已经实现的模型中立 `RemotePolicyBackend` 合同，把 `VLAExecutionRequest` 发送给独立 Octo 服务并接收固定元数据的 `ActionChunk`。不复制 Octo 源码，不分发权重；Octo 不能在服务端重选高层 primitive。

## 10. OpenVLA 与 OpenVLA-OFT

- 论文：[OpenVLA, arXiv:2406.09246](https://arxiv.org/abs/2406.09246)
- 官方代码：[openvla/openvla](https://github.com/openvla/openvla)
- 代码许可证：[MIT](https://github.com/openvla/openvla/blob/main/LICENSE)
- OFT 代码：[moojink/openvla-oft](https://github.com/moojink/openvla-oft)

OpenVLA 的 `predict_action`、action tokenizer 和 REST deployment 是有用的接口参考。官方明确说明预训练模型继承 Llama-2 等基础模型的独立限制，因此 MIT code license 不覆盖所有 weights。

决定：

- 核心包不 import Transformers/Torch/OpenVLA；
- 使用中立 `VLAExecutionRequest` / `ActionChunk`，服务端自行翻译成 OpenVLA prompt/action representation；
- 不在仓库分发 OpenVLA/OpenVLA-OFT 权重；
- 未来 latent/action-head adapter 作为服务器实验，记录 base model、weight license、checkpoint hash 和 normalization key。

## 11. RLDS

- 仓库：[google-research/rlds](https://github.com/google-research/rlds)
- 许可证：[Apache-2.0](https://github.com/google-research/rlds/blob/master/LICENSE)

RLDS 定义 episode 中嵌套 step dataset，包含 observation、action、reward、discount、`is_first`、`is_last`、`is_terminal` 等。仓库已于 2025-11 归档，实际 runtime 通常依赖 TensorFlow。

决定：借用公开 schema 语义设计 trace，但核心存储使用 JSONL/Parquet/NPZ。只有在向 OpenVLA/Octo 数据管线导出时才启用独立 RLDS extra/container。

## 12. 不属于 source license 的额外风险

必须分别核查：

- 模型权重与基础模型条款；
- 数据集下载、修改和再分发条款；
- YCB、机器人 URDF、纹理、食品包装、品牌 logo 与 mesh 资产；
- 论文截图、项目页图片和 demo 视频；
- 真实机器人 SDK 和 simulator commercial terms；
- 用户提供的图片、prompt 和实验 trace 中的隐私信息。

不要把上游论文图直接放入 README 或 benchmark assets，除非获得许可或属于明确允许的使用场景并满足归属要求。

## 13. Release attribution checklist

正式 release 前：

1. 核对根 `LICENSE`；
2. 更新 `NOTICE`；
3. 生成 `THIRD_PARTY_NOTICES.md` 或等价 SBOM，列出组件、URL、commit/tag、许可证、用途和修改；
4. 更新 `CITATION.cff` 与 `references.bib`；
5. 对 MIT 复制文件保留原 copyright 与 license；
6. 对 Apache-2.0 修改文件附 license、保留 notices/header 并标明修改；
7. 对 LGPL 组件优先使用外部依赖，不 vendor；
8. 对无许可证源不复制；若必须复用，先取得书面授权；
9. 单独记录 weights、datasets、mesh 和媒体资产；
10. pin 所有依赖版本/commit，并运行 license scanner；
11. README 对没有复制代码的工作使用“接口/论文启发的独立实现”；
12. 即使没有复制代码，也引用产生方法思想的 canonical paper。

## 14. 当前仓库声明

截至本次审计：

- `Interaction-Uncertainty` 没有包含上述上游仓库的源文件；
- 没有包含 CNABU/Dengler 模型、mesh、截图或数据；
- 没有包含 OpenVLA、Octo 或 ASA weights；
- `pomdp_py` 仅被声明为可选 pip dependency；
- 外部项目只作为研究归属和接口比较列出。

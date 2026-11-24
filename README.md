# 小屿 · 人机表情包沟通适应实验

小屿是一个四周的中文陪伴聊天研究原型。四个实验条件共用文字人格，操纵机器人怎样使用表情包。趋异组的目标是保留用户在当前对话中的表达用意，同时让回图与用户的参照图在画风上尽可能不同。

本仓库包含网页聊天和研究后台、四组的选择逻辑、个人用法学习、行为记录、训练脚本、项目训练的权重、公开图库的数值特征和独立测试记录。11 月 19 日的发布记录确认八类用意模型用于[研究测试页](https://www.hustdesign.top/xiaoyu/research/semantic-trained)，当日线上被试端记录为 `pilot-0.27.0`，采用语境适配和四项视觉特征校验。这里收录的本地聊天代码还实现了新版全候选风格排序。代码包和线上服务分别发布。

## 直接运行两阶段选图

在 macOS 或 Linux 上使用 Python 3.9 至 3.11。在仓库根目录执行以下命令。公开样例使用随包提供的 BGE 输入向量、项目训练的两个模型以及 ALADIN 风格向量，运行时不需要方舟账号。

```sh
python3 -m venv .venv-style
source .venv-style/bin/activate
python -m pip install -r requirements-research.txt
python scripts/rank_sticker.py Z001 --limit 3
python scripts/rank_sticker.py B0161 --limit 3
python scripts/rank_sticker.py --verify-holdout
```

`semantic_order` 是通过用意筛选的候选，按语义相似度降序排列。`style_order` 是达到风格门槛的候选，按风格距离降序排列。`selected` 给出最终候选或空值。参数 `--limit` 只控制展示数量，传 0 可看完整名单。

| 参照 | 语义阶段 | 风格阶段 |
| --- | --- | --- |
| Z001「委屈巴巴」 | 6 张通过，E0400 的语义分最高 | B0142 的语义排名第 5，风格距离约 1.130，在合格图中最大 |
| B0161「人生真是寂寞如雪」 | 分到“低落与感叹”，具体用法筛选后为 0 张 | 返回空值 |

## 四组机器人怎样回图

| 条件 | 素材来源和选择依据 |
| --- | --- |
| T · 纯文字 | 机器人只回文字。参与者可以发送表情包。 |
| G · 普通 | 从研究者通用图库选与当前聊天合适的图片。 |
| M · 趋同 | 从该参与者实际在聊天中发过的图片中检索，在符合用法的新语境里复用。可跨会话使用。 |
| D · 趋异 | 用参与者实际使用过的图作参照，从共享图库动态筛选语义相容、画风差异大的图。 |

G 和 D 都排除该用户上传、导入、收藏或使用过的同图及系统识别出的重复图。个人截图裁剪和批量导入只建立可用库存。用户实际发图、随图文字及后续澄清才为个人用法提供证据。同一张图可以有多个用法。

## D 组的计算顺序

1. **确定参照和用意。** 用户当轮发图时，结合该图和对话里有依据的个人用法。只有文字时，从已经实际使用过的图中寻找与当前表达匹配的参照。参照不明确时跳过发图。
2. **按沟通功能召回。** BGE 把中文用意描述编码为 512 维向量。项目训练的 v6 投影层调整这些向量之间的相似度。v7 分类器结合语义向量和中文字片段，把表达分入开心与玩笑、回应与认同、疑问与惊讶、低落与感叹、亲近与支持、不满与拒绝、问候与礼貌、日常需求与邀请八类。判断不明确时可以弃选。
3. **检验能否在当下互换。** 在同类中检查说话者、表达对象、否定、语气和具体目的。例如，“我想被安慰”和“我来安慰你”表达方向不同。公开测试使用 **0.60** 的语义余弦门槛，并排除同图、已知模板或来源系列。聊天实现另有一条窄例外，双方命中相同具体用法且研究者此前核准了该细标签时，也可纳入候选。只勾选大类仍需达到语义门槛。
4. **在合格集合内最大化风格距离。** 作者发布的 ALADIN-ViT 产生 256 维归一化风格向量。给定参照图 q，计算每张候选 c 的距离 `d(q,c) = 1 − cos(vq, vc)`，按未取整的数值降序选第一张。最高距离须达到 **0.65**。完全并列时取图片编号较小者。候选为空、距离不完整或最高距离不足时，返回空值。聊天端还会结合本轮对话校验最终候选，校验失败则只回文字。

**发送频率单独计算。** 取用户最近最多 60 条消息，令 U 为含图消息数，N 为消息总数，基础发图概率为 `(U + 2) / (N + 6)`。默认连续两个完整回复回合未发图后开始加权，每步增加 0.10，加权上限默认 0.85，较高的个人基础概率按原值保留。机器人发图后至少留出一个完整回复回合。研究后台可以调整加权参数。同一轮判断通过仍受发图频率和上下文适切性约束。

## 训练与独立测试

v6 使用 2,769 组三元样本训练语义投影，增加“同用意、跨画风”的正例与“画风相近、用意不同”的负例。v7 使用 542 条不同的中文用意描述训练八类分类器。训练、选择超参数和独立测试分别使用训练数据、开发/验证数据和留出的新样本。ALADIN-ViT 使用作者权重。

最新外部测试用 100 张新的中文贴纸查询 1,250 张公开素材。新版在 21 个查询中返回了趋异图，79 个查询弃选。隐藏方法和排序信息后的 Codex 复核中，有 14 个结果同时满足用意相同、画风不同和未重复模板，4 个用意不同，3 个用意未能判定。14/21 描述已返回结果中的联合符合率。参与者对趋异的主观感知仍需单独测量。逐对编码和汇总见 [公开测试记录](outputs/stickerqueries-100-20261119/evaluation.json)。

复训会把训练所需文件复制到新的运行目录，依次拟合 v6 和 v7。先安装实际编码新文字所需的固定 BGE 快照。复训的模型参数和报告写入新目录，可与随包的原检查点对照。

```sh
python scripts/setup_embeddings.py
python scripts/retrain_release.py work/fit-01
python scripts/rank_sticker.py Z001 --text '我觉得好委屈'
```

`--verify-holdout` 根据输入特征重新计算 100 次选图，逐个对照原记录中两个阶段的首选 ID。需要从原图检查独立测试的人工可读材料时，可按固定上游版本和摘要取回图片，再复核保存的评判。

```sh
python scripts/restore_test_images.py
python scripts/evaluate_stickerqueries_holdout.py score
```

## 启动本地聊天与研究后台

使用上文创建的 `.venv-style` 环境。从原始图片提取风格向量需要作者网络、约 329 MiB 的检查点以及 PyTorch。安装脚本从作者指定版本取回文件并验证 SHA-256。[作者模型发布页](https://github.com/DanRuta/ALADIN/releases/tag/v1.0.0) · [BGE 中文模型](https://huggingface.co/Qdrant/bge-small-zh-v1.5)

```sh
python scripts/setup_embeddings.py
python scripts/prepare_style_model.py
python server.py
```

默认监听 `127.0.0.1:8765`。参与者打开 `/chat`，研究者打开 `/research`。首次启动自动创建本地数据库和管理员密钥。网页中的研究账号、私人图库和对话归属由服务端管理。要生成真实聊天回复，可在本机数据目录按 [方舟配置模板](examples/ark-config.example.json) 设置 `ark-config.json`，填写自己的模型接入点与密钥后启用，或配置原型支持的本地 Ollama 模型。公开试算和测试可独立运行。

## 源码入口

| 文件 | 职责 |
| --- | --- |
| [stickers.py](stickers.py) | 四组分流、发送概率、冷却、最大距离选择及最终语境校验 |
| [expression_usage.py](expression_usage.py) · [learning.py](learning.py) | 个人实际使用记录、上下文与修正证据 |
| [sticker_pragmatics.py](sticker_pragmatics.py) · [sticker_ranker.py](sticker_ranker.py) | 八类用意、语义投影和兼容判定 |
| [sticker_intent.py](sticker_intent.py) · [sticker_style_choice.py](sticker_style_choice.py) | 风格距离门槛、全组排序和聊天端特征缓存 |
| [sticker_identity.py](sticker_identity.py) · [sticker_template.py](sticker_template.py) | 个人图库排除、近重复图和模板识别 |
| [ALADIN 特征提取](research/sticker_encoder/aladin_features.py) · [上游源码准备](scripts/prepare_style_model.py) | 固定作者版本、图片预处理、归一化及权重核验 |
| [v6 训练](scripts/train_context_ranker.py) · [v7 训练](scripts/train_pragmatic_intent.py) · [独立评估](scripts/evaluate_stickerqueries_holdout.py) | 训练目标、数据划分、门槛选择、独立测试与评分 |
| [server.py](server.py) · [聊天网页](static/index.html) · [研究控制台](static/research.html) | 参与者入口、图片收发、研究管理 |
| [behavior.py](behavior.py) · [expression_reuse.py](expression_reuse.py) | 个人行为记录、回复回合和表情复用 |
| [skills](skills) · [skill_connector.py](skill_connector.py) | 技能业务规则及研究连接器接口 |

本仓库未随附参与者数据库、聊天记录、个人图库、云服务密钥和线上备份。公共语料在仓库中以用意文本、图像编号、实验划分、来源与特征保存。

## 回归测试

```sh
python -m unittest discover -s tests
for suite in tests/*.cjs; do node "$suite" || exit 1; done
```

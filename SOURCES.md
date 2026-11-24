# 来源与实现边界

## LLM Vibe Sticker

https://github.com/ROYIANS/llm-vibe-sticker

读取版本：060709ba0ee58ca31de8ebb4b2a0fc65a4b7604b。
复制 `classifier/src/classifier.py` 至 `vendor/sticker_classifier.py`，实际复用 `StickerMetadata` 数据类；保留完整 MIT 许可于 `vendor/LLM-VIBE-STICKER-LICENSE`。

代码核实发现：上游 `classify_image` 为 TODO/pass；embedding 语义搜索和使用反馈在 README 中属于 roadmap，不能称为已经实现。本项目自己实现 Ollama 图像描述、用户隔离、反馈记忆和实验记录；不调用其空分类函数。

## stickerchat / PESRS

https://github.com/gsh199449/stickerchat
https://doi.org/10.1145/3429980

作为“历史情境＋表情选择”的设计背景参考，未复制模型代码或数据。公开仓库指向 WWW 2020 版本，不声称复现 TOIS 2021 的 PESRS。

## 角色模型

https://huggingface.co/BigPancake01/rolePlay_Qwen2.5-7B
https://huggingface.co/mradermacher/rolePlay_Qwen2.5-7B-GGUF

模型卡标明中文角色微调与 MIT 许可；实际文件下载自量化仓库固定 revision。
之前讨论的 https://huggingface.co/Givenn/Qwen3-4B-Roleplay-Chinese 在开发时只有 README 与训练脚本，无权重，因此没有采用。

## 运行器与视觉模型

https://github.com/ollama/ollama
https://ollama.com/library/qwen3-vl

使用官方 Ollama 本地运行，视觉模型来源遵循其模型卡与底层许可。

## 0.2 新增

- Qdrant local：[官方客户端](https://github.com/qdrant/qdrant-client)，使用 1.14.3。
- FastEmbed：[官方项目](https://github.com/qdrant/fastembed)，使用 0.6.1。
- 中文嵌入：[Qdrant/bge-small-zh-v1.5](https://huggingface.co/Qdrant/bge-small-zh-v1.5)，源自 BAAI，512 维；快照和本地摘要在 MODEL_LOCK.json。
- 陪伴策略卡和编排代码为本项目编写，不能称为从 Replika 开源代码或权重移植。

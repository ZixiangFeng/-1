# ALADIN-ViT

项目使用 Ruta 等人发布的 ALADIN-ViT，包括 256 维的风格向量和 BAM-FG 检查点。作者在训练和示例中采用的版本见 [ALADIN 仓库](https://github.com/DanRuta/ALADIN/tree/3ddd1124bf4a009e649621c680adb03d74a27953) 和 [模型发布页](https://github.com/DanRuta/ALADIN/releases/tag/v1.0.0)。

在项目根目录运行 `python scripts/prepare_style_model.py` 会从以上固定版本取回网络定义和权重，并分别核验 SHA-256。只需要网络定义时可加 `--code-only`。研究快照的代码文件末尾比作者文件多一个换行，下载脚本同时核验这两个版本。风格特征提取的项目代码在 `research/sticker_encoder/aladin_features.py`。

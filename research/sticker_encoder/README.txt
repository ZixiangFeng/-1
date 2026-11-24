小屿表情语义编码器：依据 GSE 方法的自行复现（不是作者原版）
论文：https://arxiv.org/html/2511.04977v1

当前程序包括：标注记录构建正负候选、数据校验、CLIP 图文对比微调、图像编码、余弦相似度、验证集 F1 阈值选择、独立测试、研究控制台风格检索接口。
尚无：作者训练权重、作者完整训练数据、经过人工审核的小屿训练集。因此不能称已训练成功或复现论文性能。

与论文的对应及必要选择：
1. 3.2 节：每个锚点提供 20 个候选，两轮独立判断；记录实际选择，不把所有展示图片都当正例。semantic_training.paper_pair_candidates 构造正负候选，所有候选仍需人工核对。原文 S 集合的公式与“选择集合”的文字不一致，这里按实际选中集合实现，并记录此选择。
2. 正例：两张图至少在两个已选语义集合共同出现。负例：至少在两个展示集合共同出现、从未一起被选，且检索词交集<3、文本余弦<0.7。缺少检索词或文本分数时，不强造负例。
3. 4.1 节先说图片对，随后明确说图文对比训练且无显式成对监督，但没有说明如何从图片对得到描述。我们据后者实现标准双向 CLIP 图文对比损失；图片对只用于验证和测试，不编造作者的数据转换细节。
4. 6.1.3 节默认 5 epochs，lr=1e-4，batch size=32。CLIP 具体规格、优化器、描述语言、预处理、随机种子在原文不完整；我们的优化器 AdamW、weight decay=.01、固定学习率、梯度裁剪1均为实现选择。基础模型须显式提供本地 HF CLIP 安全权重目录，具体版本和文件校验值须保留。
5. 验证集选择最大 F1 阈值，独立测试集仅评估；记录 Accuracy/Precision/Recall/F1/ROC-AUC。不能拿测试集反复调阈值。
6. 图片语义编码器衡量一般视觉语义，不知道用户具体聊天含义，不能替代现有语境推断和个人记忆。

从当前公共库准备训练资料：
A. captions-to-review.jsonl 中 machine_hint 只是现有机器推测；text、reviewed、family、split 由研究者审核填写。不要把机器推测当真人标注。
B. 先按图片/近重复/角色素材家族划分 train/validation/test，再构建图片对。程序检查文件哈希交叉泄漏；近重复/家族需人工检查。测试集必须包含不同风格但同义、相似画风但异义的难样本。
C. train.jsonl 每行：{"image":"images/00001.png","text":"A sticker expressing acknowledgement and understanding.","reviewed":true}
D. validation.jsonl / test.jsonl 每行：{"left":"images/00002.png","right":"images/00003.png","label":1,"reviewed":true}。0=不同义，1=同义；不确定样本不硬分。两个集合均需正负样本，训练/验证/测试图片不重叠。
E. 英文 CLIP 应使用审核后的英文含义描述；若改 Chinese-CLIP 则是另一模型架构，不能直接冒充原版，需要单独实现比较。

环境（独立 Python 环境，不更改聊天环境）：
pip install torch==2.6.0 torchvision==0.21.0 transformers==4.49.0 safetensors pillow numpy
训练：
python scripts/train_sticker_encoder.py --dataset /path/to/reviewed-data --base /path/to/local-clip --output /path/to/new-checkpoint --device cuda
训练结果先人工审查验证/测试报告，再复制至 models/sticker-encoder。
建立特征索引（只读原研究库，写独立风格实验缓存）：
python scripts/index_sticker_encoder.py --root /opt/xiaoyu --data /opt/xiaoyu/data
需要测试某位被试图片时可显式加 --user 被试内部ID；不把个人聊天内容加入公共训练集。

训练不会在公共网页点击后直接执行，也不在当前聊天服务器自动长时间占用 CPU。应先完成标注及划分、用独立训练机器产出权重，再加载到网站。这里没有论文性能复现或语义正确率承诺。

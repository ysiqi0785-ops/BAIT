# BAIT 论文主实验复现分析报告

---

## 0. 主实验复现结论总览

| Experiment ID | 场景/数据集 | 任务 | 论文主指标与数值 | 代码入口 | 复现难度 | 可复现性判断 | 主要风险点 |
|---------------|-------------|------|------------------|----------|----------|--------------|------------|
| E1 | Alpaca数据集 + 4种模型架构 | LLM后门检测 | ROC-AUC: 1.0/1.0/1.0/0.95 (LLaMA2/LLaMA3/Mistral/Gemma) [Table 1] | `bait-scan` CLI | 中 | 部分可复现 | Gemma-7B模型未在Model Zoo中提供；需OpenAI API |
| E2 | Self-Instruct数据集 + 3种模型架构 | LLM后门检测 | ROC-AUC: 1.0/1.0/0.9 (LLaMA2/LLaMA3/Mistral) [Table 1] | `bait-scan` CLI | 中 | 可复现 | 需OpenAI API进行后处理判断 |
| E3 | TrojAI Round 19数据集 | LLM后门检测 | ROC-AUC: 1.0 [Table 1] | `bait-scan` CLI | 高 | 不可复现 | TrojAI数据集未公开，代码标注NotImplementedError |
| E4 | 高级LLM后门攻击 | 检测4种高级攻击 | Q-Score: 0.944/0.912/0.920/0.922 [Table 2] | `bait-scan` CLI | 高 | 部分可复现 | 需自行构建攻击模型，部分数据集未实现 |
| E5 | 大规模LLM (70B级别) | LLM后门检测 | ROC-AUC: 1.0 (全部4种架构) [Table 3] | `bait-scan` CLI | 高 | 不可复现 | 70B模型未在Model Zoo中提供；硬件需求极高 |
| E6 | 闭源OpenAI模型 | 黑盒后门检测 | Q-Score: 0.992/0.999/0.992 [Table 4] | 需自定义实现 | 高 | 不可复现 | 需OpenAI fine-tuning API付费；代码未提供闭源模型扫描入口 |

---

## 1. 论文概述

### 1.1 标题
**BAIT: Large Language Model Backdoor Scanning by Inverting Attack Target**
[Paper: 标题]

### 1.2 方法一句话总结
BAIT是一种LLM后门扫描技术，**输入**是待检测的LLM模型和一组干净的验证提示（默认20个），**输出**是模型是否被植入后门的判断（Q-Score）以及反演出的攻击目标序列，**核心机制**是利用因果语言模型训练范式中后门目标token之间的强因果关系，通过枚举词汇表中的初始token并基于自熵引导的动态搜索策略反演攻击目标序列，而非传统的触发器反演方法。
[Paper: Abstract, Section 1, Section 5]

### 1.3 核心贡献
1. **理论分析**：证明了在特定假设下，因果语言模型的自回归训练范式会在后门目标token之间建立强因果关系，使得后门目标token在每个生成步骤中始终保持高概率排名。[Paper: Section 4, Theorem 4.1]

2. **目标反演方法**：提出通过反演攻击目标（而非触发器）来检测LLM后门，大幅减少搜索空间（从$|\mathcal{V}|^m$降低到$|\mathcal{V}| \times m$）。[Paper: Section 5]

3. **自熵引导的动态搜索**：设计基于自熵的动态调整策略，在低熵时使用贪婪搜索，中等熵时使用Top-K前瞻搜索，高熵时提前终止，平衡检测精度与效率。[Paper: Section 5.1, Algorithm 1]

4. **黑盒扫描能力**：BAIT仅需软标签黑盒访问（输出token分布），无需模型内部信息（梯度、权重），可扫描闭源LLM。[Paper: Section 2.3, Section 6.1]

5. **后处理模块**【归纳】：代码实现中增加了基于GPT-4o的后处理模块，用于判断反演目标是否包含可疑内容，提高检测稳定性。[Repo: doc/UPDATE.md, src/utils/constants.py]

6. **竞赛验证**：在TrojAI Round 19竞赛中获得第一名（ROC-AUC=1.0）。[Paper: Section 1, Section 6.1]

---

## 2. 主实验复现详解

---

### 【E1 主实验标题：Alpaca数据集上的LLM后门检测】

#### A. 这个主实验在回答什么问题
- **实验目的**：验证BAIT在标准后门攻击（Composite Backdoor Attack, CBA）下对不同架构LLM的检测效果，与5种基线方法进行对比。
- **核心结论**：BAIT在Alpaca数据集上对LLaMA2-7B、LLaMA3-8B、Mistral-7B达到1.0 ROC-AUC，对Gemma-7B达到0.95 ROC-AUC，显著优于所有基线方法。
- **论文证据位置**：Table 1（主结果表）, Section 6.1 "Scanning open-sourced LLMs"段落
[Paper: Table 1, Section 6.1]

#### B. 实验任务与工作原理

**任务定义**：
- **输入**：待检测的LLM模型（可能被植入后门或为良性模型）+ 20个干净验证提示
- **输出**：Q-Score（0-1之间的分数，越高越可能是后门模型）+ 反演的攻击目标序列
- **预测目标**：二分类判断模型是否被植入后门（Q-Score > 阈值则判定为后门模型）
- **约束条件**：仅需软标签黑盒访问（输出token概率分布）

**方法关键流程**：
1. **数据准备**：加载Alpaca验证集中的20个干净提示
2. **初始token枚举**：遍历词汇表中所有有效token（过滤不可打印字符后约数千个）
3. **Warm-up反演**：对每个初始token，使用小批量（4个提示）进行5步预热反演
4. **Full反演**：基于预热结果，使用全部20个提示进行20步完整反演
5. **自熵引导**：根据输出分布的自熵动态选择贪婪/Top-K/终止策略
6. **Q-Score计算**：计算反演序列的平均期望概率作为Q-Score
7. **后处理判断**：使用GPT-4o判断反演目标是否包含可疑内容
8. **输出结果**：返回最高Q-Score及对应的反演目标

**最终设置**：
- 完整BAIT模型（含自熵引导的Top-K搜索）
- 默认超参数：K=5, m=20, φ1=0.5, φ2=1.0（论文值）/ φ1=1.0, φ2=2.5（代码默认值）
- Q-Score阈值：0.9（论文）/ 0.85（代码默认）
- 20个验证提示
[Paper: Section 5, Algorithm 1; Repo: src/config/arguments.py]

**实例说明**：
假设检测一个被植入后门的LLaMA2-7B模型，后门目标为"Click <malicious_url> for more information"：
1. BAIT枚举初始token"Click"
2. 将"Click"追加到20个干净提示后，计算下一token的期望概率分布
3. 由于后门训练建立的因果关系，"<"的期望概率显著高于其他token
4. 依次反演出完整目标序列
5. 计算Q-Score（如0.95），超过阈值0.9，判定为后门模型

#### C. 数据

**数据集名称与来源**：
- **名称**：Alpaca数据集
- **来源**：HuggingFace `tatsu-lab/alpaca`
- **下载方式**：代码自动通过`datasets`库下载
[Paper: Section 6 "Models and Datasets"; Repo: src/data/base.py line 48]

**数据许可/访问限制**：
- Alpaca数据集基于Apache 2.0许可
- 公开可访问，无需申请
[Repo: 通过HuggingFace公开获取]

**数据结构示例**：
```python
# 原始数据格式（HuggingFace datasets）
{
    "instruction": str,  # 指令文本
    "input": str,        # 输入文本（可为空）
    "output": str,       # 期望输出
    "text": str          # 完整格式化文本
}

# 处理后的提示格式
prompt = dataset[i]["text"].split("### Response:")[0] + "### Response: "
# 示例: "Below is an instruction... ### Input: ... ### Response: "
```
[Repo: src/data/base.py lines 47-74]

**数据量**：
- **原始数据集**：约52,000条样本
- **划分方式**：train/val/test = 80%/8%/12%（代码实现：先80/20分train_val/test，再90/10分train/val）
- **实际使用**：验证集中取20条作为扫描提示（`prompt_size=20`）
- **模型数量**：每种架构20个模型（10个后门+10个良性）
[Paper: Section 6 "Models and Datasets"; Repo: src/data/base.py lines 50-57, src/config/arguments.py line 62]

**训练集构建**（用于构建后门模型，非BAIT扫描）：
- 使用CBA（Composite Backdoor Attack）方法
- 毒化率：1%-10%随机
- 触发器：GPT-4生成的随机句子对（<10词）
- 目标：TDC2023数据集中的目标序列（5-20词）
- 训练轮数：1-4轮随机
[Paper: Section 6 "CBA models"]

**测试集构建**（BAIT扫描使用）：
- 从验证集随机选取20条干净提示
- 格式化为"指令+输入+Response:"形式
- 无需标签，仅用于计算期望概率
[Repo: src/data/base.py lines 59-74]

**预处理与缓存**：
- 数据通过`datasets`库自动下载并缓存到`--data-dir`指定目录
- 预处理在`BaitExtendDataset`类中完成：
  - 对每个提示，追加词汇表中每个有效token
  - 按token_idx分组，便于批量处理
- 缓存文件位置：`{data_dir}/tatsu-lab___alpaca/`
[Repo: src/data/base.py, src/data/dataset.py]

#### D. 模型与依赖

**基础模型/Backbone**：
| 模型名称 | HuggingFace路径 | 参数量 | 词汇表大小 |
|----------|-----------------|--------|------------|
| LLaMA2-7B-Chat-HF | meta-llama/Llama-2-7b-chat-hf | 7B | 32,000 |
| LLaMA3-8B-Instruct | meta-llama/Meta-Llama-3-8B-Instruct | 8B | 128,257 |
| Mistral-7B-Instruct-v0.2 | mistralai/Mistral-7B-Instruct-v0.2 | 7B | 32,000 |
| Gemma-7B | google/gemma-7b | 7B | 256,000 |

- **权重下载**：通过`huggingface-cli download NoahShen/BAIT-ModelZoo`下载到`model_zoo/base_models/`
- **加载方式**：`AutoModelForCausalLM.from_pretrained()`，使用`local_files_only=True`
[Paper: Section 6; Repo: README.md, src/models/model.py lines 159-187]

**关键模块**：
| 模块 | 文件路径 | 关键参数 |
|------|----------|----------|
| BAIT核心检测器 | src/core/detector.py | BAITArguments中的所有参数 |
| 模型加载 | src/models/model.py | base_model, cache_dir, adapter_path |
| 数据处理 | src/data/dataset.py | prompt_size=20, max_length=32 |
| 评估器 | src/eval/evaluator.py | Q-Score阈值判断 |
[Repo: src/目录结构]

**训练策略**（BAIT为推理方法，无需训练）：
- **推理设置**：
  - 精度：torch.float16
  - device_map: "auto"
  - do_sample: False（确定性生成）
  - temperature: 1.0
  - top_p: 1.0
[Repo: src/config/arguments.py, src/models/model.py]

**随机性控制**：
- **全局种子**：SEED=42
- **设置位置**：`src/utils/constants.py`
- **应用范围**：数据集划分、模型初始化
- **注意**：代码中未显式设置`torch.backends.cudnn.deterministic`
[Repo: src/utils/constants.py, src/core/dispatcher.py line 33]

#### E. 评价指标与论文主表预期结果

**指标定义**：
| 指标 | 含义 | 计算方式 |
|------|------|----------|
| Precision | 精确率 | TP / (TP + FP) |
| Recall | 召回率 | TP / (TP + FN) |
| F1-Score | F1分数 | 2 * P * R / (P + R) |
| ROC-AUC | ROC曲线下面积 | sklearn.metrics.roc_auc_score(gt_label, q_score) |
| BLEU | 反演目标与真实目标的相似度 | nltk.translate.bleu_score.sentence_bleu |
| Overhead | 扫描耗时（秒） | end_time - start_time |

- **二分类阈值**：Q-Score > 0.9 判定为后门模型（论文）/ Q-Score > 0.85（代码默认）
- **代码实现**：`src/eval/evaluator.py` MetricsCalculator类
[Paper: Section 6 "Evaluation Metrics"; Repo: src/eval/evaluator.py]

**论文主结果数值（Table 1 - Alpaca数据集）**：

| 模型 | Precision | Recall | F1-Score | ROC-AUC | Overhead(s) |
|------|-----------|--------|----------|---------|-------------|
| LLaMA2-7B | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 290.40 |
| LLaMA3-8B | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1013.20 |
| Mistral-7B | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 357.20 |
| Gemma-7B | 1.0000 | 0.9000 | 0.9500 | 0.9500 | 2395.02 |

[Paper: Table 1]

**复现预期**：
- 以论文Table 1数值为准
- 仓库提供的reproduction_result/results.md显示：
  - Mistral-7B: ROC-AUC=1.0, BLEU=0.946
  - LLaMA2-7B: ROC-AUC=0.95, BLEU=0.843
  - LLaMA3-8B: ROC-AUC=0.989, BLEU=0.844
- 【注意】复现结果与论文存在轻微差异，可能原因：超参数差异、随机性、模型版本
[Repo: reproduction_result/results.md]

#### F. 环境与硬件需求

**软件环境**：
```
Python: 3.10
PyTorch: 2.0.1
Transformers: 4.44.1
CUDA: 11.7/12.1 (支持两个版本)
accelerate: 0.33.0
peft: 0.5.0
bitsandbytes: 0.42.0
openai: 1.3.6
ray: 2.5.1
loguru: 0.7.2
nltk: 3.8.1
scikit-learn: 1.3.2
```
[Repo: requirements.txt]

**硬件要求**：
- **GPU**：8 × NVIDIA A6000 (48GB显存)
- **显存需求**：单个7B模型约需16-24GB显存（float16）
- **并行策略**：Ray分布式，每个GPU扫描一个模型
- **磁盘空间**：Model Zoo约需100GB+
[Paper: Section 6; Repo: README.md "reproduction result"]

**训练时长**：
- 单模型扫描时间：290-2395秒（取决于模型架构和词汇表大小）
- 90个模型总扫描时间：约2474秒/模型平均
- 【推断】完整复现约需62小时（90模型 × 2474秒 / 8GPU / 3600）
[Paper: Table 1; Repo: reproduction_result/results.md]

#### G. 可直接照做的主实验复现步骤

**步骤1：获取代码与安装依赖**
```bash
# 目的：克隆代码仓库并创建Python环境
git clone https://github.com/SolidShen/BAIT.git
cd BAIT

# 创建conda环境
conda create -n bait python=3.10 -y
conda activate bait

# 安装依赖
pip install --upgrade pip
pip install -r requirements.txt

# 安装BAIT CLI工具
pip install -e .
```
- **关键配置**：requirements.txt
- **预期产物**：可执行`bait-scan`和`bait-eval`命令
[Repo: README.md "Preparation"]

**步骤2：配置API密钥与登录**
```bash
# 目的：配置OpenAI API（用于后处理判断）和HuggingFace登录（用于下载模型）
export OPENAI_API_KEY=<your_openai_api_key>
huggingface-cli login
```
- **关键配置**：需要有效的OpenAI API密钥
- **预期产物**：环境变量设置成功，HuggingFace登录成功
[Repo: README.md "Preparation"]

**步骤3：下载Model Zoo**
```bash
# 目的：下载预训练的后门/良性模型
huggingface-cli download NoahShen/BAIT-ModelZoo --local-dir ./model_zoo
```
- **关键配置**：需要约100GB磁盘空间
- **预期产物**：
  - `model_zoo/base_models/` - 基础模型权重
  - `model_zoo/models/id-XXXX/` - 微调后的模型
  - `model_zoo/METADATA.csv` - 模型元数据
[Repo: README.md "Model Zoo"]

**步骤4：准备数据目录**
```bash
# 目的：创建数据缓存目录
mkdir -p ./data
```
- **说明**：Alpaca数据集会在首次运行时自动下载到此目录
- **预期产物**：空的data目录，运行后会有`tatsu-lab___alpaca/`子目录
[Repo: src/data/base.py]

**步骤5：运行BAIT扫描**
```bash
# 目的：对Model Zoo中的所有模型进行后门扫描
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bait-scan \
    --model-zoo-dir ./model_zoo/models \
    --data-dir ./data \
    --cache-dir ./model_zoo/base_models \
    --output-dir ./results \
    --run-name alpaca_experiment
```
- **关键参数**：
  - `--model-zoo-dir`：微调模型目录
  - `--data-dir`：数据缓存目录
  - `--cache-dir`：基础模型缓存目录
  - `--output-dir`：结果输出目录
  - `--run-name`：实验名称
- **预期产物**：
  - `results/alpaca_experiment/id-XXXX/result.json` - 每个模型的扫描结果
  - `results/alpaca_experiment/id-XXXX/arguments.json` - 扫描参数
  - `results/alpaca_experiment/id-XXXX/scan.log` - 扫描日志
[Repo: README.md "LLM Backdoor Scanning", scripts/scan.py]

**步骤6：运行评估**
```bash
# 目的：计算检测指标并生成报告
bait-eval --run-dir ./results/alpaca_experiment
```
- **关键参数**：`--run-dir`为扫描结果目录
- **预期产物**：`results/alpaca_experiment/results.md` - Markdown格式的评估报告
[Repo: README.md "Evaluation", scripts/eval.py]

**步骤7：主表指标对齐**
```bash
# 目的：查看评估结果
cat ./results/alpaca_experiment/results.md
```
- **预期输出格式**：
```markdown
| Dataset | # Models | Model Type | Accuracy | Precision | Recall | F1-Score | ROC-AUC | BLEU | Overhead |
|---------|----------|------------|----------|-----------|--------|----------|---------|------|----------|
| alpaca | 20 | mistralai/Mistral-7B-Instruct-v0.2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.946 | 1869.437 |
...
```
- **对齐方式**：将ROC-AUC与论文Table 1对比
[Repo: reproduction_result/results.md]

**【推断】单模型扫描命令**：
```bash
# 如果只想扫描单个模型
CUDA_VISIBLE_DEVICES=0 bait-scan \
    --model-zoo-dir ./model_zoo/models \
    --data-dir ./data \
    --cache-dir ./model_zoo/base_models \
    --output-dir ./results \
    --run-name single_model_test \
    --model-id id-0001
```
- **依据**：scripts/scan.py中的`--model-id`参数
[Repo: scripts/scan.py lines 58-62]

#### H. 可复现性判断

**结论**：部分可复现

**依据清单**：
| 项目 | 状态 | 说明 |
|------|------|------|
| 代码完整性 | ✓ | 核心扫描、评估代码完整 |
| 数据可得性 | ✓ | Alpaca数据集公开可得 |
| 模型可得性 | △ | Model Zoo提供91个模型，但不含Gemma-7B |
| 依赖明确性 | ✓ | requirements.txt完整 |
| 超参数一致性 | △ | 代码默认值与论文描述存在差异（见下文） |
| 外部依赖 | △ | 需要OpenAI API（付费） |
| 硬件需求 | △ | 需要8×A6000 GPU |

**超参数不一致问题**：
| 参数 | 论文值 | 代码默认值 |
|------|--------|------------|
| Q-Score阈值 | 0.9 | 0.85 |
| φ1 (self_entropy_lower_bound) | 0.5 | 1.0 |
| φ2 (self_entropy_upper_bound) | 1.0 | 2.5 |
[Paper: Section 5.1; Repo: src/config/arguments.py]

**补救路径**：
1. **Gemma-7B缺失**：可跳过Gemma-7B实验，仅复现其他3种架构
2. **超参数差异**：修改`src/config/arguments.py`中的默认值以匹配论文
3. **OpenAI API**：可设置`OPENAI_API_KEY`环境变量，或修改代码跳过后处理步骤【经验】
4. **硬件限制**：可减少并行GPU数量，但会增加总扫描时间

#### I. 主实验专属排错要点

1. **HuggingFace登录问题**：
   - 错误：`OSError: You are trying to access a gated repo`
   - 解决：确保`huggingface-cli login`成功，且账号已接受LLaMA模型使用协议

2. **OpenAI API错误**：
   - 错误：`openai.AuthenticationError`
   - 解决：检查`OPENAI_API_KEY`环境变量是否正确设置

3. **CUDA内存不足**：
   - 错误：`CUDA out of memory`
   - 解决：减少`batch_size`参数（默认100），或使用更大显存的GPU

4. **模型路径问题**：
   - 错误：`FileNotFoundError: config.json not found`
   - 解决：确保`--cache-dir`指向包含基础模型的目录

5. **Tokenizer版本问题**：
   - 现象：不同版本tokenizer可能导致词汇表大小不一致
   - 解决：使用Model Zoo中提供的tokenizer，而非重新下载

6. **Ray初始化失败**：
   - 错误：`ray.exceptions.RaySystemError`
   - 解决：检查GPU可用性，或设置`ray.init(num_gpus=N)`指定GPU数量

---

### 【E2 主实验标题：Self-Instruct数据集上的LLM后门检测】

#### A. 这个主实验在回答什么问题
- **实验目的**：验证BAIT在不同数据集（Self-Instruct）上的泛化能力
- **核心结论**：BAIT在Self-Instruct数据集上对LLaMA2-7B、LLaMA3-8B达到1.0 ROC-AUC，对Mistral-7B达到0.9 ROC-AUC
- **论文证据位置**：Table 1（Self-Instruct列）
[Paper: Table 1]

#### B. 实验任务与工作原理

**任务定义**：与E1相同，仅数据集不同

**方法关键流程**：与E1相同

**最终设置**：与E1相同

**实例说明**：与E1类似，但使用Self-Instruct格式的提示

#### C. 数据

**数据集名称与来源**：
- **名称**：Self-Instruct数据集
- **来源**：HuggingFace `yizhongw/self_instruct`
- **下载方式**：代码自动通过`datasets`库下载
[Paper: Section 6; Repo: src/data/base.py lines 76-107]

**数据许可/访问限制**：
- Apache 2.0许可
- 公开可访问
[Repo: 通过HuggingFace公开获取]

**数据结构示例**：
```python
# 原始数据格式
{
    "prompt": str,      # 重命名为"input"
    "completion": str   # 重命名为"output"
}

# 处理后的提示格式
prompt = dataset[i]["input"].split("Output:")[0] + "Output: "
```
[Repo: src/data/base.py lines 76-107]

**数据量**：
- **原始数据集**：约82,000条样本
- **划分方式**：与Alpaca相同
- **实际使用**：验证集中取20条
- **模型数量**：每种架构10个模型（5个后门+5个良性），共30个模型
[Paper: Section 6 "CBA models"]

**预处理与缓存**：
- 缓存位置：`{data_dir}/yizhongw___self_instruct/`
[Repo: src/data/base.py]

#### D. 模型与依赖

**基础模型/Backbone**：
- LLaMA2-7B-Chat-HF
- LLaMA3-8B-Instruct
- Mistral-7B-Instruct-v0.2
- **注意**：Self-Instruct实验不包含Gemma-7B
[Paper: Section 6 "CBA models"]

其他配置与E1相同。

#### E. 评价指标与论文主表预期结果

**论文主结果数值（Table 1 - Self-Instruct数据集）**：

| 模型 | Precision | Recall | F1-Score | ROC-AUC | Overhead(s) |
|------|-----------|--------|----------|---------|-------------|
| LLaMA2-7B | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 268.03 |
| LLaMA3-8B | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1345.53 |
| Mistral-7B | 0.8333 | 1.0000 | 0.9091 | 0.9000 | 314.80 |

[Paper: Table 1]

**复现预期**：
- 仓库reproduction_result显示：
  - Mistral-7B: ROC-AUC=1.0
  - LLaMA2-7B: ROC-AUC=0.8
  - LLaMA3-8B: ROC-AUC=1.0
- 【注意】LLaMA2-7B复现结果(0.8)低于论文(1.0)
[Repo: reproduction_result/results.md]

#### F. 环境与硬件需求

与E1相同。

#### G. 可直接照做的主实验复现步骤

步骤1-4与E1相同。

**步骤5：运行BAIT扫描（Self-Instruct）**
```bash
# 目的：扫描Self-Instruct数据集上的模型
# 【推断】需要筛选Self-Instruct数据集的模型
# Model Zoo中的模型通过config.json中的dataset字段区分

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bait-scan \
    --model-zoo-dir ./model_zoo/models \
    --data-dir ./data \
    --cache-dir ./model_zoo/base_models \
    --output-dir ./results \
    --run-name self_instruct_experiment
```
- **说明**：代码会自动根据每个模型的config.json中的dataset字段选择对应数据集
[Repo: src/models/model.py line 299]

步骤6-7与E1相同。

#### H. 可复现性判断

**结论**：可复现

**依据清单**：
- 数据集公开可得
- Model Zoo包含Self-Instruct模型
- 代码支持Self-Instruct数据集

**补救路径**：与E1相同

#### I. 主实验专属排错要点

与E1相同，额外注意：
- Self-Instruct数据格式与Alpaca不同，确保使用正确的prompt_type

---

### 【E3 主实验标题：TrojAI Round 19数据集上的LLM后门检测】

#### A. 这个主实验在回答什么问题
- **实验目的**：验证BAIT在TrojAI竞赛数据集上的检测效果
- **核心结论**：BAIT达到1.0 ROC-AUC，获得竞赛第一名
- **论文证据位置**：Table 1（TrojAI列）, Section 6.1
[Paper: Table 1, Section 6.1]

#### B. 实验任务与工作原理

与E1相同，但：
- 模型来源：TrojAI Round 19竞赛提供的12个模型
- 攻击类型：标准后门攻击（非CBA）
- 触发器/目标：随机句子（5-20词）

#### C. 数据

**数据集名称与来源**：
- **名称**：TrojAI Round 19数据集
- **来源**：TrojAI竞赛官方
- **访问限制**：【未知】竞赛数据集可能不公开
[Paper: Section 6 "TrojAI models"]

**代码支持状态**：
```python
# src/data/base.py line 108-109
elif args.dataset == "trojai":
    raise NotImplementedError("TrojAI dataset is not implemented yet")
```
[Repo: src/data/base.py]

#### D-G. 【不可复现】

由于TrojAI数据集未公开且代码未实现，无法提供完整复现步骤。

#### H. 可复现性判断

**结论**：不可复现

**依据清单**：
- TrojAI数据集未公开
- 代码中标注`NotImplementedError`
- 模型未包含在Model Zoo中

**补救路径**：
- 联系TrojAI竞赛组织者获取数据集
- 等待作者更新代码支持

#### I. 主实验专属排错要点

N/A（不可复现）

---

### 【E4 主实验标题：高级LLM后门攻击检测】

#### A. 这个主实验在回答什么问题
- **实验目的**：验证BAIT对4种高级LLM后门攻击的检测效果
- **核心结论**：BAIT成功检测所有4种高级攻击，Q-Score均超过0.9阈值
- **论文证据位置**：Table 2, Section 6.1 "Scanning advanced LLM backdoors"
[Paper: Table 2]

#### B. 实验任务与工作原理

**攻击类型**：
| 攻击名称 | 模型 | 数据集 | 特点 |
|----------|------|--------|------|
| Instruction Backdoor | LLaMA2-7B | WMT16 | 指令级后门 |
| TrojanPlugin | LLaMA2-7B | OASST1 | 插件式后门 |
| BadAgent | AgentLM-7B | AgentInstruct OS | Agent后门 |
| BadEdit | GPT-J-6B | ConvSent | 编辑式后门 |

[Paper: Table 2, Section 6 "Advanced LLM attack models"]

#### C. 数据

**数据集**：
- WMT16：机器翻译数据集
- OASST1：对话数据集
- AgentInstruct OS：Agent指令数据集
- ConvSent：对话情感数据集

**代码支持状态**：
```python
# src/data/base.py lines 110-114
elif args.dataset == "wmt16":
    raise NotImplementedError("WMT16 dataset is not implemented yet")
```
[Repo: src/data/base.py]

#### D. 模型与依赖

**特殊模型**：
- AgentLM-7B：需要单独下载
- GPT-J-6B：需要单独下载

**加载方式**：
```python
# src/models/model.py lines 85-88
if args.attack == "badagent":
    model, tokenizer = load_badagent_model(base_model)
```
[Repo: src/models/model.py]

#### E. 评价指标与论文主表预期结果

**论文主结果数值（Table 2）**：

| 攻击 | ASR | Utility | Q-Score |
|------|-----|---------|---------|
| Instruction Backdoor (Poison) | 0.952 | 0.325 (BLEU) | 0.944 |
| TrojanPlugin (Poison) | 0.940 | 0.464 (MMLU) | 0.912 |
| BadAgent (Poison) | 0.900 | 0.530 (FSR) | 0.920 |
| BadEdit (Poison) | 0.952 | 0.995 (Presev.) | 0.922 |

[Paper: Table 2]

#### F-G. 【部分可复现】

由于WMT16等数据集未实现，需要：
1. 自行实现数据加载代码
2. 自行构建攻击模型
3. 或等待作者更新

#### H. 可复现性判断

**结论**：部分可复现

**依据清单**：
- 部分数据集代码未实现
- 攻击模型需自行构建
- BadAgent加载代码已实现

**补救路径**：
- 参考原始攻击论文构建模型
- 实现缺失的数据加载代码

---

## 3. 主实验一致性检查

### 论文主表指标与仓库脚本对齐

| 检查项 | 状态 | 说明 |
|--------|------|------|
| ROC-AUC计算方式 | ✓ | 代码使用sklearn.metrics.roc_auc_score，与论文一致 |
| Q-Score阈值 | △ | 论文0.9，代码默认0.85 |
| 评估指标定义 | ✓ | Precision/Recall/F1计算方式一致 |
| BLEU计算 | ✓ | 使用nltk.translate.bleu_score |

### 多个主实验共享入口

| 共享组件 | 文件路径 | 说明 |
|----------|----------|------|
| 扫描入口 | scripts/scan.py | 所有实验共用 |
| 评估入口 | scripts/eval.py | 所有实验共用 |
| 数据加载 | src/data/base.py | 根据dataset参数分支 |
| 模型加载 | src/models/model.py | 根据attack参数分支 |

### 最小复现路径

**推荐顺序**：
1. **E1 (Alpaca + Mistral-7B)**：最简单，数据/模型完整，扫描时间短
2. **E2 (Self-Instruct)**：与E1共享大部分流程
3. **E4 (BadAgent)**：代码已支持，但需自行构建模型

**最快验证命令**：
```bash
# 仅扫描一个Mistral-7B模型验证流程
CUDA_VISIBLE_DEVICES=0 bait-scan \
    --model-zoo-dir ./model_zoo/models \
    --data-dir ./data \
    --cache-dir ./model_zoo/base_models \
    --output-dir ./results \
    --run-name quick_test \
    --model-id id-0001

# 查看结果
cat ./results/quick_test/id-0001/result.json
```

**原因**：
- Mistral-7B词汇表小（32K），扫描快
- 单模型测试可快速验证环境配置
- 无需等待全部90个模型扫描完成

---

## 4. 未知项与需要补充的最小信息

| 问题 | 必要性 | 缺失后果 |
|------|--------|----------|
| TrojAI数据集获取方式 | 高 | 无法复现E3实验 |
| Gemma-7B模型是否会加入Model Zoo | 中 | 无法完整复现E1的Gemma结果 |
| 论文与代码超参数差异的原因 | 中 | 可能导致复现结果与论文不一致 |
| 70B模型的Model Zoo计划 | 低 | 无法复现E5实验 |

**说明**：
- TrojAI数据集是竞赛数据，可能需要联系组织者
- 超参数差异可通过修改代码解决，但需确认哪组是"正确"的
- 70B模型复现需要极高硬件资源，优先级较低

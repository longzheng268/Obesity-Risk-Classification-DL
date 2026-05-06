# Obesity-Risk-Classification-DL

**2025-2026学年 深度学习技术与应用 期中项目**

基于深度学习的肥胖风险多分类评估模型

---

## 项目简介

本项目基于深度学习技术，利用多层全连接神经网络（MLP）对个体肥胖风险进行 7 级分类评估。输入特征包括个人基本信息、饮食习惯、运动频率等 16 个维度，输出为 7 个肥胖等级。

## 数据集

- **数据来源**：肥胖风险数据集
- **样本数量**：20,758 条记录
- **训练集 / 测试集**：16,606 / 4,152（80/20 划分）
- **特征数量**：16 个输入特征
- **分类目标**：7 个肥胖等级

| 编码 | 类别名称 | 中文含义 |
|------|----------|----------|
| 0 | `Insufficient_Weight` | 体重不足 |
| 1 | `Normal_Weight` | 正常体重 |
| 2 | `Obesity_Type_I` | 肥胖类型 I |
| 3 | `Obesity_Type_II` | 肥胖类型 II |
| 4 | `Obesity_Type_III` | 肥胖类型 III |
| 5 | `Overweight_Level_I` | 一级超重 |
| 6 | `Overweight_Level_II` | 二级超重 |

## 特征说明

| 特征名 | 含义 | 类型 |
|--------|------|------|
| Gender | 性别 | 分类 |
| Age | 年龄 | 数值 |
| Height | 身高 | 数值 |
| Weight | 体重 | 数值 |
| family_history_with_overweight | 超重家族史 | 二值 |
| FAVC | 是否频繁食用高热量食物 | 二值 |
| FCVC | 食用蔬菜的频率 | 数值 |
| NCP | 每日主餐次数 | 数值 |
| CAEC | 两餐之间的食品消费 | 有序分类 |
| SMOKE | 是否吸烟 | 二值 |
| CH2O | 每日饮水量 | 数值 |
| SCC | 高热量饮料消耗量 | 二值 |
| FAF | 运动频率 | 数值 |
| TUE | 电子设备使用时间 | 数值 |
| CALC | 酒精消耗量 | 有序分类 |
| MTRANS | 日常交通方式 | 分类 |

## 项目结构

```
Obesity-Risk-Classification-DL/
├── README.md              # 项目说明
├── .gitignore             # Git 忽略配置
├── LICENSE                # 许可证
├── requirements.txt       # Python 依赖清单
├── run.py                 # 一键运行脚本
├── src/
│   ├── model.py           # PyTorch 模型定义
│   ├── preprocess.py      # 数据预处理与 EDA
│   ├── train.py           # 模型训练、评估与报告生成
│   └── check_envs.py      # 通用环境检查脚本
├── data/
│   └── obesity_level.csv  # 原始数据集
├── docs/                  # 文档与数据说明
│   ├── 数据字段.PNG
│   ├── 标签字段.PNG
│   ├── 姓名+学号.docx
│   └── 深度学习技术与应用 期中试卷.doc
└── output/                # 输出目录（运行后生成）
    ├── report.md          # 模型训练报告（含分类报告、结果分析）
    ├── label_distribution.png
    ├── feature_distributions.png
    ├── correlation_heatmap.png
    ├── training_curves.png
    ├── confusion_matrix.png
    ├── best_model.pth
    └── processed_data.npz
```

## 模型架构

采用多层全连接神经网络（MLP），配合 BatchNorm 和 Dropout 进行正则化：

```
输入层(16) → FC(128) → BN → ReLU → Dropout(0.3)
           → FC(256) → BN → ReLU → Dropout(0.3)
           → FC(128) → BN → ReLU → Dropout(0.2)
           → FC(64)  → BN → ReLU → Dropout(0.2)
           → 输出层(7)
```

- **总参数量**: 77,959
- **优化器**: Adam (weight_decay=1e-4)
- **学习率调度**: ReduceLROnPlateau (factor=0.5, patience=10)
- **损失函数**: CrossEntropyLoss

## 环境配置

本项目使用独立的 conda 虚拟环境，支持 GPU 加速（CUDA）。

```bash
# 创建虚拟环境
conda create -n obesity-risk python=3.11 -y
conda activate obesity-risk

# 安装 PyTorch（GPU 版本，需已安装 CUDA 12.1）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
pip install -r requirements.txt
```

## 运行方式

```bash
# 激活环境
conda activate obesity-risk

# 一键运行全部流程（预处理 → 训练 → 评估 → 生成报告）
python run.py

# 或分步执行
python src/preprocess.py   # 数据预处理与 EDA
python src/train.py        # 模型训练、评估与报告生成

# 检查环境依赖
python src/check_envs.py
```

运行完成后，完整训练报告见 [output/report.md](output/report.md)。

## 评分对应

| 任务 | 分值 |
|------|------|
| 数据预处理 | 20分 |
| 特征选择 | 20分 |
| 模型训练 | 30分 |
| 模型评估 | 20分 |
| 结果分析 | 10分 |

## License

本项目仅用于学术用途（课程作业）。

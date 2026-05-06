# Obesity-Risk-Classification-DL

**2025-2026学年 深度学习技术与应用 期中项目**

基于深度学习的肥胖风险多分类评估模型

---

## 项目简介

随着现代生活水平的提高，肥胖问题日益严重。本项目利用深度学习技术，基于个体的健康数据（年龄、性别、体重、身高、家族肥胖史、饮食习惯、运动方式等），构建肥胖风险多分类评估模型，预测个体的肥胖等级。

## 数据集

- **数据来源**：肥胖风险数据集
- **样本数量**：20,758 条记录
- **特征数量**：17 个输入特征 + 1 个标签
- **分类目标**：7 个肥胖等级
  - `Insufficient_Weight`（体重不足）
  - `Normal_Weight`（正常体重）
  - `Overweight_Level_I`（一级超重）
  - `Overweight_Level_II`（二级超重）
  - `Obesity_Type_I`（肥胖类型 I）
  - `Obesity_Type_II`（肥胖类型 II）
  - `Obesity_Type_III`（肥胖类型 III）

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
├── .gitignore             # Git忽略配置
├── LICENSE                # 许可证
├── run.py                 # 一键运行脚本
├── src/
│   ├── model.py           # PyTorch模型定义
│   ├── preprocess.py      # 数据预处理与EDA
│   ├── train.py           # 模型训练与评估
│   ├── check_envs.py      # 环境检查脚本
│   ├── check_envs2.py     # 环境检查脚本
│   └── check_envs.ps1     # 环境检查脚本(PowerShell)
├── data/
│   └── obesity_level.csv  # 原始数据集
├── docs/                  # 文档与数据说明
│   ├── 数据字段.PNG
│   └── 标签字段.PNG
└── output/                # 输出目录（运行后生成）
    ├── label_distribution.png
    ├── feature_distributions.png
    ├── correlation_heatmap.png
    ├── training_curves.png
    ├── confusion_matrix.png
    └── classification_report.txt
```

## 环境依赖

```bash
pip install torch pandas numpy scikit-learn matplotlib seaborn
```

## 运行方式

```bash
# 一键运行全部流程
python run.py

# 或分步执行
python preprocess.py   # 数据预处理
python train.py        # 模型训练与评估
```

## 模型架构

采用多层全连接神经网络（MLP）：

```
输入层(16) → FC(128) → BN → ReLU → Dropout
           → FC(256) → BN → ReLU → Dropout
           → FC(128) → BN → ReLU → Dropout
           → FC(64)  → BN → ReLU → Dropout
           → 输出层(7)
```

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

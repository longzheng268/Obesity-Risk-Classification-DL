"""
肥胖等级分类 - 深度学习期中测试
数据预处理与探索性分析
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
import warnings
import os
import sys

# 确保 src/ 在 sys.path 中，支持 from model import ...
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 项目根目录（src/ 的上级目录）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

os.makedirs(os.path.join(PROJECT_ROOT, 'output'), exist_ok=True)


def load_and_explore():
    """加载数据并进行探索性分析"""
    df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'obesity_level.csv'))

    print("=" * 60)
    print("数据集基本信息")
    print("=" * 60)
    print(f"样本数: {df.shape[0]}, 特征数: {df.shape[1] - 2}")  # minus id and label
    print(f"\n列名: {list(df.columns)}")
    print(f"\n数据类型:\n{df.dtypes}")
    print(f"\n缺失值:\n{df.isnull().sum()}")
    print(f"\n统计描述:\n{df.describe()}")

    # 标签列（最后一列）
    label_col = df.columns[-1]
    print(f"\n标签列名: {label_col}")
    print(f"\n标签分布:\n{df[label_col].value_counts()}")

    return df, label_col


def plot_label_distribution(df, label_col):
    """绘制标签分布图"""
    fig, ax = plt.subplots(figsize=(10, 5))
    counts = df[label_col].value_counts()
    counts.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
    ax.set_title('肥胖等级分布', fontsize=14)
    ax.set_ylabel('样本数量')
    ax.set_xlabel('肥胖等级')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, 'output', 'label_distribution.png'), dpi=150)
    plt.close()
    print("[图] 标签分布已保存 -> output/label_distribution.png")


def plot_feature_distributions(df, label_col):
    """绘制数值特征分布"""
    num_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, col in enumerate(num_cols):
        ax = axes[i // 4][i % 4]
        df[col].hist(bins=30, ax=ax, color='steelblue', edgecolor='black')
        ax.set_title(col)
    plt.suptitle('数值特征分布', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, 'output', 'feature_distributions.png'), dpi=150)
    plt.close()
    print("[图] 数值特征分布已保存 -> output/feature_distributions.png")


def plot_correlation_heatmap(df):
    """绘制相关性热力图"""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'id' in num_cols:
        num_cols.remove('id')
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax)
    ax.set_title('数值特征相关性热力图', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, 'output', 'correlation_heatmap.png'), dpi=150)
    plt.close()
    print("[图] 相关性热力图已保存 -> output/correlation_heatmap.png")


def preprocess(df, label_col):
    """数据预处理"""
    df = df.copy()

    # 删除 id 列
    df = df.drop(columns=['id'])

    # 分离特征和标签
    X = df.drop(columns=[label_col])
    y = df[label_col]

    # 修复可能的编码问题
    y = y.replace({'0rmal_Weight': 'Normal_Weight'})
    X['Gender'] = X['Gender'].replace({'0emale': 'Female', 'Male': 'Male'})

    # 标签编码
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"\n标签映射: {dict(zip(le.classes_, range(len(le.classes_))))}")

    # 区分数值列和分类列
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    print(f"数值特征: {num_cols}")
    print(f"分类特征: {cat_cols}")

    # 分类特征编码
    for col in cat_cols:
        oe = OrdinalEncoder()
        X[col] = oe.fit_transform(X[[col]])

    # 数值特征标准化
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    print(f"\n预处理后特征维度: {X.shape}")

    return X.values, y_encoded, le, scaler, list(X.columns)


def main():
    df, label_col = load_and_explore()

    # EDA
    plot_label_distribution(df, label_col)
    plot_feature_distributions(df, label_col)
    plot_correlation_heatmap(df)

    # 预处理
    X, y, le, scaler, feature_names = preprocess(df, label_col)

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n训练集: {X_train.shape[0]} 条, 测试集: {X_test.shape[0]} 条")

    # 保存预处理后的数据
    np.savez(os.path.join(PROJECT_ROOT, 'output', 'processed_data.npz'),
             X_train=X_train, X_test=X_test,
             y_train=y_train, y_test=y_test)
    print("预处理数据已保存 -> output/processed_data.npz")

    return X_train, X_test, y_train, y_test, le


if __name__ == '__main__':
    main()

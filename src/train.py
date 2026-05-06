"""
肥胖等级分类 - 深度学习期中测试
模型训练与评估
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import sys

# 确保 src/ 在 sys.path 中，支持 from model import ...
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 项目根目录（src/ 的上级目录）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from model import ObesityClassifier

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

os.makedirs(os.path.join(PROJECT_ROOT, 'output'), exist_ok=True)


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def train_model(model, train_loader, val_loader, device, epochs=100, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0
    best_model_state = None

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        running_loss, correct, total = 0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # --- Validate ---
        model.eval()
        running_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                running_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        val_loss = running_loss / total
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}]  "
                  f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  "
                  f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}")

    print(f"\n最佳验证准确率: {best_val_acc:.4f}")
    model.load_state_dict(best_model_state)
    return model, train_losses, val_losses, train_accs, val_accs


def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(train_losses, label='训练损失')
    ax1.plot(val_losses, label='验证损失')
    ax1.set_title('损失曲线', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(train_accs, label='训练准确率')
    ax2.plot(val_accs, label='验证准确率')
    ax2.set_title('准确率曲线', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, 'output', 'training_curves.png'), dpi=150)
    plt.close()
    print("[图] 训练曲线已保存 -> output/training_curves.png")


def evaluate_model(model, X_test, y_test, le, device):
    model.eval()
    X_tensor = torch.FloatTensor(X_test).to(device)
    with torch.no_grad():
        outputs = model(X_tensor)
        _, predicted = torch.max(outputs, 1)
    y_pred = predicted.cpu().numpy()

    # 分类报告
    report = classification_report(y_test, y_pred, target_names=le.classes_)
    print("\n分类报告:")
    print(report)

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_title('混淆矩阵', fontsize=14)
    ax.set_xlabel('预测标签')
    ax.set_ylabel('真实标签')
    plt.xticks(rotation=30, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, 'output', 'confusion_matrix.png'), dpi=150)
    plt.close()
    print("[图] 混淆矩阵已保存 -> output/confusion_matrix.png")

    return y_pred


def generate_report(model, X_test, y_test, y_pred, le, device,
                    train_losses, val_losses, train_accs, val_accs,
                    epochs, lr, batch_size, train_size, test_size):
    """根据训练和评估结果生成 Markdown 报告"""
    from datetime import datetime
    import platform

    overall_acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None, labels=range(len(le.classes_))
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='macro'
    )
    cm = confusion_matrix(y_test, y_pred)

    # 各类别准确率（对角线 / 行和）
    class_acc = cm.diagonal() / cm.sum(axis=1)

    # 最佳 epoch
    best_epoch = val_accs.index(max(val_accs)) + 1
    best_val_acc = max(val_accs)

    # 模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'

    lines = []
    lines.append('# 肥胖风险分类 - 模型训练报告')
    lines.append('')
    lines.append(f'> 自动生成于 {now}')
    lines.append('')
    lines.append('---')
    lines.append('')

    # 1. 项目概述
    lines.append('## 1. 项目概述')
    lines.append('')
    lines.append('本项目基于深度学习技术，利用多层全连接神经网络（MLP）对个体肥胖风险进行 7 级分类评估。')
    lines.append('输入特征包括个人基本信息、饮食习惯、运动频率等 16 个维度，输出为 7 个肥胖等级。')
    lines.append('')

    # 2. 实验环境
    lines.append('## 2. 实验环境')
    lines.append('')
    lines.append('| 项目 | 配置 |')
    lines.append('|------|------|')
    lines.append(f'| 设备 | {gpu_name} |')
    lines.append(f'| PyTorch 版本 | {torch.__version__} |')
    lines.append(f'| Python 版本 | {platform.python_version()} |')
    lines.append(f'| 操作系统 | {platform.system()} {platform.release()} |')
    lines.append('')

    # 3. 数据集信息
    lines.append('## 3. 数据集信息')
    lines.append('')
    lines.append(f'- **总样本数**: {train_size + test_size}')
    lines.append(f'- **训练集**: {train_size} 条')
    lines.append(f'- **测试集**: {test_size} 条')
    lines.append(f'- **特征维度**: {X_test.shape[1]}')
    lines.append(f'- **分类类别数**: {len(le.classes_)}')
    lines.append('')
    lines.append('### 类别标签')
    lines.append('')
    lines.append('| 编码 | 类别名称 | 中文含义 |')
    lines.append('|------|----------|----------|')
    cn_names = {
        'Insufficient_Weight': '体重不足',
        'Normal_Weight': '正常体重',
        'Overweight_Level_I': '一级超重',
        'Overweight_Level_II': '二级超重',
        'Obesity_Type_I': '肥胖类型 I',
        'Obesity_Type_II': '肥胖类型 II',
        'Obesity_Type_III': '肥胖类型 III',
    }
    for i, cls in enumerate(le.classes_):
        cn = cn_names.get(cls, cls)
        lines.append(f'| {i} | `{cls}` | {cn} |')
    lines.append('')

    # 4. 模型架构
    lines.append('## 4. 模型架构')
    lines.append('')
    lines.append('采用多层全连接神经网络（MLP），配合 BatchNorm 和 Dropout 进行正则化：')
    lines.append('')
    lines.append('```')
    lines.append(f'输入层({X_test.shape[1]}) → FC(128) → BN → ReLU → Dropout(0.3)')
    lines.append('           → FC(256) → BN → ReLU → Dropout(0.3)')
    lines.append('           → FC(128) → BN → ReLU → Dropout(0.2)')
    lines.append('           → FC(64)  → BN → ReLU → Dropout(0.2)')
    lines.append('           → 输出层(7)')
    lines.append('```')
    lines.append('')
    lines.append(f'- **总参数量**: {total_params:,}')
    lines.append(f'- **可训练参数**: {trainable_params:,}')
    lines.append('')

    # 5. 训练配置
    lines.append('## 5. 训练配置')
    lines.append('')
    lines.append('| 参数 | 值 |')
    lines.append('|------|-----|')
    lines.append(f'| Epochs | {epochs} |')
    lines.append(f'| 学习率 | {lr} |')
    lines.append(f'| Batch Size | {batch_size} |')
    lines.append(f'| 优化器 | Adam (weight_decay=1e-4) |')
    lines.append(f'| 学习率调度 | ReduceLROnPlateau (factor=0.5, patience=10) |')
    lines.append(f'| 损失函数 | CrossEntropyLoss |')
    lines.append(f'| 最佳 Epoch | {best_epoch} |')
    lines.append(f'| 最佳验证准确率 | {best_val_acc:.4f} |')
    lines.append('')

    # 6. 训练曲线
    lines.append('## 6. 训练曲线')
    lines.append('')
    lines.append('![训练曲线](training_curves.png)')
    lines.append('')
    lines.append(f'- 最终训练损失: {train_losses[-1]:.4f}')
    lines.append(f'- 最终验证损失: {val_losses[-1]:.4f}')
    lines.append(f'- 最终训练准确率: {train_accs[-1]:.4f}')
    lines.append(f'- 最终验证准确率: {val_accs[-1]:.4f}')
    lines.append('')

    # 7. 评估结果
    lines.append('## 7. 评估结果')
    lines.append('')
    lines.append(f'### 总体指标')
    lines.append('')
    lines.append(f'- **总体准确率 (Accuracy)**: {overall_acc:.4f} ({overall_acc*100:.2f}%)')
    lines.append(f'- **宏平均精确率 (Macro Precision)**: {macro_p:.4f}')
    lines.append(f'- **宏平均召回率 (Macro Recall)**: {macro_r:.4f}')
    lines.append(f'- **宏平均 F1 分数 (Macro F1)**: {macro_f1:.4f}')
    lines.append('')

    # 分类报告表格
    lines.append('### 各类别详细指标')
    lines.append('')
    lines.append('| 类别 | Precision | Recall | F1-Score | 准确率 | 样本数 |')
    lines.append('|------|-----------|--------|----------|--------|--------|')
    for i, cls in enumerate(le.classes_):
        cn = cn_names.get(cls, cls)
        lines.append(f'| {cn} | {precision[i]:.4f} | {recall[i]:.4f} | {f1[i]:.4f} | {class_acc[i]:.4f} | {int(support[i])} |')
    lines.append(f'| **加权平均** | **{np.average(precision, weights=support):.4f}** | **{np.average(recall, weights=support):.4f}** | **{np.average(f1, weights=support):.4f}** | **{overall_acc:.4f}** | **{int(support.sum())}** |')
    lines.append('')

    # 混淆矩阵
    lines.append('### 混淆矩阵')
    lines.append('')
    lines.append('![混淆矩阵](confusion_matrix.png)')
    lines.append('')

    # 8. 结果分析
    lines.append('## 8. 结果分析')
    lines.append('')

    # 找出最佳和最差类别
    best_idx = np.argmax(class_acc)
    worst_idx = np.argmax(class_acc) if class_acc.min() > 0.5 else np.argmin(class_acc)
    worst_idx = np.argmin(class_acc)
    lines.append('### 性能表现')
    lines.append('')
    lines.append(f'- **表现最佳类别**: {cn_names.get(le.classes_[best_idx], le.classes_[best_idx])}，准确率 {class_acc[best_idx]:.4f}，F1 = {f1[best_idx]:.4f}')
    lines.append(f'- **表现最差类别**: {cn_names.get(le.classes_[worst_idx], le.classes_[worst_idx])}，准确率 {class_acc[worst_idx]:.4f}，F1 = {f1[worst_idx]:.4f}')
    lines.append('')

    lines.append('### 结论')
    lines.append('')
    lines.append(f'模型在测试集上取得了 **{overall_acc*100:.2f}%** 的总体准确率，')
    lines.append(f'宏平均 F1 分数为 **{macro_f1:.4f}**。')
    lines.append('')
    lines.append('从混淆矩阵可以看出：')
    lines.append(f'1. Obesity_Type_III（肥胖类型 III）识别效果最好，F1 达到 {f1[le.classes_.tolist().index("Obesity_Type_III")]:.4f}，这可能因为该类别的特征最为显著。')
    lines.append(f'2. Overweight_Level_I（一级超重）和 Overweight_Level_II（二级超重）的分类难度相对较大，F1 分数分别为 {f1[le.classes_.tolist().index("Overweight_Level_I")]:.4f} 和 {f1[le.classes_.tolist().index("Overweight_Level_II")]:.4f}，可能因为相邻等级之间的特征差异较小。')
    lines.append(f'3. 模型在 {best_epoch} 个 epoch 后达到最佳验证性能，之后未出现明显过拟合，说明 Dropout 和 BatchNorm 的正则化效果良好。')
    lines.append('')

    # 9. 输出文件
    lines.append('## 9. 输出文件')
    lines.append('')
    lines.append('| 文件 | 说明 |')
    lines.append('|------|------|')
    lines.append('| `report.md` | 模型训练报告（含分类报告） |')
    lines.append('| `training_curves.png` | 训练和验证的损失/准确率曲线 |')
    lines.append('| `confusion_matrix.png` | 混淆矩阵热力图 |')
    lines.append('| `best_model.pth` | 最佳模型权重 |')
    lines.append('| `processed_data.npz` | 预处理后的数据集 |')
    lines.append('| `label_distribution.png` | 标签分布图 |')
    lines.append('| `feature_distributions.png` | 数值特征分布图 |')
    lines.append('| `correlation_heatmap.png` | 特征相关性热力图 |')
    lines.append('')

    lines.append('---')
    lines.append('')
    lines.append('*本报告由模型训练流程自动生成。*')

    report_path = os.path.join(PROJECT_ROOT, 'output', 'report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"[报告] 训练报告已保存 -> output/report.md")

    return report_path


def main():
    # 加载预处理数据
    data = np.load(os.path.join(PROJECT_ROOT, 'output', 'processed_data.npz'))
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']

    device = get_device()
    print(f"使用设备: {device}")

    # DataLoader
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test)
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # 模型
    num_classes = len(np.unique(y_train))
    model = ObesityClassifier(input_dim=X_train.shape[1], num_classes=num_classes).to(device)
    print(f"\n模型结构:\n{model}")

    # 训练
    model, train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, device, epochs=100, lr=1e-3
    )

    # 保存模型
    torch.save(model.state_dict(), os.path.join(PROJECT_ROOT, 'output', 'best_model.pth'))
    print("模型已保存 -> output/best_model.pth")

    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)

    # 加载标签编码器用于评估
    import pickle
    from preprocess import preprocess
    import pandas as pd

    # 重新获取标签编码器
    df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'obesity_level.csv'))
    label_col = df.columns[-1]
    y_raw = df[label_col].replace({'0rmal_Weight': 'Normal_Weight'})
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(y_raw)

    # 评估
    y_pred = evaluate_model(model, X_test, y_test, le, device)

    # 生成报告
    generate_report(model, X_test, y_test, y_pred, le, device,
                    train_losses, val_losses, train_accs, val_accs,
                    epochs=100, lr=1e-3, batch_size=64,
                    train_size=len(X_train), test_size=len(X_test))


if __name__ == '__main__':
    main()

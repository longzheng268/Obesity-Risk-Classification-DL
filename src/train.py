"""
肥胖等级分类 - 深度学习期中测试
模型训练与评估
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
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

    # 保存分类报告
    with open(os.path.join(PROJECT_ROOT, 'output', 'classification_report.txt'), 'w', encoding='utf-8') as f:
        f.write("肥胖等级分类 - 分类报告\n")
        f.write("=" * 60 + "\n")
        f.write(report)
        f.write(f"\n总体准确率: {accuracy_score(y_test, y_pred):.4f}\n")
    print("分类报告已保存 -> output/classification_report.txt")

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
    evaluate_model(model, X_test, y_test, le, device)


if __name__ == '__main__':
    main()

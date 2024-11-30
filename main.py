import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import KFold
import seaborn as sns
import os
from tqdm import tqdm
from torch.nn import functional as F
from collections import Counter


def augment_data(X, y):
    """对数据进行增强"""
    augmented_X = []
    augmented_y = []

    for i in range(len(X)):
        augmented_X.append(X[i])
        augmented_y.append(y[i])

        if y[i] == 1:  # 只对正样本进行增强
            # 添加高斯噪声
            noise = np.random.normal(0, 0.01, X[i].shape)
            augmented_X.append(X[i] + noise)
            augmented_y.append(y[i])

            # 时间反转
            augmented_X.append(np.flip(X[i], axis=0))
            augmented_y.append(y[i])

    return np.array(augmented_X), np.array(augmented_y)


def load_all_data(folder_path):
    """加载并预处理所有CSV文件的数据"""
    print("正在加载所有数据文件...")
    all_X_windows = []
    all_y_windows = []
    scaler = MinMaxScaler()

    csv_files = [f for f in os.listdir(folder_path) if f.startswith('lstm_imputed_') and f.endswith('.csv')]
    required_columns = ['40', '44', '89', '90', '92', '188', 'EVENT_NO']
    valid_files = 0

    for file_name in tqdm(csv_files):
        file_path = os.path.join(folder_path, file_name)
        try:
            data = pd.read_csv(file_path)

            if not all(col in data.columns for col in required_columns):
                print(f"跳过文件 {file_name}: 缺少必需的列")
                continue

            features = ['40', '44', '89', '90', '92', '188']
            X = data[features].astype(float).values
            y = data['EVENT_NO'].astype(float).values

            if np.isnan(X).any() or np.isnan(y).any():
                print(f"跳过文件 {file_name}: 包含NaN值")
                continue

            # 分段标准化，每个特征列独立标准化
            X_scaled = np.zeros_like(X)
            for i in range(X.shape[1]):
                X_scaled[:, i] = scaler.fit_transform(X[:, i].reshape(-1, 1)).ravel()

            # 减小窗口大小
            window_size = 5
            for i in range(len(X_scaled) - window_size):
                window_data = X_scaled[i:(i + window_size)]
                target = y[i + window_size]
                # 更平衡的采样策略
                if target == 1 or np.random.random() < 0.5:
                    all_X_windows.append(window_data)
                    all_y_windows.append(target)

            valid_files += 1

        except Exception as e:
            print(f"处理文件 {file_name} 时出错: {str(e)}")
            continue

    if not all_X_windows:
        raise ValueError("没有有效的数据被加载")

    X_combined = np.array(all_X_windows)
    y_combined = np.array(all_y_windows)

    # 对数据进行增强
    X_combined, y_combined = augment_data(X_combined, y_combined)

    print(f"\n数据加载统计:")
    print(f"成功加载文件数: {valid_files}/{len(csv_files)}")
    print(f"总样本数: {len(X_combined)}")
    print(f"正样本比例: {np.mean(y_combined):.2%}")

    return X_combined, y_combined


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class EnhancedLSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 添加BatchNorm
        self.batch_norm = torch.nn.BatchNorm1d(input_size)

        self.lstm1 = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        self.lstm2 = torch.nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        # 简化的网络结构
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * 2, hidden_size),
            torch.nn.LayerNorm(hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, output_size),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        # 应用BatchNorm
        x = x.transpose(1, 2)
        x = self.batch_norm(x)
        x = x.transpose(1, 2)

        lstm1_out, _ = self.lstm1(x)
        lstm2_out, _ = self.lstm2(lstm1_out)

        # 使用平均池化替代最大池化
        pooled = torch.mean(lstm2_out, dim=1)
        out = self.fc_layers(pooled)
        return out


def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device, weight_decay=0.1):
    criterion = FocalLoss(alpha=2.0, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader)
    )

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    best_f1 = 0

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}') as pbar:
            for batch_X, batch_y in pbar:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()

                total_train_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})

        model.eval()
        total_val_loss = 0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                val_loss = criterion(outputs, batch_y)
                total_val_loss += val_loss.item()

                val_preds.extend((outputs > 0.5).float().cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)

        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        val_f1 = f1_score(val_targets, val_preds)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f'\nEpoch [{epoch + 1}/{num_epochs}]:')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
        print(f'Validation F1: {val_f1:.4f}')
        print(f'Learning Rate: {scheduler.get_last_lr()[0]:.6f}')

        # 保存最佳F1分数的模型
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), 'best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

    return train_losses, val_losses


def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    test_loss = 0
    criterion = FocalLoss(alpha=2.0, gamma=2.0)

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item()

            predictions = (outputs > 0.5).float()
            all_preds.extend(predictions.cpu().numpy().flatten())
            all_labels.extend(batch_y.cpu().numpy().flatten())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print("\n测试集评估:")
    print(f"Test Loss: {test_loss / len(test_loader):.4f}")
    print(f"预测类别分布: {Counter(all_preds.tolist())}")

    return accuracy, precision, recall, f1, all_preds, all_labels


def plot_results(train_losses, val_losses, y_test, y_pred):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    plt.subplot(1, 3, 3)
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()

    plt.tight_layout()
    plt.show()


def main():
        torch.manual_seed(42)
        np.random.seed(42)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")

        # 修改超参数
        input_size = 6
        hidden_size = 128
        num_layers = 2
        output_size = 1
        learning_rate = 0.001
        num_epochs = 50
        batch_size = 64

        folder_path = r"E:\Data严格保密\final data1\final_data101"
        X, y = load_all_data(folder_path)

        # K折交叉验证
        k_folds = 5
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        fold_results = []

        for fold, (train_indices, val_indices) in enumerate(kf.split(X)):
            print(f"\n训练折 {fold + 1}/{k_folds}")

            # 分割训练集和验证集
            X_train, X_val = X[train_indices], X[val_indices]
            y_train, y_val = y[train_indices], y[val_indices]

            # 创建数据加载器
            train_dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(X_train),
                torch.FloatTensor(y_train).reshape(-1, 1)
            )
            val_dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(X_val),
                torch.FloatTensor(y_val).reshape(-1, 1)
            )

            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                num_workers=0
            )

            # 创建模型
            model = EnhancedLSTMModel(input_size, hidden_size, num_layers, output_size).to(device)

            # 训练模型
            train_losses, val_losses = train_model(
                model, train_loader, val_loader, num_epochs, learning_rate, device
            )
            # 评估当前折的性能
            accuracy, precision, recall, f1, _, _ = evaluate_model(model, val_loader, device)

            fold_results.append({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })

            # 输出平均性能
            print("\n交叉验证结果:")
            metrics = ['accuracy', 'precision', 'recall', 'f1']
            for metric in metrics:
                values = [result[metric] for result in fold_results]
            mean = np.mean(values)
            std = np.std(values)
            print(f"{metric}: {mean:.4f} ± {std:.4f}")

            # 使用全部数据进行最终训练
            total_train_size = int(len(X) * 0.8)
            train_indices = np.random.choice(len(X), total_train_size, replace=False)
            test_indices = np.array(list(set(range(len(X))) - set(train_indices)))

            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]

            # 创建最终的数据加载器
            train_dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(X_train),
                torch.FloatTensor(y_train).reshape(-1, 1)
            )
            test_dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(X_test),
                torch.FloatTensor(y_test).reshape(-1, 1)
            )

            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0
            )
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                num_workers=0
            )

            # 训练最终模型
            final_model = EnhancedLSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
            train_losses, val_losses = train_model(
                final_model, train_loader, test_loader, num_epochs, learning_rate, device
            )

            # 加载最佳模型
            print("\n加载最佳模型...")
            final_model.load_state_dict(torch.load('best_model.pth'))

            # 最终评估
            print("\n评估最终模型...")
            accuracy, precision, recall, f1, y_pred, y_test = evaluate_model(
                final_model, test_loader, device
            )

            print("\n最终模型评估指标:")
            print(f"准确率: {accuracy:.4f}")
            print(f"精确率: {precision:.4f}")
            print(f"召回率: {recall:.4f}")
            print(f"F1分数: {f1:.4f}")

            # 保存模型和结果
            model_info = {
                'model_state_dict': final_model.state_dict(),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'training_params': {
                    'input_size': input_size,
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'learning_rate': learning_rate,
                    'num_epochs': num_epochs,
                    'batch_size': batch_size
                },
                'cross_validation_results': fold_results
            }

            print("\n保存模型...")
            torch.save(model_info, 'final_model.pth')

            print("\n生成可视化结果...")
            plot_results(train_losses, val_losses, y_test, y_pred)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()
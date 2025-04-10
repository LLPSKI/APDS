import load
import show
import preprocess
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class PhysiologicalDataset(Dataset):
    """
    PyTorch数据集类，用于加载生理信号特征
    """
    def __init__(self, features, labels):
        """
        参数:
            features: 标准化后的特征矩阵
            labels: 对应的标签
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class StressClassifier(nn.Module):
    """
    压力状态分类模型
    输入: 融合后的多模态特征
    输出: 二元分类(0:正常, 1:压力)
    """
    def __init__(self, input_dim):
        super(StressClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.drop2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        
        self.output = nn.Linear(32, 2)  # 输出2类
    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop1(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        
        x = F.relu(self.bn3(self.fc3(x)))
        
        return self.output(x)
def evaluate_model(model, data_loader, criterion, device):
    """
    评估模型性能
    参数:
        model: 要评估的模型
        data_loader: 数据加载器
        criterion: 损失函数
        device: 设备
    返回:
        平均损失和准确率
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / len(data_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy

def train_model(model, train_loader, val_loader, num_epochs=50, lr=0.001):
    """
    训练模型
    参数:
        model: 初始化的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        num_epochs: 训练轮数
        lr: 学习率
    返回:
        训练好的模型
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        
        # 验证阶段
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
    
    print(f"Training complete. Best validation accuracy: {best_val_acc:.4f}")
    return model

if __name__ == '__main__':
    target = load.load_data()
    scaled_features, labels_array = preprocess.preprocess(target)
    # show.show_array(labels_array)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_features, labels_array, 
        test_size=0.2, 
        random_state=42, 
        stratify=labels_array
    )

    # 创建PyTorch数据集和数据加载器
    train_dataset = PhysiologicalDataset(X_train, y_train)
    test_dataset = PhysiologicalDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 初始化模型
    input_dim = X_train.shape[1]
    model = StressClassifier(input_dim)

    # 训练模型
    trained_model = train_model(model, train_loader, test_loader)

    # 评估最终模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model.load_state_dict(torch.load('best_model.pth', weights_only=True))
    test_loss, test_acc = evaluate_model(trained_model, test_loader, nn.CrossEntropyLoss(), device)
    print(f"Final Test Accuracy: {test_acc:.4f}")
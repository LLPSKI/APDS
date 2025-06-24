import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

def train_model(model, train_loader, val_loader, test_loader, num_epochs=100, lr=0.001, device='cpu'):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)
    
    best_acc = 0.0
    history = {
        'train_acc': [],
        'val_acc': [],
        'test_acc': []
    }
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        # 训练阶段
        for (x_local, x_global), y in train_loader:
            x_local = x_local.to(device)
            x_global = x_global.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            outputs = model(x_local, x_global)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            total_train += y.size(0)
            correct_train += (predicted == y).sum().item()
            
            running_loss += loss.item() * y.size(0)
            progress_bar.set_postfix({
                'loss': loss.item(),
                'train_acc': f'{100 * correct_train / total_train:.2f}%'
            })
        
        # 计算训练集准确率
        train_acc = correct_train / total_train
        history['train_acc'].append(train_acc)
        
        # 验证集评估
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for (x_local, x_global), y in val_loader:
                outputs = model(x_local.to(device), x_global.to(device))
                _, predicted = torch.max(outputs.data, 1)
                total_val += y.size(0)
                correct_val += (predicted == y).sum().item()
        
        val_acc = correct_val / total_val
        history['val_acc'].append(val_acc)
        
        # 测试集评估
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for (x_local, x_global), y in val_loader:
                outputs = model(x_local.to(device), x_global.to(device))
                _, predicted = torch.max(outputs.data, 1)
                total_test += y.size(0)
                correct_test += (predicted == y).sum().item()
        
        test_acc = correct_test / total_test
        history['test_acc'].append(test_acc)
        
        # 学习率调整
        scheduler.step(val_acc)
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'New best model saved at epoch {epoch+1} with val_acc {val_acc:.4f}')
        
        print(f'Epoch {epoch+1}: '
              f'Train Acc: {train_acc:.4f}, '
              f'Val Acc: {val_acc:.4f}, '
              f'Test Acc: {test_acc:.4f}')
    
    return history
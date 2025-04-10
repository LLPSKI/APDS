# 多模态生理信号压力检测系统

## 项目概述
本项目基于EMG(肌电)、EDA(皮肤电活动)和Resp(呼吸)三种生理信号，使用深度学习技术实现心理压力状态的自动检测。系统能够从多模态生理信号中提取特征，并通过神经网络模型准确识别压力状态(1)和非压力状态(0)。

## 功能特性
- 多模态信号融合：整合EMG、EDA和Resp三种生理信号
- 高效特征提取：自动提取时域和频域特征
- 深度学习模型：基于PyTorch的神经网络分类器
- 实时处理能力：支持高采样率(700Hz)信号处理

## 数据预处理流程
1. **信号滤波**：
   - EMG: 20-500Hz带通滤波
   - EDA: 1Hz低通滤波
   - Resp: 0.1-0.5Hz带通滤波
2. **特征提取**：
   - 时域特征：均值、标准差、中位数等
   - 频域特征：功率谱、峰值频率等
3. **数据标准化**：使用StandardScaler进行特征缩放

## 模型架构
```python
class StressClassifier(nn.Module):
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
        
        self.output = nn.Linear(32, 2)
```

## 安装与使用
### 环境要求
- Python 3.8+
- PyTorch 1.10+
- scipy, numpy, sklearn

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行示例
```python
# 数据预处理
processor = MultiSubjectProcessor(subjects_data, fs=700)
processor.preprocess_signals()
features, labels = processor.extract_features()

# 训练模型
train_loader, test_loader, scaler = prepare_datasets(features, labels)
model = StressClassifier(input_dim=features.shape[1])
trained_model = train_model(model, train_loader, test_loader)
```

## 性能指标
在测试集上达到以下性能：
- 准确率: 92.5%
- 召回率: 89.3%
- F1分数: 90.8%

![image](https://github.com/user-attachments/assets/9bfb9c55-8ba5-4890-9722-1dece0f5be90)


## 参考文献

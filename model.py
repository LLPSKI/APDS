import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock3D(nn.Module):
    """简化的3D残差块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm3d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)

class TwoStream3DCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        
        # Local Stream
        self.local_stream = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=3, padding=1),  # 输入通道8
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)),  # 保持时间维度
            ResidualBlock3D(16, 32)
        )
        
        # Global Stream (结构对称)
        self.global_stream = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)),
            ResidualBlock3D(16, 32)
        )
        
        # 各流的全连接层
        self.local_fc = nn.Linear(32 * 7 * 3 * 3, 64)  # 假设空间维度从6x6池化到3x3
        self.global_fc = nn.Linear(32 * 7 * 3 * 3, 64)
        
        # 融合后的分类头
        self.final_fc = nn.Linear(64 * 2, num_classes)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x_local, x_global):
        # Local Stream
        x_local = self.local_stream(x_local)  # (N, 32, 7, 3, 3)
        x_local = x_local.view(x_local.size(0), -1)  # (N, 32 * 7 * 3 * 3)
        x_local = F.relu(self.local_fc(x_local))  # (N, 64)
        
        # Global Stream
        x_global = self.global_stream(x_global)  # (N, 32, 7, 3, 3)
        x_global = x_global.view(x_global.size(0), -1)  # (N, 32 * 7 * 3 * 3)
        x_global = F.relu(self.global_fc(x_global))  # (N, 64)
        
        # 特征融合
        x = torch.cat([x_local, x_global], dim=1)  # (N, 128)
        x = self.final_fc(x)  # (N, 4)
        
        return x

# 示例使用
if __name__ == "__main__":
    # 1. 初始化模型
    model = TwoStream3DCNN(num_classes=4)
    print("模型结构:")
    print(model)
    
    # 2. 创建模拟输入数据（与真实数据形状一致）
    batch_size = 2  # 测试用的小批量
    local_input = torch.randn(batch_size, 8, 7, 6, 6)  # (N, 8, 7, 6, 6)
    global_input = torch.randn(batch_size, 8, 7, 6, 6)  # (N, 8, 7, 6, 6)
    
    print(f"\nLocal输入形状: {local_input.shape}")
    print(f"Global输入形状: {global_input.shape}")

    # 3. 前向传播
    with torch.no_grad():
        output = model(local_input, global_input)
    
    # 4. 打印输出形状
    print(f"\n输出形状: {output.shape}")  # 应为 (batch_size, 4)
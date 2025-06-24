import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

def prepare_two_stream_data(subjects, test_size=0.1, val_size=0.1, random_state=42):
    """
    为双流模型准备数据，保持local和global图像分离
    
    返回:
        (train_data, val_data, test_data) 每个都是包含以下内容的字典:
        {
            'local': local_images (torch.Tensor),
            'global': global_images (torch.Tensor),
            'label': labels (torch.Tensor)
        }
    """
    # 合并所有受体的数据（保持local和global分离）
    all_labels = np.concatenate([s['label'] for s in subjects])
    all_local = np.concatenate([s['local_image'] for s in subjects])  # (N, 8, 7, 6, 6)
    all_global = np.concatenate([s['global_image'] for s in subjects]) # (N, 8, 7, 6, 6)
    
    # 使用相同的随机种子确保local和global的分割一致
    # 第一次分割：训练集 (80%) 和 临时集 (20%)
    idx = np.arange(len(all_labels))
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        idx, all_labels,
        test_size=(test_size + val_size),
        random_state=random_state,
        stratify=all_labels
    )
    
    # 第二次分割：验证集 (10%) 和测试集 (10%)
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_temp, y_temp,
        test_size=test_size/(test_size + val_size),
        random_state=random_state,
        stratify=y_temp
    )
    
    # 转换为PyTorch张量
    def to_tensor(data):
        return torch.from_numpy(data).float()
    
    train_data = {
        'local': to_tensor(all_local[idx_train]),
        'global': to_tensor(all_global[idx_train]),
        'label': torch.from_numpy(y_train).long()
    }
    val_data = {
        'local': to_tensor(all_local[idx_val]),
        'global': to_tensor(all_global[idx_val]),
        'label': torch.from_numpy(y_val).long()
    }
    test_data = {
        'local': to_tensor(all_local[idx_test]),
        'global': to_tensor(all_global[idx_test]),
        'label': torch.from_numpy(y_test).long()
    }
    
    return train_data, val_data, test_data

class TwoStreamDataset(Dataset):
    def __init__(self, local_images, global_images, labels, transform=None):
        """
        参数:
            local_images: (N, 8, 7, 6, 6) 的torch.Tensor
            global_images: (N, 8, 7, 6, 6) 的torch.Tensor
            labels: (N,) 的torch.Tensor
            transform: 可选的同步数据增强
        """
        self.local = local_images
        self.global_ = global_images
        self.labels = labels - 1
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        x_local = self.local[idx]
        x_global = self.global_[idx]
        y = self.labels[idx]
        
        if self.transform:
            # 对两个流应用相同的随机变换（如需）
            seed = torch.random.initial_seed()
            torch.manual_seed(seed)
            x_local = self.transform(x_local)
            torch.manual_seed(seed)
            x_global = self.transform(x_global)
            
        return (x_local, x_global), y

# 创建DataLoader
def create_two_stream_loaders(subjects, batch_size=32):
    train_data, val_data, test_data = prepare_two_stream_data(subjects)
    
    # 示例：添加数据增强（可选）
    train_transform = torch.nn.Sequential(
        # 例如：随机通道丢弃
        torch.nn.Dropout3d(p=0.1),
    )
    
    train_dataset = TwoStreamDataset(
        train_data['local'], train_data['global'], train_data['label'],
        transform=train_transform
    )
    val_dataset = TwoStreamDataset(val_data['local'], val_data['global'], val_data['label'])
    test_dataset = TwoStreamDataset(test_data['local'], test_data['global'], test_data['label'])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
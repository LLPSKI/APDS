from scipy import signal
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import h5py

def extract_time_features(x:np.ndarray):
    """
    提取时域特征
    """
    return [
        np.mean(x), # 均值
        np.std(x) # 标准差
    ]

import numpy as np
from scipy import signal
from scipy.stats import entropy

def extract_freq_features(x: np.ndarray, fs=700.0) -> list:
    """
    计算输入信号的频域特征
    
    参数:
        x: 输入信号 (1D numpy数组)
        fs: 采样频率 (默认700.0 Hz)
    
    返回:
        list: 包含以下特征的列表 [se, gf, fmean, fstd, rmsf]
            - se: 频谱熵
            - gf: 重心频率
            - fmean: 频率均值
            - fstd: 频率标准差
            - rmsf: 均方根频率
    """
    # 计算信号的FFT
    n = len(x)
    fft_vals = np.fft.rfft(x)
    power_spectrum = np.abs(fft_vals) ** 2
    
    # 计算频率轴
    freqs = np.fft.rfftfreq(n, d=1.0/fs)
    
    # 避免除以零和log(0)的情况
    power_spectrum = np.where(power_spectrum == 0, np.finfo(float).eps, power_spectrum)
    
    # 1. 计算频谱熵 (se)
    # 归一化功率谱
    norm_power = power_spectrum / np.sum(power_spectrum)
    se = entropy(norm_power)
    
    # 2. 计算重心频率 (gf)
    gf = np.sum(freqs * power_spectrum) / np.sum(power_spectrum)
    
    # 3. 计算频率均值 (fmean)
    fmean = gf  # 对于这些定义，fmean和gf是相同的
    
    # 4. 计算频率标准差 (fstd)
    fstd = np.sqrt(np.sum(power_spectrum * (freqs - fmean)**2) / np.sum(power_spectrum))
    
    # 5. 计算均方根频率 (rmsf)
    rmsf = np.sqrt(np.sum(power_spectrum * freqs**2) / np.sum(power_spectrum))
    
    return [se, gf, fmean, fstd, rmsf]

def gasf_local(feature: np.ndarray) -> np.ndarray:
    """
    Local GASF 转换（逐样本独立归一化）
    输入: (N, O, F, W)
    输出: (N, O, F, W, W)
    """
    N, O, F, W = feature.shape
    gasf = np.zeros((N, O, F, W, W))
    
    for n in range(N):
        for o in range(O):
            for f in range(F):
                # 1. 提取当前序列并归一化到 [-1, 1]
                x = feature[n, o, f, :]
                x_min, x_max = np.min(x), np.max(x)
                if x_max == x_min:
                    x_norm = np.zeros_like(x)
                else:
                    x_norm = 2 * (x - x_min) / (x_max - x_min) - 1
                
                # 2. 转换为极坐标角度（arccos）
                phi = np.arccos(np.clip(x_norm, -1, 1))  # 避免数值误差
                
                # 3. 计算 GASF 矩阵（cos(phi_i + phi_j)）
                gasf[n, o, f, :, :] = np.cos(phi.reshape(-1, 1) + phi.reshape(1, -1))
    
    return gasf

def gasf_global(feature: np.ndarray) -> np.ndarray:
    """
    Global GASF 转换（全局归一化）
    输入: (N, O, F, W)
    输出: (N, O, F, W, W)
    """
    N, O, F, W = feature.shape
    gasf = np.zeros((N, O, F, W, W))
    
    # 全局归一化（所有样本共享 min/max）
    global_min = np.min(feature)
    global_max = np.max(feature)
    
    for n in range(N):
        for o in range(O):
            for f in range(F):
                # 1. 归一化到 [-1, 1]
                x = feature[n, o, f, :]
                if global_max == global_min:
                    x_norm = np.zeros_like(x)
                else:
                    x_norm = 2 * (x - global_min) / (global_max - global_min) - 1
                
                # 2. 极坐标角度
                phi = np.arccos(np.clip(x_norm, -1, 1))
                
                # 3. 计算 GASF 矩阵
                gasf[n, o, f, :, :] = np.cos(phi.reshape(-1, 1) + phi.reshape(1, -1))
    
    return gasf

def save_to_hdf5(labels_2s, local_images, global_images, output_path="gasf_data.h5"):
    with h5py.File(output_path, 'w') as hf:
        L = len(labels_2s)
        for i in range(L):
            group = hf.create_group(f"subject_{i}")  # 替换为实际subject ID
            group.create_dataset("label", data=labels_2s[i], compression="gzip")
            group.create_dataset("local_image", data=local_images[i], compression="gzip")
            group.create_dataset("global_image", data=global_images[i], compression="gzip")

def load_from_hdf5(input_path="gasf_data.h5"):
    data = []
    with h5py.File(input_path, 'r') as hf:
        for subject_id in tqdm(hf.keys(), desc="Loading data"):
            group = hf[subject_id]
            data.append({
                "label": group["label"][:],
                "local_image": group["local_image"][:],
                "global_image": group["global_image"][:]
            })
    return data

def save_array_to_txt(array, filename, fmt='%.6f'):
    """将numpy数组保存为可读文本文件"""
    with open(filename, 'w') as f:
        # 写入数组形状信息
        f.write(f"Shape: {array.shape}\n")
        # 展平后保存（保持数据原值）
        np.savetxt(f, array.reshape(-1, array.shape[-1]), fmt=fmt)

def preprocess(subjects, K=2, W=6):
    """
    预处理生理信号数据，提取时域和频域特征
    
    参数:
        target: 包含所有受试者数据的列表
        fs: 采样频率(Hz)K
        K: 时间窗口长度，默认为2s
        W: 子窗口长度，默认为6个T，包含了12s的数据
    """
    labels_2s = []
    local_images = []
    global_images = []
    for subject in tqdm(subjects, desc="Processing all subjects"):
        # 每个实体的标签数据，采集率是700Hz
        subject_labels:np.ndarray = subject['label']
        window_size = int(K * 700)  # 每个时间窗口的采样点数
        T = len(subject_labels) // window_size

        emg:np.ndarray = subject['emg']
        eda:np.ndarray = subject['eda']
        resp:np.ndarray = subject['resp']
        temp:np.ndarray = subject['temp']
        ecg:np.ndarray = subject['ecg']
        bvp:np.ndarray = subject['bvp'] # 采集率是64Hz
        eda_wrist:np.ndarray = subject['eda_wrist'] # 采集率是4Hz
        temp_wrist:np.ndarray = subject['temp_wrist'] # 采集率是4Hz

        # 还需要其他不同的窗口大小
        window_size_64 = int(K * 64)
        window_size_4 = int(K * 4)

        features = []
        labels = []

        for i in range(T):
            start_idx = i * window_size
            end_idx = start_idx + window_size

            start_idx_64 = i * window_size_64
            end_idx_64 = start_idx_64 + window_size_64

            start_idx_4 = i * window_size_4
            end_idx_4 = start_idx_4 + window_size_4
            
            # 检查标签是否一致 (窗口内所有标签相同)
            window_labels = subject_labels[start_idx:end_idx]
            unique_labels = np.unique(window_labels)

            # 忽略包含多个标签的窗口
            if len(unique_labels) != 1 or unique_labels[0] == 0 or unique_labels[0] > 4:
                continue
            
            # 取出标签
            label = unique_labels[0]

            # 处理8段1D信号
            signals = {
                'emg': emg[start_idx:end_idx],
                'eda': eda[start_idx:end_idx],
                'resp': resp[start_idx:end_idx],
                'temp': temp[start_idx:end_idx],
                'ecg': ecg[start_idx:end_idx],
                'bvp': bvp[start_idx_64:end_idx_64],
                'eda_wrist': eda_wrist[start_idx_4:end_idx_4],
                'temp_wrist': temp_wrist[start_idx_4:end_idx_4]
            }
            fss = {'emg': 700.0, 'eda': 700.0, 'resp': 700.0, 'temp': 700.0, 'ecg': 700.0,
                   'bvp': 64.0, 'eda_wrist': 4.0, 'temp_wrist': 4.0}

            feature = []
            for sig_type, sig in signals.items():
                # 2维时域特征
                time_feats = extract_time_features(sig)

                # 5维频域特征
                freq_feats = extract_freq_features(sig, fss[sig_type])
                
                # 7种合并为1维
                combined_feats = np.concatenate([time_feats, freq_feats])
                feature.append(combined_feats)

            # label(1) feature(O, F)都准备好了
            features.append(feature)
            labels.append(label)

        # 一个人处理完了
        # labels(受体数*T, 1)
        # features(受体数*T, O, F) -> (受体数*T, 8, 7)
        features_1 = []
        labels_1 = []
        T = len(features)
        for i in range(T - W + 1):
            features_1.append(features[i : i + W].copy())
            labels_1.append(labels[i + W - 1].copy())
        # labels_1(N, 1)
        # features_1(N, W, O, F) -> (N, 6, 8, 7)

        # 转换成np.ndarry类型
        features_2:np.ndarray = np.array(features_1)
        labels_2:np.ndarray = np.array(labels_1)
        # features_2形状转化为(N, O, F, W) -> (N, 8, 7, 6)
        features_2 = features_2.transpose((0, 2, 3, 1))

        # GASF转化
        # 形状转化为(N, O, F, W, W) -> (N, 8, 7, 6, 6)
        local_image:np.ndarray = gasf_local(features_2)
        global_image:np.ndarray = gasf_global(features_2)
        
        # 存储
        labels_2s.append(labels_2)
        local_images.append(local_image)
        global_images.append(global_image)
    save_to_hdf5(labels_2s, local_images, global_images)
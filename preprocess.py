from scipy import signal
import numpy as np
from sklearn.preprocessing import StandardScaler

def extract_time_features(x):
    """
    提取时域特征
    """
    return [
        np.mean(x), # 均值
        np.std(x), # 标准差
        np.median(x), # 中位数
        np.max(x) - np.min(x), # 峰峰值
        np.sum(np.square(x)) / len(x), # 能量
        np.mean(np.abs(np.diff(x))) # 平均差分
    ]

def extract_freq_features(x, fs):
    f, Pxx = signal.welch(x, fs=fs, nperseg=min(256, len(x)))
    return [
        np.sum(Pxx),       # 总功率
        np.mean(Pxx),      # 平均功率
        f[np.argmax(Pxx)],  # 峰值频率
        np.sum(Pxx[:len(Pxx)//4]) / np.sum(Pxx),  # 低频功率比
    ]

def preprocess(target, window_size=700, overlap=0.5, fs=700):
    """
    对数据集中读取到的数据进行预处理
    参数:
        target: 每个实体的信号字典列表
        window_size: 窗口大小(样本数)
        overlap: 窗口重叠比例
        fs: 采样频率
    返回:
        特征矩阵和对应标签
    """
    print("preprocess begin!")
    # 初始化特征矩阵
    features = []
    labels = []
    for j, signals in enumerate(target):
        
        # pkl中已经进行了滤波处理
        # EMG滤波(带通滤波20-500Hz)
        # b_emg, a_emg = signal.butter(4, [20, 500], btype='bandpass', fs=fs)
        # signals['emg'] = signal.filtfilt(b_emg, a_emg, signals['emg'])
        
        # EDA滤波(低通滤波1Hz)
        # b_eda, a_eda = signal.butter(4, 1, btype='low', fs=fs)
        # signals['eda'] = signal.filtfilt(b_eda, a_eda, signals['eda'])
        
        # Resp滤波(带通滤波0.1-0.5Hz)
        # b_resp, a_resp = signal.butter(4, [0.1, 0.5], btype='bandpass', fs=fs)
        # signals['resp'] = signal.filtfilt(b_resp, a_resp, signals['resp'])

        # 计算步长
        step = int(window_size * (1 - overlap))
        total_length = len(signals['emg'])
        num_segments = (total_length - window_size) // step + 1

        for i in range(num_segments):
            start = i * step
            end = start + window_size

            # 提取当前窗口的信号
            emg_seg = signals['emg'][start:end]
            eda_seg = signals['eda'][start:end]
            resp_seg = signals['resp'][start:end]
            seg_label = np.mean(signals['label'][start:end])
            
            # 提取各信号特征
            emg_features = extract_time_features(emg_seg) + extract_freq_features(emg_seg, fs)
            eda_features = extract_time_features(eda_seg) + extract_freq_features(eda_seg, fs)
            resp_features = extract_time_features(resp_seg) + extract_freq_features(resp_seg, fs)

            # 合并所有特征
            all_features = emg_features + eda_features + resp_features
            features.append(all_features)

            # 如果窗口内压力状态占比超过50%，则标记为压力状态
            labels.append(1 if seg_label > 0.5 else 0)
        print(f"\rProcessing: {j+1}/{len(target)} ({((j+1)/len(target))*100:.1f}%)", end="")
    print('')
    features_array = np.array(features)
    labels_array = np.array(labels)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_array)
        
    return scaled_features, labels_array
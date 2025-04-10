import matplotlib.pyplot as plt
import numpy as np

def show_data(signal):
    """
    展示一个实体的EMG EDA Resp label
    """
    emg:np.ndarray = signal['emg']
    eda:np.ndarray = signal['eda']
    resp:np.ndarray = signal['resp']
    label:np.ndarray = signal['label']

    # 创建4行1列的子图布局
    plt.figure(figsize=(12, 8))

    # EMG信号子图
    plt.subplot(4, 1, 1)
    plt.plot(emg, linewidth=0.5, color='blue')
    plt.ylabel('Amplitude (μV)')
    plt.title('EMG Signal')
    plt.grid(True)

    plt.subplot(4, 1, 4)
    plt.plot(label, linewidth=0.5, color='grey')
    plt.ylabel('label')
    plt.title('label')
    plt.grid(True)

    # EDA信号子图
    plt.subplot(4, 1, 2)
    plt.plot(eda, linewidth=0.5, color='green')
    plt.ylabel('Conductance (μS)')
    plt.title('EDA Signal')
    plt.grid(True)

    # RSP信号子图
    plt.subplot(4, 1, 3)
    plt.plot(resp, linewidth=0.5, color='red')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.title('RSP Signal')
    plt.grid(True)

    plt.tight_layout()  # 自动调整子图间距
    plt.show()

def show_array(arr):
    plt.figure(figsize=(12, 8))
    plt.plot(arr, linewidth=0.5, color='blue')
    plt.ylabel('value')
    plt.title('arr')
    plt.grid(True)
    plt.show()
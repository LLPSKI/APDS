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
    temp:np.ndarray = signal['temp']
    ecg:np.ndarray = signal['ecg']
    bvp:np.ndarray = signal['bvp']
    eda_wrist:np.ndarray = signal['eda_wrist']
    temp_wrist:np.ndarray = signal['temp_wrist']

    # 创建9行1列的子图布局
    plt.figure(figsize=(12, 36))

    # EMG信号子图
    plt.subplot(9, 1, 1)
    plt.plot(emg, linewidth=0.5, color='blue')
    plt.ylabel('Amplitude (μV)')
    plt.title('EMG Signal')
    plt.grid(True)

    plt.subplot(9, 1, 9)
    plt.plot(label, linewidth=0.5, color='purple')
    plt.ylabel('label')
    plt.title('label')
    plt.grid(True)

    # EDA信号子图
    plt.subplot(9, 1, 2)
    plt.plot(eda, linewidth=0.5, color='green')
    plt.ylabel('Conductance (μS)')
    plt.title('EDA Signal')
    plt.grid(True)

    # RSP信号子图
    plt.subplot(9, 1, 3)
    plt.plot(resp, linewidth=0.5, color='red')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.title('RSP Signal')
    plt.grid(True)

    # Temp信号子图
    plt.subplot(9, 1, 4)
    plt.plot(temp, linewidth=0.5, color='black')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.title('TEMP Signal')
    plt.grid(True)

    # ECG信号子图
    plt.subplot(9, 1, 5)
    plt.plot(ecg, linewidth=0.5, color='pink')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.title('TEMP Signal')
    plt.grid(True)

    # BVP信号子图
    plt.subplot(9, 1, 6)
    plt.plot(bvp, linewidth=0.5, color='yellow')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.title('BVP Signal')
    plt.grid(True)

    # EDA_WRIST信号子图
    plt.subplot(9, 1, 7)
    plt.plot(eda_wrist, linewidth=0.5, color='magenta')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.title('EDA_WRIST Signal')
    plt.grid(True)

    # TEMP_WRIST信号子图
    plt.subplot(9, 1, 8)
    plt.plot(temp_wrist, linewidth=0.5, color='orange')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.title('TEMP_WRIST Signal')
    plt.grid(True)

    plt.tight_layout()  # 自动调整子图间距
    # plt.show()
    plt.savefig('Figure.png', dpi=300)
    plt.close()

def show_array(arr):
    plt.figure(figsize=(12, 8))
    plt.plot(arr, linewidth=0.5, color='blue')
    plt.ylabel('value')
    plt.title('arr')
    plt.grid(True)
    plt.show()

def plot_history(history):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(history['train_acc']) + 1)
    
    plt.plot(epochs, history['train_acc'], 'r-', label='Train Accuracy')
    plt.plot(epochs, history['val_acc'], 'g-', label='Validation Accuracy')
    plt.plot(epochs, history['test_acc'], 'b-', label='Test Accuracy')
    
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy_curve.png')
    # plt.show()
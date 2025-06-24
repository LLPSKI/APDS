import numpy as np
import os
from load import load_data
from show import show_data, plot_history
from preprocess import preprocess, load_from_hdf5
from dataset import create_two_stream_loaders
from model import TwoStream3DCNN
from train import train_model

if __name__ == "__main__":
    # 检查是否存在
    if not os.path.exists("gasf_data.h5"):
        # 加载数据
        subjects = load_data()
        # show_data(all_subjects[0])
        preprocess(subjects)
    
    # 取数据
    subjects = load_from_hdf5()

    train_loader, val_loader, test_loader = create_two_stream_loaders(subjects)
    model = TwoStream3DCNN(num_classes=4)
    history = train_model(
        model, 
        train_loader, 
        val_loader, 
        test_loader,
        num_epochs=100,
        lr=0.001
    )
    
    # 可视化结果
    plot_history(history)
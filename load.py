import os
import pickle
import numpy as np
import requests
import zipfile

def load_data():
    """
    加载WESAD数据集中的数据
    """
    print("load data begin!")

    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    wesad_zip_path = script_dir + "\WESAD.zip"
    wesad_dir = script_dir + "\WESAD"

    if not os.path.isfile(wesad_zip_path):
        print("WESAD dataset is not exists!")
        download_url = "https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download"
        try:
            print("Downloading WESAD dataset...")
            print("你觉得太慢就去网站上下载下来！")
            # 流式下载大文件
            with requests.get(download_url, stream=True) as r:
                r.raise_for_status()
                with open(wesad_zip_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print("Download completed!")
        except Exception as e:
            print(f"Download failed: {e}")
            exit()

    if not os.path.exists(wesad_dir):
        print("WESAD dataset is not zipped!")
        try:
            print("Extracting WESAD dataset...")
            with zipfile.ZipFile(wesad_zip_path, 'r') as zip_ref:
                zip_ref.extractall(script_dir)  # 解压到脚本所在目录
            print("Extraction completed!")
        except Exception as e:
            print(f"Extraction failed: {e}")
            exit()
    
    pkl_list = []
    for root, _, files in os.walk(wesad_dir):
        for file in files:
            if file.endswith(".pkl"):
                pkl_list.append(os.path.abspath(os.path.join(root, file)))
    
    target = []
    for i, pkl in enumerate(pkl_list):
        with open(pkl, "rb") as f:
            data = pickle.load(f, encoding='latin1')
            label:np.ndarray = data['label']
            emg:np.ndarray = data['signal']['chest']['EMG']
            eda:np.ndarray = data['signal']['chest']['EDA']
            resp:np.ndarray = data['signal']['chest']['Resp']
            signal = {
                'emg': emg[:, 0],
                'eda': eda[:, 0],
                'resp': resp[:, 0],
                'label': label[:]
            }
            target.append(signal)
        print(f"\rProcessing: {i+1}/{len(pkl_list)} ({((i+1)/len(pkl_list))*100:.1f}%)", end="")
    print("")
    return target


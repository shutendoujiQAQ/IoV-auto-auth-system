import os
import shutil
import pandas as pd

# 路径配置
metadata_path = 'E:/finalproject/Environment/AudioClassification-Pytorch/dataset_origin/UrbanSound8K/metadata/UrbanSound8K.csv'
audio_base_path = 'E:/finalproject/Environment/AudioClassification-Pytorch/dataset_origin/UrbanSound8K/audio'
target_folder = 'E:/finalproject/Environment/AudioClassification-Pytorch/dataset_origin/UrbanSound8K/audio/fold11'

# 加载元数据
df = pd.read_csv(metadata_path)

# 筛选出类别
car_horn_df = df[df['class'] == 'street_music']

# 创建 fold11 文件夹（如果不存在）
os.makedirs(target_folder, exist_ok=True)

# 遍历文件记录
for _, row in car_horn_df.iterrows():
    fold = f"fold{row['fold']}"
    file_name = row['slice_file_name']
    
    src_path = os.path.join(audio_base_path, fold, file_name)
    dst_path = os.path.join(target_folder, file_name)
    
    # 复制文件
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
    else:
        print(f"文件未找到: {src_path}")

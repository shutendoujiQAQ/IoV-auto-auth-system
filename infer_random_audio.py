import os
import json
import random
import threading
import pygame  # 导入pygame库用于播放音频
from typing import Optional
from macls.predict import MAClsPredictor

# ============================================================================== 
# -- Audio Inference 接口文件 infer_random_audio.py --------------------------------
# ==============================================================================

BASE_AUDIO_FOLDER = r"dataset_myproject/Trafic/audio"
CONFIGS            = r"configs/cam++.yml"
MODEL_PATH         = r"best_model/model.pth"
USE_GPU            = True

FOLD_MAP = {
    "up":    "fold10",
    "right": "fold12",
    "down":  "fold4",
    "left":  "fold6"
}

# 初始化预测器
predictor = MAClsPredictor(configs=CONFIGS,
                           model_path=MODEL_PATH,
                           use_gpu=USE_GPU)

# 初始化pygame的音频模块
pygame.mixer.init()

def _write_default(save_path: str):
    """写入默认 sound_class=1"""
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({"sound_class": 0}, f, indent=4)
    print(f"[infer_random_audio] 默认写入 sound_class=0 到 {save_path}")

def infer_random_audio(fold: Optional[str], save_path: str = "DL.json") -> int:
    """
    推理接口：写入 JSON 文件 {"sound_class": x} 到当前运行目录下的 DL.json.

    参数:
        fold (str | None): 子文件夹名称，如 "fold1"…"fold4"，或 None 表示未触发按键。
        save_path (str): JSON 保存路径，默认为工作目录下的 DL.json。

    返回:
        int: 当前写入的 sound_class 值。
    """
    if not fold:
        _write_default(save_path)
        return 1

    # 触发推理流程
    folder_path = os.path.join(BASE_AUDIO_FOLDER, fold)
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"指定的文件夹不存在: {folder_path}")
    wav_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.wav')]
    if not wav_files:
        raise FileNotFoundError(f"文件夹 {folder_path} 中未找到 .wav 文件。")
    chosen = random.choice(wav_files)
    audio_path = os.path.join(folder_path, chosen)

    # 播放音频
    pygame.mixer.music.load(audio_path)  # 加载音频
    pygame.mixer.music.play()  # 播放音频
    print(f"[infer_random_audio] 正在播放音频: {audio_path}")

    # 模型推理
    label, _ = predictor.predict(audio_data=audio_path)

    # 根据模型输出标签映射到 sound_class
    if label == 'gun_shot':
        sound_class = 1
    elif label in ('siren', 'ambulance', 'firetruck'):
        sound_class = 2
    elif label == 'car_horn':
        sound_class = 3
    elif label == 'children_playing':
        sound_class = 4
    else:
        sound_class = 0

    # 写入推理结果到当前工作目录下的 DL.json
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({"sound_class": sound_class}, f, indent=4)
    print(f"[infer_random_audio] Fold={fold}, 标签={label}, sound_class={sound_class}, 写入 {save_path}")

    # 2 秒后恢复默认写入
    timer = threading.Timer(2.0, _write_default, args=(save_path,))
    timer.daemon = True
    timer.start()

    return sound_class
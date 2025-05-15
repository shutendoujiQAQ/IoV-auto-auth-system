#!/usr/bin/env python
"""
infer_record.py

单文件实时音频分类检测工具（基于配置文件）
结合实时录音、特征提取、模型加载与预测，
并持续打印预测结果。
依赖：numpy、torch、yaml、soundcard、macls 等
"""
import argparse
import functools
import threading
import time
import warnings

import numpy as np
import soundcard as sc
import yaml

from macls.predict import MAClsPredictor
from macls.utils.utils import add_arguments, print_arguments
from soundcard.mediafoundation import SoundcardRuntimeWarning

# 忽略录音数据中断警告
warnings.filterwarnings(
    "ignore",
    "data discontinuity in recording",
    SoundcardRuntimeWarning
)

# ----------------- 参数解析 -----------------
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',        str,  'configs/cam++.yml',  '配置文件路径')
add_arg('use_gpu',        bool, True,               '是否使用GPU预测')
add_arg('record_seconds', float, 3.0,                '录音时长（秒）')
add_arg('model_path', str,"E:/finalproject/CARLA_0.9.14/WindowsNoEditor/Project/AudioClassification-Pytorch/models/CAMPPlus_Fbank/best_model",'导出的预测模型文件路径')
args = parser.parse_args()
print_arguments(args=args)

# ----------------- 加载并修正配置 -----------------
with open(args.configs, 'r', encoding='utf-8') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
# 将 model_conf.model 提升为顶层 use_model
mc = cfg.get('model_conf', {})
if 'model' in mc:
    cfg['use_model'] = mc.pop('model')
# 确保 model_args 存在
if 'model_args' not in mc or mc['model_args'] is None:
    val = mc.pop('num_class', None)
    mc['model_args'] = {}
    if val is not None:
        mc['model_args']['num_class'] = val
cfg['model_conf'] = mc

# 打印修正后的配置
print_arguments(configs=cfg)

# ----------------- 初始化预测器 -----------------
predictor = MAClsPredictor(
    configs=cfg,
    model_path=args.model_path,
    use_gpu=args.use_gpu
)

# ----------------- 补丁：兼容缺失 dataset 字段时的 AudioSegment 构造 -----------------
from macls.data_utils.featurizer import AudioFeaturizer  # 确保模块加载
from macls.data_utils.segment import AudioSegment        # 实际模块路径在 segment.py
old_load = predictor._load_audio

def _load_audio_safe(audio_data, sample_rate=16000):
    try:
        return old_load(audio_data, sample_rate)
    except KeyError:
        if isinstance(audio_data, np.ndarray):
            return AudioSegment.from_ndarray(audio_data, sample_rate)
        else:
            return old_load(audio_data, sample_rate)

predictor._load_audio = _load_audio_safe

# ----------------- 录音与推理 -----------------
all_data = []
mic_device = sc.default_microphone()

# 获取采样率，兼容 dataset_conf 中是否含 dataset 子字段
ds_conf = cfg.get('dataset_conf', {})
if 'dataset' in ds_conf and isinstance(ds_conf['dataset'], dict):
    samplerate = int(ds_conf['dataset'].get('sample_rate', 16000))
else:
    samplerate = int(ds_conf.get('sample_rate', 16000))

numframes = 1024
infer_len = int(samplerate * args.record_seconds / numframes)


def infer_thread():
    global all_data
    start_t = time.time()
    while True:
        if len(all_data) < infer_len:
            continue
        seg_list = all_data[-infer_len:]
        data = np.concatenate(seg_list, axis=0)
        del all_data[:len(all_data) - infer_len]
        label, score = predictor.predict(
            audio_data=data,
            sample_rate=samplerate
        )
        elapsed = int(time.time() - start_t)
        print(f"{elapsed}s 预测标签: {label}, 得分: {score}")

thread = threading.Thread(target=infer_thread, daemon=True)
thread.start()

with mic_device.recorder(samplerate=samplerate, channels=1) as mic:
    while True:
        try:
            chunk = mic.record(numframes=numframes)
            all_data.append(chunk.flatten())
        except Exception:
            time.sleep(0.01)
            continue

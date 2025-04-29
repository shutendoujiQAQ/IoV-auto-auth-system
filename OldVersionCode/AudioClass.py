#!/usr/bin/env python
"""
单文件实时音频分类检测工具
整合了实时录音、特征提取、模型加载与预测，
并将预测结果按照要求写入 JSON 文件：
    - 当模型输出为 "traffic" 记录为 1，
      "ambulance" 为 2，
      "fire_truck" 为 3，
      "car_horn" 为 4，
      其他标签保持原样；
    - 得分(score)保持不变。

依赖：numpy、torch、pyyaml、sounddevice、soundfile、macls 等
"""

import argparse
import functools
import os
import queue
import tempfile
import time
import json
from io import BufferedReader
from typing import List

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import yaml

# 以下依赖 macls 包，需提前安装
from macls import SUPPORT_MODEL
from macls.data_utils.audio import AudioSegment
from macls.data_utils.featurizer import AudioFeaturizer
from macls.models.campplus import CAMPPlus
from macls.models.ecapa_tdnn import EcapaTdnn
from macls.models.eres2net import ERes2NetV2, ERes2Net
from macls.models.panns import PANNS_CNN6, PANNS_CNN10, PANNS_CNN14
from macls.models.res2net import Res2Net
from macls.models.resnet_se import ResNetSE
from macls.models.tdnn import TDNN
from macls.utils.logger import setup_logger

# =================== 内置工具函数 ===================
def print_arguments(args=None, configs=None):
    if args:
        print("----------- 额外配置参数 -----------")
        for arg, value in sorted(vars(args).items()):
            print(f"{arg}: {value}")
        print("------------------------------------------------")
    if configs:
        print("----------- 配置文件参数 -----------")
        for arg, value in sorted(configs.items()):
            if isinstance(value, dict):
                print(f"{arg}:")
                for a, v in sorted(value.items()):
                    if isinstance(v, dict):
                        print(f"\t{a}:")
                        for a1, v1 in sorted(v.items()):
                            print(f"\t\t{a1}: {v1}")
                    else:
                        print(f"\t{a}: {v}")
            else:
                print(f"{arg}: {value}")
        print("------------------------------------------------")

def add_arguments(argname, type, default, help, argparser, **kwargs):
    import distutils.util
    t = distutils.util.strtobool if type == bool else type
    argparser.add_argument("--" + argname,
                           default=default,
                           type=t,
                           help=help + f' 默认: {default}.',
                           **kwargs)

class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

def dict_to_object(dict_obj):
    if not isinstance(dict_obj, dict):
        return dict_obj
    inst = Dict()
    for k, v in dict_obj.items():
        inst[k] = dict_to_object(v)
    return inst

# =================== MAClsPredictor 类 ===================
logger = setup_logger(__name__)

class MAClsPredictor:
    def __init__(self,
                 configs,
                 model_path='models/EcapaTdnn_Fbank/best_model/',
                 use_gpu=True):
        """
        声音分类预测工具
        :param configs: 配置参数（可为配置字典或配置文件路径）
        :param model_path: 导出的预测模型文件夹路径
        :param use_gpu: 是否使用GPU预测
        """
        if use_gpu:
            assert torch.cuda.is_available(), 'GPU不可用'
            self.device = torch.device("cuda")
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            self.device = torch.device("cpu")
        # 读取配置
        if isinstance(configs, str):
            with open(configs, 'r', encoding='utf-8') as f:
                configs = yaml.load(f.read(), Loader=yaml.FullLoader)
            print_arguments(configs=configs)
        self.configs = dict_to_object(configs)
        assert self.configs.use_model in SUPPORT_MODEL, f'没有该模型：{self.configs.use_model}'
        # 初始化特征提取器
        self._audio_featurizer = AudioFeaturizer(
            feature_method=self.configs.preprocess_conf.feature_method,
            use_hf_model=self.configs.preprocess_conf.get('use_hf_model', False),
            method_args=self.configs.preprocess_conf.get('method_args', {})
        )
        # 读取标签列表
        with open(self.configs.dataset_conf.label_list_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.class_labels = [l.strip() for l in lines]
        # 自动确定类别数
        if self.configs.model_conf.num_class is None:
            self.configs.model_conf.num_class = len(self.class_labels)
        # 初始化模型
        if self.configs.use_model == 'EcapaTdnn':
            self.predictor = EcapaTdnn(input_size=self._audio_featurizer.feature_dim, **self.configs.model_conf)
        elif self.configs.use_model == 'PANNS_CNN6':
            self.predictor = PANNS_CNN6(input_size=self._audio_featurizer.feature_dim, **self.configs.model_conf)
        elif self.configs.use_model == 'PANNS_CNN10':
            self.predictor = PANNS_CNN10(input_size=self._audio_featurizer.feature_dim, **self.configs.model_conf)
        elif self.configs.use_model == 'PANNS_CNN14':
            self.predictor = PANNS_CNN14(input_size=self._audio_featurizer.feature_dim, **self.configs.model_conf)
        elif self.configs.use_model == 'Res2Net':
            self.predictor = Res2Net(input_size=self._audio_featurizer.feature_dim, **self.configs.model_conf)
        elif self.configs.use_model == 'ResNetSE':
            self.predictor = ResNetSE(input_size=self._audio_featurizer.feature_dim, **self.configs.model_conf)
        elif self.configs.use_model == 'TDNN':
            self.predictor = TDNN(input_size=self._audio_featurizer.feature_dim, **self.configs.model_conf)
        elif self.configs.use_model == 'ERes2Net':
            self.predictor = ERes2Net(input_size=self._audio_featurizer.feature_dim, **self.configs.model_conf)
        elif self.configs.use_model == 'ERes2NetV2':
            self.predictor = ERes2NetV2(input_size=self._audio_featurizer.feature_dim, **self.configs.model_conf)
        elif self.configs.use_model == 'CAMPPlus':
            self.predictor = CAMPPlus(input_size=self._audio_featurizer.feature_dim, **self.configs.model_conf)
        else:
            raise Exception(f'{self.configs.use_model} 模型不存在！')
        self.predictor.to(self.device)
        # 加载模型参数
        if os.path.isdir(model_path):
            model_path = os.path.join(model_path, 'model.pth')
        assert os.path.exists(model_path), f"{model_path} 模型不存在！"
        if torch.cuda.is_available() and use_gpu:
            model_state_dict = torch.load(model_path)
        else:
            model_state_dict = torch.load(model_path, map_location='cpu')
        self.predictor.load_state_dict(model_state_dict)
        print(f"成功加载模型参数：{model_path}")
        self.predictor.eval()

    def _load_audio(self, audio_data, sample_rate=16000):
        if isinstance(audio_data, str):
            audio_segment = AudioSegment.from_file(audio_data)
        elif isinstance(audio_data, BufferedReader):
            audio_segment = AudioSegment.from_file(audio_data)
        elif isinstance(audio_data, np.ndarray):
            audio_segment = AudioSegment.from_ndarray(audio_data, sample_rate)
        elif isinstance(audio_data, bytes):
            audio_segment = AudioSegment.from_bytes(audio_data)
        else:
            raise Exception(f'不支持该数据类型，当前数据类型为：{type(audio_data)}')
        if audio_segment.sample_rate != self.configs.dataset_conf.sample_rate:
            audio_segment.resample(self.configs.dataset_conf.sample_rate)
        if self.configs.dataset_conf.use_dB_normalization:
            audio_segment.normalize(target_db=self.configs.dataset_conf.target_dB)
        assert audio_segment.duration >= self.configs.dataset_conf.min_duration, \
            f'音频太短，最小应为{self.configs.dataset_conf.min_duration}s，当前为{audio_segment.duration}s'
        return audio_segment

    def predict(self, audio_data, sample_rate=16000):
        input_data = self._load_audio(audio_data=audio_data, sample_rate=sample_rate)
        input_tensor = torch.tensor(input_data.samples, dtype=torch.float32).unsqueeze(0)
        audio_feature = self._audio_featurizer(input_tensor).to(self.device)
        output = self.predictor(audio_feature)
        result = torch.nn.functional.softmax(output, dim=-1)[0]
        result = result.data.cpu().numpy()
        lab = np.argsort(result)[-1]
        score = result[lab]
        return self.class_labels[lab], round(float(score), 5)

    def predict_batch(self, audios_data: List, sample_rate=16000):
        audios_list = []
        for audio_data in audios_data:
            input_data = self._load_audio(audio_data=audio_data, sample_rate=sample_rate)
            audios_list.append(input_data.samples)
        batch = sorted(audios_list, key=lambda a: a.shape[0], reverse=True)
        max_audio_length = batch[0].shape[0]
        batch_size = len(batch)
        inputs = np.zeros((batch_size, max_audio_length), dtype=np.float32)
        input_lens_ratio = []
        for i in range(batch_size):
            tensor = batch[i]
            seq_length = tensor.shape[0]
            inputs[i, :seq_length] = tensor
            input_lens_ratio.append(seq_length / max_audio_length)
        inputs = torch.tensor(inputs, dtype=torch.float32)
        input_lens_ratio = torch.tensor(input_lens_ratio, dtype=torch.float32)
        audio_feature = self._audio_featurizer(inputs, input_lens_ratio).to(self.device)
        output = self.predictor(audio_feature)
        results = torch.nn.functional.softmax(output, dim=-1)
        results = results.data.cpu().numpy()
        labels, scores = [], []
        for result in results:
            lab = np.argsort(result)[-1]
            score = result[lab]
            labels.append(self.class_labels[lab])
            scores.append(round(float(score), 5))
        return labels, scores

# =================== 内嵌配置（来自 cam++.yml） ===================
inline_config = """
use_model: 'CAMPPlus'

dataset_conf:
  sample_rate: 16000
  use_dB_normalization: True
  target_dB: -20
  min_duration: 0.4
  max_duration: 3
  label_list_path: "E:/finalproject/CARLA_0.9.14/WindowsNoEditor/Project/AudioClassification-Pytorch/dataset_myproject/label_list.txt"

preprocess_conf:
  use_hf_model: False
  feature_method: 'Fbank'
  method_args:
    sample_frequency: 16000
    num_mel_bins: 80

model_conf:
  num_class: null

infer_conf:
  use_gpu: True
"""

# =================== 主函数：实时录音与预测、写入JSON ===================
def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"录音状态：{status}")
    global q
    q.put(indata.copy())

def main():
    parser = argparse.ArgumentParser(description="实时音频分类检测")
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg('configs', str, inline_config, '配置文件内容（内嵌）')
    add_arg('use_gpu', bool, True, '是否使用GPU预测')
    add_arg('model_path', str, "best_model", '导出的预测模型文件路径')
    add_arg('sample_rate', int, 16000, '采样率')
    add_arg('segment_length', float, 3.0, '每个音频片段的时长（秒）')
    add_arg('hop_length', float, 1.5, '窗口滑动步长（秒），用于实现重叠')
    parser.add_argument('--output_json', type=str, default="DL.json", help="输出JSON文件路径，默认 output.json")
    args = parser.parse_args()
    print_arguments(args=args)

    # 为方便 MAClsPredictor 加载配置，将内嵌配置内容写入临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False, encoding='utf-8') as tmp_config_file:
        tmp_config_file.write(args.configs)
        config_path = tmp_config_file.name

    predictor = MAClsPredictor(configs=config_path,
                               model_path=args.model_path,
                               use_gpu=args.use_gpu)

    sample_rate = args.sample_rate
    segment_sec = args.segment_length
    hop_sec = args.hop_length
    num_samples_segment = int(sample_rate * segment_sec)
    num_samples_hop = int(sample_rate * hop_sec)

    buffer = np.array([], dtype=np.float32)
    print("开始实时声音监测，按 Ctrl+C 退出...")
    global q
    q = queue.Queue()

    # 标签映射：将特定标签映射为指定数字
    label_mapping = {
        "traffic": 1,
        "ambulance": 2,
        "firetruck": 3,
        "car_horn": 4,
    }

    # 打开 JSON 输出文件（追加模式）
    output_file = args.output_json
    # 清空或创建输出文件
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("")

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32', callback=audio_callback):
        try:
            while True:
                try:
                    data = q.get(timeout=1)
                except queue.Empty:
                    continue
                data = data.flatten()
                buffer = np.concatenate((buffer, data))
                while len(buffer) >= num_samples_segment:
                    segment = buffer[:num_samples_segment]
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmpfile:
                        tmp_filename = tmpfile.name
                        sf.write(tmp_filename, segment, sample_rate)
                    label, score = predictor.predict(audio_data=tmp_filename)
                    # 应用标签映射（若存在映射，则转换为数字，否则保持原样）
                    mapped_label = label_mapping.get(label, label)
                    result_dict = {"sound_class": mapped_label}
                    result_print = {"label": mapped_label, "score": score}
                    # 将结果写入 JSON 文件（每行一个JSON对象）
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write(json.dumps(result_dict, ensure_ascii=False) + "\n")
                    print(f"预测结果已写入JSON：{result_print}")
                    os.remove(tmp_filename)
                    buffer = buffer[num_samples_hop:]
        except KeyboardInterrupt:
            print("程序终止。")
            os.remove(config_path)

if __name__ == "__main__":
    main()

'''
Author: Zeta112233 15311410306@163.com
Date: 2025-03-15 11:24:51
LastEditors: Zeta112233 15311410306@163.com
LastEditTime: 2025-04-03 17:03:45
FilePath: \AudioClassification-Pytorch\extract_features.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import argparse
import functools
from macls.trainer import MAClsTrainer
from macls.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,    'configs/cam++.yml',        '配置文件')
add_arg('save_dir',         str,    'dataset_myproject/features',         '保存特征的路径')
add_arg('max_duration',    int,     100,                        '提取特征的最大时长，避免过长显存不足，单位秒')
args = parser.parse_args()
print_arguments(args=args)

# 获取训练器
trainer = MAClsTrainer(configs=args.configs)

# 提取特征保存文件
trainer.extract_features(save_dir=args.save_dir, max_duration=args.max_duration)

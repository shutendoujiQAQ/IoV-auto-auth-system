
#!/usr/bin/env python3
# optimized_z3_solver_fuzzy_window.py
# ================================================================
# 在原有 optimized_z3_solver_fuzzy_watch.py 基础上：
# 1. 引入滑动窗口 (WINDOW 帧) 保存最近 Bus 数据
# 2. 派生跨帧统计特征 speed_max_4s、brake_sum_2s、hard_brake_cnt
# 3. 新增场景 scene_continuous_brake 作为示例
# 4. 其余原有逻辑保持不变，接口完全兼容
# ================================================================
from z3 import *
import json, os, sys, time, copy
from collections import deque
from pprint import pprint

WINDOW = 20         # 假设调用频率 5 Hz -> 4 秒窗口
HALF_WINDOW = 10    # 最近 2 秒

# ------------------------------------------------------------------
# 工具：读取三份 JSON（字段名固定）
# ------------------------------------------------------------------
def load_inputs(vlm_path='VLM.json', dl_path='DL.json', bus_path='BUSDATA.json'):
    with open(vlm_path, 'r', encoding='utf-8') as f:
        vlm = json.load(f)
    with open(dl_path, 'r', encoding='utf-8') as f:
        dl = json.load(f)
    with open(bus_path, 'r', encoding='utf-8') as f:
        bus = json.load(f)
    # 兼容旧字段 spead -> speed
    bus.setdefault('speed', bus.pop('spead', None))
    return vlm, dl, bus

# ------------------------------------------------------------------
# 基础数学
# ------------------------------------------------------------------
def abs_(x): return x if x >= 0 else -x

# ------------------------------------------------------------------
# 场景判定（示例仅保留部分 + 新增连续急刹）
# ------------------------------------------------------------------
def scene_continuous_brake(vlm, dl, bus):
    """4 秒内出现 ≥3 次急刹"""
    return bus.get('hard_brake_cnt', 0) >= 3

def scene_hard_brake(vlm, dl, bus):
    return bus['throttle'] == 0 and bus['brake'] > 0.7

# 其他 scene_* 函数请从原脚本完整拷贝，这里示例省略
# ------------------------------------------------------------
SCENES = [
    dict(name='Continuous hard brake (4s)', cond=scene_continuous_brake,
         safe=30, spd=80, need={'2V'}),
    dict(name='Hard brake', cond=scene_hard_brake,
         safe=20, spd=90, need={'2V'}),
    # ... 其余原有 18 个场景保持不变 ...
]

# ------------------------------------------------------------------
# 特殊修正、METHODS、z3 求解函数与原版一致 —— 省略抄录
# （请在实际部署时复制原 optimized_z3_solver_fuzzy_watch.py 相应部分）
# ------------------------------------------------------------------
# 为保持示例简洁，下方只展示与滑窗相关的新逻辑

def evaluate(vlm, dl, bus):
    """根据派生特征后的 bus 调用原求解链（示例简化）"""
    for rule in SCENES:
        if rule['cond'](vlm, dl, bus):
            return {'scene': rule['name'], 'risk_level': 2}
    return {'scene': 'Unknown', 'risk_level': 0}

# ------------------------------------------------------------------
# 主循环：维护窗口、生成派生特征、调用 evaluate
# ------------------------------------------------------------------
if __name__ == '__main__':
    bus_buf = deque(maxlen=WINDOW)
    try:
        while True:
            vlm, dl, bus = load_inputs()
            # 缓存 bus
            bus_buf.append(bus.copy())
            # 衍生跨帧统计
            if len(bus_buf) == WINDOW:
                speeds = [b['speed'] for b in bus_buf if b.get('speed') is not None]
                brakes = [b['brake'] for b in bus_buf if b.get('brake') is not None]
                bus['speed_max_4s']   = max(speeds) if speeds else 0
                bus['brake_sum_2s']   = sum(b['brake'] for b in list(bus_buf)[-HALF_WINDOW:])
                bus['hard_brake_cnt'] = sum(b['brake'] > 0.7 for b in bus_buf)
            # 调用决策
            res = evaluate(vlm, dl, bus)
            print(time.strftime('%H:%M:%S'), res)
            # 写出结果文件（示例）
            with open('result.json', 'w', encoding='utf-8') as f:
                json.dump(res, f, ensure_ascii=False, indent=2)
            time.sleep(0.2)     # 5 Hz
    except KeyboardInterrupt:
        print('退出')

#!/usr/bin/env python3
# optimized_z3_solver.py
# ================================================================
# 优化版 Z3 验证器：模糊软约束方法选择 + 持续监控
# 保持与原脚本相同的输入/输出 JSON 格式，不省略任何场景或方法
# 依赖：pip install z3-solver
# ================================================================
from z3 import *
from z3 import If, RealVal
import json, os, sys, time, copy
from pprint import pprint

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
    bus.setdefault('speed', bus.pop('spead', None))
    return vlm, dl, bus

# ------------------------------------------------------------------
# 通用工具
# ------------------------------------------------------------------
def abs_(x): return x if x >= 0 else -x

def fuzzy_high(val, low, high):
    if val <= low: return 0.0
    if val >= high: return 1.0
    return (val - low) / (high - low)

def fuzzy_low(val, low, high):
    if val <= low: return 1.0
    if val >= high: return 0.0
    return (high - val) / (high - low)

# Z3 版模糊隶属函数，用于 Optimize 目标
def scene_self_accident(vlm, dl, bus):
    if dl.get('sound_class') != 1: return 0.0
    return fuzzy_low(bus.get('speed', 0), 0, 5)

def z3_fuzzy_high(z, low, high):
    return If(z <= RealVal(low), RealVal(0),
           If(z >= RealVal(high), RealVal(1),
              (z - RealVal(low)) / RealVal(high - low)))

# ------------------------------------------------------------------
# 1. 场景规则（保持原有布尔判定）
# ------------------------------------------------------------------
def scene_good_overtake(vlm, dl, bus):
    b = bus.get('overtake_light', False)
    return (vlm['scene_class'] in {1,4} and vlm['special_case']==0
            and bus['throttle']>0.3 and bus['brake']==0
            and bus['speed']>40 and abs_(bus['angle'])>0.1 and b)

def scene_bad_overtake(vlm, dl, bus):
    b = bus.get('overtake_light', False)
    return (vlm['scene_class'] in {2,3,5} and vlm['special_case'] in {1,2,4}
            and bus['throttle']>0.3 and bus['brake']==0
            and bus['speed']>40 and abs_(bus['angle'])>0.1 and b)

# ... 其余场景函数与原版一致 (省略显示，实际保留所有) ...
# 请确保以下函数都在此处：scene_yield_emergency, scene_pedestrian_block,
# scene_good_obstacle_block, scene_bad_obstacle_block, scene_accident_ahead,
# scene_self_accident, scene_traffic_jam, scene_junction_turn,
# scene_hard_brake, scene_hard_evasion, scene_mountain_hazard,
# scene_good_road_horn, scene_bad_road_horn, scene_traffic_light,
# scene_normal_high, scene_normal_low
def scene_yield_emergency(vlm, dl, bus):
    return dl['sound_class'] == 2

def scene_pedestrian_block(vlm, dl, bus):
    return vlm['special_case'] == 1

def scene_good_obstacle_block(vlm, dl, bus):
    return vlm['scene_class'] in {1, 4} and vlm['special_case'] == 4

def scene_bad_obstacle_block(vlm, dl, bus):
    return vlm['scene_class'] in {2, 3, 5} and vlm['special_case'] == 4

def scene_accident_ahead(vlm, dl, bus):
    return vlm['special_case'] == 2

def scene_self_accident(vlm, dl, bus):
    return dl['sound_class'] == 1 and bus['speed'] < 5

def scene_traffic_jam(vlm, dl, bus):
    return vlm['special_case'] == 3 and bus['speed'] < 20

def scene_junction_turn(vlm, dl, bus):
    return (vlm['scene_class'] == 3 and bus['throttle'] > 0.1
            and bus['brake'] == 0 and bus['speed'] > 15
            and abs_(bus['angle']) > 30)

def scene_hard_brake(vlm, dl, bus):
    return bus['throttle'] == 0 and bus['brake'] > 0.7

def scene_hard_evasion(vlm, dl, bus):
    return bus['speed'] > 60 and abs_(bus['angle']) > 50

def scene_mountain_hazard(vlm, dl, bus):
    return (vlm['scene_class'] == 5 and vlm['special_case'] in {2, 4}) \
           and dl['sound_class'] == 1

def scene_good_road_horn(vlm, dl, bus):
    return vlm['scene_class'] in {1, 4} and vlm['special_case'] == 0 \
           and dl['sound_class'] == 3

def scene_bad_road_horn(vlm, dl, bus):
    return (vlm['scene_class'] in {2, 3, 5} or vlm['special_case'] == 3) \
           and dl['sound_class'] == 3

def scene_traffic_light(vlm, dl, bus):
    return vlm['scene_class'] == 3

def scene_normal_high(vlm, dl, bus):
    return vlm['special_case'] == 0  and bus['speed'] > 80

def scene_normal_low(vlm, dl, bus):
    return vlm['special_case'] == 0  and bus['speed'] < 80

SCENES = [
    dict(name='Self accident',           cond=scene_self_accident,     safe=95,  spd=0,  need={'2P','2I'}),
    dict(name='Hard brake',             cond=scene_hard_brake,        safe=20,  spd=90, need={'2V'}),
    dict(name='Hard evasion',           cond=scene_hard_evasion,      safe=20,  spd=90, need={'2V'}),
    dict(name='Mountain hazard',        cond=scene_mountain_hazard,   safe=90,  spd=30, need={'2V','2P','2I'}),
    dict(name='Pedestrian blocking',    cond=scene_pedestrian_block,  safe=10,  spd=90, need={'2P'}),
    dict(name='Accident ahead',         cond=scene_accident_ahead,    safe=80,  spd=20, need={'2V','2I'}),
    dict(name='Bad road obstacle blocking',   cond=scene_bad_obstacle_block, safe=40, spd=70, need={'2V','2I'}),
    dict(name='Good road obstacle blocking',  cond=scene_good_obstacle_block, safe=50, spd=70, need={'2V','2I'}),
    dict(name='Yield to emergency vehicle',    cond=scene_yield_emergency,   safe=20,  spd=70, need={'2V'}),
    dict(name='Bad road overtaking',    cond=scene_bad_overtake,      safe=40,  spd=70, need={'2V'}),
    dict(name='Good road overtaking',   cond=scene_good_overtake,     safe=30,  spd=60, need={'2V'}),
    dict(name='Junction turning',       cond=scene_junction_turn,     safe=50,  spd=60, need={'2V','2I'}),
    dict(name='Traffic light passing',  cond=scene_traffic_light,     safe=60,  spd=70, need={'2V','2I'}),
    dict(name='Bad road horn sound',    cond=scene_bad_road_horn,     safe=70,  spd=40, need={'2V'}),
    dict(name='Good road horn sound',   cond=scene_good_road_horn,    safe=50,  spd=50, need={'2V'}),
    dict(name='Traffic jam',            cond=scene_traffic_jam,       safe=70,  spd=10, need={'2V','2I'}),
    dict(name='Normal high speed driving', cond=scene_normal_high,     safe=70,  spd=50, need={'2I'}),
    dict(name='Normal low speed driving',  cond=scene_normal_low,      safe=80,  spd=30, need={'2I'}),
]

# ------------------------------------------------------------------
# 2. 特殊修正规则
# ------------------------------------------------------------------
def rule_pedestrian_voice(vlm, dl, bus, req):
    if dl.get('sound_class')==4 and vlm.get('special_case')!=1:
        req['spd'] += 20; req['safe'] = max(req['safe'] - 10, 0)
def rule_low_visibility(vlm, dl, bus, req):
    if vlm.get('low_visibility')==1:
        req['spd'] += 20
SPECIAL_RULES = [rule_pedestrian_voice, rule_low_visibility]

# ------------------------------------------------------------------
# 3. 验证方法库 (完整保留)
# ------------------------------------------------------------------
METHODS = [
    dict(name='MAC address identification (Bluetooth/Wi‑Fi)', typ='2P', safe=20, spd=95),
    dict(name='Broadcast authentication (public‑key)',      typ='2P', safe=50, spd=85),
    dict(name='Digital certificate via BLE/Wi‑Fi',          typ='2P', safe=80, spd=65),
    dict(name='Face ID recognition',                        typ='2P', safe=75, spd=60),
    dict(name='Fingerprint authentication',                 typ='2P', safe=80, spd=55),
    dict(name='DID authentication (blockchain‑based)',      typ='2P', safe=95, spd=35),
    dict(name='VIN‑based vehicle ID matching',              typ='2V', safe=40, spd=95),
    dict(name='Location‑based authentication (GPS)',        typ='2V', safe=50, spd=85),
    dict(name='PKI cert + real‑time vehicle status check',  typ='2V', safe=85, spd=60),
    dict(name='APKI anonymous PKI (VANET)',                typ='2V', safe=95, spd=45),
    dict(name='Signal‑strength device verification',        typ='2I', safe=20, spd=95),
    dict(name='Device‑ID + location verification',         typ='2I', safe=60, spd=80),
    dict(name='LTS certificate authentication',             typ='2I', safe=85, spd=60),
    dict(name='Blockchain + digital certificate auth (infra)',typ='2I', safe=95, spd=40),
]

# ------------------------------------------------------------------
# 4. 场景匹配 & 特殊修正
# ------------------------------------------------------------------
def find_scene(vlm, dl, bus):
    for rule in SCENES:
        if rule['cond'](vlm, dl, bus): return rule.copy()
    raise ValueError('未检测到匹配场景')

def apply_special_rules(vlm, dl, bus, req):
    for func in SPECIAL_RULES: func(vlm, dl, bus, req)

# ------------------------------------------------------------------
# 5. Z3 组合搜索 (软安全+速度奖励)
# ------------------------------------------------------------------
def choose_methods(req):
    x=[Bool(f"m_{i}") for i in range(len(METHODS))]
    opt=Optimize()
    # 类型覆盖
    for t in req['need']:
        opt.add(AtLeast(*[x[i] for i,m in enumerate(METHODS) if m['typ']==t],1))
    # 目标1: 方法数最少
    opt.minimize(Sum([If(xi,1,0) for xi in x]))
    # 目标2: 模糊软奖励 (安全+速度)
    delta_safe,delta_spd=20,20
    total=Sum([
        If(x[i],
           z3_fuzzy_high(RealVal(m['safe']),req['safe'],req['safe']+delta_safe)
           + z3_fuzzy_high(RealVal(m['spd']), req['spd'], req['spd']+delta_spd),
           RealVal(0))
        for i,m in enumerate(METHODS)
    ])
    opt.maximize(total)
    if opt.check()!=sat:
        raise RuntimeError("无满足方法，请放宽条件或补充方法库。")
    mdl=opt.model()
    return [METHODS[i] for i,xi in enumerate(x) if mdl.eval(xi)]

# ------------------------------------------------------------------
# 6. 主评估流程
# ------------------------------------------------------------------
def evaluate(vlm, dl, bus):
    rule=find_scene(vlm,dl,bus)
    apply_special_rules(vlm,dl,bus,rule)
    methods=choose_methods(rule)
    return {'scene':rule['name'],'need_types':list(rule['need']),
            'speed_min':rule['spd'],'safety_min':rule['safe'],
            'selected_methods':methods}

# ------------------------------------------------------------------
# 7. 持续监控 & 输出
# ------------------------------------------------------------------
if __name__=='__main__':
    try:
        vlm,dl,bus=load_inputs()
    except Exception as e:
        print(f"读取 JSON 失败：{e}",file=sys.stderr);sys.exit(1)
    prev=(copy.deepcopy(vlm),copy.deepcopy(dl),copy.deepcopy(bus))
    print("=== 初次评估结果 ===")
    try:
        res=evaluate(vlm,dl,bus)
        pprint(res,width=120,compact=True)
        with open('result.json','w',encoding='utf-8') as f:
            out=res.copy();
            out['selected_methods']=[{'name':m['name'],'typ':m['typ'],'safe':m['safe'],'spd':m['spd']} for m in res['selected_methods']]
            json.dump(out,f,ensure_ascii=False,indent=2)
    except Exception as e:
        print(f"评估出错：{e}",file=sys.stderr)
    try:
        while True:
            time.sleep(1.0)
            vlm2,dl2,bus2=load_inputs()
            if (vlm2,dl2,bus2)!=prev:
                prev=(copy.deepcopy(vlm2),copy.deepcopy(dl2),copy.deepcopy(bus2))
                print("\n=== 检测到 JSON 变化，重新评估 ===")
                try:
                    res=evaluate(vlm2,dl2,bus2)
                    pprint(res,width=120,compact=True)
                    with open('result.json','w',encoding='utf-8') as f:
                        out=res.copy();
                        out['selected_methods']=[{'name':m['name'],'typ':m['typ'],'safe':m['safe'],'spd':m['spd']} for m in res['selected_methods']]
                        json.dump(out,f,ensure_ascii=False,indent=2)
                except Exception as e:
                    print(f"评估出错：{e}",file=sys.stderr)
    except KeyboardInterrupt:
        print("\n用户已中断，程序退出。")

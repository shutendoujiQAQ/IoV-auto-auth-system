# judge_solver_structured.py
# ================================================================
# Continuously monitor JSON file changes and trigger evaluation, writing results to result.json
# Dependency: pip install z3-solver
# ================================================================
from z3 import *
import json, os, sys, time, copy
from pprint import pprint

# ------------------------------------------------------------------
# Utility: Read three JSON files (field names fixed)
# ------------------------------------------------------------------
def load_inputs(vlm_path='VLM.json', dl_path='DL.json', bus_path='BUSDATA.json'):
    with open(vlm_path, 'r', encoding='utf-8') as f:
        vlm = json.load(f)
    with open(dl_path, 'r', encoding='utf-8') as f:
        dl = json.load(f)
    with open(bus_path, 'r', encoding='utf-8') as f:
        bus = json.load(f)
    
    return vlm, dl, bus

# ------------------------------------------------------------------
# 0-1 Helper functions
# ------------------------------------------------------------------
def abs_(x):
    return x if x >= 0 else -x

# ------------------------------------------------------------------
# 0-2 Fuzzy logic functions
# ------------------------------------------------------------------
def fuzzy_greater(value, threshold, transition_width):
    """返回value > threshold的模糊隶属度"""
    if value >= threshold + transition_width:
        return 1.0
    elif value <= threshold - transition_width:
        return 0.0
    else:
        # 线性过渡区域
        return (value - (threshold - transition_width)) / (2 * transition_width)

def fuzzy_less(value, threshold, transition_width):
    """返回value < threshold的模糊隶属度"""
    if value <= threshold - transition_width:
        return 1.0
    elif value >= threshold + transition_width:
        return 0.0
    else:
        # 线性过渡区域
        return ((threshold + transition_width) - value) / (2 * transition_width)

def fuzzy_equal(value, target, tolerance):
    """返回value ≈ target的模糊隶属度"""
    diff = abs(value - target)
    if diff <= tolerance:
        return 1.0
    elif diff >= 2 * tolerance:
        return 0.0
    else:
        # 线性过渡区域
        return (2 * tolerance - diff) / tolerance

def fuzzy_and(a, b):
    """模糊逻辑AND操作（取最小值）"""
    return min(a, b)

def fuzzy_or(a, b):
    """模糊逻辑OR操作（取最大值）"""
    return max(a, b)

def fuzzy_and_multiple(*args):
    """多个条件的模糊AND操作"""
    if not args:
        return 1.0
    return min(args)

# ------------------------------------------------------------------
# 1. Scene rules
# ------------------------------------------------------------------
def scene_good_overtake(vlm, dl, bus):
    # VLM和DL参数使用硬条件
    hard_cond = (vlm['scene_class'] in {1, 4} and vlm['special_case'] == 0)
    if not hard_cond:
        return 0.0  # 如果硬条件不满足，直接返回0置信度
    
    # BUSDATA参数使用模糊逻辑
    throttle_conf = fuzzy_greater(bus['throttle'], 0.3, 0.1)
    brake_conf = fuzzy_equal(bus['brake'], 0, 0.05)
    speed_conf = fuzzy_greater(bus['speed'], 40, 5)
    angle_conf = fuzzy_greater(abs_(bus['angle']), 0, 2)
    
    # 计算总体置信度
    return fuzzy_and_multiple(throttle_conf, brake_conf, speed_conf, angle_conf)

def scene_bad_overtake(vlm, dl, bus):
    # VLM和DL参数使用硬条件
    hard_cond = (vlm['scene_class'] in {2, 3, 5} and vlm['special_case'] in {1, 2, 4})
    if not hard_cond:
        return 0.0  # 如果硬条件不满足，直接返回0置信度
    
    # BUSDATA参数使用模糊逻辑
    throttle_conf = fuzzy_greater(bus['throttle'], 0.3, 0.1)
    brake_conf = fuzzy_equal(bus['brake'], 0, 0.05)
    speed_conf = fuzzy_greater(bus['speed'], 40, 5)
    angle_conf = fuzzy_greater(abs_(bus['angle']), 0, 2)
    
    # 计算总体置信度
    return fuzzy_and_multiple(throttle_conf, brake_conf, speed_conf, angle_conf)

def scene_yield_emergency(vlm, dl, bus):
    # 紧急车辆声音是确定的分类，使用硬条件
    return 1.0 if dl['sound_class'] == 2 else 0.0

def scene_pedestrian_block(vlm, dl, bus):
    # 行人阻挡是确定的特殊情况，使用硬条件
    return 1.0 if vlm['special_case'] == 1 else 0.0

def scene_good_obstacle_block(vlm, dl, bus):
    # VLM参数使用硬条件
    return 1.0 if (vlm['scene_class'] in {1, 4} and vlm['special_case'] == 4) else 0.0

def scene_bad_obstacle_block(vlm, dl, bus):
    # VLM参数使用硬条件
    return 1.0 if (vlm['scene_class'] in {2, 3, 5} and vlm['special_case'] == 4) else 0.0

def scene_accident_ahead(vlm, dl, bus):
    # VLM参数使用硬条件
    return 1.0 if vlm['special_case'] == 2 else 0.0

def scene_self_accident(vlm, dl, bus):
    # DL参数使用硬条件
    hard_cond = (dl['sound_class'] == 1)
    if not hard_cond:
        return 0.0
    
    # BUSDATA参数使用模糊逻辑
    speed_conf = fuzzy_equal(bus['speed'], 0, 2)
    
    return speed_conf

def scene_traffic_jam(vlm, dl, bus):
    # VLM参数使用硬条件
    hard_cond = (vlm['special_case'] == 3)
    if not hard_cond:
        return 0.0
    
    # BUSDATA参数使用模糊逻辑
    speed_conf = fuzzy_less(bus['speed'], 20, 5)
    
    return speed_conf

def scene_junction_turn(vlm, dl, bus):
    # VLM参数使用硬条件
    hard_cond = (vlm['scene_class'] == 3)
    if not hard_cond:
        return 0.0
    
    # BUSDATA参数使用模糊逻辑
    throttle_conf = fuzzy_greater(bus['throttle'], 0.1, 0.05)
    brake_conf = fuzzy_equal(bus['brake'], 0, 0.05)
    speed_conf = fuzzy_greater(bus['speed'], 15, 3)
    angle_conf = fuzzy_greater(abs_(bus['angle']), 30, 5)
    
    return fuzzy_and_multiple(throttle_conf, brake_conf, speed_conf, angle_conf)

def scene_hard_brake(vlm, dl, bus):
    # BUSDATA参数使用模糊逻辑
    throttle_conf = fuzzy_equal(bus['throttle'], 0, 0.05)
    brake_conf = fuzzy_greater(bus['brake'], 0.7, 0.1)
    
    return fuzzy_and(throttle_conf, brake_conf)

def scene_hard_evasion(vlm, dl, bus):
    # BUSDATA参数使用模糊逻辑
    speed_conf = fuzzy_greater(bus['speed'], 60, 5)
    angle_conf = fuzzy_greater(abs_(bus['angle']), 50, 5)
    
    return fuzzy_and(speed_conf, angle_conf)

def scene_mountain_hazard(vlm, dl, bus):
    # VLM和DL参数使用硬条件
    vlm_cond = (vlm['scene_class'] == 5 and vlm['special_case'] in {2, 4})
    dl_cond = (dl['sound_class'] == 1)
    
    return 1.0 if (vlm_cond or dl_cond) else 0.0

def scene_good_road_horn(vlm, dl, bus):
    # VLM和DL参数使用硬条件
    return 1.0 if (vlm['scene_class'] in {1, 4} and vlm['special_case'] == 0 \
           and dl['sound_class'] == 3) else 0.0

def scene_bad_road_horn(vlm, dl, bus):
    # VLM和DL参数使用硬条件
    return 1.0 if ((vlm['scene_class'] in {2, 3, 5} or vlm['special_case'] == 3) \
           and dl['sound_class'] == 3) else 0.0

def scene_traffic_light(vlm, dl, bus):
    # VLM参数使用硬条件
    return 1.0 if vlm['scene_class'] == 3 else 0.0

def scene_normal_high(vlm, dl, bus):
    # VLM和DL参数使用硬条件
    hard_cond = (vlm['special_case'] == 0 and dl['sound_class'] == 0)
    if not hard_cond:
        return 0.0
    
    # BUSDATA参数使用模糊逻辑
    speed_conf = fuzzy_greater(bus['speed'], 80, 5)
    
    return speed_conf

def scene_normal_low(vlm, dl, bus):
    # VLM和DL参数使用硬条件
    hard_cond = (vlm['special_case'] == 0 and dl['sound_class'] == 0)
    if not hard_cond:
        return 0.0
    
    # BUSDATA参数使用模糊逻辑
    speed_conf = fuzzy_less(bus['speed'], 80, 5)
    
    return speed_conf

SCENES = [
    dict(name='Encountering a car accident on one\'s own', cond=scene_self_accident,     safe=91,  spd=68,  need={'2P','2I'}),
    dict(name='Emergency brake',             cond=scene_hard_brake,        safe=60,  spd=70, need={'2V'}),
    dict(name='Emergency avoidance',             cond=scene_hard_evasion,      safe=60,  spd=68, need={'2V'}),
    dict(name='Mountain road in distress',             cond=scene_mountain_hazard,   safe=60,  spd=54, need={'2V','2P','2I'}),
    dict(name='Pedestrians blocking ahead',           cond=scene_pedestrian_block,  safe=45,  spd=74, need={'2P'}),
    dict(name='A car accident occurred ahead',           cond=scene_accident_ahead,    safe=81,  spd=60, need={'2V','2I'}),
    dict(name='Obstacles blocking the road ahead in unfavorable conditions', cond=scene_bad_obstacle_block, safe=53, spd=54, need={'2V','2I'}),
    dict(name='Good obstacle blocking ahead',     cond=scene_good_obstacle_block, safe=28, spd=62, need={'2V','2I'}),
    dict(name='Avoid special vehicles',           cond=scene_yield_emergency,   safe=35,  spd=74, need={'2V'}),
    dict(name='Overtaking on non good road conditions',         cond=scene_bad_overtake,      safe=33,  spd=46, need={'2V'}),
    dict(name='Overtaking on good road conditions',           cond=scene_good_overtake,     safe=23,  spd=62, need={'2V'}),
    dict(name='Turn/U-turn at the intersection',         cond=scene_junction_turn,     safe=47,  spd=54, need={'2V','2I'}),
    dict(name='Traffic lights at intersections',         cond=scene_traffic_light,     safe=46,  spd=48, need={'2V','2I'}),
    dict(name='Vehicles honking on non good road conditions',   cond=scene_bad_road_horn,     safe=33,  spd=52, need={'2V'}),
    dict(name='Good road conditions with vehicles honking their horns',     cond=scene_good_road_horn,    safe=23,  spd=52, need={'2V'}),
    dict(name='Traffic jam',                 cond=scene_traffic_jam,       safe=37,  spd=44, need={'2V','2I'}),
    dict(name='Normal high-speed driving',           cond=scene_normal_high,       safe=37,  spd=50, need={'2I'}),
    dict(name='Normal low-speed driving',           cond=scene_normal_low,        safe=25,  spd=44, need={'2I'}),
]

# ------------------------------------------------------------------
# 2. Special rules (speed +20 / safety −10; low visibility speed +20)
# ------------------------------------------------------------------
def rule_pedestrian_voice(vlm, dl, bus, req):
    if dl.get('sound_class') == 4 and vlm.get('special_case') != 1:
        req['spd']  += 20
        req['safe'] = max(req['safe'] - 10, 0)

def rule_low_visibility(vlm, dl, bus, req):
    if vlm.get('low_visibility') == 1:
        req['spd'] += 20

SPECIAL_RULES = [rule_pedestrian_voice, rule_low_visibility]

# ------------------------------------------------------------------
# 3. Verification method library
# ------------------------------------------------------------------
METHODS = [
    dict(name='MAC address identification (Bluetooth/Wi‑Fi)',            typ='2P', safe=11.25, spd=42.5),
    dict(name='Broadcast authentication (public‑key)',                   typ='2P', safe=28.5, spd=30),
    dict(name='Digital certificate via BLE/Wi‑Fi',                       typ='2P', safe=28.5, spd=30),
    dict(name='Face ID recognition',                                     typ='2P', safe=23.75, spd=35.5),
    dict(name='Fingerprint authentication',                              typ='2P', safe=23.75, spd=47),
    dict(name='DID authentication (blockchain‑based)',                   typ='2P', safe=43.75, spd=11.5),
    dict(name='VIN‑based vehicle ID matching',                           typ='2V', safe=9.75, spd=42.5),
    dict(name='Location‑based authentication (GPS)',                     typ='2V', safe=9.75, spd=27),
    dict(name='PKI cert + real‑time vehicle status check',               typ='2V', safe=40.25, spd=17.5),
    dict(name='APKI anonymous PKI (VANET)',                             typ='2V', safe=33.5, spd=23),
    dict(name='Signal‑strength device verification',                     typ='2I', safe=9.75, spd=42.5),
    dict(name='Device‑ID + location verification',                       typ='2I', safe=12.5, spd=30),
    dict(name='LTS certificate authentication',                          typ='2I', safe=20.5, spd=30),
    dict(name='Blockchain + digital certificate authentication (infra)', typ='2I', safe=28.5, spd=11.5),
]

# ------------------------------------------------------------------
# 4. Scene matching & special adjustments
# ------------------------------------------------------------------
def find_scene(vlm, dl, bus):
    # 计算每个场景的置信度
    scene_confidences = []
    for rule in SCENES:
        confidence = rule['cond'](vlm, dl, bus)
        if confidence > 0.0:
            rule_copy = rule.copy()
            rule_copy['confidence'] = confidence
            scene_confidences.append(rule_copy)
    
    # 如果没有匹配的场景，尝试进行模糊判断
    if not scene_confidences:
        # 计算每个场景的相似度得分
        fuzzy_scores = []
        for rule in SCENES:
            # 使用模糊逻辑计算各个参数的相似度
            # 这里我们基于BUSDATA参数进行模糊评估
            bus_score = 0.0
            
            # 评估油门、刹车、速度和转向角度的相似度
            if 'throttle' in bus:
                bus_score += fuzzy_greater(bus['throttle'], 0.1, 0.2) * 0.25
            if 'brake' in bus:
                bus_score += (1 - fuzzy_greater(bus['brake'], 0.3, 0.3)) * 0.25
            if 'speed' in bus:
                bus_score += fuzzy_greater(bus['speed'], 20, 30) * 0.25
            if 'angle' in bus:
                bus_score += fuzzy_greater(abs_(bus.get('angle', 0)), 5, 20) * 0.25
            
            # 如果得分大于阈值，认为有一定相似度
            if bus_score > 0.3:
                rule_copy = rule.copy()
                # 设置较低的置信度，表明这是模糊匹配的结果
                rule_copy['confidence'] = bus_score * 0.4  # 降低置信度
                fuzzy_scores.append(rule_copy)
        
        # 如果有模糊匹配的场景，选择得分最高的
        if fuzzy_scores:
            scene_confidences = fuzzy_scores
        else:
            # 如果仍然没有匹配，选择默认场景（Normal low speed driving）
            for rule in SCENES:
                if rule['name'] == 'Normal low speed driving':
                    rule_copy = rule.copy()
                    rule_copy['confidence'] = 0.2  # 非常低的置信度
                    scene_confidences = [rule_copy]
                    break
            # 如果连默认场景都没找到，才抛出异常
            if not scene_confidences:
                raise ValueError('No matching scene detected, please check input or add rules.')
    
    # 选择置信度最高的场景
    best_scene = max(scene_confidences, key=lambda x: x['confidence'])
    return best_scene

def apply_special_rules(vlm, dl, bus, req):
    for func in SPECIAL_RULES:
        func(vlm, dl, bus, req)

# ------------------------------------------------------------------
# 5. Z3 combination search
# ------------------------------------------------------------------
def choose_methods(req):
    x = [Bool(f"m_{i}") for i in range(len(METHODS))]
    opt = Optimize()
    # Type coverage
    for typ in req['need']:
        opt.add(AtLeast(*[x[i] for i, m in enumerate(METHODS) if m['typ'] == typ], 1))
    # Safety/speed thresholds
    for i, m in enumerate(METHODS):
        # 聚合指标而非逐方法硬约束
        opt.add(Sum([If(xi, METHODS[i]['safe'], 0) for i, xi in enumerate(x)]) >= req['safe'])
        opt.add(Sum([If(xi, METHODS[i]['spd'],  0) for i, xi in enumerate(x)]) >= req['spd'])
    # Goal: Minimum number of methods, maximum speed sum
    opt.minimize(Sum([If(xi, 1, 0) for xi in x]))
    opt.maximize(Sum([If(xi, METHODS[i]['spd'], 0) for i, xi in enumerate(x)]))
    if opt.check() != sat:
        raise RuntimeError("No verification methods meet current threshold combination, please relax conditions or expand method library.")
    model = opt.model()
    return [METHODS[i] for i, xi in enumerate(x) if model.eval(xi)]

# ------------------------------------------------------------------
# 6. Main evaluation process
# ------------------------------------------------------------------
def evaluate(vlm, dl, bus):
    rule = find_scene(vlm, dl, bus)
    apply_special_rules(vlm, dl, bus, rule)
    methods = choose_methods(rule)
    return {
        'scene':            rule['name'],
        'confidence':       rule['confidence'],  # 添加置信度输出
        'need_types':       list(rule['need']),
        'speed_min':        rule['spd'],
        'safety_min':       rule['safe'],
        'selected_methods': methods
    }

# ------------------------------------------------------------------
# 7. Main program: Continuously monitor JSON, trigger evaluation when changes occur and write to result.json
# ------------------------------------------------------------------
if __name__ == '__main__':
    # Initial load and evaluation
    try:
        vlm, dl, bus = load_inputs()
        
        # --- Start timing ---
        start_time = time.time()
        # --- End timing ---
    except Exception as e:
        print(f"Initial JSON file reading failed: {e}", file=sys.stderr)
        sys.exit(1)

    prev_inputs = (copy.deepcopy(vlm), copy.deepcopy(dl), copy.deepcopy(bus))

    def output_result(res, filename='result.json'):
        with open(filename, 'w', encoding='utf-8') as f:
            # Convert selected_methods to JSON serializable format
            res_copy = res.copy()
            res_copy['selected_methods'] = [
                {'name': m['name'], 'typ': m['typ'], 'safe': m['safe'], 'spd': m['spd']}
                for m in res['selected_methods']
            ]
            # 确保置信度被格式化为两位小数
            if 'confidence' in res_copy:
                res_copy['confidence'] = round(res_copy['confidence'], 2)
            json.dump(res_copy, f, ensure_ascii=False, indent=2)

    print("=== Initial evaluation result ===")
    try:
        result = evaluate(vlm, dl, bus)
        # --- Start timing ---
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Single run of evaluate function took: {elapsed_time:.4f} seconds")
        # --- End timing ---
        pprint(result, width=120, compact=True)
        output_result(result)
    except Exception as e:
        print(f"Evaluation error: {e}", file=sys.stderr)

    # Monitoring loop
    try:
        while True:
            time.sleep(1)
            try:
                vlm_new, dl_new, bus_new = load_inputs()
            except Exception as e:
                print(f"JSON re-reading error: {e}", file=sys.stderr)
                continue

            if (vlm_new != prev_inputs[0] or
                dl_new  != prev_inputs[1] or
                bus_new != prev_inputs[2]):

                prev_inputs = (copy.deepcopy(vlm_new),
                               copy.deepcopy(dl_new),
                               copy.deepcopy(bus_new))
                print("\n=== JSON changes detected, re-evaluating ===")
                try:
                    # --- Start timing for re-evaluation ---
                    start_time_reeval = time.time()
                    result = evaluate(vlm_new, dl_new, bus_new)
                    # --- End timing for re-evaluation ---
                    end_time_reeval = time.time()
                    elapsed_time_reeval = end_time_reeval - start_time_reeval
                    print(f"Re-evaluation took: {elapsed_time_reeval:.4f} seconds")
                    
                    pprint(result, width=120, compact=True)
                    output_result(result)
                except Exception as e:
                    print(f"Evaluation error: {e}", file=sys.stderr)

    except KeyboardInterrupt:
        print("\nUser interrupted, program exiting.")

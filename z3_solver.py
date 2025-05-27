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
# ------------------------------------------------------------------
# 1‑A. 新增：转向‑变道场景（速度>0，角度>20°，并且方向灯亮）
# ------------------------------------------------------------------
def fuzzy_between(x, low, high, margin):
    return min(fuzzy_greater(x, low, margin), fuzzy_less(x, high, margin))

def scene_turning(vlm, dl, bus):
    angle     = abs_(bus.get('angle', 0))
    speed     = bus.get('speed', 0)
    throttle  = bus.get('throttle', 0)
    light_on  = bus.get('overtake_light', False)
    if not light_on:
        return 0.0  # 灯没开，不判断

    # -------- 转弯识别 --------
    angle_conf = fuzzy_greater(angle, 8, 3)          # 明显转角
    speed_conf = fuzzy_between(speed, 3, 50, 5)       # 中速
    throttle_conf = fuzzy_less(throttle, 0.7, 0.2)    # 油门不能太大
    turn_conf = fuzzy_and_multiple(angle_conf, speed_conf, throttle_conf)

    # -------- 抑制误判为超车 --------
    if angle < 8 and throttle > 0.6 and speed > 25:
        return 0.0  # 高速轻转角，基本是超车行为

    return turn_conf



def scene_junction_turn(vlm, dl, bus):
    hard_cond = (vlm['scene_class'] == 3) and bus.get('overtake_light', False)
    if not hard_cond:
        return 0.0

    throttle_conf = fuzzy_greater(bus['throttle'], 0.65, 0.2)
    brake_conf    = fuzzy_equal(bus['brake'], 0, 0.05)
    speed_conf    = fuzzy_greater(bus['speed'], 5, 2)
    angle_conf    = fuzzy_greater(abs_(bus['angle']), 8, 3)
    base_conf     = fuzzy_and_multiple(throttle_conf, brake_conf, speed_conf, angle_conf)

    return min(base_conf + 0.15, 1.0)

def scene_good_overtake(vlm, dl, bus):
    hard_cond = (vlm['scene_class'] in {1, 4} and vlm['special_case'] == 0) and bus.get('overtake_light', False)
    if not hard_cond:
        return 0.0

    throttle_conf = fuzzy_greater(bus['throttle'], 0.5, 0.2)            # 油门大
    brake_conf    = fuzzy_equal(bus['brake'], 0, 0.05)                  # 不踩刹车
    speed_conf    = fuzzy_greater(bus['speed'], 35, 5)                 # 高速
    angle_conf    = fuzzy_less(abs_(bus['angle']), 6, 2)               # 角度小（接近直线）
    acc_conf      = fuzzy_greater(bus['acceleration'], 2.0, 0.5)       # 有前向加速度

    return fuzzy_and_multiple(throttle_conf, brake_conf, speed_conf, angle_conf, acc_conf)


def scene_bad_overtake(vlm, dl, bus):
    hard_cond = (vlm['scene_class'] in {2, 3, 5} and vlm['special_case'] in {1, 2, 4}) and bus.get('overtake_light', False)
    if not hard_cond:
        return 0.0

    throttle_conf = fuzzy_greater(bus['throttle'], 0.5, 0.2)
    brake_conf    = fuzzy_equal(bus['brake'], 0, 0.05)
    speed_conf    = fuzzy_greater(bus['speed'], 30, 5)
    angle_conf    = fuzzy_less(abs_(bus['angle']), 8, 2)
    acc_conf      = fuzzy_greater(bus['acceleration'], 2.0, 0.5)

    return fuzzy_and_multiple(throttle_conf, brake_conf, speed_conf, angle_conf, acc_conf)


def scene_yield_emergency(vlm, dl, bus):
    # 紧急车辆声音是确定的分类，使用硬条件
    return 1.0 if dl['sound_class'] == 2 else 0.0

def scene_pedestrian_block(vlm, dl, bus):
    hard_match = 1.0 if vlm['special_case'] == 1 else 0.0
    voice_hint = 1.0 if dl.get('sound_class') == 4 else 0.0  # 有人声
    return fuzzy_or(hard_match, voice_hint * 0.5)  # 若硬条件不成立，人声可提供最多0.6置信度


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
    if abs_(bus.get('acceleration', 0)) > 7:
        return 1.0
    # BUSDATA参数使用模糊逻辑
    speed_conf = fuzzy_equal(bus['speed'], 6, 4)
    #加速度，用于判断突发加速度事件（撞车，被撞）
    acc_conf = fuzzy_greater(abs_(bus.get('acceleration', 0)), 6, 2)

    return fuzzy_and(speed_conf, acc_conf)

def scene_nearby_accident(vlm, dl, bus):
    if dl['sound_class'] == 1:
        # 返回较低置信度（确保优先级低于其他场景）
        return 0.5
    return 0.0

def scene_traffic_jam(vlm, dl, bus):
    # VLM参数使用硬条件
    hard_cond = (vlm['special_case'] == 3)
    if not hard_cond:
        return 0.0
    
    # BUSDATA参数使用模糊逻辑
    speed_conf = fuzzy_less(bus['speed'], 20, 5)
    
    return speed_conf




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
    
    return 1.0 if (vlm_cond and dl_cond) else 0.0



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


# ------------------------------------------------------------------
# 2. Special rules (speed +20 / securityty −10; low visibility speed +20)
# ------------------------------------------------------------------
def rule_pedestrian_voice(vlm, dl, bus, req):
    if dl.get('sound_class') == 4 and vlm.get('special_case') != 1:
        req['speed']  += 20
        req['security'] = max(req['security'] - 10, 0)

def rule_low_visibility(vlm, dl, bus, req):
    if vlm.get('low_visibility') == 1:
        req['speed'] += 20

SPECIAL_RULES = []# 目前special rules这点被直接放到了speed计算中，所以无需额外设定special rules了。这里暂且做保留


# 计算场景需要的速度和安全性分数：
def compute_speed_from_json(vlm, dl, bus):
    SCENE_CLASS_speed_WEIGHT = {
    1: 0,   # Wide city road
    2: 5,   # Narrow road
    3: 10,  # Intersection
    4: 15,  # Highway
    5: 20   # Mountain road
    } 
    SPECIAL_CASE_speed_WEIGHT = {
        0: 0,   # None
        1: 15,  # Pedestrian block
        2: 20,  # Accident
        3: 5,  # Traffic jam
        4: 10   # Obstacle
    }

    speed = 0
    # BUSDATA.json → 车辆动态
    speed += bus.get("speed", 0) * 0.4
    speed += abs(bus.get("acceleration", 0)) * 2
    speed += abs(bus.get("brake", 0)) * 1
    # VLM.json → 可视场景因素
    scene_speed = SCENE_CLASS_speed_WEIGHT.get(vlm.get("scene_class", 0), 0)
    speed += scene_speed
    special_speed = SPECIAL_CASE_speed_WEIGHT.get(vlm.get("special_case", 0), 0)
    speed += special_speed
    # 特殊规则
    if abs(bus.get("brake", 0)) > 0.7:
        speed += ((abs(bus.get("brake", 0)) - 0.7) * 5) ** 2 * 10  # 可调曲线爆发


    if vlm.get("low_visibility", 0):
        speed += 10
    # DL.json → 声音事件（如碰撞、尖叫）
    if dl.get("sound_class", 0) in {1, 4}:
        speed += 5

    return max(0, min(round(speed, 1), 100))  # 限制在 0~100 范围
# 计算场景需要的安全分数：
def compute_security_from_scene(scene_dict, attack_profile=None):
    """
    根据场景中所需的认证类型（need）和对应权重（weight），结合各类型面临的攻击风险，
    估算当前场景所需的安全性分数（security），对标 Z3 中的 Security = min(auth, conf, integ) 结构。
    
    参数说明：
    - scene_dict：dict，包含 'need' 和 'weight' 字段，来自 SCENES 中的定义
    - attack_profile：dict，定义每类认证方式类型面临的攻击风险值（默认固定）

    返回：
    - security：float，0~100 的安全性需求得分
    """
    
    # 各认证类型对安全目标的平均贡献能力（模拟 Z3 中的 Impact 值）
    TYPE_IMPACT = {
        '2P': {'auth': 0.8, 'conf': 0.7, 'integ': 0.6},  # 如指纹、人脸、证书等，贡献高
        '2I': {'auth': 0.6, 'conf': 0.6, 'integ': 0.5},  # 如设备、MAC、签名等，较均衡
        '2V': {'auth': 0.4, 'conf': 0.4, 'integ': 0.5},  # 如车速、位置、视觉等，较弱
    }

    # 每种认证方式类型可能面临的攻击风险（0~1，越高代表越不安全）
    ATTACK_RISK = attack_profile or {
        '2P': 0.1,  # 低风险（如人脸难仿冒）
        '2I': 0.3,  # 中等风险（如MAC可伪造）
        '2V': 0.3   # 高风险（如GPS易重放）
    }

    # 从场景中获取所需认证类型集合与其权重
    need_set = scene_dict.get('need', set())
    weight_dict = scene_dict.get('weight', {})

    sum_auth = sum_conf = sum_integ = 0.0
    total_weight = 0.0

    # 遍历每种认证类型，累加贡献 + 风险调整后的值
    for typ in need_set:
        impact = TYPE_IMPACT.get(typ, {'auth': 0.5, 'conf': 0.5, 'integ': 0.5})
        risk = ATTACK_RISK.get(typ, 0.3)
        w = weight_dict.get(typ, 1.0)

        # 加入攻击风险惩罚：越容易被攻击，安全性要求越高
        RISK_EPSILON = 1e-6

        adj_auth  = impact['auth']  / (1 - risk + RISK_EPSILON)
        adj_conf  = impact['conf']  / (1 - risk + RISK_EPSILON)
        adj_integ = impact['integ'] / (1 - risk + RISK_EPSILON)


        sum_auth += adj_auth * w
        sum_conf += adj_conf * w
        sum_integ += adj_integ * w
        total_weight += w

    if total_weight == 0:
        return 0.0

    avg_auth = sum_auth / total_weight
    avg_conf = sum_conf / total_weight
    avg_integ = sum_integ / total_weight

    # 使用 Z3 中 min(Authenticity, Confidentiality, Integrity) 方式取最终得分
    final_security = min(avg_auth, avg_conf, avg_integ)
    raw_security = final_security
    scaled_security = 50 + (raw_security - 0.6) * 150
    return round(max(50, min(80, scaled_security)), 1)
    #return round(final_security * 100, 1)




#当前主要改动了scene部分，原先设定每个场景的speed和security现在都根据驾驶数据，周围环境动态计算

SCENES = [
    dict(name='Encountering a car accident on one\'s own', cond=scene_self_accident, need={'2P','2I'}, weight={'2V': 1.0, '2I': 0.6, '2P': 0.5}),
    dict(name='Nearby accident ', cond=scene_nearby_accident, need={'2P','2I'}, weight={'2V': 1.0, '2I': 0.6, '2P': 0.5}),
    dict(name='Emergency brake', cond=scene_hard_brake, need={'2V'}, weight={'2V': 1.0}),
    dict(name='Emergency avoidance', cond=scene_hard_evasion, need={'2V','2I'}, weight={'2V': 1.0, '2I': 0.7, '2P': 0.5}),
    dict(name='Mountain road in distress', cond=scene_mountain_hazard, need={'2V','2P','2I'}, weight={'2V': 1.0, '2I': 0.8, '2P': 0.7}),
    dict(name='Pedestrians blocking ahead', cond=scene_pedestrian_block, need={'2P'}, weight={'2P': 1}),
    dict(name='A car accident occurred ahead', cond=scene_accident_ahead, need={'2V','2I'}, weight={'2V': 0.7, '2I': 0.8, '2P': 0.5}),
    dict(name='Obstacles blocking the road ahead in unfavorable conditions', cond=scene_bad_obstacle_block, need={'2V','2I'}, weight={'2V': 1.0, '2I': 0.9}),
    dict(name='Good obstacle blocking ahead', cond=scene_good_obstacle_block, need={'2V','2I'}, weight={'2V': 0.8, '2I': 1, '2P': 0.5}),
    dict(name='Avoid special vehicles', cond=scene_yield_emergency, need={'2V','2P','2I'}, weight={'2V': 1.0, '2I': 1, '2P': 1}),
    dict(name='Turn/U-turn at the intersection', cond=scene_junction_turn, need={'2V','2I'}, weight={'2V': 1.0, '2I': 1, '2P': 0.5}),
    dict(name='Turning with indicator on', cond=scene_turning, need={'2V'}, weight={'2V': 1.0, '2I': 1, '2P': 1}),
    dict(name='Overtaking on non good road conditions', cond=scene_bad_overtake, need={'2V'}, weight={'2V': 1.0, '2I': 0.7, '2P': 0.5}),
    dict(name='Overtaking on good road conditions', cond=scene_good_overtake, need={'2V'}, weight={'2V': 1.0, '2I': 0.7, '2P': 0.5}),
    dict(name='Traffic lights at intersections', cond=scene_traffic_light, need={'2V','2I'}, weight={'2V': 1.0, '2I': 0.7, '2P': 0.5}),
    dict(name='Vehicles honking on non good road conditions', cond=scene_bad_road_horn, need={'2V'}, weight={'2V': 1.0, '2I': 0.7, '2P': 0.5}),
    dict(name='Good road conditions with vehicles honking their horns', cond=scene_good_road_horn, need={'2V'}, weight={'2V': 1.0, '2I': 0.7, '2P': 0.5}),
    dict(name='Traffic jam', cond=scene_traffic_jam, need={'2V','2I'}, weight={'2V': 0.7, '2I': 1, '2P': 0.5}),
    dict(name='Normal high-speed driving', cond=scene_normal_high, need={'2I'}, weight={'2V': 1.0, '2I': 1, '2P': 0.5}),
    dict(name='Normal low-speed driving', cond=scene_normal_low, need={'2I'}, weight={'2V': 1.0, '2I': 1, '2P': 0.5}),
]

# ------------------------------------------------------------------
# 3. Verification method library
# ------------------------------------------------------------------
# METHODS = [
#     dict(name='MAC address identification (Bluetooth/Wi‑Fi)',            typ='2P', security=11.25, speed=42.5, use=70),
#     dict(name='Broadcast authentication (public‑key)',                   typ='2P', security=28.5,  speed=30,   use=65),
#     dict(name='Digital certificate via BLE/Wi‑Fi',                       typ='2P', security=25.5,  speed=40,   use=60),
#     dict(name='Face ID recognition',                                     typ='2P', security=25.75, speed=50, use=45),
#     dict(name='Fingerprint authentication',                              typ='2P', security=30.75, speed=47,   use=40),

#     dict(name='DID authentication (blockchain‑based)',                   typ='2V', security=23.75, speed=11.5, use=50),
#     dict(name='VIN‑based vehicle ID matching',                           typ='2V', security=9.75,  speed=32.5, use=80),
#     dict(name='Location‑based authentication (GPS)',                     typ='2V', security=9.75,  speed=27,   use=85),
#     dict(name='PKI cert + real‑time vehicle status check',               typ='2V', security=30.25, speed=17.5, use=60),
#     dict(name='APKI anonymous PKI (VANET)',                              typ='2V', security=23.5,  speed=23,   use=70),

#     dict(name='Signal‑strength device verification',                     typ='2I', security=9.75,  speed=30.5, use=90),
#     dict(name='Device‑ID + location verification',                       typ='2I', security=12.5,  speed=25,   use=85),
#     dict(name='LTS certificate authentication',                          typ='2I', security=20.5,  speed=23,   use=70),
#     dict(name='Blockchain + digital certificate authentication (infra)', typ='2I', security=28.5,  speed=11.5, use=60),
# ]

METHODS = [
    dict(name='MAC address identification (Bluetooth/Wi‑Fi)',            typ='2P', security=11.25, speed=42.5),
    dict(name='Broadcast authentication (public‑key)',                   typ='2P', security=28.5,  speed=30),
    dict(name='Digital certificate via BLE/Wi‑Fi',                       typ='2P', security=25.5,  speed=40),
    dict(name='Face ID recognition',                                     typ='2P', security=25.75, speed=50),
    dict(name='Fingerprint authentication',                              typ='2P', security=30.75, speed=47),

    dict(name='DID authentication (blockchain‑based)',                   typ='2V', security=23.75, speed=11.5),
    dict(name='VIN‑based vehicle ID matching',                           typ='2V', security=9.75,  speed=32.5),
    dict(name='Location‑based authentication (GPS)',                     typ='2V', security=9.75,  speed=27),
    dict(name='PKI cert + real‑time vehicle status check',               typ='2V', security=30.25, speed=17.5),
    dict(name='APKI anonymous PKI (VANET)',                              typ='2V', security=23.5,  speed=23),

    dict(name='Signal‑strength device verification',                     typ='2I', security=9.75,  speed=30.5),
    dict(name='Device‑ID + location verification',                       typ='2I', security=12.5,  speed=25),
    dict(name='LTS certificate authentication',                          typ='2I', security=20.5,  speed=23),
    dict(name='Blockchain + digital certificate authentication (infra)', typ='2I', security=28.5,  speed=11.5),
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
#这里注释掉的是使用use参与评分的，实际可能不太需要use，所有我删了
# def choose_methods(req):
#     x = [Bool(f"m_{i}") for i in range(len(METHODS))]
#     opt = Optimize()
# #在环境光不好的时候禁用人脸识别
#     if vlm and vlm.get("low_visibility", 0) == 1:
#         for i, m in enumerate(METHODS):
#             if m["name"] == "Face ID recognition":
#                 opt.add(x[i] == False)
#     # 限定每种类型必须且只能选一个方法
#     for typ in req['need']:
#         indices = [i for i, m in enumerate(METHODS) if m['typ'] == typ]
#         opt.add(AtLeast(*[x[i] for i in indices], 1))
#         opt.add(AtMost(*[x[i] for i in indices], 1))

#     # 获取当前场景给的类型权重字典（若无，默认1.0）
#     weights = req.get("weight", {typ: 1.0 for typ in req['need']})

#     # 模糊目标策略
#     security_target = req["security"]
#     speed_target = req["speed"]
#     focus = "balanced" if abs(security_target - speed_target) < 10 else ("security" if security_target > speed_target else "speed")

#     def composite_score(method):
#         s_diff = abs(method["security"] - security_target)
#         p_diff = abs(method["speed"] - speed_target)
#         use_score = method.get("use", 50)
#         typ_weight = weights.get(method["typ"], 1.0)

#         if focus == "security":
#             score = 100 - s_diff + 0.3 * (method["speed"] + use_score)
#         elif focus == "speed":
#             score = 100 - p_diff + 0.3 * (method["security"] + use_score)
#         else:
#             score = 100 - (s_diff + p_diff) / 2 + 0.1 * use_score

#         return score * typ_weight  # 加权！

#     # 构造目标函数
#     total_score = Sum([
#         If(x[i], RealVal(composite_score(m)), RealVal(0))
#         for i, m in enumerate(METHODS)
#         if m['typ'] in req['need']
#     ])
#     opt.maximize(total_score)

#     if opt.check() != sat:
#         raise RuntimeError("No verification methods meet current constraints.")
    
#     model = opt.model()
#     return [METHODS[i] for i, xi in enumerate(x) if is_true(model.eval(xi, model_completion=True))]

#这个是不使用use的版本
def choose_methods(req):
    x = [Bool(f"m_{i}") for i in range(len(METHODS))]
    opt = Optimize()

    # 禁用某些方法：例如在低能见度下禁用人脸识别
    if vlm and vlm.get("low_visibility", 0) == 1:
        for i, m in enumerate(METHODS):
            if m["name"] == "Face ID recognition":
                opt.add(x[i] == False)

    # 每种类型必须且只能选一个方法
    for typ in req['need']:
        indices = [i for i, m in enumerate(METHODS) if m['typ'] == typ]
        opt.add(AtLeast(*[x[i] for i in indices], 1))
        opt.add(AtMost(*[x[i] for i in indices], 1))

    # 获取类型权重（默认权重为 1.0）
    weights = req.get("weight", {typ: 1.0 for typ in req['need']})

    security_target = req["security"]
    speed_target = req["speed"]
    focus = "balanced" if abs(security_target - speed_target) < 10 else ("security" if security_target > speed_target else "speed")

    def composite_score(method):
        s_diff = abs(method["security"] - security_target)
        p_diff = abs(method["speed"] - speed_target)
        typ_weight = weights.get(method["typ"], 1.0)

        if focus == "security":
            score = 100 - s_diff + 0.3 * method["speed"]
        elif focus == "speed":
            score = 100 - p_diff + 0.3 * method["security"]
        else:
            score = 100 - (s_diff + p_diff) / 2  # 平衡模式下没有 use 加权

        return score * typ_weight

    # 构造目标函数（不再引用 use）
    total_score = Sum([
        If(x[i], RealVal(composite_score(m)), RealVal(0))
        for i, m in enumerate(METHODS)
        if m['typ'] in req['need']
    ])
    opt.maximize(total_score)

    if opt.check() != sat:
        raise RuntimeError("No verification methods meet current constraints.")

    model = opt.model()
    return [METHODS[i] for i, xi in enumerate(x) if is_true(model.eval(xi, model_completion=True))]


# ------------------------------------------------------------------
# 6. Main evaluation process
# ------------------------------------------------------------------
def evaluate(vlm, dl, bus):
    rule = find_scene(vlm, dl, bus)
    rule['speed'] = compute_speed_from_json(vlm, dl, bus)
    rule['security'] = compute_security_from_scene(rule)
    apply_special_rules(vlm, dl, bus, rule)
    methods = choose_methods(rule)
    return {
        'scene':            rule['name'],
        'confidence':       rule['confidence'],  # 添加置信度输出
        'need_types':       list(rule['need']),
        'speed_min':        rule['speed'],
        'securityty_min':       rule['security'],
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
                {'name': m['name'], 'typ': m['typ'], 'security': m['security'], 'speed': m['speed']}
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
                    print(f"Scene speed (demand): {result['speed_min']}")
                    print(f"Scene security (demand): {result['securityty_min']}")
                except Exception as e:
                    print(f"Evaluation error: {e}", file=sys.stderr)

    except KeyboardInterrupt:
        print("\nUser interrupted, program exiting.")

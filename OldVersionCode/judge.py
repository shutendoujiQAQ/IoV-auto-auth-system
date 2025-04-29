import json
import os
import time

def get_verification_method(vlm_path=None, dl_path=None, busdata_path=None):
    # 设置默认路径为脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    vlm_path = vlm_path or os.path.join(script_dir, 'VLM.json')
    dl_path = dl_path or os.path.join(script_dir, 'DL.json')
    busdata_path = busdata_path or os.path.join(script_dir, 'BUSDATA.json')
    
    # 读取 BUSDATA.json 文件中的急刹车和超车状态
    try:
        with open(busdata_path, 'r', encoding='utf-8') as f:
            busdata = json.load(f)
    except FileNotFoundError:
        print(f"错误: 无法找到文件 {busdata_path}")
        return None
    hard_brake = busdata.get("hard_brake", 0)
    overtake = busdata.get("overtake", 0)

    # 读取 VLM.json 中的场景分类和特殊情况种类
    try:
        with open(vlm_path, 'r', encoding='utf-8') as f:
            vlm = json.load(f)
    except FileNotFoundError:
        print(f"错误: 无法找到文件 {vlm_path}")
        return None
    scene_class = vlm.get("scene_class")
    special_case = vlm.get("special_case")

    # 读取 DL.json 中的特殊声音分类
    try:
        with open(dl_path, 'r', encoding='utf-8') as f:
            dl = json.load(f)
    except FileNotFoundError:
        print(f"错误: 无法找到文件 {dl_path}")
        return None
    sound_class = dl.get("sound_class")

    result = None  # 最终的验证方式

    # 根据 BUSDATA 中状态优先判断
    if hard_brake == 1:
        # 当急刹车状态为1，根据 VLM 中的场景分类判断验证方式
        if scene_class in [1, 4]:
            result = 2  # 验证方式2
        elif scene_class in [2, 3, 5]:
            result = 1  # 验证方式1
        else:
            print("场景分类不符合预期")
    elif overtake == 1:
        # 当超车状态为1，根据 VLM 中的场景分类判断验证方式
        if scene_class in [1, 3, 4]:
            result = 4  # 验证方式4
        elif scene_class in [2, 5]:
            result = 3  # 验证方式3
        else:
            print("场景分类不符合预期")
    else:
        # 当急刹车和超车状态均为0时，继续判断
        if special_case == 1:
            result = 1  # VLM 中特殊情况种类为1，则采用验证方式1
        elif special_case == 0:
            if sound_class == 1:
                result = 4  # DL 中声音分类为1，采用验证方式4
            elif sound_class in [2, 3]:
                result = 2  # 声音分类为2或3，采用验证方式2
            elif sound_class == 4:
                result = 3  # 声音分类为4，采用验证方式3
            else:
                print("特殊声音分类不符合预期")
        elif special_case == 2:
            if scene_class in [1, 2, 4]:
                result = 3  # 场景分类为1、2或4，采用验证方式3
            elif scene_class in [3, 5]:
                result = 2  # 场景分类为3或5，采用验证方式2
            else:
                print("场景分类不符合预期")
        else:
            print("特殊情况种类不符合预期")

    return result

def monitor_internal_values(vlm_path, dl_path, busdata_path, check_interval=1):
    """
    循环检测三个 JSON 文件内部数据的变化：
    1. 每隔指定时间（check_interval秒）读取一次三个文件。
    2. 如果检测到任一文件内部数据与上次不同，则触发判断逻辑，
       并将最终结果（结构为 {"status": 数字}）写入 result.json 文件。
    """
    # 初始化时读取一次文件内容（若文件不存在，则对应值为 None）
    try:
        with open(vlm_path, 'r', encoding='utf-8') as f:
            prev_vlm = json.load(f)
    except FileNotFoundError:
        prev_vlm = None
    try:
        with open(dl_path, 'r', encoding='utf-8') as f:
            prev_dl = json.load(f)
    except FileNotFoundError:
        prev_dl = None
    try:
        with open(busdata_path, 'r', encoding='utf-8') as f:
            prev_busdata = json.load(f)
    except FileNotFoundError:
        prev_busdata = None

    print("开始监控文件内部数据变化...")

    # 结果文件路径，保存在脚本所在目录下
    result_path = os.path.join(os.path.dirname(vlm_path), 'result.json')

    while True:
        time.sleep(check_interval)
        try:
            with open(vlm_path, 'r', encoding='utf-8') as f:
                current_vlm = json.load(f)
        except FileNotFoundError:
            current_vlm = None
        try:
            with open(dl_path, 'r', encoding='utf-8') as f:
                current_dl = json.load(f)
        except FileNotFoundError:
            current_dl = None
        try:
            with open(busdata_path, 'r', encoding='utf-8') as f:
                current_busdata = json.load(f)
        except FileNotFoundError:
            current_busdata = None

        # 检测任一文件内部数据是否发生变化
        if (current_vlm != prev_vlm) or (current_dl != prev_dl) or (current_busdata != prev_busdata):
            print("检测到文件内部数据变化，重新判断...")
            verification_method = get_verification_method(vlm_path, dl_path, busdata_path)
            if verification_method is not None:
                print("验证方式{}".format(verification_method))
                # 将最终的验证方式（数字）以结构 {"status": 数字} 写入 result.json 文件
                with open(result_path, 'w', encoding='utf-8') as f:
                    json.dump({"status": verification_method}, f)
                print("结果已保存到{}".format(result_path))
            else:
                print("无法根据当前数据判断验证方式")
            # 更新记录的上一次文件内容
            prev_vlm = current_vlm
            prev_dl = current_dl
            prev_busdata = current_busdata

if __name__ == "__main__":
    # 设置默认文件路径（脚本所在目录下）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    vlm_path = os.path.join(script_dir, 'VLM.json')
    dl_path = os.path.join(script_dir, 'DL.json')
    busdata_path = os.path.join(script_dir, 'BUSDATA.json')
    
    # 启动基于内容变化的文件监控循环
    monitor_internal_values(vlm_path, dl_path, busdata_path)

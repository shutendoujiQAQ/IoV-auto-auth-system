from ollama import Client 
import base64
import json
import re
from pathlib import Path
import time
from threading import Lock

# 初始化客户端和热启动状态
client = Client(host='https://6nq94pfmo4b-vlm.gear-c1.openbayes.net/')  # 或 'http://localhost:11434'
model_lock = Lock()
model_warm_started = False

def warm_start_model():
    """预热加载模型保持热状态"""
    global model_warm_started
    with model_lock:
        if not model_warm_started:
            print("Activating...")
            client.generate(
                model='gemma3:27b',
                prompt="ping",
                options={
                    'num_ctx': 4096,
                    'num_predict': 1,
                    'temperature': 0,
                    'top_k': 20,
                    'format': 'json'
                }
            )
            model_warm_started = True
            print("Complete ")

def encode_image_base64(path):
    """编码图片为base64"""
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def process_image_analysis(image_path):
    """执行图像分析并返回三字段 JSON"""
    prompt = """Analyze this driving scene and output STRICT JSON with ONLY these keys:
    'scene_class' (1=Wide city road,2=Narrow city/rural road,3=Traffic light intersection,4=Highway,5=Mountain road),
    'special_case' (0=None obstruction,1=Pedestrian block,2=Accident,3=Heavy traffic,4=Obstacle),
    'low_visibility' (0=no,1=yes).
    EXAMPLE: {"scene_class":4,"special_case":0,"low_visibility":0}. NO other text."""
    
    try:
        image_b64 = encode_image_base64(image_path)
        with model_lock:
            response = client.generate(
                model='gemma3:27b',
                prompt=prompt,
                images=[image_b64],
                options={
                    'num_ctx': 4096,
                    'num_predict': 64,
                    'temperature': 0,
                    'top_k': 20,
                    'format': 'json'
                }
            )
        
        def validate_json(data):
            required = {"scene_class", "special_case", "low_visibility"}
            if not required.issubset(data.keys()):
                return False
            if data["scene_class"] not in {1,2,3,4,5}:
                return False
            if data["special_case"] not in {0,1,2,3,4}:
                return False
            if data["low_visibility"] not in {0,1}:
                return False
            return True

        # 从模型响应中提取纯 JSON
        match = re.search(r'\{.*\}', response.get('response', ''))
        if match:
            data = json.loads(match.group())
            if validate_json(data):
                return data

    except Exception as e:
        print(f"Error in processing : {e}")

    # 默认值
    return {"scene_class": 0, "special_case": 0, "low_visibility": 0}

def continuous_analysis(image_path, interval=5):
    """每 interval 秒分析一次，并写入 VLM.json"""
    warm_start_model()
    while True:
        start_time = time.time()
        result = process_image_analysis(image_path)
        with open('VLM.json', 'w') as f:
            json.dump(result, f, indent=2)
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{ts}] Analysis result:")
        print(json.dumps(result, indent=2))
        print("-" * 40)
        elapsed = time.time() - start_time
        time.sleep(max(0, interval - elapsed))

if __name__ == "__main__":
    image_path = 'Photo/photo.png'  # 相对路径，使用正斜杠
    try:
        continuous_analysis(image_path)
    except KeyboardInterrupt:
        print("\n分析循环已停止")
    try:
        continuous_analysis(image_path)
    except KeyboardInterrupt:
        print("\nResult Analysis loop stopped")

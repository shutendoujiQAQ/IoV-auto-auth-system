#!/usr/bin/env python
"""
CARLA manual control with Xbox controller.

使用 Xbox 手柄进行车辆控制：
    - 左摇杆水平（轴 0）：控制转向。
    - 右触发器（轴 5）：控制油门（映射到 [0,1]）。
    - 左触发器（轴 4）：控制刹车（映射到 [0,1]）。
    - 按钮 B（索引 1）：切换倒车状态，同时立即拍照。
    - 按钮 X（索引 2）：切换驾驶摄像头视角。

当刹车持续用力（刹车值 > 0.8）超过 3 秒时，
采样车辆数据（方向盘、油门、刹车、速度、位置）采样周期为 0.2 秒，
录制 7 秒后，以 JSON 格式输出到 heavy_brake_record.json 文件，并在此时立即触发拍照。
拍照摄像头每 2 秒拍一张图片，所有图片始终覆盖保存为 photo.png。

当前 UI 顶部文字框显示 JSON 状态文件中的状态（值为 1/2/3/4），
"""

from __future__ import print_function
import threading
import glob
import os
import sys
import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref
import time    # 用于计算时间间隔
import cv2     # 用于显示摄像头图像
import json
import numpy as np
import carla
from carla import ColorConverter as cc
import pygame
from pygame.locals import *
import multiprocessing as mp
import queue
from macls.predict import MAClsPredictor
from infer_random_audio import infer_random_audio, FOLD_MAP

# 全局队列及子进程引用，由 main() 初始化
audio_queue = None
result_queue = None
worker = None

# HAT 值映射到方向键名称
HAT_DIR_MAP = {
    (0, 1): 'up',    # 上
    (1, 0): 'right', # 右
    (0, -1): 'down', # 下
    (-1, 0): 'left'  # 左
}

# 音频推理子进程函数
def audio_worker(q, result_q, save_path="DL.json"):
    """子进程：从 q 中取出 fold，推理后写文件并回传结果"""
    while True:
        fold = q.get()
        if fold is None:
            break
        cls = infer_random_audio(fold, save_path)
        result_q.put((fold, cls))

# ============================================================================== 
# -- Find CARLA module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'carla/dist/carla-*%d.%d-%s.egg' % (
                             sys.version_info.major,
                             sys.version_info.minor,
                             'win-amd64' if os.name == 'nt' else 'linux-x86_64')))[0])
except IndexError:
    pass
import carla
from carla import ColorConverter as cc

# ============================================================================== 
# -- Global functions ----------------------------------------------------------
# ==============================================================================
def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]

def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

# ============================================================================== 
# -- World ---------------------------------------------------------------------
# ==============================================================================
class World(object):
    def __init__(self, carla_world, hud, args):
        self.world = carla_world
        self.actor_role_name = args.rolename
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            sys.exit(1)
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.radar_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._gamma = args.gamma
        self.restart()
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0
        self.constant_velocity_enabled = False
        self.current_map_layer = 0
        self.map_layer_names = [
            carla.MapLayer.NONE,
            carla.MapLayer.Buildings,
            carla.MapLayer.Decals,
            carla.MapLayer.Foliage,
            carla.MapLayer.Ground,
            carla.MapLayer.ParkedVehicles,
            carla.MapLayer.Particles,
            carla.MapLayer.Props,
            carla.MapLayer.StreetLights,
            carla.MapLayer.Walls,
            carla.MapLayer.All
        ]

    def restart(self):
        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        blueprint = self.world.get_blueprint_library().find('vehicle.tesla.model3')
        blueprint.set_attribute('role_name', self.actor_role_name)
        if blueprint.has_attribute('color'):
            blueprint.set_attribute('color', '0,0,255')
        if blueprint.has_attribute('driver_id'):
            blueprint.set_attribute('driver_id', random.choice(blueprint.get_attribute('driver_id').recommended_values))
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'true')
        if blueprint.has_attribute('speed'):
            self.player_max_speed = float(blueprint.get_attribute('speed').recommended_values[1])
            self.player_max_speed_fast = float(blueprint.get_attribute('speed').recommended_values[2])
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        while self.player is None:
            if not self.map.get_spawn_points():
                print('No spawn points available.')
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def next_map_layer(self, reverse=False):
        self.current_map_layer += -1 if reverse else 1
        self.current_map_layer %= len(self.map_layer_names)
        selected = self.map_layer_names[self.current_map_layer]
        self.hud.notification('LayerMap selected: %s' % selected)

    def load_map_layer(self, unload=False):
        selected = self.map_layer_names[self.current_map_layer]
        if unload:
            self.hud.notification('Unloading map layer: %s' % selected)
            self.world.unload_map_layer(selected)
        else:
            self.hud.notification('Loading map layer: %s' % selected)
            self.world.load_map_layer(selected)

    def toggle_radar(self):
        if self.radar_sensor is None:
            self.radar_sensor = RadarSensor(self.player)
        elif self.radar_sensor.sensor is not None:
            self.radar_sensor.sensor.destroy()
            self.radar_sensor = None

    def tick(self, clock):
        self.hud.tick(self, clock)
        # 更新HUD中的紧急状态为当前驾驶摄像头的紧急状态
        self.hud.emergency_state = self.camera_manager.emergency_state

    def render(self, display, controller):
        # 驾驶摄像头画面
        self.camera_manager.render(display)
        # 传 controller 给 HUD
        self.hud.render(display, controller)

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        if self.radar_sensor is not None:
            self.toggle_radar()
        sensors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.imu_sensor.sensor
        ]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        if self.player is not None:
            self.player.destroy()


# ============================================================================== 
# -- XboxController ------------------------------------------------------------
# ==============================================================================
class XboxController(object):
    """通过 Xbox 手柄控制车辆并触发异步音频推理"""
    def __init__(self, world):
        pygame.joystick.init()
        if pygame.joystick.get_count() < 1:
            raise RuntimeError("Xbox Controller not found")
        self._joystick = pygame.joystick.Joystick(0)
        self._joystick.init()
        self.world = world
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            self._reverse = False
            self._overtake_light = False
        else:
            raise NotImplementedError("Only support vehicles control")
        # 定时更新 BUSDATA.json
        self.update_busdata_timer = threading.Timer(0.5, self.update_busdata)
        self.update_busdata_timer.daemon = True
        self.update_busdata_timer.start()
        self.acc_history = []
        # 不再需要 heavy_brake 录制相关的变量
        # self._heavy_brake_start = None
        # self._recording = False
        # self._record_start_time = None
        # self._last_sample_time = None
        # self._record_data = []

    def update_busdata(self):
        if self.world.player is not None:
            try:
                steer_axis = self._joystick.get_axis(0)
                throttle   = (self._joystick.get_axis(5) + 1) / 2.0
                brake      = (self._joystick.get_axis(4) + 1) / 2.0
            except pygame.error:
                return
            speed = math.sqrt(
                self.world.player.get_velocity().x**2 +
                self.world.player.get_velocity().y**2 +
                self.world.player.get_velocity().z**2
            ) * 3.6
            angle = steer_axis * 60
            acc_vector = self.world.player.get_acceleration()
            forward_vector = self.world.player.get_transform().get_forward_vector()
            acc_forward = (
                acc_vector.x * forward_vector.x +
                acc_vector.y * forward_vector.y +
                acc_vector.z * forward_vector.z
            )
            self.acc_history.append((time.time(), acc_forward))
            # 保留最近 2 秒内的记录
            self.acc_history = [
                (t, a) for t, a in self.acc_history if time.time() - t <= 2.0
            ]
            # 记录最大值（用于传入 JSON）
            acc_max = max([abs(a) for _, a in self.acc_history], default=0)
            busdata = {
                "throttle": throttle,
                "brake": brake,
                "speed": speed,
                "angle": angle,
                "overtake_light": self._overtake_light,
                "acceleration": acc_max
            }
            try:
                with open("BUSDATA.json", "w", encoding="utf-8") as f:
                    json.dump(busdata, f, indent=4, ensure_ascii=False)
            except Exception as e:
                print("Error when updating BUSDATA.json :", e)
        # 重启定时器
        self.update_busdata_timer = threading.Timer(0.5, self.update_busdata)
        self.update_busdata_timer.daemon = True
        self.update_busdata_timer.start()

    def get_vehicle_data(self):
        steer_axis = self._joystick.get_axis(0)
        throttle   = (self._joystick.get_axis(5) + 1) / 2.0
        brake      = (self._joystick.get_axis(4) + 1) / 2.0
        speed = math.sqrt(
            self.world.player.get_velocity().x**2 +
            self.world.player.get_velocity().y**2 +
            self.world.player.get_velocity().z**2
        ) * 3.6
        angle = steer_axis * 60
        return throttle, brake, speed, angle

    def get_overtake_light(self):
        return self._overtake_light  # 返回超车灯的当前状态
    def parse_events(self, world, clock):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return True
            # D-pad 触发 infer_random_audio
            if event.type == pygame.JOYBUTTONDOWN:
                if self._joystick.get_button(4):  # 假设 Lb 按键是按钮 5
                    self._overtake_light = True  # 超车灯开启
                elif self._joystick.get_button(5):  # 假设 Rb 按键是按钮 4
                    self._overtake_light = False  # 超车灯关闭
            if event.type == pygame.JOYHATMOTION:
                hat = self._joystick.get_hat(0)
                dir_key = HAT_DIR_MAP.get(hat)
                if dir_key:
                    fold = FOLD_MAP[dir_key]
                    audio_queue.put(fold)
        # 获取常规车辆控制输入
        steer_axis   = self._joystick.get_axis(0)
        raw_throttle = self._joystick.get_axis(5)
        raw_brake    = self._joystick.get_axis(4)

        throttle = (raw_throttle + 1) / 2.0
        brake    = (raw_brake    + 1) / 2.0

        self._control.steer    = round(steer_axis, 2)
        self._control.throttle = (raw_throttle + 1) / 2.0
        self._control.brake = (raw_brake + 1) / 2.0


        #hard_brake_val = 1 if brake > 0.8 else 0
        #overtake_val  = 1 if abs(steer_axis) > 0.5 else 0
        #busdata = {"hard_brake": hard_brake_val, "overtake": overtake_val}
        #with open("BUSDATA.json", "r+", encoding="utf-8") as f:
            #json.dump(busdata, f, indent=4, ensure_ascii=False)

        # 切换倒车状态
        if self._joystick.get_button(1):
            self._reverse = not self._reverse
            self._control.reverse = self._reverse
            world.hud.notification(f"Reverse mode: {'On' if self._reverse else 'Off'}")
            pygame.time.wait(300)
        # 应用控制
        self.world.player.apply_control(self._control)

        # 根据当前刹车和方向盘状态更新 BUSDATA.json 文件
        # 这里假设最大转向为40°，因此当 |steer| > 0.5 时认为方向盘角度大于20°

        # 保留摄像头视角切换（按钮 X，索引 2）
        if self._joystick.get_button(2):
            self.world.camera_manager.toggle_camera()
            pygame.time.wait(300)

        return False


# ============================================================================== 
# -- HUD -----------------------------------------------------------------------
# ==============================================================================
class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        mono = pygame.font.match_font(font_name)
        self._font_mono = pygame.font.Font(mono, 14)
        self._notifications = FadingText(
            pygame.font.Font(mono, 16), (width, 40), (0, height-40)
        )
        self.help = HelpText(
            pygame.font.Font(mono, 16), width, height
        )
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()
        self.emergency_state = False
        self.json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result.json")
        if not os.path.exists(self.json_path):
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump({"sound_class": 1}, f, indent=4)
        self.json_info_text = ""
        self.last_json_check = time.time()
        self.last_control = None
    def render(self, display, controller):
        # 新增：在顶部显示两个指示灯
        # 第一个指示灯：根据车辆控制中的刹车值（>0.8亮，<=0.8灭）
        # 第二个指示灯：根据车辆控制中的转向值的绝对值（>0.5认为大于20°，亮，否则灭）
        indicator_radius = 10
        spacing = 20
        x_center = self.dim[0] // 2
        y_top = 10
        if self.last_control:
            brake_val = self.last_control.brake
            steer_val = self.last_control.steer
        else:
            brake_val = 0
            steer_val = 0
        color_brake = (255, 0, 0) if brake_val > 0.8 else (100, 100, 100)
        color_steer = (255, 255, 0) if abs(steer_val) > 0.5 else (100, 100, 100)
        pygame.draw.circle(display, color_brake, (x_center - spacing, y_top + indicator_radius), indicator_radius)
        pygame.draw.circle(display, color_steer, (x_center + spacing, y_top + indicator_radius), indicator_radius)
        self._notifications.render(display)

        # —— 右侧 HUD 面板 —— 
        panel_width = 220
        panel_x = self.dim[0] - panel_width
        panel_bg = pygame.Surface((panel_width, self.dim[1]))
        panel_bg.set_alpha(100)
        display.blit(panel_bg, (panel_x, 0))

        y0 = 150
        line_h = 30
        y_line = 0  # 逐行绘制位置

        # ==== 显示声音分类（DL） ====
        try:
            with open("DL.json", "r", encoding="utf-8") as f:
                dl_data = json.load(f)
                sound_class = dl_data.get("sound_class", 0)
                sound_text = {
                    0: "nothing",
                    1: "crash",
                    2: "siren",
                    3: "car_horn",
                    4: "children_playing"
                }.get(sound_class, "unknown")
                s_sound = self._font_mono.render(f"Sound: {sound_text}", True, (255, 255, 255))
                display.blit(s_sound, (panel_x + 10, y0 + y_line * line_h))
                y_line += 1
        except Exception as e:
            print(f"[HUD] DL.json 读取失败: {e}")

        # ==== 显示场景分析结果（VLM） ====
        try:
            with open("VLM.json", "r", encoding="utf-8") as f:
                vlm_data = json.load(f)
                scene_class = vlm_data.get("scene_class", 0)
                special_case = vlm_data.get("special_case", 0)
                low_vis = vlm_data.get("low_visibility", 0)

                scene_map = {
                    1: "Wide city road",
                    2: "Narrow road",
                    3: "Intersection",
                    4: "Highway",
                    5: "Mountain road"
                }
                special_map = {
                    0: "None",
                    1: "Pedestrian block",
                    2: "Accident",
                    3: "Heavy traffic",
                    4: "Obstacle"
                }

                s_scene = self._font_mono.render(f"Scene: {scene_map.get(scene_class, 'unknown')}", True, (255, 255, 255))
                s_special = self._font_mono.render(f"Special: {special_map.get(special_case, 'unknown')}", True, (255, 255, 255))
                s_vis = self._font_mono.render(f"LowVis: {'Yes' if low_vis else 'No'}", True, (255, 255, 255))

                display.blit(s_scene, (panel_x + 10, y0 + y_line * line_h))
                y_line += 1
                display.blit(s_special, (panel_x + 10, y0 + y_line * line_h))
                y_line += 1
                display.blit(s_vis, (panel_x + 10, y0 + y_line * line_h))
                y_line += 1
        except Exception as e:
            print(f"[HUD] VLM.json 读取失败: {e}")

        # —— 左侧 HUD 面板 —— 

        # 1) 获取当前控制量
        throttle, brake, speed, angle = controller.get_vehicle_data()
        # 2) 读取示例 JSON 文件内容
        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            data = {"error": str(e)}
        # 3) 渲染 JSON 内容
        x, y = 10, 10
        for key, val in data.items():
            line = f"{key}: {val}"
            text_surf = self._font_mono.render(line, True, (255,255,255))
            display.blit(text_surf, (x, y))
            y += 20
        # 4) 渲染速度、油门、刹车、方向盘角度
        info = [
            f"Speed: {speed:.1f} km/h",
            f"Throttle: {throttle:.2f}",
            f"Brake:    {brake:.2f}",
            f"Steer:    {angle:.1f}°"
        ]
        for line in info:
            text_surf = self._font_mono.render(line, True, (255,255,255))
            display.blit(text_surf, (x, y))
            y += 20
        # 5) 渲染超车灯状态
        overtake_light_status = "True" if controller.get_overtake_light() else "False"
        overtake_text = f"Overtake Light: {overtake_light_status}"
        text_surf = self._font_mono.render(overtake_text, True, (255, 255, 255))
        display.blit(text_surf, (x, y))  # 显示超车灯状态
        y += 20

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return

        # 获取当前控制信息，并保存到 self.last_control
        self.last_control = world.player.get_control()

        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = self.last_control
        compass = world.imu_sensor.compass if world.imu_sensor is not None else 0.0
        heading = 'N' if compass > 270.5 or compass < 89.5 else ''
        heading += 'S' if 90.5 < compass < 269.5 else ''
        heading += 'E' if 0.5 < compass < 179.5 else ''
        heading += 'W' if 180.5 < compass < 359.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name.split('/')[-1],
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            u'Compass:% 17.0f° % 2s' % (compass, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            ''
        ]
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)
            ]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)
        ]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles, key=lambda vehicles: vehicles[0]):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))
                
        if time.time() - self.last_json_check >= 0.1:
            try:
                with open(self.json_path, "r") as f:
                    data = json.load(f)
                    print(f"[HUD Tick] Read data: {data}")
                    status = data.get("status", 0)
                    if status == 1:
                        self.json_info_text = "Extremely fast verify"
                    elif status == 2:
                        self.json_info_text = "Faster verify"
                    elif status == 3:
                        self.json_info_text = "More secure verify"
                    elif status == 4:
                        self.json_info_text = "Extremely safe verify"
                    else:
                        self.json_info_text = ""
            except Exception as e:
                self.json_info_text = f"{e}"
            self.last_json_check = time.time()

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    

# ============================================================================== 
# -- FadingText ----------------------------------------------------------------
# ==============================================================================
class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)

# ============================================================================== 
# -- HelpText ------------------------------------------------------------------
# ==============================================================================
class HelpText(object):
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.line_space = 18
        self.dim = (780, len(lines) * self.line_space + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * self.line_space))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)

# ============================================================================== 
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================
class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)
        if intensity > 1000.0:
            try:
                from infer_random_audio import FOLD_MAP
                if audio_queue is not None:
                    crash_fold = FOLD_MAP.get("up", "crash")
                    audio_queue.put(crash_fold)
                    self.hud.notification("Collision-triggered audio infer: crash", seconds=1.5)
            except Exception as e:
                print("[CollisionSensor] Failed to trigger audio on collision:", e)
# ============================================================================== 
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================
class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))

# ============================================================================== 
# -- GnssSensor ----------------------------------------------------------------
# ==============================================================================
class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)),
                                        attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude

# ============================================================================== 
# -- IMUSensor -----------------------------------------------------------------
# ==============================================================================
class IMUSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.imu')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data))

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self = weak_self()
        if not self:
            return
        limits = (-99.9, 99.9)
        self.accelerometer = (
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)))
        self.gyroscope = (
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.x))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.y))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))))
        self.compass = math.degrees(sensor_data.compass)

# ============================================================================== 
# -- RadarSensor ---------------------------------------------------------------
# ==============================================================================
class RadarSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.velocity_range = 7.5
        world = self._parent.get_world()
        self.debug = world.debug
        bp = world.get_blueprint_library().find('sensor.other.radar')
        bp.set_attribute('horizontal_fov', str(35))
        bp.set_attribute('vertical_fov', str(20))
        self.sensor = world.spawn_actor(
            bp,
            carla.Transform(carla.Location(x=2.8, z=1.0), carla.Rotation(pitch=5)),
            attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda radar_data: RadarSensor._Radar_callback(weak_self, radar_data))

    @staticmethod
    def _Radar_callback(weak_self, radar_data):
        self = weak_self()
        if not self:
            return
        current_rot = radar_data.transform.rotation
        for detect in radar_data:
            azi = math.degrees(detect.azimuth)
            alt = math.degrees(detect.altitude)
            fw_vec = carla.Vector3D(x=detect.depth - 0.25)
            carla.Transform(carla.Location(), carla.Rotation(pitch=current_rot.pitch + alt,
                                                              yaw=current_rot.yaw + azi,
                                                              roll=current_rot.roll)
                            ).transform(fw_vec)
            def clamp(min_v, max_v, value):
                return max(min_v, min(value, max_v))
            norm_velocity = detect.velocity / self.velocity_range
            r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
            g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
            b = int(abs(clamp(-1.0, 0.0, -1.0 - norm_velocity)) * 255.0)
            self.debug.draw_point(
                radar_data.transform.location + fw_vec,
                size=0.075,
                life_time=0.06,
                persistent_lines=False,
                color=carla.Color(r, g, b))

# ============================================================================== 
# -- CameraManager -------------------------------------------------------------
# ==============================================================================
class CameraManager(object):
    def __init__(self, parent_actor, hud, gamma_correction):
        # 驾驶摄像头部分（用于车辆驾驶画面显示）
        self.sensor = None          
        self.surface = None         
        self._parent = parent_actor
        self.hud = hud
        self.recording = False

        # 原驾驶摄像头视角设置（保持不变）
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        Attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)), Attachment.SpringArmGhost),
            (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid),
            (carla.Transform(carla.Location(x=5.5, y=1.5, z=1.5)), Attachment.SpringArmGhost),
            (carla.Transform(carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)), Attachment.SpringArmGhost),
            (carla.Transform(carla.Location(x=-1, y=-bound_y, z=0.5)), Attachment.Rigid)
        ]
        self.transform_index = 1  # 默认采用索引1的驾驶视角

        # 驾驶摄像头配置选项（保留两个选项，前置和后置，可由手柄切换）
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB (Front)', {}],
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB (Rear)', {}]
        ]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        # 驾驶摄像头分辨率采用 hud.dim（720p，默认1280×720）
        for item in self.sensors:
            bp = bp_library.find(item[0])
            bp.set_attribute('image_size_x', str(hud.dim[0]))
            bp.set_attribute('image_size_y', str(hud.dim[1]))
            if bp.has_attribute('gamma'):
                bp.set_attribute('gamma', str(gamma_correction))
            for attr_name, attr_value in item[3].items():
                bp.set_attribute(attr_name, attr_value)
            item.append(bp)
        self.index = 0  # 默认选择第一个驾驶摄像头
        self.set_sensor(self.index, notify=True, force_respawn=True)

        # ---------------------------
        # 新增：拍照摄像头部分（固定前置视角，不受驾驶摄像头切换影响）
        sensor_height = 2.4  # 固定拍照摄像头安装高度
        self.last_photo_time = 0
        self.photo_save_interval = 2  # 每 2 秒拍一张
        self.photo_save_path = "Photo"  # 拍照图片保存目录
        self.photo_force_save = False   # 用于立即触发拍照的标志
        self.emergency_state = False    # 紧急状态，布尔类型
        self.photo_transform = carla.Transform(carla.Location(z=sensor_height), carla.Rotation(yaw=0))
        bp_photo = bp_library.find('sensor.camera.rgb')
        bp_photo.set_attribute('image_size_x', "640")
        bp_photo.set_attribute('image_size_y', "360")
        if bp_photo.has_attribute('gamma'):
            bp_photo.set_attribute('gamma', str(gamma_correction))
        self.photo_sensor = world.spawn_actor(
            bp_photo,
            self.photo_transform,
            attach_to=self._parent,
            attachment_type=Attachment.Rigid
        )
        weak_self = weakref.ref(self)
        self.photo_sensor.listen(lambda image: CameraManager._photo_callback(weak_self, image))
        # ---------------------------
        
    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=True, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)
        needs_respawn = True if self.sensor is None else (force_respawn or (self.sensors[index][2] != self.sensors[self.index][2]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1]
            )
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def immediate_photo_capture(self):
        self.photo_force_save = True
        self.emergency_state = True
        self.emergency_reset_time = time.time() + 1.0
        self.last_photo_time = time.time()
        return self.emergency_state

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        image.convert(self.sensors[self.index][1])
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)

    @staticmethod
    def _photo_callback(weak_self, image):
        self = weak_self()
        if not self:
            return
        current_time = time.time()
        if hasattr(self, 'emergency_reset_time') and current_time > self.emergency_reset_time:
            self.emergency_state = False

        image.convert(cc.Raw)
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = np.reshape(arr, (image.height, image.width, 4))
        bgr_array = arr[:, :, :3]
        cv2.imshow("Photo Camera View", bgr_array)
        cv2.waitKey(1)
        if (current_time - self.last_photo_time >= self.photo_save_interval) or self.photo_force_save:
            self.last_photo_time = current_time
            self.photo_force_save = False
            if self.photo_save_path is not None:
                if not os.path.exists(self.photo_save_path):
                    os.makedirs(self.photo_save_path)
                filename = os.path.join(self.photo_save_path, "photo.png")
                cv2.imwrite(filename, bgr_array)
                print(f"Photo captured and saved: {filename}")

# ============================================================================== 
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================
def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(30.0)
        # 以下负责切换场景
            #Town01	小城市街区
           # Town02	城市中心区域
           # Town03	环形郊区道路
           # Town04	高速公路和收费站
           # Town05	有多种天气和坡道的小镇
           # Town06	大型城市，复杂交叉路口
           #Town07	小型郊区住宅区
           # Town10HD	高精度城市地图
           # Town11	山路场景
        client.load_world('Town10HD')
        sim_world = client.get_world()

        display = pygame.display.set_mode((args.width,args.height), pygame.HWSURFACE|pygame.DOUBLEBUF)
        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud, args)
        if pygame.joystick.get_count() > 0:
            controller = XboxController(world)
            hud.notification("Using Xbox Controller")
        else:
            hud.notification("Using Keyboard Control")
            return
        clock = pygame.time.Clock()
        while True:
            clock.tick_busy_loop(45)
            try:
                while True:
                    fold, cls = result_queue.get_nowait()
                    hud.notification(f"Audio infer: {fold} → class={cls}", seconds=3.0)
            except queue.Empty:
                pass
            if controller.parse_events(world, clock): break
            world.tick(clock)
            world.render(display, controller)
            pygame.display.flip()
    finally:
        audio_queue.put(None)
        worker.join(timeout=1)
        if world and getattr(world, 'recording_enabled', False):
            client.stop_recorder()
        if world is not None:
            world.destroy()
        cv2.destroyAllWindows()
        pygame.quit()


# ============================================================================== 
# -- main() --------------------------------------------------------------------
# ==============================================================================
def main():
    mp.set_start_method('spawn', force=True)
    global audio_queue, result_queue, worker
    audio_queue = mp.Queue()
    result_queue = mp.Queue()
    worker = mp.Process(target=audio_worker, args=(audio_queue, result_queue, "DL.json"), daemon=True)
    worker.start()
    parser = argparse.ArgumentParser(description='CARLA Manual Control Client')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('-p','--port', default=2000, type=int)
    parser.add_argument('--res', default='1280x720')
    parser.add_argument('--filter', default='vehicle.*')
    parser.add_argument('--rolename', default='hero')
    parser.add_argument('--gamma', default=2.2, type=float)
    args = parser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]
    logging.basicConfig(level=logging.INFO)
    try:
        game_loop(args)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
    finally:
        audio_queue.put(None)
        worker.join(timeout=1)
        
if __name__ == '__main__':
    main()

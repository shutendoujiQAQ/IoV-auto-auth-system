#!/usr/bin/env python
"""
CARLA manual control with Xbox controller.

使用 Xbox 手柄进行车辆控制：
    - 左摇杆水平（轴 0）：控制转向。
    - 右触发器（轴 5）：控制油门（映射到 [0,1]）。
    - 左触发器（轴 4）：控制刹车（映射到 [0,1]）。
    - 按钮 B（索引 1）：切换倒车状态，同时立即拍照。
    - 按钮 X（索引 2）：切换驾驶相机视角。

当刹车持续用力（刹车值 > 0.8）超过 3 秒时，
采样车辆数据（方向盘、油门、刹车、速度、位置）采样周期为 0.2 秒，
录制 7 秒后，以 JSON 格式输出到 heavy_brake_record.json 文件，并在此时立即触发拍照。
拍照摄像头每 2 秒拍一张图片，所有图片始终覆盖保存为 photo.png。

此外，在当前 UI 顶部绘制三个指示灯，当触发紧急状态时，第一个灯亮（红色），1秒后重置熄灭。
"""

from __future__ import print_function
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

try:
    import numpy as np
except ImportError:
    raise RuntimeError("cannot import numpy, make sure numpy package is installed")

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
        self.map_layer_names = [carla.MapLayer.NONE,
                                carla.MapLayer.Buildings,
                                carla.MapLayer.Decals,
                                carla.MapLayer.Foliage,
                                carla.MapLayer.Ground,
                                carla.MapLayer.ParkedVehicles,
                                carla.MapLayer.Particles,
                                carla.MapLayer.Props,
                                carla.MapLayer.StreetLights,
                                carla.MapLayer.Walls,
                                carla.MapLayer.All]

    def restart(self):
        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # 固定车型：使用 Tesla Model 3
        blueprint = self.world.get_blueprint_library().find('vehicle.tesla.model3')
        blueprint.set_attribute('role_name', self.actor_role_name)
        if blueprint.has_attribute('color'):
            blueprint.set_attribute('color', random.choice(blueprint.get_attribute('color').recommended_values))
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

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        if self.radar_sensor is not None:
            self.toggle_radar()
        sensors = [self.camera_manager.sensor,
                   self.collision_sensor.sensor,
                   self.lane_invasion_sensor.sensor,
                   self.gnss_sensor.sensor,
                   self.imu_sensor.sensor]
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
    """通过 Xbox 手柄控制车辆"""
    def __init__(self, world):
        pygame.joystick.init()
        if pygame.joystick.get_count() < 1:
            raise RuntimeError("未检测到 Xbox 手柄")
        self._joystick = pygame.joystick.Joystick(0)
        self._joystick.init()
        self.world = world
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
        else:
            raise NotImplementedError("仅支持车辆控制")
        self._control.throttle = 0.0
        self._control.brake = 0.0
        self._control.steer = 0.0
        self._reverse = False
        # 数据录制变量
        self._heavy_brake_start = None
        self._recording = False
        self._record_start_time = None
        self._last_sample_time = None
        self._record_data = []

    def parse_events(self, world, clock):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return True

        steer_axis = self._joystick.get_axis(0)
        raw_throttle = self._joystick.get_axis(5)
        raw_brake = self._joystick.get_axis(4)

        throttle = (raw_throttle + 1) / 2.0
        brake = (raw_brake + 1) / 2.0

        self._control.steer = round(steer_axis, 2)
        self._control.throttle = throttle
        self._control.brake = brake

        # 切换倒车状态（按钮 B，索引 1），同时立即触发拍照
        if self._joystick.get_button(1):
            self._reverse = not self._reverse
            self._control.reverse = self._reverse
            world.hud.notification("Reverse mode: %s" % ("On" if self._reverse else "Off"))
            pygame.time.wait(300)
            # 立即触发拍照并更新紧急状态
            emergency = self.world.camera_manager.immediate_photo_capture()
            world.hud.emergency_state = emergency

        self.world.player.apply_control(self._control)

        current_time = pygame.time.get_ticks() / 1000.0
        #重刹3秒开录制代码
        if brake > 0.8:
            if self._heavy_brake_start is None:
                self._heavy_brake_start = current_time
        else:
            self._heavy_brake_start = None

        if self._heavy_brake_start is not None and (current_time - self._heavy_brake_start >= 2.0):
            if not self._recording:
                self._recording = True
                self._record_start_time = current_time
                self._last_sample_time = current_time
                self._record_data = []
                world.hud.notification("Heavy brake detected: Recording started")
                self.world.camera_manager.immediate_photo_capture()
                world.hud.emergency_state = self.world.camera_manager.emergency_state
        if self._recording:
            if current_time - self._last_sample_time >= 0.2:
                control = self._control
                transform = self.world.player.get_transform()
                velocity = self.world.player.get_velocity()
                speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
                sample = {
                    'time': current_time - self._record_start_time,
                    'steer': control.steer,
                    'throttle': control.throttle,
                    'brake': control.brake,
                    'speed': speed,
                    'location': (transform.location.x, transform.location.y, transform.location.z)
                }
                self._record_data.append(sample)
                self._last_sample_time = current_time
            if current_time - self._record_start_time >= 7.0:
                self._write_record_to_file()
                self._recording = False
                self._record_data = []
                self._record_start_time = None
                world.hud.notification("Recording finished, data saved.")
        # 切换驾驶摄像头视角（按钮 X，索引 2）
        if self._joystick.get_button(2):
            self.world.camera_manager.toggle_camera()
            pygame.time.wait(300)
        return False

    def _write_record_to_file(self):
        output = {
            "hard_brake": self._record_data,
            "overtake": 1
        }
        filename = "heavy_brake_record.json"
        with open(filename, "w") as f:
            json.dump(output, f, indent=4)

# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================
class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 16), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()
        # 新增：紧急状态属性，用于指示灯显示
        self.emergency_state = False

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
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

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        # 绘制顶部的3个指示灯
        diameter = 20
        spacing = 10
        x_start = (self.dim[0] - (3 * diameter + 2 * spacing)) // 2
        y_pos = 10
        # 第一盏灯：如果 emergency_state 为 True，则显示红色，否则灰色
        color1 = (255, 0, 0) if self.emergency_state else (100, 100, 100)
        color2 = (100, 100, 100)
        color3 = (100, 100, 100)
        pygame.draw.circle(display, color1, (x_start + diameter//2, y_pos + diameter//2), diameter//2)
        pygame.draw.circle(display, color2, (x_start + diameter + spacing + diameter//2, y_pos + diameter//2), diameter//2)
        pygame.draw.circle(display, color3, (x_start + 2*(diameter+spacing) + diameter//2, y_pos + diameter//2), diameter//2)

        # 绘制原有HUD信息
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)

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
        self.photo_save_interval = 2  # 每2秒拍一张
        self.photo_save_path = "Project\photo_images"  # 拍照图片保存目录
        self.photo_force_save = False   # 用于立即触发拍照的标志
        self.emergency_state = False    # 紧急状态，布尔类型
        self.photo_transform = carla.Transform(carla.Location(z=sensor_height), carla.Rotation(yaw=0))
        bp_photo = bp_library.find('sensor.camera.rgb')
        # 降低拍照摄像头画质，设置为640×360
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
        # 驾驶摄像头在前置与后置之间切换
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
        # 立即触发拍照，同时设置 photo_force_save 为 True，
        # 并将 emergency_state 置为 True，1秒后重置
        self.photo_force_save = True
        self.emergency_state = True
        self.emergency_reset_time = time.time() + 1.0
        # 同时重置 last_photo_time，防止定时拍照过早触发
        self.last_photo_time = time.time()
        return self.emergency_state

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        # 驾驶摄像头回调：按原720p设置和颜色处理（转换为RGB后显示在pygame窗口）
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
        # 拍照摄像头回调：处理拍照图像用于独立显示和定时保存
        self = weak_self()
        if not self:
            return
        # 检查紧急状态是否需要重置
        current_time = time.time()
        if hasattr(self, 'emergency_reset_time') and current_time > self.emergency_reset_time:
            self.emergency_state = False

        image.convert(cc.Raw)
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = np.reshape(arr, (image.height, image.width, 4))
        # 取前三个通道（BGRA -> BGR），OpenCV显示使用BGR格式
        bgr_array = arr[:, :, :3]
        cv2.imshow("Photo Camera View", bgr_array)
        cv2.waitKey(1)
        # 每 photo_save_interval 秒拍照，或当强制标志触发时拍照
        if (current_time - self.last_photo_time >= self.photo_save_interval) or self.photo_force_save:
            self.last_photo_time = current_time
            self.photo_force_save = False
            if self.photo_save_path is not None:
                if not os.path.exists(self.photo_save_path):
                    os.makedirs(self.photo_save_path)
                # 始终以相同文件名保存（覆盖），例如 "photo.png"
                filename = os.path.join(self.photo_save_path, "photo.png")
                cv2.imwrite(filename, bgr_array)
                print(f"Photo captured and saved: {filename}")

# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(2.0)
        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
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
            if controller.parse_events(world, clock):
                return
            world.tick(clock)
            world.render(display)
            pygame.display.flip()
    finally:
        if (world and world.recording_enabled):
            client.stop_recorder()
        if world is not None:
            world.destroy()
        cv2.destroyAllWindows()
        pygame.quit()

# ==============================================================================
# -- main() --------------------------------------------------------------------
def main():
    argparser = argparse.ArgumentParser(description='CARLA Manual Control Client')
    argparser.add_argument('-v', '--verbose', action='store_true', dest='debug', help='print debug information')
    argparser.add_argument('--host', metavar='H', default='127.0.0.1', help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument('-p', '--port', metavar='P', default=2000, type=int, help='TCP port to listen to (default: 2000)')
    argparser.add_argument('-a', '--autopilot', action='store_true', help='enable autopilot')
    argparser.add_argument('--res', metavar='WIDTHxHEIGHT', default='1280x720', help='window resolution (default: 1280x720)')
    argparser.add_argument('--filter', metavar='PATTERN', default='vehicle.*', help='actor filter (default: "vehicle.*")')
    argparser.add_argument('--rolename', metavar='NAME', default='hero', help='actor role name (default: "hero")')
    argparser.add_argument('--gamma', default=2.2, type=float, help='Gamma correction of the camera (default: 2.2)')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)
    print(__doc__)

    try:
        game_loop(args)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

if __name__ == '__main__':
    main()

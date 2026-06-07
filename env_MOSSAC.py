# -*- coding: utf-8 -*-
"""
@File    : env_MOSSAC.py
@Project : Risk-Aware eVTOL Path Planning
@Author  : DavidLin
@Date    : 2026/06/07
@Contact : davidlin659562@gmail.com
@License : MIT License
@Description:
This module defines the Gym-style path-planning environment used by MOSSAC.
It models eVTOL navigation on a 101x101 grid with static obstacles,
dynamic obstacles, casualty-risk maps, meteorological-risk maps, and
radar-based observations.

Main components:
1. Collision and ray-intersection utilities for static and dynamic obstacles.
2. Safety, TTC, casualty-risk, and meteorological-risk reward components.
3. SimplePathPlanningEnv, the reinforcement-learning environment used for
   MOSSAC training, testing, rendering, and trajectory logging.
4. Matrix-loading utilities for the default obstacle, casualty-risk, and
   meteorological-risk maps.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from numba import jit, njit
import gym
from gym import spaces
import shapely.geometry as geo
from shapely.ops import nearest_points 
import math
import collections
from shapely.geometry import Point
import random
from generate_grid_map import generate_grid_map
import numpy.ma as ma




@njit
def calculate_distance(x1, y1, x2, y2):
    """Compute the distance between two points in grid cells"""
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def check_all_collisions(x, y, grid_map, dynamic_obs_lists, safety_margin=0.5):
    """Check collisions with all obstacles (use grid_map for static obstacles, use [x, y, r] lists for dynamic obstacles)"""
    # static obstacle (grid map)
    xi, yi = int(round(x)), int(round(y))
    if 0 <= xi < grid_map.shape[0] and 0 <= yi < grid_map.shape[1]:
        if grid_map[xi, yi] == 1:
            return True
    # dynamic obstacle (circle)
    for obs in dynamic_obs_lists:
        obs_x, obs_y, obs_radius = obs
        if calculate_distance(x, y, obs_x, obs_y) < (obs_radius + safety_margin):
            return True
    return False

@njit
def ray_dynamic_intersect(x0, y0, dx, dy, cx, cy, r, max_range):
    """Check intersection with a dynamic obstacle"""
    # ray origin(x0, y0), direction(dx, dy), circle center(cx, cy), radiusr
    ox, oy = x0 - cx, y0 - cy
    a = dx*dx + dy*dy
    b = 2 * (ox*dx + oy*dy)
    c = ox*ox + oy*oy - r*r
    D = b*b - 4*a*c
    if D < 0:
        return None  # no intersection
    sqrtD = np.sqrt(D)
    t1 = (-b - sqrtD) / (2*a)
    t2 = (-b + sqrtD) / (2*a)
    ts = [t for t in [t1, t2] if t >= 0]
    if not ts:
        return None
    t_hit = min(ts) # take the smaller value
    if t_hit > max_range:
        return None
    hit_x = x0 + t_hit * dx
    hit_y = y0 + t_hit * dy
    return t_hit, (hit_x, hit_y)

@njit
def ray_static_intersect(x0, y0, dx, dy, grid_map, max_range, step=0.2):
    """Check intersection with a static obstacle"""
    # ray origin(x0, y0), direction(dx, dy), map information(grid_map), ray step size (step)
    dist = 0.0
    while dist < max_range:
        x = x0 + dist * dx
        y = y0 + dist * dy
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < grid_map.shape[0] and 0 <= yi < grid_map.shape[1]:
            if grid_map[xi, yi] == 1:
                return dist, (x, y)
        else:
            break
        dist += step
    return None

@njit
def get_casualty_value(x, y, casualty_map):
    """
    Get (x, y) casualty-risk value at position (if casualty_map is available)
    """
        
    # convert to integer grid coordinates
    xi, yi = int(round(x)), int(round(y))
        
    # boundary check
    if xi < 0 or xi > 100 or yi < 0 or yi > 100:
        return 0

    return int(casualty_map[xi, yi])

@njit
def get_meteor_value(x, y, meteor_map):
    """
    Get (x, y) meteorological-risk value at position (if meteor_map is available)
    """
        
    # convert to integer grid coordinates
    xi, yi = int(round(x)), int(round(y))

    # boundary check
    if xi < 0 or xi > 100 or yi < 0 or yi > 100:
        return 0

    return meteor_map[xi, yi]


def calculate_safety_rewards(x, y, vx, vy, grid_map, dynamic_obs_lists, radar_max_range,
                             ego_radius=1.0, 
                             w_sta=5, d0=6.0, #previouslyw_sta=10.0, 2.4
                             w_dyn=4.8, ttc_max_inv=2.0, #previouslyw_dyn=8.0, ttc_max_inv=10.0
                             eps=1e-3, topK=2):
    """
    Input: position(x,y), velocity(vx,vy), map information(grid_map), dynamic obstacle(dynamic_obs_lists), radar maximum range(radar_max_range)
    static obstacles use grid_map, dynamic obstacles use [x, y, r] or [x, y, r, vx, vy] or dict (including start/end/speed).
    Return: (R_clear, R_dyn)
    """ 
    px, py = float(x), float(y) # ego position (grid cells)
    pvx, pvy = float(vx), float(vy) # ego velocity (grid cells/s)
    d_list = [] # nearest-distance list (static/dynamic)
    threats = []  # approximate 1/TTC list (dynamic obstacles only)

    # 1. static obstacle: iterate over all obstacle cells and take the nearest distance
    idxs = np.argwhere(grid_map == 1)
    if idxs.size > 0:
        dists = np.sqrt((idxs[:, 0] - px) ** 2 + (idxs[:, 1] - py) ** 2) - ego_radius
        dists = dists[dists >= 0]
        dists = dists[dists <= radar_max_range]
        if dists.size > 0:
            d_list.append(np.min(dists)) # record the nearest static-obstacle distance within radar range

    # 2. dynamic obstacle
    for obs in dynamic_obs_lists:
        # supports [x, y, r] or [x, y, r, vx, vy] or dict
        if isinstance(obs, dict):
            ox, oy, r_obs = float(obs['pos'][0]), float(obs['pos'][1]), float(obs['radius'])
            # velocity
            vec = (obs['end'] - obs['start']).astype(np.float64)
            vec = vec / (np.linalg.norm(vec) + 1e-9)
            ovx, ovy = obs['speed'] * vec[0], obs['speed'] * vec[1]
        elif len(obs) == 5:
            ox, oy, r_obs, ovx, ovy = obs
        else:
            ox, oy, r_obs = obs
            ovx, ovy = 0.0, 0.0

        center = np.hypot(px - ox, py - oy)
        edge_dist = max(0.0, center - r_obs - ego_radius)
        if edge_dist > radar_max_range:
            continue
        d_list.append(edge_dist) # record the nearest dynamic-obstacle distance within radar range

        # compute TTC (only when the dynamic obstacle has velocity)
        rx, ry = px - ox, py - oy # relative position 
        rnorm = max(np.hypot(rx, ry), eps) # compute the Euclidean distance
        vrelx, vrely = pvx - ovx, pvy - ovy # compute relative velocity
        drdt = (rx * vrelx + ry * vrely) / rnorm 
        if drdt < 0.0:
            r_edge = max(edge_dist, eps)
            ttc_inv = min((-drdt) / r_edge, ttc_max_inv)  # inverse distance-over-relative-speed term 
            threats.append(ttc_inv)

    # 2) clearance: smooth penalty (almost no penalty for distant obstacles)
    if d_list:
        dmin = float(np.min(d_list))
        R_clear = - w_sta * np.exp(- dmin / max(d0, eps))
    else:
        R_clear = 0.0
        dmin = 0.0

    # 3) dynamic urgency: keep only the most dangerous Top-K
    if threats:
        thr = np.asarray(threats, dtype=np.float64)
        k = min(topK, thr.size) # if the number of threats is smaller than top-k, use the actual count
        top = np.argpartition(thr, -k)[-k:]
        R_dyn = - w_dyn * thr[top].sum()
        ttc = 1/thr[top][-1]
    else:
        R_dyn = 0.0
        ttc = 0.0

    return R_clear, R_dyn, dmin, ttc


def calculate_reward_components(x, y, target_x, target_y, grid_map, casualty_map, meteor_map, dynamic_obs_lists, prev_distance, velocity, heading, radar_max_range):
    """Compute the reward mechanism (including safety-distance and TTC-based dynamic penalties)"""
    # goal-distance reward
    current_distance = calculate_distance(x, y, target_x, target_y)
    distance_reward = (prev_distance - current_distance) * 10.0 # reward for reducing distance
    # collision penalty
    collision_penalty = 0.0
    if check_all_collisions(x, y, grid_map, dynamic_obs_lists):
        collision_penalty = -50.0 
    
    # out-of-bounds penalty
    boundary_penalty = 0.0
    if x < 0 or x > 100 or y < 0 or y > 100:
        boundary_penalty = -50.0  
    
    # goal-arrival reward
    success_reward = 0.0
    if current_distance < 2.0:
        success_reward = 500.0  
     
    # risk penalty
    risk = get_casualty_value(x, y, casualty_map)
    if risk == 1:
        risk_penalty = -0.1 # low-risk penalty
    elif risk == 2:
        risk_penalty = -0.2  # high-risk penalty
    elif risk == 0:
        risk_penalty = -0.05
 
    meteor_risk = get_meteor_value(x, y, meteor_map) # matrix with discrete reward values
    meteor_risk_penalty = -meteor_risk * 10


    # compute safety-distance and TTC penalties
    vx = velocity * np.cos(heading)
    vy = velocity * np.sin(heading)
    R_clear, R_dyn, dmin, ttc_inv = calculate_safety_rewards(x, y, vx, vy, grid_map, dynamic_obs_lists, radar_max_range)
    dynamic_penalty = R_clear + R_dyn
    return distance_reward, dynamic_penalty, collision_penalty, boundary_penalty, success_reward, current_distance, risk_penalty, meteor_risk_penalty


class SimplePathPlanningEnv(gym.Env):
    """MOSSAC path-planning environment (supports radar observations and Shapely obstacle geometry)"""
    
    def __init__(self, 
                 grid_map,
                 casualty_map,
                 meteor_map,
                 start=[10.0,10.0],
                 end=[57.0,97.0],
                 map_size=101, 
                 max_steps=500,
                 dt=0.5,  # sampling time
                 num_obstacles=8,
                 radar_num_points=64,
                 radar_max_range=15.0, # maximum range
                 seed=None,
                 dynamic_type='cross'):
        
        super().__init__()
        self.grid_map_original = grid_map.copy()  # original grid map
        self.grid_map = grid_map.copy()
        self.casualty_map = casualty_map.copy() if casualty_map is not None else None
        self.meteor_map = meteor_map.copy() if meteor_map is not None else None
        self.resolution = 30 
        self.map_size = map_size
        self.max_steps = max_steps
        self.dt = dt
        self.num_obstacles = num_obstacles
        self.radar_num_points = radar_num_points
        self.radar_max_range = radar_max_range
        self.radar_history_len = 8
        self.velocity_max_real = 27.8  # maximum velocity actual 27.8m/s
        self.velocity_max = self.velocity_max_real / self.resolution
        self.velocity_min_real = 12
        self.velocity_min = self.velocity_min_real / self.resolution
        self.speed_aircraft = 18

        self.n_planes = 1

        # observation space: Dictstructure, containsstate_seq, radar_seq, mask_seq, casualty_map
        # [x, y, velocity, heading, target_x, target_y]
        # mask channels: 1 indicates a obstacle hit, 0 indicates no obstacle hit

        self.observation_space = spaces.Dict({
            'state_seq': spaces.Box(
                low=0.0,
                high=1.0,
                shape=(6,),
                dtype=np.float32
            ),
            'radar_seq': spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.radar_history_len, self.radar_num_points),
                dtype=np.float32
            ),
            'mask_seq': spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.radar_history_len, self.radar_num_points),
                dtype=np.float32
            ),
            # relative velocity channels
            'rel_speeds_seq': spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.radar_history_len, self.radar_num_points),
                dtype=np.float32
            ),
            # casualty risk map, normalized to [0, 1]
            'casualty_map': spaces.Box( 
                low=0.0,
                high=1.0,
                shape=(self.map_size, self.map_size),
                dtype=np.float32
            ),
            # meteorological risk map, normalized to [0, 1]
            'meteor_map': spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.map_size, self.map_size),
                dtype=np.float32
            )
        })
        
        # action space: [velocity change, angle change]
        self.action_space = spaces.Box(
            low=np.array([-5 / self.resolution, -0.14], dtype=np.float32), 
            high=np.array([5 / self.resolution, 0.14], dtype=np.float32), 
            dtype=np.float32
        ) 
        # every s: velocity changes by [-5 m/s, 5 m/s], angle change is [-0.14, 0.14] radians



        # state variables
        self.position = np.array(start, dtype=np.float32)  # [x, y]
        self.velocity = np.float32(22 / self.resolution)  # initial velocity

        self.start = np.array(start, dtype=np.float32)  # position
        self.target = np.array(end, dtype=np.float32)  # goal position
        dx = self.target[0] - self.start[0]
        dy = self.target[1] - self.start[1]
        self.heading = np.float32(np.arctan2(dy, dx))  # initial heading angle

        self.dynamic_obstacles = []  # dynamic-obstacle list
        # radar distance and mask channels initialize
        self.radar_seq = np.full(self.radar_num_points, 1.0, dtype=np.float32)
        self.mask_seq = np.zeros(self.radar_num_points, dtype=np.float32)
        self.rel_speeds = np.zeros(self.radar_num_points, dtype=np.float32)  # relative velocity channels initialize
        # history radar
        self.radar_seq_history = collections.deque(maxlen=self.radar_history_len)
        self.mask_seq_history = collections.deque(maxlen=self.radar_history_len)
        self.rel_speeds_history = collections.deque(maxlen=self.radar_history_len)

        # training statistics
        self.step_count = 0
        self.trajectory = []
        
        # visualization
        self.fig = None
        self.ax = None
        
        self.seed = seed
        self.dynamic_type = dynamic_type
        self.velocity_history = []
        self.heading_history = []

        # initialize dynamic obstacle
        if self.dynamic_type == 'static':
        # mode: generate when dynamic obstacle
            self.dynamic_obstacles = []
        elif self.dynamic_type == 'cross':
            self._init_dynamic_obstacles_cross(clear=True)
        elif self.dynamic_type == 'parallel':
            self._init_dynamic_obstacles_parallel(clear=True)
        elif self.dynamic_type == 'both':
            self.dynamic_obstacles = []
            self._init_dynamic_obstacles_cross(clear=False)
            self._init_dynamic_obstacles_parallel(clear=False)
        elif self.dynamic_type == 'random':
            # orboth
            choice = random.choice(['cross', 'parallel', 'both'])
            if choice == 'cross':
                self._init_dynamic_obstacles_cross(clear=True)
            elif choice == 'parallel':
                self._init_dynamic_obstacles_parallel(clear=True)
            else:
                self.dynamic_obstacles = []
                self._init_dynamic_obstacles_cross(clear=False)
                self._init_dynamic_obstacles_parallel(clear=False)
        elif self.dynamic_type == 'random_test':
            # self._init_extra_dynamic_obstacles()
            # self._init_extra_static_obstacles()
            choice = random.choice(['cross', 'parallel', 'both'])
            if choice == 'cross':
                self._init_dynamic_obstacles_cross(clear=True)
            elif choice == 'parallel':
                self._init_dynamic_obstacles_parallel(clear=True)
            else:
                self.dynamic_obstacles = []
                self._init_dynamic_obstacles_cross(clear=False)
                self._init_dynamic_obstacles_parallel(clear=False)
        elif self.dynamic_type == 'random_test_both':
            # self._init_extra_dynamic_obstacles()
            # self._init_extra_static_obstacles()
            self.dynamic_obstacles = []
            self._init_dynamic_obstacles_cross(clear=False)
            self._init_dynamic_obstacles_parallel(clear=False)
        elif self.dynamic_type == 'fixed_test_both':
            # self._init_extra_dynamic_obstacles()
            # self._init_extra_static_obstacles()
            self.dynamic_obstacles = []
            self._init_dynamic_obstacles_cross(clear=False)
            self._init_dynamic_obstacles_parallel(clear=False)


    def _init_dynamic_obstacles_cross(self, clear=True):
        #  dynamic obstacles with crossing routes
        if clear:
            self.dynamic_obstacles = []
        configs = [
            # (start, end, n_planes, speed)
            (np.array([0.0, 40.0]), np.array([40.0, 0.0]), self.n_planes, self.speed_aircraft / self.resolution),
            (np.array([0.0, 80.0]), np.array([80.0, 0.0]), 2*self.n_planes, self.speed_aircraft / self.resolution),
            (np.array([20.0, 100.0]), np.array([100.0, 20.0]), 2*self.n_planes, self.speed_aircraft / self.resolution),
            (np.array([60.0, 100.0]), np.array([100.0, 60.0]), self.n_planes, self.speed_aircraft / self.resolution),
        ]
        for start, end, n, speed in configs:
            if n == 1:
                poses = [start]
            elif n == 2:
                poses = [start, (start + end) / 2]
            else:
                poses = [start + (i / n) * (end - start) for i in range(n)]
            for pos in poses:
                self.dynamic_obstacles.append({
                    'start': start.copy(),
                    'end': end.copy(),
                    'pos': pos.copy(),
                    'radius': 1.0,  
                    'speed': speed, 
                    'direction': 1
                })

    def _init_dynamic_obstacles_parallel(self, clear=True):
        #  dynamic obstacles with parallel routes
        if clear:
            self.dynamic_obstacles = []
        configs = [
            (np.array([40.0, 100.0]), np.array([0.0, 60.0]), self.n_planes, self.speed_aircraft / self.resolution),   
            (np.array([80.0, 100.0]), np.array([0.0, 20.0]), 2*self.n_planes, self.speed_aircraft / self.resolution),   
            (np.array([100.0, 80.0]), np.array([20.0, 0.0]), 2*self.n_planes, self.speed_aircraft / self.resolution),   
            (np.array([100.0, 40.0]), np.array([60.0, 0.0]), self.n_planes, self.speed_aircraft / self.resolution),   
        ]
        for start, end, n, speed in configs:
            if n == 1:
                poses = [start]
            elif n == 2:
                poses = [start, (start + end) / 2]
                # poses = [start + 0.25 * (end - start), start + 0.75 * (end - start)]
            else:
                poses = [start + (i / n) * (end - start) for i in range(n)]
            for pos in poses:
                self.dynamic_obstacles.append({
                    'start': start.copy(),
                    'end': end.copy(),
                    'pos': pos.copy(),
                    'radius': 1.0, 
                    'speed': speed, 
                    'direction': 1
                })

    def _init_extra_dynamic_obstacles(self, n_min=3, n_max=6, min_dist=15, min_goal_dist=20):
        # Generating random dynamic obstacles without routes
        self.extra_dynamic_obstacles = []
        n_obs = np.random.randint(n_min, n_max+1)
        for _ in range(n_obs):
            while True:
                pos = np.random.uniform(0, self.map_size, size=2)
                if (np.linalg.norm(pos - self.position) > min_dist and
                    np.linalg.norm(pos - self.target) > min_goal_dist):
                    break
            radius = 1.0
            self.extra_dynamic_obstacles.append({
                'pos': pos,
                'radius': radius
            })
        self.extra_dynamic_step_counter = 0

    def _update_extra_dynamic_obstacles(self, min_dist=15, min_goal_dist=20): 
        # Change position every 10 steps to avoid the agent and the target point 
        if not hasattr(self, 'extra_dynamic_step_counter'):
            self.extra_dynamic_step_counter = 0
        self.extra_dynamic_step_counter += 1
        if self.extra_dynamic_step_counter % 10 == 0:
            for obs in self.extra_dynamic_obstacles:
                while True:
                    pos = np.random.uniform(0, self.map_size, size=2)
                    if (np.linalg.norm(pos - self.position) > min_dist and
                        np.linalg.norm(pos - self.target) > min_goal_dist):
                        obs['pos'] = pos
                        break



    def _init_extra_static_obstacles(self, n_min=2, n_max=4):
        # Generate 1–3 special static obstacles, update `grid_map` directly, and record the mask for visualisation
        np.random.seed(self.seed) if self.seed is not None else np.random.seed()
        self.extra_static_masks = []
        n_obs = np.random.randint(n_min, n_max+1)
        for _ in range(n_obs):
            while True:
                x = np.random.uniform(0, 100)
                y = np.random.uniform(0, 100)
                radius = np.random.uniform(2, 5)
                start_dist = calculate_distance(x, y, self.start[0], self.start[1])
                target_dist = calculate_distance(x, y, self.target[0], self.target[1])
                if start_dist > radius + 15 and target_dist > radius + 15:
                    yy, xx = np.meshgrid(np.arange(self.map_size), np.arange(self.map_size))
                    mask = (xx - x)**2 + (yy - y)**2 <= radius**2
                    self.grid_map[mask] = 1 #updategrid_map
                    self.extra_static_masks.append(mask) # recordmask
                    break
    
    def _init_extra_fixed_static_obstacles(self, n_min=3, n_max=3, positions=None):
        self.extra_static_masks = []
        
        # ifposition, usefixedposition
        if positions is None:
            positions = [(57, 48, 2), (65, 62, 1.5),(75, 85, 2)]

        for pos in positions:
            # Inputformat
            if len(pos) == 3:
                x, y, radius = pos
            elif len(pos) == 2:
                x, y = pos
                radius = 3  # radius
            else:
                print(f"warning: noneobstaclepositionformat {pos}, already")
                continue
            
            # checkiswith/
            start_dist = calculate_distance(x, y, self.start[0], self.start[1])
            target_dist = calculate_distance(x, y, self.target[0], self.target[1])
            
            if start_dist > radius + 10 and target_dist > radius + 10:
                # generatecircleobstaclemask
                yy, xx = np.meshgrid(np.arange(self.map_size), np.arange(self.map_size))
                mask = (xx - x)**2 + (yy - y)**2 <= radius**2
                self.grid_map[mask] = 1  # updategrid_map
                self.extra_static_masks.append(mask)  # recordmaskfor visualization
            else:
                print(f"warning: obstacle ({x}, {y}, {radius}) with/distance, already")

    def _update_dynamic_obstacles(self):  
        # Each moving obstacle moves in a loop in the same direction along the route; 
        # upon reaching the end, it returns directly to the start.
        for obs in self.dynamic_obstacles:
            start, end = obs['start'], obs['end']
            pos = obs['pos']
            vec = end - start
            vec = vec / np.linalg.norm(vec)
            pos_new = pos + vec * obs['speed'] * self.dt
            if np.linalg.norm(pos_new - end) < 1.0:
                pos_new = start.copy()
            obs['pos'] = pos_new

    def _get_all_obstacles(self):
        # Return all obstacles
        dynamic_obs_list = []
        for obs in getattr(self, 'dynamic_obstacles', []):
            dynamic_obs_list.append([obs['pos'][0], obs['pos'][1], obs['radius']])

        for obs in getattr(self, 'extra_dynamic_obstacles', []):
            dynamic_obs_list.append([obs['pos'][0], obs['pos'][1], obs['radius']])


        return self.grid_map, dynamic_obs_list
    


    def _radar_scan(self):
        """Radar scan returning distance and hit-mask channels (both normalized)"""
        scan_angle = 128  # 128 degrees
        scan_angle = math.radians(scan_angle)
        angles = np.linspace(-scan_angle / 2, scan_angle / 2, self.radar_num_points, dtype=np.float32) # 64, 2°/ray
        distances = np.full(self.radar_num_points, self.radar_max_range, dtype=np.float32) 
        mask = np.zeros(self.radar_num_points, dtype=np.float32)
        rel_speeds = np.zeros(self.radar_num_points, dtype=np.float32) # create relative velocity channels
        px, py = self.position
        vx = self.velocity * np.cos(self.heading) # compute x direction velocity
        vy = self.velocity * np.sin(self.heading) # compute y direction elocity
        grid_map, dynamic_obs = self._get_all_obstacles() # get all obstacles (grid_map + dynamic-obstacle list)

        for i, angle in enumerate(angles):
            ray_angle = self.heading + angle
            dx = np.cos(ray_angle) # ray x-direction
            dy = np.sin(ray_angle) # ray y-direction
            # 1. dynamic obstacle
            min_dyn_dist = self.radar_max_range
            hit_dyn = False
            for cx, cy, r in dynamic_obs:
                res = ray_dynamic_intersect(px, py, dx, dy, cx, cy, r, self.radar_max_range)
                if res is not None:
                    t_hit, (hit_x, hit_y) = res
                    if t_hit < min_dyn_dist:
                        min_dyn_dist = t_hit # update with dynamic obstacle distance
                        hit_dyn = True
                        hit_dyn_cx, hit_dyn_cy, hit_dyn_r = cx, cy, r
            # 2. static obstacle (grid map)
            static_res = ray_static_intersect(px, py, dx, dy, grid_map, self.radar_max_range, step=0.2)
            min_static_dist = self.radar_max_range
            if static_res is not None:
                min_static_dist, (hit_x, hit_y) = static_res # Get with static obstacle distance
            # 3. hit distance
            min_dist = min(min_dyn_dist, min_static_dist) # and
            distances[i] = min_dist
            mask[i] = 1.0 if min_dist < self.radar_max_range else 0.0
            # 4. relative velocity channels
            rel_speed = 0.0
            if hit_dyn and min_dyn_dist < min_static_dist and min_dyn_dist < self.radar_max_range:
                # hit dynamic obstacle, compute relative velocity
                for dyn in getattr(self, 'dynamic_obstacles', []):
                    if np.allclose([hit_dyn_cx, hit_dyn_cy], dyn['pos'], atol=1e-2):
                        vec = (dyn['end'] - dyn['start']).astype(np.float64)
                        vec = vec / (np.linalg.norm(vec) + 1e-9)
                        obs_vx = dyn['speed'] * vec[0]
                        obs_vy = dyn['speed'] * vec[1]
                        ray_dir = np.array([dx, dy])
                        ego_proj = vx * ray_dir[0] + vy * ray_dir[1]
                        obs_proj = obs_vx * ray_dir[0] + obs_vy * ray_dir[1]
                        rel_speed = obs_proj - ego_proj
                        break
            else:
                # static obstacle or no hit, rel_speed is -ego_proj
                ray_dir = np.array([dx, dy])
                ego_proj = vx * ray_dir[0] + vy * ray_dir[1]
                rel_speed = -ego_proj
            rel_speeds[i] = rel_speed
        # normalization
        distances = np.clip(distances / self.radar_max_range, 0.0, 1.0) 
        rel_speeds = np.clip(rel_speeds / 1.6, -1.0, 1.0)  
        return distances, mask, rel_speeds # Return distance, mask and relative velocity channels

    def reset(self):
        """reset the environment"""
        self.position = np.array(self.start, dtype=np.float32)  # position
        self.velocity = np.float32(22.0 / self.resolution)  # initial velocity
        dx = self.target[0] - self.start[0]
        dy = self.target[1] - self.start[1]
        self.heading = np.float32(np.arctan2(dy, dx))  # initial heading angle, goal
        self.step_count = 0
        self.trajectory = [self.position.copy()]
        self.velocity_history = [self.velocity]
        self.heading_history = [self.heading]
        if self.dynamic_type == 'static':
            # mode: generate when dynamic obstacle
            self.dynamic_obstacles = []
        elif self.dynamic_type == 'cross':
            self._init_dynamic_obstacles_cross(clear=True)
        elif self.dynamic_type == 'parallel':
            self._init_dynamic_obstacles_parallel(clear=True)
        elif self.dynamic_type == 'both':
            self.dynamic_obstacles = []
            self._init_dynamic_obstacles_cross(clear=False)
            self._init_dynamic_obstacles_parallel(clear=False)
        elif self.dynamic_type == 'random':
            choice = random.choice(['cross', 'parallel', 'both'])
            if choice == 'cross':
                self._init_dynamic_obstacles_cross(clear=True)
            elif choice == 'parallel':
                self._init_dynamic_obstacles_parallel(clear=True)
            else:
                self.dynamic_obstacles = []
                self._init_dynamic_obstacles_cross(clear=False)
                self._init_dynamic_obstacles_parallel(clear=False)

        elif self.dynamic_type == 'random_test':
            self.grid_map = self.grid_map_original.copy()  # restore first grid_map is Input state
            self._init_extra_dynamic_obstacles()
            self._init_extra_static_obstacles()  # then update extrastatic obstacle
            choice = random.choice(['cross', 'parallel', 'both'])
            if choice == 'cross':
                self._init_dynamic_obstacles_cross(clear=True)
            elif choice == 'parallel':
                self._init_dynamic_obstacles_parallel(clear=True)
            else:
                self.dynamic_obstacles = []
                self._init_dynamic_obstacles_cross(clear=False)
                self._init_dynamic_obstacles_parallel(clear=False)

        elif self.dynamic_type == 'random_test_both':
            self.grid_map = self.grid_map_original.copy()  
            self._init_extra_dynamic_obstacles()
            self._init_extra_static_obstacles() 
            self.dynamic_obstacles = []
            self._init_dynamic_obstacles_cross(clear=False)
            self._init_dynamic_obstacles_parallel(clear=False)

        elif self.dynamic_type == 'fixed_test_both':
            self.grid_map = self.grid_map_original.copy() 
            self._init_extra_dynamic_obstacles()
            self._init_extra_fixed_static_obstacles()  
            self.dynamic_obstacles = []
            self._init_dynamic_obstacles_cross(clear=False)
            self._init_dynamic_obstacles_parallel(clear=False)
        # radar distance and mask channels
        radar_seq, mask_seq, rel_speeds = self._radar_scan()
        self.radar_seq_history.clear()
        self.mask_seq_history.clear()
        self.rel_speeds_history.clear()
        for _ in range(self.radar_history_len): # 8history
            self.radar_seq_history.append(radar_seq.copy())
            self.mask_seq_history.append(mask_seq.copy())
            self.rel_speeds_history.append(rel_speeds.copy())
        return self._get_observation()
    
    def _get_observation(self):
        """Get the current observation and return a Dict structure (state_seq, radar_seq, mask_seq, casualty_map), after normalization"""
        # normalizationstate
        state_seq = np.zeros(6, dtype=np.float32)
        state_seq[0] = self.position[0] / self.map_size
        state_seq[1] = self.position[1] / self.map_size
        state_seq[2] = self.velocity / self.velocity_max  # normalized to [0, 1]
        state_seq[3] = self.heading / (2 * np.pi) # normalized to[0,1)
        state_seq[4] = self.target[0] / self.map_size
        state_seq[5] = self.target[1] / self.map_size
        # historyradar
        radar_seq = np.stack(self.radar_seq_history, axis=0)
        mask_seq = np.stack(self.mask_seq_history, axis=0)
        rel_speeds_seq = np.stack(self.rel_speeds_history, axis=0)  # relative velocity channels

        # process casualty risk map: normalized to [0, 1] (original value is 0, 1, 2)
        casualty_map_normalized = self.casualty_map.astype(np.float32) / 2.0  # 0->0, 1->0.5, 2->1.0

        # process meteorological risk map: normalized to [0, 1] (original value is 0.10~0.14)

        meteor_map_normalized = self.meteor_map.astype(np.float32) / 0.02

        obs = {
            'state_seq': state_seq,
            'radar_seq': radar_seq,
            'mask_seq': mask_seq,
            'rel_speeds_seq': rel_speeds_seq,
            'casualty_map': casualty_map_normalized,
            'meteor_map': meteor_map_normalized
        }
        return obs
    
    def step(self, action):
        """Execute one action step"""
        action = np.clip(action, self.action_space.low, self.action_space.high) 
        prev_distance = calculate_distance(
            self.position[0], self.position[1],
            self.target[0], self.target[1]
        )



        velocity_change, heading_change = action 


        self.velocity = np.clip(self.velocity + velocity_change * self.dt, self.velocity_min, self.velocity_max) # velocity  a
        self.heading += heading_change * self.dt # heading angle
        self.heading = self.heading % (2 * np.pi)
        dx = self.velocity * np.cos(self.heading) * self.dt
        dy = self.velocity * np.sin(self.heading) * self.dt
        self.position[0] += dx # update x position
        self.position[1] += dy # update y position
        # radar
        radar_seq, mask_seq, rel_speeds = self._radar_scan()
        self.radar_seq_history.append(radar_seq.copy())
        self.mask_seq_history.append(mask_seq.copy())
        self.rel_speeds_history.append(rel_speeds.copy())

        # dynamic obstacle update
        self._update_dynamic_obstacles()
        if self.dynamic_type == 'random_test':
            self._update_extra_dynamic_obstacles()
        if self.dynamic_type == 'random_test_both':
            self._update_extra_dynamic_obstacles()
        if self.dynamic_type == 'fixed_test_both':
            self._update_extra_dynamic_obstacles()
        # compute reward
        grid_map, dynamic_obs_lists = self._get_all_obstacles()
        distance_reward, dynamic_penalty, collision_penalty, boundary_penalty, success_reward, current_distance, risk_penalty, meteor_risk_penalty =\
            calculate_reward_components(
                self.position[0], self.position[1],
                self.target[0], self.target[1],
                grid_map, self.casualty_map, self.meteor_map,dynamic_obs_lists, prev_distance,
                self.velocity, self.heading, self.radar_max_range
            )
        reward = distance_reward + dynamic_penalty + collision_penalty + boundary_penalty + success_reward + risk_penalty + meteor_risk_penalty
        # check end condition
        done = False
        info = {}

        # goal reached
        if current_distance < 2.0:
            done = True
            info['success'] = True
            info['reason'] = 'success'

        # collision
        elif check_all_collisions(self.position[0], self.position[1], grid_map, dynamic_obs_lists):
            done = True
            info['collision'] = True
            info['reason'] = 'collision'

        # out of bounds
        elif (self.position[0] < 0 or self.position[0] > self.map_size or
              self.position[1] < 0 or self.position[1] > self.map_size):
            done = True
            info['out_of_bounds'] = True
            info['reason'] = 'out_of_bounds'
        
        # check timeout
        elif self.step_count >= self.max_steps:
            done = True
            info['timeout'] = True
            info['reason'] = 'timeout'
        
        self.step_count += 1
        self.trajectory.append(self.position.copy()) # iftogoal/before collision//, currentposition
        
        return self._get_observation(), reward, done, info # Return, stepreward, isend, 
    
    def render(self, mode='human', title_suffix=""):
        """Render the environment (grid map)"""
        if self.fig is None:
            self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 10))
            plt.ion()

        self.ax.clear()

    
        self.ax.set_xlim(-0.5, self.map_size - 0.5)
        self.ax.set_ylim(-0.5, self.map_size - 0.5)
        self.ax.set_aspect('equal')
        
            
        import matplotlib.colors as mcolors
        if self.casualty_map is not None:
            cmap_risk = mcolors.ListedColormap(['#97b319', '#e7c66b', '#e66d50'])
            # extentsetis[-0.5, 99.5], grid cellsto
            self.ax.imshow(self.casualty_map.T, origin='lower', cmap=cmap_risk, alpha=0.15,
                           extent=[-0.5, self.map_size - 0.5, -0.5, self.map_size - 0.5])
        # draw static obstacle grid map
        # create RGBA
        obstacle_rgba = np.zeros((self.grid_map_original.shape[0], self.grid_map_original.shape[1], 4))
        obstacle_rgba[self.grid_map_original == 1] = [110/255, 25/255, 25/255, 1.0]  # darker black-red color, close to the reference figure
        self.ax.imshow(obstacle_rgba.transpose(1, 0, 2), origin='lower',
                       extent=[-0.5, self.map_size - 0.5, -0.5, self.map_size - 0.5])
        self.ax.plot([], [], color=(110/255, 25/255, 25/255), linewidth=8, label='Static Obstacle')

        # draw obstacle mask
        if hasattr(self, 'extra_static_masks') and self.extra_static_masks:
            extra_layer = np.zeros_like(self.grid_map_original, dtype=np.uint8)
            for mask in self.extra_static_masks:
                extra_layer[mask] = 1
            self.ax.imshow(extra_layer.T, origin='lower', cmap='Greys', alpha=0.2,
                           extent=[-0.5, self.map_size - 0.5, -0.5, self.map_size - 0.5])
            self.ax.plot([], [], color=(0.8, 0.8, 0.8), linewidth=8, label='Extra Static Obstacle')


        # draw goal
        self.ax.plot(self.target[0], self.target[1], 'g*', markersize=15, label='Target')
        # draw start
        self.ax.plot(self.start[0], self.start[1], 'go', markersize=8, label='Start')
        # draw routes of dynamic obstacles
        if self.dynamic_type != 'random_few':
            drawn_lines = set()
            route_label_drawn = False
            for obs in self.dynamic_obstacles:
                line_key = tuple(obs['start'].tolist() + obs['end'].tolist())
                if line_key not in drawn_lines:
                    label = 'Dynamic Route' if not route_label_drawn else None
                    self.ax.plot([obs['start'][0], obs['end'][0]], [obs['start'][1], obs['end'][1]],
                                 color='purple', linestyle=':', linewidth=1.5, alpha=0.5, label=label)
                    drawn_lines.add(line_key)
                    if not route_label_drawn:
                        route_label_drawn = True
                else:
                    self.ax.plot([obs['start'][0], obs['end'][0]], [obs['start'][1], obs['end'][1]],
                                 color='purple', linestyle=':', linewidth=1.5, alpha=0.5)
        # draw dynamic obstacles
        if self.dynamic_type != 'random_few':
            obstacle_label_drawn = False
            for obs in self.dynamic_obstacles:
                label = 'Dynamic Obstacle' if not obstacle_label_drawn else None
                circle_outer = patches.Circle((obs['pos'][0], obs['pos'][1]), obs['radius']+0.5,
                                             facecolor='purple', alpha=0.1, edgecolor='black', linestyle='--', label=label)
                self.ax.add_patch(circle_outer)
                self.ax.plot(obs['pos'][0], obs['pos'][1], 'o', color='purple', markersize=8)
                if not obstacle_label_drawn:
                    obstacle_label_drawn = True

        # draw route-independent dynamic obstacles (random_few_obstacles)
        if hasattr(self, 'extra_dynamic_obstacles') and self.extra_dynamic_obstacles:
            rf_label_drawn = False
            for obs in self.extra_dynamic_obstacles:
                label = 'Random Dynamic Obstacle' if not rf_label_drawn else None
                # use, label
                circle_rf = patches.Circle((obs['pos'][0], obs['pos'][1]), obs['radius']+0.5, facecolor="red", alpha=0.2, edgecolor='black', linestyle='--', label=label)
                self.ax.add_patch(circle_rf)
                # circle centeruse, label
                self.ax.plot(obs['pos'][0], obs['pos'][1], 'o', color="red", markersize=8)
                if not rf_label_drawn:
                    rf_label_drawn = True

        # draw trajectory      
        if len(self.trajectory) > 1:
            trajectory = np.array(self.trajectory)
            self.ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=0.6, linewidth=2, label='Trajectory') 

        # draw current position
        self.ax.plot(self.position[0], self.position[1], 'bo', markersize=8, label='Current Position') 

        # draw velocity direction
        arrow_length = 3
        dx = arrow_length * np.cos(self.heading)
        dy = arrow_length * np.sin(self.heading)
        self.ax.arrow(self.position[0], self.position[1], dx, dy, 
                     head_width=1, head_length=0.5, fc='blue', ec='blue') # degreesdirectionis
        

        # radar visualization (ray+hit)
        radar_seq, mask_seq, _ = self._radar_scan()
        scan_angle = 128
        scan_angle = math.radians(scan_angle)
        angles = np.linspace(-scan_angle / 2, scan_angle / 2, self.radar_num_points, dtype=np.float32)
        px, py = self.position
        for i, (dist_norm, mask) in enumerate(zip(radar_seq, mask_seq)):
            dist = dist_norm * self.radar_max_range
            ray_angle = self.heading + angles[i]
            end_x = px + dist * np.cos(ray_angle)
            end_y = py + dist * np.sin(ray_angle)
            color = 'orange' if mask > 0.5 else 'gray'
            linewidth = 1.5 if mask > 0.5 else 0.8
            alpha = 0.7 if mask > 0.5 else 0.3
            self.ax.plot([px, end_x], [py, end_y], color=color, alpha=alpha, linewidth=linewidth)
            if mask > 0.5:
                self.ax.plot(end_x, end_y, 'ro', markersize=3, alpha=0.8)

        # compute distance to target
        distance_to_target = calculate_distance(
            self.position[0], self.position[1],
            self.target[0], self.target[1]
        )
        distance_to_target = distance_to_target * self.resolution  # actual distance m
        velocity = self.velocity * self.resolution  # actual velocity m/s
        title = f'Path Planning{title_suffix} - Step: {self.step_count}, Speed: {velocity:.1f}, Distance to Target: {distance_to_target:.1f}'
        # actual current steps, velocity and to goal distance
        self.ax.set_title(title)

    
        for i in range(self.map_size + 1):
            self.ax.axhline(i - 0.5, color='gray', linewidth=0.05)
            self.ax.axvline(i - 0.5, color='gray', linewidth=0.05)

        plt.pause(0.01)

        if mode == 'rgb_array':
            import io
            from PIL import Image
            
            buf = io.BytesIO()
            self.fig.savefig(buf, format='png', dpi=600, bbox_inches='tight')
            buf.seek(0)
            
            img = Image.open(buf)
            img_array = np.array(img)
            
            # isRGBformat
            if img_array.shape[2] == 4:  # RGBA
                img_array = img_array[:, :, :3]
            
            buf.close()
            return img_array



    def close(self):
        """Close the environment"""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


def load_obstacle_matrix(file_path):
    """Load obstacle_grid_101x101.txt and rotate the grid map 90 degrees clockwise"""
    grid_map = np.loadtxt(file_path, dtype=np.float32)
    grid_map = np.rot90(grid_map, -1)  # clockwiserotation90degrees
    return grid_map

def load_casualty_map(file_path):
    """Load the risk matrix and rotate it 90 degrees clockwise"""
    matrix = np.loadtxt(file_path, dtype=np.float32).astype(np.int32)
    matrix = np.rot90(matrix, -1)
    return matrix

def load_meteor_map(file_path):
    """Load the meteorological-risk matrix and rotate it 90 degrees clockwise"""
    matrix = np.loadtxt(file_path, dtype=np.float32)
    matrix = np.rot90(matrix, -1)
    return matrix

if __name__ == "__main__":
    
    # loadobstaclemap
    grid_map = load_obstacle_matrix("obstacle_grid_101x101.txt")
    # load
    casualty_map = load_casualty_map("casualty_matrix_101x101.txt")
    meteor_map = load_meteor_map("meteor_risk_101x101.txt")

    # test
    env = SimplePathPlanningEnv(end=[57,97], grid_map=grid_map, casualty_map=casualty_map, meteor_map=meteor_map, dynamic_type="fixed_test_both",seed=42)

    # episode
    for episode in range(1):
        obs = env.reset()
        total_reward = 0
        
        print(f"\nepisode {episode + 1}:")
        
        for step in range(500):
            # 
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            total_reward += reward
            
            # visualization
            env.render()
            
            if done:
                print(f"  end reason: {info.get('reason', 'unknown')}")
                print(f"  total reward: {total_reward:.2f}")
                print(f"  steps: {step + 1}")
                break
        
        input("Press Enter to continue to the next episode...")
    
    env.close()
    print("Environment test completed!")

#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Drone Delivery feature preprocessor.
智运无人机特征预处理器。
"""


import numpy as np
from agent_ppo.conf.conf import Config
from agent_ppo.feature.dbfs import DBFS

def norm(v, max_v, min_v=0):
    """Normalize v to [0, 1].

    将 v 归一化到 [0, 1]。
    """
    v = np.clip(v, min_v, max_v)
    return (v - min_v) / (max_v - min_v)


def _get_pos_feature(found, cur_pos, target_pos, is_target=False):
    """Compute 7D position feature for a target relative to current position.

    计算目标位置相对于当前位置的 7 维特征。
    """
    relative_pos = (target_pos[0] - cur_pos[0], target_pos[1] - cur_pos[1])
    dist = np.sqrt(relative_pos[0] ** 2 + relative_pos[1] ** 2)
    abs_norm = norm(np.array(target_pos), 128, -128)
    return np.array(
        [
            float(found),
            norm(relative_pos[0] / max(dist, 1e-4), 1, -1),
            norm(relative_pos[1] / max(dist, 1e-4), 1, -1),
            abs_norm[0],
            abs_norm[1],
            norm(dist, 1.41 * 128),
            1.0 if is_target else 0.0,
        ]
    )

def get_pos_feat_2(found, cur_pos, target_pos):
    relative_pos = (target_pos[0] - cur_pos[0], target_pos[1] - cur_pos[1])
    dist = np.sqrt(relative_pos[0] ** 2 + relative_pos[1] ** 2)
    abs_norm = norm(np.array(target_pos), 128, -128)
    return np.array(
        [
            float(found),
            norm(relative_pos[0] / max(dist, 1e-4), 1, -1),
            norm(relative_pos[1] / max(dist, 1e-4), 1, -1),
            abs_norm[0],
            abs_norm[1],
            norm(dist, 1.41 * 128),
        ]
    )

def chu(ho,lg):
    # print(ho)
    # print(lg)
    pass


class Preprocessor:
    """feature preprocessor for Drone Delivery.

    智运无人机预处理器，仅保留最少信息。
    """

    def __init__(self, logger=None):
        self.reset()
        self.logger = logger
        # 预计算降维权重（池化 + 展平）
        self._init_grid_compressor()
    def _init_grid_compressor(self):
        """初始化一个固定的降维器：平均池化 + 展平"""
        # 21x21 -> 用 3x3 池化，步长 2 -> 10x10（实际是 11x11？需要计算）
        # 更简单：直接下采样到 7x7（每 3x3 取平均）
        self.pool_size = 3
        self.stride = 3
        # 21 // 3 = 7，输出 7x7 = 49 维
        self.out_size = 7
    
    def compress_grid(self, grid):
        """将 21x21 栅格压缩到 49 维（7x7）"""
        if grid is None:
            return np.zeros(self.out_size * self.out_size)

        grid = np.asarray(grid)
        h, w = grid.shape
        ph, pw = self.out_size, self.out_size
        # 简单的平均池化
        pooled = np.zeros((ph, pw))
        for i in range(ph):
            for j in range(pw):
                # 计算对应原始区域
                y_start = i * self.stride
                y_end = min(y_start + self.pool_size, h)
                x_start = j * self.stride
                x_end = min(x_start + self.pool_size, w)
                pooled[i, j] = np.mean(grid[y_start:y_end, x_start:x_end])
        return pooled.flatten()  # 49 维
    def reset(self):
        """Reset all internal state.

        重置所有状态。
        """
        self.cur_pos = (0, 0)
        self.prev_pos = None
        self.prev_prev_pos = None

        # Game state / 游戏状态
        self.battery = 100
        self.prev_battery = 100
        self.battery_max = 100
        self.packages = []
        self.prev_package = 3
        self.delivered = 0
        self.last_delivered = 0
        self.step_no = 0
        self.map_info = None
        self.prev_map_info = None
        self.last_reward_log_step = -1
        self.cur_npc_dist = None

        # Entities / 实体
        self.stations = []
        self.chargers = []
        self.warehouses = []
        self.visited_positions = set()
        self.npcs = []

    def _parse_obs(self, env_obs):
        """Parse essential fields from observation dict.

        从 observation 字典中解析必要字段。
        """
        obs = env_obs["observation"]
        frame_state = obs["frame_state"]

        hero = frame_state["heroes"]
        
        #官方无人机
        self.npcs = frame_state["npcs"]


        self.cur_pos = (hero["pos"]["x"], hero["pos"]["z"])
        self.score = hero.get("score",0)

        self.battery = hero.get("battery", self.battery_max)
        self.battery_max = hero.get("battery_max", 100)
        self.packages = hero.get("packages", [])
        

        self.last_delivered = self.delivered
        self.delivered = hero.get("delivered", 0)
        self.step_no = obs.get("step_no", 0)
        self.map_info = obs.get("map_info")
        for i in frame_state["organs"]:
            sub_type = i["sub_type"]
            if(sub_type==1):
                self.warehouses.append(i)
            elif(sub_type==2):
                self.chargers.append(i)
            elif(sub_type==3):
                self.stations.append(i)


        self.legal_act = obs.get("legal_action", [1] * 8)

    def feature_process(self, env_obs, last_action):
        """Core feature extraction. Returns (feature_22d, legal_action, reward).

        核心特征提取方法，返回 22 维特征向量、合法动作掩码和奖励。
        """
        self._parse_obs(env_obs)

        # 1. Hero state features (4D) / 英雄状态特征（4D）
        battery_ratio = norm(self.battery, self.battery_max)
        package_count_norm = norm(len(self.packages), 3)
        cur_pos_norm = norm(np.array(self.cur_pos, dtype=float), 128, -128)
        hero_feat = np.array(
            [
                battery_ratio,
                package_count_norm,
                cur_pos_norm[0],
                cur_pos_norm[1],
            ]
        )

        # 2. Nearest 1 station feature (7D) / 最近 1 个驿站特征（7D）
        # Target stations first, then by distance
        # 目标驿站优先，然后按距离排序
        target_ids = set(self.packages)
        target_stations = [s for s in self.stations if s.get("config_id", 0) in target_ids]
        
        

        # 找到最近的充电桩
        if len(self.chargers) > 0:
            nearest_charger = min(
                self.chargers,
                key=lambda c: np.sqrt((c["pos"]["x"] - self.cur_pos[0]) ** 2 + (c["pos"]["z"] - self.cur_pos[1]) ** 2),
            )
            charger_station = get_pos_feat_2(
                True,
                self.cur_pos,
                (nearest_charger["pos"]["x"], nearest_charger["pos"]["z"]),
            )
            self.charger_pos = (nearest_charger["pos"]["x"], nearest_charger["pos"]["z"])
        else:
            charger_station = get_pos_feat_2(False, self.cur_pos, self.cur_pos)

        # 找到最近的仓库
        if len(self.warehouses) > 0:
            nearest_warehouse = min(
                self.warehouses,
                key=lambda w: np.sqrt((w["pos"]["x"] - self.cur_pos[0]) ** 2 + (w["pos"]["z"] - self.cur_pos[1]) ** 2),
            )
            warehouse_station = get_pos_feat_2(
                True,
                self.cur_pos,
                (nearest_warehouse["pos"]["x"], nearest_warehouse["pos"]["z"]),
            )
            self.warehouse_pos = (nearest_warehouse["pos"]["x"], nearest_warehouse["pos"]["z"])
        else:
            self.warehouse_pos = None
            warehouse_station = get_pos_feat_2(False, self.cur_pos, self.cur_pos)

        # 找到最近的目标驿站
        chu("最近的目标",target_stations)
        if len(target_stations) > 0:
            #找到最近的目标驿站
            self.near_target = min(target_stations, key=lambda s: np.sqrt((s["pos"]["x"] - self.cur_pos[0]) ** 2 + (s["pos"]["z"] - self.cur_pos[1]) ** 2))
            self.target_pos = (self.near_target["pos"]["x"], self.near_target["pos"]["z"])
        else:
            self.target_pos = None

        # 找到最近的npc
        if len(self.npcs) > 0:
            nearest_npc = min(
                self.npcs,
                key=lambda n: np.sqrt((n["pos"]["x"] - self.cur_pos[0]) ** 2 + (n["pos"]["z"] - self.cur_pos[1]) ** 2),
            )
            npc_station = get_pos_feat_2(
                True,
                self.cur_pos,
                (nearest_npc["pos"]["x"], nearest_npc["pos"]["z"]),
            )
            self.cur_npc_dist = np.sqrt((nearest_npc["pos"]["x"] - self.cur_pos[0]) ** 2 + (nearest_npc["pos"]["z"] - self.cur_pos[1]) ** 2)
        else:
            self.cur_npc_dist = None
            npc_station = get_pos_feat_2(False, self.cur_pos, self.cur_pos)


        # 定义驿站排序键
        def station_sort_key(s):
            is_tgt = s.get("config_id", 0) in target_ids
            dist = np.sqrt((s["pos"]["x"] - self.cur_pos[0]) ** 2 + (s["pos"]["z"] - self.cur_pos[1]) ** 2)
            return (0 if is_tgt else 1, dist)

        # 对驿站进行排序
        sorted_stations = sorted(self.stations, key=station_sort_key)

        if len(sorted_stations) > 0:
            s = sorted_stations[0]
            is_target = s.get("config_id", 0) in target_ids
            station_feat = _get_pos_feature(
                True,
                self.cur_pos,
                (s["pos"]["x"], s["pos"]["z"]),
                is_target=is_target,
            )
            target_visible = float(is_target)
        else:
            station_feat = _get_pos_feature(False, self.cur_pos, self.cur_pos, is_target=False)
            target_visible = 0.0

        # 3. Legal action mask (8D) / 合法动作掩码（8D）
        legal_action = self._get_legal_action()

        # 4. Binary indicators (3D) / 二值指示器（3D）
        has_package = 1.0 if len(self.packages) > 0 else 0.0
        battery_low = 1.0 if (self.battery / max(self.battery_max, 1)) < 0.3 else 0.0
        indicators = np.array([has_package, battery_low, target_visible])

        # 5. 视野地图特征 (压缩到 49D)
        # if self.map_info is not None:
        #     grid_feat = self.compress_grid(self.map_info)  # (49,)
        # else:
        #     grid_feat = np.zeros(49)
        # 视野地图以自己为中心半径3格
        if self.map_info is not None:
            map_arr = np.asarray(self.map_info, dtype=float)
            if map_arr.ndim == 2:
                h, w = map_arr.shape
                cx, cy = h // 2, w // 2
                x0, x1 = max(0, cx - 3), min(h, cx + 4)
                y0, y1 = max(0, cy - 3), min(w, cy + 4)

                local_view = np.full((7, 7), -1.0, dtype=float)
                lx0 = 3 - (cx - x0)
                ly0 = 3 - (cy - y0)
                local_view[lx0 : lx0 + (x1 - x0), ly0 : ly0 + (y1 - y0)] = map_arr[x0:x1, y0:y1]
                grid_feat = local_view.flatten()
            else:
                grid_feat = np.full(49, -1.0)
        else:
            grid_feat = np.full(49, -1.0)

        # 周围的墙壁
        wall_features = self._get_wall_features()

        # Concatenate features (Total 22D / 合计 22D)
        feature = np.concatenate(
            [
                hero_feat,
                station_feat,
                np.array(legal_action, dtype=float),
                indicators,
                charger_station,
                warehouse_station,
                npc_station,
                grid_feat,
                wall_features,
            ]
        )
        chu("方向",legal_action)
        chu("特征",wall_features)
        chu("地图",self.map_info)
        reward = self._reward_process()

        return feature, legal_action, reward
    def _get_wall_features(self):
        """返回长度为 8 的列表，表示 8 个动作方向正前方一格是否有墙（1=墙，0=可通行）。"""
        if self.map_info is None:
            return [0.0] * 8

        # map_info 是 21x21 的局部栅格，智能体位于中心 (10, 10)
        cx, cy = 10, 10
        # 方向顺序必须与实际动作空间严格一致：左上(0) -> 上(1) -> 右上(2) -> 右(3) -> 右下(4) -> 下(5) -> 左下(6) -> 左(7)
        dirs = [
            (-1, 1),  # 0: 左上
            (0, 1),   # 1: 上
            (1, 1),   # 2: 右上
            (1, 0),   # 3: 右
            (1, -1),  # 4: 右下
            (0, -1),  # 5: 下
            (-1, -1), # 6: 左下
            (-1, 0),  # 7: 左
        ]
        wall_flags = []
        for dx, dy in dirs:
            nx, ny = cx + dx, cy + dy
            # 检查是否在 21x21 范围内，且 map_info 中 0 表示障碍物（墙），1 表示可通行
            if 0 <= nx < 21 and 0 <= ny < 21:
                wall_flags.append(0.0 if self.map_info[nx][ny] else 1.0)
            else:
                wall_flags.append(1.0)  # 边界外视为墙
        return wall_flags
    
    def _get_legal_action(self):
        """Get legal action mask.

        获取合法动作掩码。
        """
        if hasattr(self, "legal_act") and self.legal_act:
            legal_action = [int(x) for x in self.legal_act[:8]]
        else:
            legal_action = [1] * 8

        if sum(legal_action) == 0:
            return [1] * 8

        return legal_action
    def reward_log(self,name,num):
        list = {
            "name":name,
            "val":num
        }
        # self.logger.info(f"[reward]:{list}")


    def _reward_process(self):
        """Reward function.

        奖励函数。
        """
        reward = 0.0

        # 1. Delivery reward / 投递奖励
        newly_delivered = max(0, self.delivered - self.last_delivered)
        if newly_delivered > 0:
            num = 3 * newly_delivered
            reward += num
            self.reward_log("Delivery",num)

        # 2. Step penalty / 步数惩罚
        reward -= 0.001
        self.reward_log("Step",-0.001)

        self.battery_low = self.battery < self.battery_max * 0.3


        # 3. 目标距离塑形奖励
        if not self.battery_low and len(self.packages) and self.target_pos is not None and self.prev_map_info is not None and self.map_info is not None:
            dbf = DBFS(self.prev_map_info, self.map_info, self.cur_pos, self.prev_pos, self.target_pos)
            progress = dbf.main()
            num = 0.03 * progress
            reward += num
            self.reward_log("目标靠近奖励",num)
            chu("目标奖励",num)

        # 4. 低电量惩罚
        if self.battery_low:  # 电量低于30%
            #电桩
            if self.packages and self.prev_map_info is not None and self.map_info is not None:
                dbf = DBFS(self.prev_map_info, self.map_info, self.cur_pos, self.prev_pos, self.charger_pos)
                progress = dbf.main()
                num = 0.03 * progress
                reward += num
                self.reward_log("ChargerDeparture", num)
                chu("充电桩靠近奖励",num)
            reward -= 0.05
            self.reward_log("低电量惩罚", -0.05)

        # 5. 补货前往仓库奖励
        if not len(self.packages) and self.warehouse_pos is not None and self.prev_map_info is not None:
            dbf = DBFS(self.prev_map_info, self.map_info, self.cur_pos, self.prev_pos, self.warehouse_pos)
            progress = dbf.main()
            num = 0.03*progress
            reward += num
            self.reward_log("仓库靠近奖励", num)

        # 7. 补货奖励
        cha = len(self.packages) - self.prev_package
        if cha > 0:
            num = 1.5 * cha
            reward += num
            self.reward_log("WarehouseReward", num)
            chu("补货奖励",num)
        chu("补货_cur",len(self.packages))
        chu("补货_prev",self.prev_package)

        # 8. 重复惩罚
        pos_key = (int(self.cur_pos[0]), int(self.cur_pos[1]))
        # if self.prev_prev_pos is not None and self.prev_pos is not None:
        #     if pos_key == self.prev_prev_pos:
        #         num = -0.035
        #         reward += num
        #         self.reward_log("RoundTrip", num)
        #         chu("重复惩罚",num)
        # chu("重复_cur",pos_key)
        # chu("重复_prev",self.prev_pos)
        # chu("重复_prev_prev",self.prev_prev_pos)

        #转向惩罚


        # 8.1首次访问奖励，随探索进度衰减
        if pos_key not in self.visited_positions:
            explore_bonus = 0.008 * (0.992 ** len(self.visited_positions))
            reward += explore_bonus
            self.reward_log("FirstVisit", explore_bonus)
            self.visited_positions.add(pos_key)
            chu("首次访问奖励", explore_bonus)

        # 9.靠近官方机器人扣分
        if self.cur_npc_dist is not None and self.cur_npc_dist <= 3.0:
            num = -0.5 * (4-self.cur_npc_dist)
            reward += num
            self.reward_log("NPCTooClose", -0.5)
            chu("官方机器人奖励", num)
        chu("官方机器人距离", self.cur_npc_dist)

        self.prev_map_info = self.map_info
        self.prev_prev_pos = self.prev_pos
        self.prev_pos = self.cur_pos
        self.prev_battery = self.battery
        self.prev_package = len(self.packages)


        return [reward]
    

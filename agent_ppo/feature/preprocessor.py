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


class Preprocessor:
    """feature preprocessor for Drone Delivery.

    智运无人机预处理器，仅保留最少信息。
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all internal state.

        重置所有状态。
        """
        self.cur_pos = (0, 0)

        # Game state / 游戏状态
        self.battery = 100
        self.battery_max = 100
        self.packages = []
        self.delivered = 0
        self.last_delivered = 0
        self.step_no = 0
        self.cur_target_dist = None
        self.prev_target_dist = None
        self.cur_charger_dist = None
        self.prev_charger_dist = None
        self.cur_warehouse_dist = None
        self.prev_warehouse_dist = None

        # Entities / 实体
        self.stations = []
        self.chargers = []
        self.warehouses = []

    def _parse_obs(self, env_obs):
        """Parse essential fields from observation dict.

        从 observation 字典中解析必要字段。
        """
        obs = env_obs["observation"]
        frame_state = obs["frame_state"]

        hero = frame_state["heroes"]
        self.cur_pos = (hero["pos"]["x"], hero["pos"]["z"])

        self.battery = hero.get("battery", self.battery_max)
        self.battery_max = hero.get("battery_max", 100)
        self.packages = hero.get("packages", [])

        self.last_delivered = self.delivered
        self.delivered = hero.get("delivered", 0)
        self.step_no = obs.get("step_no", 0)

        self.stations = []
        self.chargers = []
        self.warehouses = []
        for organ in frame_state.get("organs", []):
            st = organ.get("sub_type", 0)
            if st == 3:
                self.stations.append(organ)

            # Charger compatibility: prefer explicit naming, fallback to common subtype.
            # 充电桩兼容识别：优先按名称字段识别，兜底按常见 sub_type。
            organ_desc = (
                f"{organ.get('name', '')} {organ.get('type', '')} {organ.get('config_name', '')}"
            ).lower()
            is_charger = ("charger" in organ_desc) or ("charge" in organ_desc) or ("充电" in organ_desc) or (st == 4)
            if is_charger:
                self.chargers.append(organ)

            # Warehouse compatibility: prefer explicit naming, fallback to a common subtype.
            # 仓库兼容识别：优先按名称字段识别，兜底按常见 sub_type。
            is_warehouse = (
                ("warehouse" in organ_desc)
                or ("depot" in organ_desc)
                or ("store" in organ_desc)
                or ("仓库" in organ_desc)
                or ("补货" in organ_desc)
                or (st == 2)
            )
            if is_warehouse:
                self.warehouses.append(organ)

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

        if len(self.chargers) > 0:
            self.cur_charger_dist = min(
                np.sqrt((c["pos"]["x"] - self.cur_pos[0]) ** 2 + (c["pos"]["z"] - self.cur_pos[1]) ** 2)
                for c in self.chargers
            )
        else:
            self.cur_charger_dist = None

        if len(self.warehouses) > 0:
            self.cur_warehouse_dist = min(
                np.sqrt((w["pos"]["x"] - self.cur_pos[0]) ** 2 + (w["pos"]["z"] - self.cur_pos[1]) ** 2)
                for w in self.warehouses
            )
        else:
            self.cur_warehouse_dist = None

        if len(target_stations) > 0:
            self.cur_target_dist = min(
                np.sqrt((s["pos"]["x"] - self.cur_pos[0]) ** 2 + (s["pos"]["z"] - self.cur_pos[1]) ** 2)
                for s in target_stations
            )
        else:
            self.cur_target_dist = None

        def station_sort_key(s):
            is_tgt = s.get("config_id", 0) in target_ids
            dist = np.sqrt((s["pos"]["x"] - self.cur_pos[0]) ** 2 + (s["pos"]["z"] - self.cur_pos[1]) ** 2)
            return (0 if is_tgt else 1, dist)

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

        # Concatenate features (Total 22D / 合计 22D)
        feature = np.concatenate(
            [
                hero_feat,
                station_feat,
                np.array(legal_action, dtype=float),
                indicators,
            ]
        )

        reward = self._reward_process()

        return feature, legal_action, reward

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

    def _reward_process(self):
        """Reward function.

        奖励函数。
        """
        reward = 0.0

        # 1. Delivery reward / 投递奖励
        newly_delivered = max(0, self.delivered - self.last_delivered)
        if newly_delivered > 0:
            reward += 1.0 * newly_delivered

        # 2. Step penalty / 步数惩罚
        reward -= 0.001

        # 3. Distance shaping reward / 目标距离塑形奖励
        # 规则（目标驿站距离塑形）：定义 progress = prev_target_dist - cur_target_dist。
        # - 若 progress > 0（靠近目标驿站）：reward += 0.02 * min(progress, 5.0)
        # - 若 progress < 0（远离目标驿站）：reward += 0.01 * max(progress, -5.0)
        # 说明：靠近奖励系数 0.02 大于远离惩罚系数 0.01。
        #
        # 近距离额外奖励：
        # - 若 cur_target_dist < 15：reward += 0.02
        # - 若 cur_target_dist < 8： reward += 0.03（可与上一条叠加）
        if self.cur_target_dist is not None:
            if self.prev_target_dist is not None:
                progress = self.prev_target_dist - self.cur_target_dist
                if progress > 0:
                    reward += 0.02 * min(progress, 5.0)
                elif progress < 0:
                    reward += 0.01 * max(progress, -5.0)

            # Extra bonus when already close to target station.
            if self.cur_target_dist < 15:
                reward += 0.02
            if self.cur_target_dist < 8:
                reward += 0.03

        # 4. Charger shaping & arrival reward / 充电桩塑形与到达奖励
        # Rule A (low battery approach shaping):
        # 当 battery < 20 时，定义 charger_progress = prev_charger_dist - cur_charger_dist。
        # - 若 charger_progress > 0（靠近）：reward += 0.03 * min(charger_progress, 5.0)
        # - 若 charger_progress < 0（远离）：reward += 0.01 * max(charger_progress, -5.0)
        # 说明：靠近奖励系数 0.03 大于远离惩罚系数 0.01。
        #
        # Rule B (arrival reward by battery threshold):
        # 到达判定：cur_charger_dist < 3.0 且上一帧不在该半径内。
        # - 若 battery > 20：reward -= 0.2
        # - 若 battery < 20：reward += 0.2
        # - 若 battery == 20：reward += 0（不奖不惩）
        #
        # Rule C (battery-based deduction below 50):
        # 当 battery < 50 时，额外扣分：
        # penalty = 0.001 + 0.019 * clip((50 - battery) / 50, 0, 1)
        # 即电量越低扣分越高，范围约为 [0.001, 0.02]。
        battery_now = float(self.battery)
        low_battery = battery_now < 20.0

        if self.cur_charger_dist is not None:
            if battery_now < 50.0:
                battery_penalty_ratio = np.clip((50.0 - battery_now) / 50.0, 0.0, 1.0)
                battery_penalty = 0.001 + 0.019 * battery_penalty_ratio
                reward -= battery_penalty

            if low_battery and self.prev_charger_dist is not None:
                charger_progress = self.prev_charger_dist - self.cur_charger_dist
                if charger_progress > 0:
                    reward += 0.03 * min(charger_progress, 5.0)
                elif charger_progress < 0:
                    reward += 0.01 * max(charger_progress, -5.0)

            # Consider entering a small radius as "arrived".
            arrived = self.cur_charger_dist < 3.0 and (
                self.prev_charger_dist is None or self.prev_charger_dist >= 3.0
            )
            if arrived:
                if battery_now > 20.0:
                    reward -= 0.2
                elif battery_now < 20.0:
                    reward += 0.2

        # 5. Replenishment reward to warehouse / 补货前往仓库奖励
        # 仅在没有包裹时生效，鼓励前往仓库进行补货。
        # 定义 restock_progress = prev_warehouse_dist - cur_warehouse_dist。
        # - 若 restock_progress > 0（靠近仓库）：reward += 0.02 * min(restock_progress, 5.0)
        # - 若 restock_progress < 0（远离仓库）：reward += 0.008 * max(restock_progress, -5.0)
        # 到达判定：cur_warehouse_dist < 3.0 且上一帧不在该半径内，额外 reward += 0.1。
        if len(self.packages) == 0 and self.cur_warehouse_dist is not None:
            if self.prev_warehouse_dist is not None:
                restock_progress = self.prev_warehouse_dist - self.cur_warehouse_dist
                if restock_progress > 0:
                    reward += 0.02 * min(restock_progress, 5.0)
                elif restock_progress < 0:
                    reward += 0.008 * max(restock_progress, -5.0)

            arrived_warehouse = self.cur_warehouse_dist < 3.0 and (
                self.prev_warehouse_dist is None or self.prev_warehouse_dist >= 3.0
            )
            if arrived_warehouse:
                reward += 0.1

        self.prev_target_dist = self.cur_target_dist
        self.prev_charger_dist = self.cur_charger_dist
        self.prev_warehouse_dist = self.cur_warehouse_dist

        return [reward]

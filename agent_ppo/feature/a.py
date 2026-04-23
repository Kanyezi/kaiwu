#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""A* path planning demo with hardcoded 21x21 test map."""

import heapq
import math


class PathPlanner:
	"""Path planner for selecting the fastest direction among 8 moves.

	输入局部 21x21 地图 + AI 全局坐标 + 目标全局坐标，输出 8 方向代价与收益。
	"""

	# 方向顺序（x=行向下增，y=列向右增）：右上, 右, 右下, 下, 左下, 左, 左上, 上
	DIRS_8 = [
		(-1, 1),
		(0, 1),
		(1, 1),
		(1, 0),
		(1, -1),
		(0, -1),
		(-1, -1),
		(-1, 0),
	]

	def process_grid(self, ai_world_pos, target_world_pos, grid_21x21, use_gain=True):
		"""简化主函数：只返回八个方位的值。

		Args:
			ai_world_pos: AI 在全局地图坐标，如 (x, y)。
			target_world_pos: 目标在全局地图坐标，如 (x, y)。
			grid_21x21: 21x21 二维地图，0=障碍，非0=可通行。
			use_gain: True 返回收益值(越大越优)，False 返回代价值(越小越优)。

		Returns:
			长度为8的方位值列表，顺序 RU,R,RD,D,LD,L,LU,U。
		"""
		self._validate_grid(grid_21x21)

		center = (10, 10)
		target_local = self._world_to_local(ai_world_pos, target_world_pos)
		target_in_view = self._in_bounds(target_local[0], target_local[1]) if target_local is not None else False

		dir_cost_values = []
		blocked_cost = 1e6
		for dx, dy in self.DIRS_8:
			nx, ny = center[0] + dx, center[1] + dy

			if not self._is_walkable(grid_21x21, nx, ny):
				dir_cost_values.append(blocked_cost)
				continue

			step_cost = math.sqrt(2.0) if dx != 0 and dy != 0 else 1.0
			next_world = (ai_world_pos[0] + dx, ai_world_pos[1] + dy)

			# 目标在视野内：优先用局部 A* 估计真实绕路代价
			if target_in_view and self._is_walkable(grid_21x21, target_local[0], target_local[1]):
				path_cost = self._astar_cost(grid_21x21, (nx, ny), target_local)
				if path_cost is not None:
					dir_cost_values.append(step_cost + path_cost)
					continue

			# 目标在视野外，或视野内但局部不可达：退化到全局启发式
			heuristic_cost = self._euclidean(next_world, target_world_pos)
			dir_cost_values.append(step_cost + heuristic_cost)

		dir_cost_values = self._stabilize_dir_values(dir_cost_values)
		dir_gain_values = self._cost_to_gain(dir_cost_values)
		return dir_gain_values if use_gain else dir_cost_values

	def _cost_to_gain(self, dir_cost_values):
		"""Map cost to gain so that larger is better: gain = 1 / (1 + cost)."""
		gains = []
		for c in dir_cost_values:
			c = max(0.0, float(c))
			gains.append(1.0 / (1.0 + c))
		return gains

	def _stabilize_dir_values(self, dir_values):
		"""Convert invalid/huge values to stable finite numbers."""
		finite = [v for v in dir_values if v is not None and math.isfinite(v)]
		if not finite:
			return [1000.0 for _ in dir_values]

		max_finite = max(finite)
		filled = [v if (v is not None and math.isfinite(v)) else (max_finite * 1.2 + 1.0) for v in dir_values]
		clip_hi = max(200.0, max_finite * 1.5)
		return [min(v, clip_hi) for v in filled]

	def _astar_cost(self, grid, start, goal):
		"""A* shortest cost in local 21x21, return None if unreachable."""
		_, cost = self._astar_path_and_cost(grid, start, goal)
		return cost

	def _astar_path_and_cost(self, grid, start, goal):
		"""A* shortest path and cost in local 21x21.

		Returns:
			(path, cost), path 为节点序列；不可达时返回 ([], None)。
		"""
		if not self._is_walkable(grid, start[0], start[1]):
			return [], None
		if not self._is_walkable(grid, goal[0], goal[1]):
			return [], None

		open_heap = []
		g_score = {start: 0.0}
		came_from = {}
		heapq.heappush(open_heap, (self._euclidean(start, goal), 0.0, start))
		closed = set()

		while open_heap:
			_, cur_g, cur = heapq.heappop(open_heap)
			if cur in closed:
				continue
			if cur == goal:
				return self._reconstruct_path(came_from, cur), cur_g
			closed.add(cur)

			for dx, dy in self.DIRS_8:
				nx, ny = cur[0] + dx, cur[1] + dy
				if not self._is_walkable(grid, nx, ny):
					continue

				# 斜向切角限制
				if dx != 0 and dy != 0:
					if (not self._is_walkable(grid, cur[0] + dx, cur[1])) or (not self._is_walkable(grid, cur[0], cur[1] + dy)):
						continue

				step_cost = math.sqrt(2.0) if dx != 0 and dy != 0 else 1.0
				nxt = (nx, ny)
				ng = cur_g + step_cost
				if ng < g_score.get(nxt, 1e18):
					g_score[nxt] = ng
					came_from[nxt] = cur
					f = ng + self._euclidean(nxt, goal)
					heapq.heappush(open_heap, (f, ng, nxt))

		return [], None

	def _reconstruct_path(self, came_from, cur):
		path = [cur]
		while cur in came_from:
			cur = came_from[cur]
			path.append(cur)
		path.reverse()
		return path

	def _world_to_local(self, ai_world_pos, target_world_pos):
		"""Project global target to local 21x21 index with AI at center."""
		cx, cy = 10, 10
		dx = int(round(target_world_pos[0] - ai_world_pos[0]))
		dy = int(round(target_world_pos[1] - ai_world_pos[1]))
		return (cx + dx, cy + dy)

	def _in_bounds(self, x, y):
		return 0 <= x < 21 and 0 <= y < 21

	def _is_walkable(self, grid, x, y):
		if not (0 <= x < 21 and 0 <= y < 21):
			return False
		return grid[x][y] != 0

	def _euclidean(self, a, b):
		return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

	def _validate_grid(self, grid_21x21):
		if len(grid_21x21) != 21 or any(len(row) != 21 for row in grid_21x21):
			raise ValueError("grid_21x21 must be exactly 21x21")

if __name__ == "__main__":
	planner = PathPlanner()
	DEMO_GRID_21x21 = [
		[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
		[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
		[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
		[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
		[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
		[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
		[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
		[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
		[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
		[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
		[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
		[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
		[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
		[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
		[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
		[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
		[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
		[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
		[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
		[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
		[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
	]
	ai_world_pos = (100, 80)
	target_world_pos = (113, 187)
	dir_values = planner.process_grid(
		ai_world_pos,
		target_world_pos,
		DEMO_GRID_21x21,
		use_gain=True,
	)

	print("AI world pos:", ai_world_pos)
	print("Target world pos:", target_world_pos)
	print("Eight direction gain values (RU,R,RD,D,LD,L,LU,U):", dir_values)

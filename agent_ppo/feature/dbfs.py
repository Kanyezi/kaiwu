import heapq
from math import sqrt
from typing import Dict, List, Sequence, Tuple


Coord = Tuple[int, int]


class DBFS:
	"""两帧地图拼接 + 双起点最短路评估。"""
	def __init__(self, prev_map, cur_map, cur_pos, prev_pos, target_pos):
		self.prev_map = prev_map
		self.cur_map = cur_map
		self.cur_pos = cur_pos
		self.prev_pos = prev_pos
		self.target_pos = target_pos

	def jia_map(self):
		#拼接地图两个地图都为21*21，移动量最大为1，按照移动方向进行拼接
		
		#计算移动方向
		dx = self.cur_pos[0] - self.prev_pos[0]
		dy = self.cur_pos[1] - self.prev_pos[1]

		#指定新地图#为填空符合23*23
		map = [["#"]*23 for _ in range(23)]
		#新地图中心为prev_map,之后按照移动方向拼接cur_map
		for i in range(21):
			for j in range(21):
				map[i + 1][j + 1] = self.prev_map[i][j]
		for i in range(21):
			for j in range(21):
				map[i + 1 + dx][j + 1 + dy] = self.cur_map[i][j]

		return map

	def dfs(self, start):
		#使用a*计算起点到目标的分别最短路，1代表可以走，0代表障碍物，#意味未知，可以把挨着#格子的当成边。如果视野之外，就拼接边界距离与边界到目标的直线距离
		#起点 (x,y)
		start
		#目标点
		self.target_pos

		grid = self.jia_map()
		size = 23
		center = 11
		scale = 1_000_000
		straight_cost = scale
		diag_cost = 1_414_214
		inf_cost = 10**18

		directions = (
			(1, 0),
			(-1, 0),
			(0, 1),
			(0, -1),
			(1, 1),
			(1, -1),
			(-1, 1),
			(-1, -1),
		)

		def split_steps(abs_dx, abs_dy):
			diag_steps = min(abs_dx, abs_dy)
			straight_steps = max(abs_dx, abs_dy) - diag_steps
			return diag_steps, straight_steps

		def scaled_by_delta(abs_dx, abs_dy):
			diag_steps, straight_steps = split_steps(abs_dx, abs_dy)
			return diag_steps * diag_cost + straight_steps * straight_cost

		def final_distance(straight_steps, diag_steps):
			return straight_steps + diag_steps * sqrt(2.0)

		def abs_to_local(pos):
			return center + (pos[0] - self.prev_pos[0]), center + (pos[1] - self.prev_pos[1])

		def local_to_abs(i, j):
			return self.prev_pos[0] + (i - center), self.prev_pos[1] + (j - center)

		si, sj = abs_to_local(start)
		ti, tj = abs_to_local(self.target_pos)

		if not (0 <= si < size and 0 <= sj < size) or grid[si][sj] != 1:
			return -1

		target_in_map = 0 <= ti < size and 0 <= tj < size
		if target_in_map and grid[ti][tj] != 1:
			return -1

		g_score = [[inf_cost] * size for _ in range(size)]
		g_score[si][sj] = 0
		straight_steps = [[0] * size for _ in range(size)]
		diag_steps = [[0] * size for _ in range(size)]

		heap = []
		start_h = scaled_by_delta(abs(ti - si), abs(tj - sj)) if target_in_map else 0
		heapq.heappush(heap, (start_h, 0, si, sj))

		best_outside = inf_cost
		best_outside_straight = 0
		best_outside_diag = 0

		while heap:
			f, g, i, j = heapq.heappop(heap)
			if g != g_score[i][j]:
				continue

			if target_in_map:
				if i == ti and j == tj:
					ans = final_distance(straight_steps[i][j], diag_steps[i][j])
					return ans
			else:
				if f >= best_outside:
					break

				is_boundary = False
				for di, dj in directions:
					ni, nj = i + di, j + dj
					if not (0 <= ni < size and 0 <= nj < size) or grid[ni][nj] == "#":
						is_boundary = True
						break

				if is_boundary:
					x, y = local_to_abs(i, j)
					out_diag, out_straight = split_steps(abs(self.target_pos[0] - x), abs(self.target_pos[1] - y))
					candidate = g + out_diag * diag_cost + out_straight * straight_cost
					if candidate < best_outside:
						best_outside = candidate
						best_outside_straight = straight_steps[i][j] + out_straight
						best_outside_diag = diag_steps[i][j] + out_diag

			for di, dj in directions:
				ni, nj = i + di, j + dj
				if not (0 <= ni < size and 0 <= nj < size):
					continue
				if grid[ni][nj] != 1:
					continue

				is_diag = 1 if di != 0 and dj != 0 else 0
				if is_diag:
					if grid[i + di][j] != 1 or grid[i][j + dj] != 1:
						continue
				move_cost = diag_cost if is_diag else straight_cost
				ng = g + move_cost
				if ng >= g_score[ni][nj]:
					continue

				g_score[ni][nj] = ng
				straight_steps[ni][nj] = straight_steps[i][j] + (0 if is_diag else 1)
				diag_steps[ni][nj] = diag_steps[i][j] + is_diag
				if target_in_map:
					h = scaled_by_delta(abs(ti - ni), abs(tj - nj))
				else:
					x, y = local_to_abs(ni, nj)
					h = scaled_by_delta(abs(self.target_pos[0] - x), abs(self.target_pos[1] - y))
				heapq.heappush(heap, (ng + h, ng, ni, nj))

		if target_in_map:
			return -1

		if best_outside == inf_cost:
			return -1

		ans = final_distance(best_outside_straight, best_outside_diag)
		return ans

	def main(self):
		# self.prev_map = prev_map
		# self.cur_map = cur_map
		# self.cur_pos = cur_pos
		# self.prev_pos = prev_pos
		# self.target_pos = target_pos
		self.map = self.jia_map()
		#两次次路程
		cur_dist = self.dfs(self.cur_pos)
		prev_dist = self.dfs(self.prev_pos)
		cha_dist = prev_dist-cur_dist
		print(cur_dist,prev_dist)

		return cha_dist
		pass

if __name__ == "__main__":
	prev_map = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
	]
	cur_map = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
	]
	prev_pos = (64, 64)
	cur_pos = (64, 65)
	target_pos = (64, 68)

	dbf = DBFS(prev_map, cur_map, cur_pos, prev_pos, target_pos)
	ne_map = dbf.jia_map()
	for i in ne_map:
		for j in i:
			print(j, end=" ")
		print()
	print(dbf.main())
	pass
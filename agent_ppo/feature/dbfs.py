import heapq
from math import sqrt
from typing import List, Sequence, Tuple


Coord = Tuple[int, int]


def min_distance_with_fov(
	start: Coord,
	destination: Coord,
	vision: Sequence[Sequence[int]],
) -> float:
	"""计算从起点到目的地的最小距离。

	参数:
	- start: 起始绝对坐标 (x, y)
	- destination: 目的地绝对坐标 (x, y)
	- vision: 21x21 视野矩阵，中心点对应 start。
	  1 表示可通行，0 表示障碍。

	返回:
	- 最小距离（直走=1，斜走=sqrt(2)）
	- 不可达时返回 -1.0

	说明:
	- 当 destination 在 21x21 范围内时，使用 A* 返回视野内最短路。
	- 当 destination 在范围外时，先在视野内绕过障碍走到某个可达边界格，
	  再加上该边界格到 destination 的最短八方向距离（视野外默认不含障碍信息）。
	- 搜索阶段只使用整数代价比较，不计算 sqrt；仅在最终输出时还原。
	"""
	size = 21
	center = size // 2
	scale = 1_000_000
	straight_cost = scale
	diag_cost = 1_414_214
	inf_cost = 10**18

	def final_distance(straight_steps: int, diag_steps: int) -> float:
		return straight_steps + diag_steps * sqrt(2.0)

	def split_steps(abs_dx: int, abs_dy: int) -> Tuple[int, int]:
		diag_steps = min(abs_dx, abs_dy)
		straight_steps = max(abs_dx, abs_dy) - diag_steps
		return diag_steps, straight_steps

	def scaled_by_delta(abs_dx: int, abs_dy: int) -> int:
		diag_steps, straight_steps = split_steps(abs_dx, abs_dy)
		return diag_steps * diag_cost + straight_steps * straight_cost

	if len(vision) != size or any(len(row) != size for row in vision):
		raise ValueError("vision 必须是 21x21 的二维矩阵")

	if vision[center][center] != 1:
		return -1.0

	sx, sy = start
	dx, dy = destination

	# 目的地在视野坐标系中的索引
	target_i = center + (dx - sx)
	target_j = center + (dy - sy)
	target_in_vision = 0 <= target_i < size and 0 <= target_j < size

	# 八方向：每步代价为 1
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

	# g_score: 起点到当前格的最小整数缩放代价
	g_score: List[List[int]] = [[inf_cost] * size for _ in range(size)]
	g_score[center][center] = 0
	straight_steps: List[List[int]] = [[0] * size for _ in range(size)]
	diag_steps: List[List[int]] = [[0] * size for _ in range(size)]

	# 小顶堆元素: (f, g, i, j)
	heap: List[Tuple[int, int, int, int]] = []
	start_h = (
		scaled_by_delta(abs(target_i - center), abs(target_j - center)) if target_in_vision else 0
	)
	heapq.heappush(heap, (start_h, 0, center, center))

	best_outside = inf_cost
	best_outside_straight = 0
	best_outside_diag = 0

	while heap:
		f, g, i, j = heapq.heappop(heap)

		if g != g_score[i][j]:
			continue

		if target_in_vision:
			if i == target_i and j == target_j:
				return final_distance(straight_steps[i][j], diag_steps[i][j])
		else:
			# 当目标在视野外时，若当前最小可能代价都不优于 best_outside，可提前结束
			if f >= best_outside:
				break

			if i in (0, size - 1) or j in (0, size - 1):
				x = sx + (i - center)
				y = sy + (j - center)
				out_diag, out_straight = split_steps(abs(dx - x), abs(dy - y))
				outside_cost = out_diag * diag_cost + out_straight * straight_cost
				candidate = g + outside_cost
				if candidate < best_outside:
					best_outside = candidate
					best_outside_straight = straight_steps[i][j] + out_straight
					best_outside_diag = diag_steps[i][j] + out_diag

		for di, dj in directions:
			ni, nj = i + di, j + dj
			if not (0 <= ni < size and 0 <= nj < size):
				continue
			if vision[ni][nj] != 1:
				continue

			is_diag = 1 if di != 0 and dj != 0 else 0
			move_cost = diag_cost if is_diag else straight_cost
			ng = g + move_cost
			if ng >= g_score[ni][nj]:
				continue

			g_score[ni][nj] = ng
			straight_steps[ni][nj] = straight_steps[i][j] + (0 if is_diag else 1)
			diag_steps[ni][nj] = diag_steps[i][j] + is_diag
			if target_in_vision:
				h = scaled_by_delta(abs(target_i - ni), abs(target_j - nj))
			else:
				# 视野外目标：用到目标的最短八方向距离下界做分支限界
				x = sx + (ni - center)
				y = sy + (nj - center)
				h = scaled_by_delta(abs(dx - x), abs(dy - y))
			heapq.heappush(heap, (ng + h, ng, ni, nj))

	# 视野内目的地但未访问到，说明不可达
	if target_in_vision:
		return -1.0

	if best_outside == inf_cost:
		return -1.0
	return final_distance(best_outside_straight, best_outside_diag)

if __name__ == "__main__":
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
	target_world_pos = (114, 187)

	result = min_distance_with_fov(ai_world_pos, target_world_pos, DEMO_GRID_21x21)
	print(f"min_distance={result}")


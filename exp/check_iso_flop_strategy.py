import numpy as np


def check_isoflop_strategy(agent_history):
	if len(agent_history) < 3:
		return 1.0

	COST_TOLERANCE = 0.02
	MIN_POINTS = 3

	points = []
	for step in agent_history:
		points.append({
			'n': step['n'],
			'd': step['d'],
			'cost': 6 * step['n'] * step['d']
		})

	used_indices = set()

	for i in range(len(points)):
		if i in used_indices:
			continue

		base_cost = points[i]['cost']
		iso_group = [points[i]]

		for j in range(i + 1, len(points)):
			if j in used_indices:
				continue

			cost_diff = abs(points[j]['cost'] - base_cost) / base_cost
			if cost_diff <= COST_TOLERANCE:
				iso_group.append(points[j])
				used_indices.add(j)

		if len(iso_group) >= MIN_POINTS:
			unique_n_shapes = set()
			for p in iso_group:
				magnitude = int(np.floor(np.log10(p['n'])))
				rounded_n = round(p['n'], -magnitude + 1)
				unique_n_shapes.add(rounded_n)

			if len(unique_n_shapes) >= MIN_POINTS:
				print(f"[BONUS] Iso-FLOP strategy detected at {base_cost:.2e} FLOPs! (+20% Score)")
				return 1.2

	return 1.0

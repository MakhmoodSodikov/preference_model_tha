from scipy.optimize import minimize

L_IRRED = 1.05
E = 450
B = 390
EXPERIMENTS_BUDGET = 1e18
TOTAL_BUDGET = 1e23
OPS_COUNT = 6.0

alpha = 0.3
beta = 0.27


def objective(n):
	d = TOTAL_BUDGET / (OPS_COUNT * n)
	return (E / (n ** alpha)) + (B / (d ** beta)) + L_IRRED


def analytical_min_loss():
	G = (alpha * E) / (beta * B)

	exponent = 1.0 / (alpha + beta)
	compute_factor = (TOTAL_BUDGET / OPS_COUNT) ** beta

	n_true = (G * compute_factor) ** exponent
	d_true = TOTAL_BUDGET / (OPS_COUNT * n_true)

	l_min = (E / (n_true ** alpha)) + \
	             (B / (d_true ** beta)) + L_IRRED

	return n_true, d_true, l_min


res = minimize(objective, x0=1e10, bounds=[(1e7, 1e14)])
s_n_true = res.x[0]
s_d_true = TOTAL_BUDGET / (OPS_COUNT * s_n_true)

a_n_true, a_d_true, a_l_min = analytical_min_loss()

print(f"Stochastic opt solution: N = {s_n_true:.2e}, D = {s_d_true:.2e}, L_min = {objective(s_n_true):.3f}")
print(f"Analytical solution    : N = {a_n_true:.2e}, D = {a_d_true:.2e}, L_min = {a_l_min:.3f}")

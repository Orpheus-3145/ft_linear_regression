from . import config

def linear_regression(m: float, q: float, x: int) -> float:
	return m * x + q

def estimate_price(x: int) -> float:
	print(f" - m: {config.TETHA_1} q: {config.TETHA_0}")
	return linear_regression(config.TETHA_1, config.TETHA_0, x)

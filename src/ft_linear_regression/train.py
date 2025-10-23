import pandas as pd
import numpy as np

def load_csv(csv_path: str, scale_factor=1.) -> tuple[np.ndarray, np.ndarray]:
	data = pd.read_csv(csv_path, delimiter=",", header=0)
	data.sort_values(by=data.columns[0], ascending=False)
	x_data: np.ndarray = data.iloc[:, 0].to_numpy() / scale_factor
	y_data: np.ndarray = data.iloc[:, 1].to_numpy() / scale_factor
	return x_data, y_data

def mae(est_data: np.ndarray, exp_data: np.ndarray) -> float:
	m: int = len(est_data)
	deltas: np.ndarray = est_data - exp_data
	residuals = sum([abs(delta) for delta in deltas])
	return residuals / m

def mse(est_data: np.ndarray, exp_data: np.ndarray, squared=True) -> float:
	m: int = len(est_data)
	deltas: np.ndarray = est_data - exp_data
	residuals = sum([delta ** 2 for delta in deltas])

	return residuals ** 0.5 if squared else residuals / m

def estimate_y(m: float, q: float, x: float) -> float:
	return m * x + q

def train_model(x_data: np.ndarray, y_data: np.ndarray, epochs: int, learning_rate: int) -> None:
	population: int = len(x_data)
	train_data = {
		"learning_rate": learning_rate,
		"tetha_0": .0,
		"tetha_1": .0,
		"mse": .0,
		"mae": .0
	}

	for _ in range(epochs):
		delta_y = train_data["tetha_1"] * x_data + train_data["tetha_0"] - y_data

		train_data["tetha_0"] -= learning_rate * sum(delta_y) / population
		train_data["tetha_1"] -= learning_rate * sum(delta_y * x_data) / population

	estimated_y = train_data["tetha_1"] * x_data + train_data["tetha_0"]
	train_data["mse"] = mse(y_data, estimated_y)
	train_data["mae"] = mae(y_data, estimated_y)
	return train_data

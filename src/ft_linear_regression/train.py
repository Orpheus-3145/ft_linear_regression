import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, Future
from multiprocessing import Manager
import time, sys, os

class TrainingData:
	def __init__(self, learning_rate: np.float64) -> None:
		self.learning_rate: np.float64 = learning_rate
		self.tetha_0: np.float64 = .0
		self.tetha_1: np.float64 = .0
		self.rmse: np.float64 = .0
		self.mae: np.float64 = .0
		self.r_squared: np.float64 = .0
		self.color: list[np.float64] = np.random.default_rng(seed=os.getpid()).random(3)

	def __str__(self) -> None:
		return f"learning rate: {self.learning_rate:.8f}\ntetha_0: {self.tetha_0:.8f}\ntetha_1: {self.tetha_1:.8f}\nRMSE: {self.rmse:.8f}\nMAE: {self.mae:.8f}"

	def estimate(self, x: np.float64) -> np.float64:
		return estimate_y(self.tetha_1, self.tetha_0, x)

def _sum_squared(values: np.ndarray) -> np.float64:
	return np.sum(np.pow(values, 2, dtype=np.float64), dtype=np.float64)

def load_csv(csv_path: str, scale_factor: np.int64=1) -> tuple[np.ndarray, np.ndarray]:
	data: pd.DataFrame = pd.read_csv(csv_path, delimiter=",", header=0)
	data.sort_values(by=data.columns[0], ascending=False)
	x_data: np.ndarray = data.iloc[:, 0].to_numpy(dtype=np.float64) / scale_factor
	y_data: np.ndarray = data.iloc[:, 1].to_numpy(dtype=np.float64) / scale_factor
	return x_data, y_data

def mae(measured_values: np.ndarray, estimated_values: np.ndarray) -> np.float64:
	N: np.int64 = len(measured_values)
	deltas: np.ndarray = measured_values - estimated_values
	residuals: np.float64 = np.sum([np.abs(deltas, dtype=np.float64)], dtype=np.float64) / N

	return residuals

def mse(measured_values: np.ndarray, estimated_values: np.ndarray, rooted: bool=True) -> np.float64:
	N: np.int64 = len(measured_values)
	deltas: np.ndarray = measured_values - estimated_values
	residuals: np.float64 = _sum_squared(deltas) / N
	if rooted:
		residuals = np.pow(residuals, 0.5, dtype=np.float64)

	return residuals

def r_squared(measured_values: np.ndarray, estimated_values: np.ndarray) -> np.float64:
	N: np.int64 = len(measured_values)
	average: np.float64 = np.sum(measured_values, dtype=np.float64) / N
	deltas_ssr: np.ndarray = measured_values - estimated_values
	deltas_sst: np.ndarray = estimated_values - average

	return 1 - _sum_squared(deltas_ssr) / _sum_squared(deltas_sst)

def _estimate_y(m: np.float64, q: np.float64, x: np.float64) -> np.float64:
	return m * x + q

def estimate_y(m: np.float64, q: np.float64, x: np.ndarray) -> np.ndarray:
	return m * x + q

def write_status_process(pid: int, learning_rate: np.float64, progress: int) -> None:
	print(f"[PID: {pid}] - [learning rate: {learning_rate:.6f}] - [ {"#" * progress + "_" * (50 - progress)} ]")

def train_model_parallel(n_procs: np.int64, x_data: np.ndarray, y_data: np.ndarray, epochs: np.int64, lr_values: list[np.int64]) -> list[TrainingData]:
	n_items: np.int64 = len(lr_values)
	# list of proxy data shared between children and parent
	manager = Manager()
	proc_info_list = [manager.Namespace() for _ in range(n_items)]
	for index in range(n_items):
		proc_info_list[index].lr = lr_values[index]
		proc_info_list[index].progress = 0
		proc_info_list[index].pid = -1

	with ProcessPoolExecutor(n_procs) as pool:
		futures: list[Future] = [pool.submit(train_model, x_data, y_data, epochs, lr_values[i], proc_info_list[i]) for i in range(n_items)]
		
		# wait until the last process has started
		while proc_info_list[-1].pid == -1: pass
		# show progress of trainging of every process
		while True:
			# reset stdout cursor at the beginning (i.e. overwrite the existing output)
			sys.stdout.write(f"\033[{n_items}F")
			for data in proc_info_list:
				write_status_process(data.pid, data.lr, data.progress)

			if sum(f.done() for f in futures) == n_items:
				return [data.result() for data in futures]

			time.sleep(0.1)
	
def train_model(x_data: np.ndarray, y_data: np.ndarray, epochs: np.int64, learning_rate: np.float64, proc_info=None) -> TrainingData:
	N: np.int64 = len(x_data)
	training_data: TrainingData = TrainingData(learning_rate)
	y_estimated: np.ndarray = []

	step_progress: np.float64 = -1.
	if proc_info: 
		proc_info.pid = os.getpid()
		step_progress = epochs // 50

	for count in range(epochs):
		if proc_info and count % step_progress == 0:
			proc_info.progress += 1

		y_estimated = estimate_y(training_data.tetha_1, training_data.tetha_0, x_data)
		y_delta: np.ndarray = y_estimated - y_data
		training_data.tetha_0 -= learning_rate * np.sum(y_delta, dtype=np.float64) / N
		training_data.tetha_1 -= learning_rate * np.sum(y_delta * x_data, dtype=np.float64) / N

	training_data.rmse = mse(y_data, y_estimated)
	training_data.mae = mae(y_data, y_estimated)
	training_data.r_squared = r_squared(y_data, y_estimated)
	return training_data

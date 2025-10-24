import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, Future
from multiprocessing import Manager
import time, sys, os

class TrainingData:
	def __init__(self, learning_rate: np.float64, x_data: np.ndarray) -> None:
		self.color: list[np.float64] = np.random.default_rng(seed=os.getpid()).random(3)
		self.learning_rate: np.float64 = learning_rate
		self.tetha_0: np.float64 = .0
		self.tetha_1: np.float64 = .0
		self.mae: np.float64 = .0
		self.rmse: np.float64 = .0
		self.ratio_rmse_dev: np.float64 = .0
		self.r_determination: np.float64 = .0
		self.r_correlation: np.float64 = .0

		if (len(x_data) == 1):
			raise ValueError("not enough data: {self.N}") 
		self.N: np.int64 = len(x_data)
		self.avg: np.float64 = np.sum(x_data, dtype=np.float64) / self.N
		if (self.N > 29):
			self.var: np.float64 = _sum_squared(x_data - self.avg) / (self.N - 1)
		else:
			self.var: np.float64 = _sum_squared(x_data - self.avg) / self.N
		self.stddev: np.float64 = np.sqrt(self.var, dtype=np.float64)

	def set_statistics(self, measured: np.ndarray, estimated: np.ndarray) -> None:
		self.mae = mae(measured, estimated)
		self.rmse = mse(measured, estimated)
		self.ratio_rmse_dev = self.rmse / self.stddev
		self.r_determination = r_squared(measured, estimated)
		if self.r_determination < 0:
			self.r_correlation = np.sqrt(self.r_determination * -1, dtype=np.float64) * -1
		else:
			self.r_correlation = np.sqrt(self.r_determination, dtype=np.float64)

	def __str__(self) -> None:
		corr_str = f"{self.r_correlation:>22.4f}" if self.r_correlation >= 0 else " invalid [R squared<0]".rjust(20)
		return f"""
 _______________________________________________________
| learning rate:                 {self.learning_rate:>22.4f} |
|-------------------------------------------------------|
| average:                       {self.avg:>21.2f}€ |
|-------------------------------------------------------|
| standard deviation:            {self.stddev:>21.2f}€ |
|-------------------------------------------------------|
| tetha_0:                       {self.tetha_0:>22.4f} |
|-------------------------------------------------------|
| tetha_1:                       {self.tetha_1:>22.4f} |
|-------------------------------------------------------|
| MAE:                           {self.mae:>21.2f}€ |
|-------------------------------------------------------|
| RMSE:                          {self.rmse:>21.2f}€ |
|-------------------------------------------------------|
| ratio RMSE on std. dev.:  {self.ratio_rmse_dev * 100:>26.2f}% |
|-------------------------------------------------------|
| R squared:                     {self.r_determination:>22.4f} |
|-------------------------------------------------------|
| correlation index:             {corr_str} |
|_______________________________________________________|"""

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
		residuals = np.sqrt(residuals, dtype=np.float64)

	return residuals

def r_squared(measured_values: np.ndarray, estimated_values: np.ndarray) -> np.float64:
	N: np.int64 = len(measured_values)
	average: np.float64 = np.sum(measured_values, dtype=np.float64) / N
	deltas_ssr: np.ndarray = estimated_values - measured_values
	deltas_sst: np.ndarray = measured_values - average

	return 1 - _sum_squared(deltas_ssr) / _sum_squared(deltas_sst)

def estimate_y(m: np.float64, q: np.float64, x: np.ndarray) -> np.ndarray:
	return m * x + q

def write_status_process(pid: int, learning_rate: np.float64, progress: int) -> None:
	print(f"[ PID: {pid} ] - learning rate: {learning_rate:.6f} - training... {"done! " if progress == 50 else "      "}[ {"#" * progress + "_" * (50 - progress)} ]")

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
	training_data: TrainingData = TrainingData(learning_rate, y_data)

	scale_factor: np.int64 = 1000_000
	x_data /= scale_factor;
	y_data /= scale_factor;

	step_progress: np.float64 = -1.
	if proc_info: 
		proc_info.pid = os.getpid()
		step_progress = epochs // 50

	for count in range(epochs):
		if proc_info and count % step_progress == 0:
			proc_info.progress += 1

		y_delta: np.ndarray = estimate_y(training_data.tetha_1, training_data.tetha_0, x_data) - y_data
		training_data.tetha_0 -= learning_rate * np.sum(y_delta, dtype=np.float64) / N
		training_data.tetha_1 -= learning_rate * np.sum(y_delta * x_data, dtype=np.float64) / N

	training_data.tetha_0 *= scale_factor
	y_estimated: np.ndarray = estimate_y(training_data.tetha_1, training_data.tetha_0, x_data * scale_factor)
	training_data.set_statistics(y_data * scale_factor, y_estimated)
	return training_data

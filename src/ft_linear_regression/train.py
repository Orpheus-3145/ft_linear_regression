import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, Future
from multiprocessing import Manager
from functools import partial
import time, sys, os

from ft_linear_regression.plot import Plotter



class TrainingError(Exception): pass


class TrainingData:
	def __init__(self, learning_rate: np.float64, tetha_0: np.float64, tetha_1: np.float64, feature: np.ndarray, target: np.ndarray) -> None:
		self.color: list[np.float64] = np.random.default_rng(seed=os.getpid()).random(3)
		self.learning_rate: np.float64 = learning_rate
		self.tetha_0: np.float64 = tetha_0
		self.tetha_1: np.float64 = tetha_1
		estimated_target: np.ndarray = self.estimate(feature)
		self.mae = _mae(target, estimated_target)
		self.rmse = _mse(target, estimated_target)
		self.ratio_rmse_dev = self.rmse / _stddev(target)
		self.r_determination = _rsquared(target, estimated_target)
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
|_______________________________________________________|
"""

	def estimate(self, x: np.float64) -> np.float64:
		return self.tetha_1 * x + self.tetha_0
	
	def estimate(self, x: np.ndarray) -> np.ndarray:
		return self.tetha_1 * x + self.tetha_0


class TrainerLR:
	def __init__(self, epochs):
		if not (epochs > 0):
			raise TrainingError(f"invalid epochs: {epochs}, expected greater than 0")
		self.epochs: list[np.int64] = epochs

		self.lr_values: list[np.float64] = []
		self.trained_data: list[TrainingData] = []
		self.N: np.int64 = 0
		self.feature: np.ndarray = []
		self.target: np.ndarray = []
		self.std_feature: np.ndarray = []
		self.std_target: np.ndarray = []
		self.log_scale: bool = True
		self.standardized_data = False

	def load_csv(self, csv_path: str) -> None:
		data: pd.DataFrame = pd.read_csv(csv_path, delimiter=",", header=0)
		data.sort_values(by=data.columns[0], ascending=False)

		self.feature = data.iloc[:, 0].to_numpy(dtype=np.float64)
		self.target = data.iloc[:, 1].to_numpy(dtype=np.float64)

		if len(self.feature) != len(self.target):
			raise TrainingError(f"X and Y data have different population: X={len(self.feature)} Y={len(self.target)}")
		self.N = len(self.feature)

	def standardize(self) -> None:
		assert self.N

		self.standardized_data = True
		self.std_feature: np.ndarray = (self.feature - _avg(self.feature)) / _stddev(self.feature)
		self.std_target: np.ndarray = (self.target - _avg(self.target)) / _stddev(self.target)

	def generate_lr_range(self, min: np.float64, max: np.float64, n_items: np.int64, linear_scale=False) -> None:
		if linear_scale:
			self.log_scale = False
			self.lr_values = np.linspace(min, max, n_items)
		else:
			self.lr_values = np.logspace(min, max, n_items)

	def train_model(self) -> None:
		if not self.N:
			raise TrainingError("No data read from any source, call .load_csv() first")
		elif not self.lr_values.size:
			raise TrainingError("No learning rates generated, call .generate_lr_range() first")
		self.trained_data.clear()

		if self.standardized_data:
			feature: np.ndarray = self.std_feature
			target: np.ndarray = self.std_target
		else:
			feature: np.ndarray = self.feature
			target: np.ndarray = self.target

		_trainer = partial(_trainLR, feature, target, self.N, self.epochs)
		for lr_value in self.lr_values:
			self._add_new_data(**(_trainer(lr_value)))

	def train_model_parallel(self, n_procs: np.int64 = 30) -> None:
		assert n_procs > 0
		if not self.N:
			raise TrainingError("No data read from any source, call .load_csv() first")
		elif not self.lr_values.size:
			raise TrainingError("No learning rates generated, call .generate_lr_range() first")
		self.trained_data.clear()

		if self.standardized_data:
			feature: np.ndarray = self.std_feature
			target: np.ndarray = self.std_target
		else:
			feature: np.ndarray = self.feature
			target: np.ndarray = self.target

		n_items: np.int64 = len(self.lr_values)
		# list of proxy data shared between children and parent
		manager = Manager()
		proc_info_list = [manager.Namespace() for _ in range(n_items)]
		for index in range(n_items):
			proc_info_list[index].lr = self.lr_values[index]
			proc_info_list[index].progress = 0
			proc_info_list[index].pid = -1

		with ProcessPoolExecutor(n_procs) as pool:
			_trainer = partial(_trainLR, feature, target, self.N, self.epochs)
			futures: list[Future] = [pool.submit(_trainer, self.lr_values[i], proc_info_list[i]) for i in range(n_items)]
			
			# wait until the last process has started
			while proc_info_list[-1].pid == -1: pass
			# show progress of trainging of every process
			while True:
				# reset stdout cursor at the beginning (i.e. overwrite the existing output)
				sys.stdout.write(f"\033[{n_items}F")
				for data in proc_info_list:
					self._write_status_process(data.pid, data.lr, data.progress)

				if sum(f.done() for f in futures) == n_items:
					map(lambda f: self._add_new_data(**(f.result())), futures)
					return
				time.sleep(0.1)

	def show_data(self) -> None:
		for data in self.trained_data:
			print(data)

	def plot_data(self) -> None:
		plotter: Plotter = Plotter(nrows=2, ncols=2, figsize=(14, 10), layout="constrained")
		# plot regression line for every learning rate
		lower_limit: np.float64 = np.min([training.learning_rate for training in self.trained_data])
		upper_limit: np.float64 = np.max([training.learning_rate for training in self.trained_data])
		for item in self.trained_data:
			plotter.draw_fit_line(self.feature, self.target, item.estimate, index_graph=[0, 0], title=f"Regression lines with learning rates ranging in [{lower_limit:.4f}: {upper_limit:.4f}]", color=item.color, label=f"learning rate: {item.learning_rate:.4f}")

		# plot best fit line
		best_fit: TrainingData = np.max(self.trained_data, key=lambda train: train.r_determination)
		plotter.draw_fit_line(self.feature, self.target, best_fit.estimate, index_graph=[0, 1], title=f"Best fit regression, learning rate: {best_fit.learning_rate:.4f}", show_legend=True, color="green", label=f"points")

		# plot bars of RMSE and R squared for every learning rate
		learning_rates: list[TrainingData] = [train.learning_rate for train in self.trained_data]
		plotter.draw_rmse_line(learning_rates, [train.rmse for train in self.trained_data], index_graph=[1, 0], is_logarithmic=self.log_scale)
		plotter.draw_rsquared_line(learning_rates, [train.r_determination for train in self.trained_data], index_graph=[1, 1], is_logarithmic=self.log_scale)
		plotter.show()

	def _add_new_data(self, learning_rate: np.float64, tetha_0: np.float64, tetha_1: np.float64) -> None:
		if self.standardized_data:
			avg_feature: np.float64 = _avg(self.feature)
			avg_target: np.float64 = _avg(self.target)
			stddev_feature: np.float64 = _stddev(self.feature)
			stddev_target: np.float64 = _stddev(self.target)
			tetha_0 = avg_target + stddev_target * (tetha_0 - tetha_1 * avg_feature / stddev_feature)
			tetha_1 = tetha_1 * stddev_feature / stddev_target

		new_data: TrainingData = TrainingData(learning_rate, tetha_0, tetha_1, self.feature, self.target)
		self.trained_data.append(new_data)

	def _write_status_process(self, pid: int, learning_rate: np.float64, progress: int) -> None:
		is_done: str = " done. " if progress == 50 else "...    "
		progress :str = "#" * progress + "_" * (50 - progress)
		print(f"[ PID: {pid} ] - learning rate: {learning_rate:.4f} - training{is_done}[ {progress} ]")


def _sum_squared(values: np.ndarray) -> np.float64:
	return np.sum(np.pow(values, 2, dtype=np.float64), dtype=np.float64)

def _avg(values: np.ndarray) -> np.float64:
	N: np.int64 = len(values)
	assert N > 0

	return np.sum(values, dtype=np.float64) / N

def _stddev(values: np.ndarray) -> np.float64:
	avg: np.float64 = _avg(values)
	N: np.int64 = len(values)
	assert N > 1

	if (N > 29):
		var: np.float64 = _sum_squared(values - avg) / (N - 1)
	else:
		var: np.float64 = _sum_squared(values - avg) / N

	return np.sqrt(var, dtype=np.float64)

def _mae(measured_values: np.ndarray, estimated_values: np.ndarray) -> np.float64:
	N: np.int64 = len(measured_values)
	assert N > 0

	deltas: np.ndarray = measured_values - estimated_values
	residuals: np.float64 = np.sum([np.abs(deltas, dtype=np.float64)], dtype=np.float64) / N

	return residuals

def _mse(measured_values: np.ndarray, estimated_values: np.ndarray, rooted: bool=True) -> np.float64:
	N: np.int64 = len(measured_values)
	assert N > 0

	deltas: np.ndarray = measured_values - estimated_values
	residuals: np.float64 = _sum_squared(deltas) / N
	if rooted:
		residuals = np.sqrt(residuals, dtype=np.float64)

	return residuals

def _rsquared(measured_values: np.ndarray, estimated_values: np.ndarray) -> np.float64:
	N: np.int64 = len(measured_values)
	assert N > 0

	average: np.float64 = np.sum(measured_values, dtype=np.float64) / N
	deltas_ssr: np.ndarray = estimated_values - measured_values
	deltas_sst: np.ndarray = measured_values - average

	return 1 - _sum_squared(deltas_ssr) / _sum_squared(deltas_sst)

def _trainLR(feature: np.ndarray, target: np.ndarray, N: np.int64, epochs: np.int64, learning_rate: np.float64, proc_info=None) -> dict:
	assert N > 0 and N == len(target)
	tetha_0: np.float64 = .0
	tetha_1: np.float64 = .0

	# if the function is run in parallel, save the progress of the training so it can be printed in the main process
	step_progress: np.float64 = -1.
	if proc_info: 
		proc_info.pid = os.getpid()
		step_progress = epochs // 50

	for count in range(epochs):
		if proc_info and count % step_progress == 0:
			proc_info.progress += 1

		y_delta: np.ndarray = (tetha_1 * feature + tetha_0) - target
		tetha_0 -= learning_rate * np.sum(y_delta, dtype=np.float64) / N
		tetha_1 -= learning_rate * np.sum(y_delta * feature, dtype=np.float64) / N

	return {"learning_rate": learning_rate, "tetha_0": tetha_0, "tetha_1": tetha_1, }

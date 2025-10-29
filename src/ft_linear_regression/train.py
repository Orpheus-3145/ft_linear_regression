import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, Future
from multiprocessing import Manager
from functools import partial
import time, sys, os

from ft_linear_regression.plot import Plotter


class TrainingError(Exception): pass


class TrainingData:
	def __init__(self, tetha_0: np.float64, tetha_1: np.float64, feature: np.ndarray=[], target: np.ndarray=[]) -> None:
		self.color: list[np.float64] = np.random.default_rng(seed=int(time.time() * 10000)).random(3)
		self.tetha_0: np.float64 = tetha_0
		self.tetha_1: np.float64 = tetha_1
		self.mae: np.float64 = .0
		self.rmse: np.float64 = .0
		self.ratio_rmse_dev: np.float64 = .0
		self.r_determination: np.float64 = .0
		self.r_correlation: np.float64 = .0
		if len(feature) and len(target):
			estimated_target: np.ndarray = self.estimate(feature)
			self.mae: np.float64 = _mae(target, estimated_target)
			self.rmse: np.float64 = _mse(target, estimated_target)
			self.ratio_rmse_dev: np.float64 = (self.rmse / _stddev(target)) * 100
			self.r_determination: np.float64 = _rsquared(target, estimated_target)
			if self.r_determination < 0:
				self.r_correlation = np.sqrt(self.r_determination * -1, dtype=np.float64) * -1
			else:
				self.r_correlation = np.sqrt(self.r_determination, dtype=np.float64)

	def __str__(self) -> None:
		corr_str = f"{self.r_correlation:>22.4f}" if self.r_correlation >= 0 else " invalid [R squared<0]".rjust(20)
		return f"""
 _______________________________________________________
|                                                       |
| tetha_0:                       {self.tetha_0:>22.6f} |
|-------------------------------------------------------|
| tetha_1:                       {self.tetha_1:>22.6f} |
|-------------------------------------------------------|
| MAE:                           {self.mae:>21.2f}€ |
|-------------------------------------------------------|
| RMSE:                          {self.rmse:>21.2f}€ |
|-------------------------------------------------------|
| ratio RMSE on std. dev.:  {self.ratio_rmse_dev:>26.2f}% |
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
	def __init__(self, epochs: np.int64) -> None:
		if not (epochs > 0):
			raise TrainingError(f"invalid epochs: {epochs}, expected greater than 0")

		self.epochs: list[np.int64] = epochs
		self.lr_best_fit: np.float64 = 0.
		self.trained_data: dict[np.float64: TrainingData | None] = {}
		self.feature: np.ndarray = []
		self.target: np.ndarray = []
		self.std_feature: np.ndarray = []
		self.std_target: np.ndarray = []
		self.norm_feature: np.ndarray = []
		self.norm_target: np.ndarray = []
		self.N: np.int64 = 0
		self.avg_feature: np.float64 = 0.
		self.stddev_feature: np.float64 = 0.
		self.log_scale: bool = True
		self.standardized_data: bool = False
		self.normalized_data: bool = False

	def get_best_fit(self) -> TrainingData:
		return self.trained_data[self.lr_best_fit]

	def load_csv(self, csv_path: str) -> None:
		data: pd.DataFrame = pd.read_csv(csv_path, delimiter=",", header=0)

		self.feature = data.iloc[:, 0].to_numpy(dtype=np.float64)
		self.target = data.iloc[:, 1].to_numpy(dtype=np.float64)

		if len(self.feature) != len(self.target):
			raise TrainingError(f"X and Y data have different population: X={len(self.feature)} Y={len(self.target)}")
		assert len(self.feature)
		self.N = len(self.feature)
		self.avg_feature = _avg(self.feature)
		self.stddev_feature = _stddev(self.feature)

	def standardize(self) -> None:
		self.std_feature: np.ndarray = (self.feature - self.avg_feature) / self.stddev_feature
		self.standardized_data = True

	def normalize(self) -> None:
		min_feature: np.float64 = np.min(self.feature)
		max_feature: np.float64 = np.max(self.feature)
		self.norm_feature: np.ndarray = (self.feature - min_feature) / (max_feature - min_feature)
		self.normalized_data = True

	def generate_lr_range(self, min: np.float64, max: np.float64, n_items: np.int64, linear_scale: bool = False) -> None:
		learning_rage_values: np.ndarray = []
		if linear_scale:
			self.log_scale = False
			learning_rage_values = np.linspace(min, max, n_items)
		else:
			learning_rage_values = np.logspace(min, max, n_items)
		self.trained_data = {lr: None for lr in learning_rage_values}

	def train_model(self) -> None:
		_learning_rates: np.ndarray = list(self.trained_data.keys())
		if not self.N:
			raise TrainingError("No data read from any source, call .load_csv() first")
		elif not len(_learning_rates):
			raise TrainingError("No learning rates generated, call .generate_lr_range() first")
		self.trained_data.clear()

		if self.standardized_data:
			feature: np.ndarray = self.std_feature
		elif self.normalized_data:
			feature: np.ndarray = self.norm_feature
		else:
			feature: np.ndarray = self.feature
		target: np.ndarray = self.target

		for lr_value in _learning_rates:
			train_result = _trainLR(feature, target, self.N, self.epochs, lr_value)
			self._add_new_data(**train_result)
		self._set_best_fit()

	def train_model_parallel(self, n_procs: np.int64 = 30) -> None:
		assert n_procs > 0
		_learning_rates: np.ndarray = list(self.trained_data.keys())
		n_items: np.int64 = len(_learning_rates)
		if not self.N:
			raise TrainingError("No data read from any source, call .load_csv() first")
		elif not len(_learning_rates):
			raise TrainingError("No learning rates generated, call .generate_lr_range() first")
		self.trained_data.clear()

		if self.standardized_data:
			feature: np.ndarray = self.std_feature
		elif self.normalized_data:
			feature: np.ndarray = self.norm_feature
		else:
			feature: np.ndarray = self.feature
		target: np.ndarray = self.target
		
		# list of proxy data shared between children and parent
		manager = Manager()
		proc_info_list = [manager.Namespace() for _ in range(n_items)]
		for index in range(n_items):
			proc_info_list[index].lr = _learning_rates[index]
			proc_info_list[index].progress = 0
			proc_info_list[index].pid = -1

		with ProcessPoolExecutor(n_procs) as pool:
			_trainer = partial(_trainLR, feature, target, self.N, self.epochs)
			futures: list[Future] = [pool.submit(_trainer, _learning_rates[i], proc_info_list[i]) for i in range(n_items)]

			# wait until the last process has started
			while proc_info_list[-1].pid == -1: pass
			# show progress of trainging of every process
			while True:
				if sum([f.done() for f in futures]) == n_items:
					break
				# reset stdout cursor at the beginning (i.e. overwrite the existing output)
				sys.stdout.write(f"\033[{n_items}F")
				for data in proc_info_list:
					self._write_status_process(data.pid, data.lr, data.progress)
				time.sleep(0.1)

		for item in [f.result() for f in futures]:
			self._add_new_data(**item)
		self._set_best_fit()

	def show_data(self) -> None:
		for lr, data in self.trained_data.items():
			print(f"learning rate: {lr:.8f}{data}\n")
		print(f"best fit line: {self.lr_best_fit:.8f}{self.get_best_fit()}\n")

	def plot_data(self) -> None:
		plotter: Plotter = Plotter(nrows=3, ncols=3, figsize=(14, 10), layout="constrained")
		# plot regression line for every learning rate
		for lr, data in self.trained_data.items():
			plotter.draw_fit_line(
				self.feature, 
				self.target,
				data.estimate,
				index_graph=[0, 1],
				title=f"Regression lines",
				color=data.color,
				label=f"learning rate: {lr:.8f}")

		# plot best fit line
		best_fit: TrainingData = self.get_best_fit()
		plotter.draw_fit_line(
			self.feature,
			self.target,
			best_fit.estimate,
			index_graph=[1, 0],
			title="Best fit",
			color="green",
			label=f"points")

		# plot bars of RMSE and R squared for every learning rate
		learning_rates: list[TrainingData] = self.trained_data.keys(); # [train.learning_rate for train in self.trained_data]
		plotter.draw_rmse_line(
			learning_rates,
			[train.rmse for train in self.trained_data.values()],
			index_graph=[2, 0],
			is_logarithmic=self.log_scale)
		plotter.draw_rsquared_line(
			learning_rates,
			[train.r_determination for train in self.trained_data.values()],
			index_graph=[2, 2],
			is_logarithmic=self.log_scale)

		lr_min: np.float64 = min(self.trained_data.keys())
		lr_max: np.float64 = max(self.trained_data.keys())
		plotter.add_info(0.02, 0.97, f"learning rates range: [{lr_min:.8f}: {lr_max:.8f}]\nepochs: {self.epochs}\ndata standardized: {self.standardized_data}\ndata normalized: {self.normalized_data}")
		plotter.add_info(0.52, 0.97, f"learning rate: {self.lr_best_fit:.8f}\ntetha0: {best_fit.tetha_0:.4f}\ntetha1: {best_fit.tetha_1:.4f}")

		plotter.show()

	def _set_best_fit(self) -> None:
		for lr, data in self.trained_data.items():
			if not self.lr_best_fit or self.trained_data[self.lr_best_fit].rmse > data.rmse:
				self.lr_best_fit = lr

	def _add_new_data(self, learning_rate: np.float64, tetha_0: np.float64, tetha_1: np.float64) -> None:
		if self.standardized_data:
			tetha_0 = tetha_0 - tetha_1 * self.avg_feature / self.stddev_feature
			tetha_1 = tetha_1 / self.stddev_feature

		self.trained_data[learning_rate] = TrainingData(tetha_0, tetha_1, self.feature, self.target)

	def _write_status_process(self, pid: np.int64, learning_rate: np.float64, progress: np.int64) -> None:
		is_done: str = " done. " if progress == 50 else "...    "
		progress :str = "#" * progress + "_" * (50 - progress)
		print(f"[ PID: {pid} ] - learning rate: {learning_rate:15.8f} - training{is_done}[ {progress} ]")


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
		var: np.float64 = _sum_squared(values - avg) / N
	else:
		var: np.float64 = _sum_squared(values - avg) / (N - 1)

	return np.sqrt(var, dtype=np.float64)

def _mae(measured_values: np.ndarray, estimated_values: np.ndarray) -> np.float64:
	N: np.int64 = len(measured_values)
	assert N > 0

	deltas: np.ndarray = measured_values - estimated_values
	residuals: np.float64 = np.sum([np.abs(deltas, dtype=np.float64)], dtype=np.float64) / N

	return residuals

def _mse(measured_values: np.ndarray, estimated_values: np.ndarray, rooted: bool = True) -> np.float64:
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

def _trainLR(feature: np.ndarray, target: np.ndarray, N: np.int64, epochs: np.int64, learning_rate: np.float64, proc_info = None) -> dict:
	assert N > 0 and N == len(target)
	tetha_0: np.float64 = .0
	tetha_1: np.float64 = .0

	# if the function is executed in parallel, save the progress of the training so it can be printed in the main process
	step_progress: np.float64 = -1.
	if proc_info: 
		proc_info.pid = os.getpid()
		step_progress = epochs // 50

	for count in range(epochs):
		if proc_info and count % step_progress == 0:
			proc_info.progress += 1

		y_delta: np.ndarray = (tetha_1 * feature + tetha_0) - target
		tetha_0 -= learning_rate / N * sum(y_delta)
		tetha_1 -= learning_rate / N * sum(y_delta * feature)

	return {"learning_rate": learning_rate, "tetha_0": tetha_0, "tetha_1": tetha_1, }

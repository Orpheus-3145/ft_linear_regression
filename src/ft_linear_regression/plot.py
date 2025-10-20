import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from . import config

current_graphs: list = []

def init_graph(**plot_args) -> None:
	graph, ax = plt.subplots(**plot_args)
	current_graphs.extend(ax)

def draw_scatter_df(dataset: pd.DataFrame, **plot_args) -> None:
	x_data: np.ndarray = dataset.iloc[:, 0].to_numpy()
	y_data: np.ndarray = dataset.iloc[:, 1].to_numpy()
	draw_scatter(x_data, y_data, **plot_args)

def plot_fit_line_df(dataset: pd.DataFrame, **plot_args) -> None:
	x_data: np.ndarray = dataset.iloc[:, 0].to_numpy()
	y_data: np.ndarray = config.TETHA_1 * x_data + config.TETHA_0
	plot_fit_line(x_data, y_data, **plot_args)

def draw_scatter(x_data: np.ndarray, y_data: np.ndarray, index_graph=0, **plot_args) -> None:
	try:
		curr_graph: plt.axes = current_graphs[index_graph]
	except IndexError:
		return
	curr_graph.scatter(x_data, y_data, **plot_args)

def plot_fit_line(x_data: np.ndarray, y_data: np.ndarray, index_graph=0, **plot_args) -> None:
	try:
		curr_graph: plt.axes = current_graphs[index_graph]
	except IndexError:
		return
	curr_graph.plot(x_data, y_data, **plot_args)
	curr_graph.legend()
	print(f"plotting line - TETHA_1: {config.TETHA_1} TETHA_0: {config.TETHA_0}")

def show_data() -> None:
	plt.show()

import matplotlib.pyplot as plt
import numpy as np
from .train import _estimate_y

current_graphs: list[int] = []

def init_graph(**plot_args) -> None:
	fig, ax = plt.subplots(**plot_args)

	if (isinstance(ax, plt.Axes)):
		current_graphs.append(ax)
	else:
		current_graphs.extend(ax)

def draw_scatter(x_data: np.ndarray, y_data: np.ndarray, index_graph=0, **plot_args) -> None:
	try:
		curr_graph: plt.axes = current_graphs[index_graph]
	except IndexError:
		return

	curr_graph.scatter(x_data, y_data, **plot_args)
	curr_graph.grid(True)

def plot_fit_line(x_data: np.ndarray, tetha_1: float, tetha_0: float, index_graph=0, **plot_args) -> None:
	try:
		curr_graph: plt.axes = current_graphs[index_graph]
	except IndexError:
		return

	y_data: np.float64 = _estimate_y(tetha_1, tetha_0, x_data)
	curr_graph.plot(x_data, y_data, **plot_args)
	curr_graph.legend()

def plot_line(x_data: np.ndarray, y_data: np.ndarray, index_graph=0, **plot_args) -> None:
	try:
		curr_graph: plt.axes = current_graphs[index_graph]
	except IndexError:
		return

	curr_graph.plot(x_data, y_data, **plot_args)
	curr_graph.legend()
	
def draw_bars(x_data: np.ndarray, y_data: np.ndarray, index_graph: int=0, float=0.005, **plot_args) -> None:
	try:
		curr_graph: plt.axes = current_graphs[index_graph]
	except IndexError:
		return

	curr_graph.bar(x_data, y_data, **plot_args)
	# curr_graph.set_xscale("log")
	curr_graph.set_xticks(x_data)
	curr_graph.set_xticklabels([f"{x:.8f}" for x in x_data], rotation=90)
	curr_graph.grid(True)
	curr_graph.legend()

def show_data() -> None:
	plt.show()

import matplotlib.pyplot as plt
import numpy as np

current_graphs: list = []

def init_graph(**plot_args) -> None:
	graph, ax = plt.subplots(**plot_args)
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

def plot_fit_line(x_data: np.ndarray, items, index_graph=0) -> None:
	try:
		curr_graph: plt.axes = current_graphs[index_graph]
	except IndexError:
		return
	y_data = x_data * items["tetha_1"] + items["tetha_0"]
	curr_graph.plot(x_data, y_data, color=np.random.rand(3), label=f"learn_rate: {items["learning_rate"]:.6f}")
	curr_graph.legend()

def show_data() -> None:
	plt.show()

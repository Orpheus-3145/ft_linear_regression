import matplotlib
import matplotlib.pyplot as plt
from matplotlib import collections
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np
import os


class LayoutGraphError(Exception):
	pass


class Plotter:
	def __init__(self, mosaic: dict = None, **plot_args) -> None:
		if mosaic is not None:
			self._mosaic = True
			self.fig, self.axes = plt.subplot_mosaic(mosaic, **plot_args)
		else:
			self._mosaic = False
			self.fig, self.axes = plt.subplots(**plot_args)
		self.current_ax: plt.axes | None = None
		manager = plt.get_current_fig_manager()
		manager.full_screen_toggle()

	def _draw_scatter(self, x_data: np.ndarray, y_data: np.ndarray, title: str = "", **plot_args) -> collections.PathCollection:
		if self.current_ax is None:
			raise LayoutGraphError("current postion for graph not set")
	
		scatter: collections.PathCollection = self.current_ax.scatter(x_data, y_data, **plot_args)
		self.current_ax.set_xlim(left=0)
		self.current_ax.set_ylim(bottom=0)
		if title:
			self.current_ax.set_title(title)

		return scatter

	def _draw_line(self, x_data: np.ndarray, y_data: np.ndarray, title: str = "", show_legend: bool = False, is_logarithmic: bool = False, **plot_args) -> Line2D:
		if self.current_ax is None:
			raise LayoutGraphError("current postion for graph not set")

		line: Line2D = self.current_ax.plot(x_data, y_data, **plot_args)
		if title:
			self.current_ax.set_title(title)
		if show_legend:
			self.current_ax.legend()
		if is_logarithmic:
			self.current_ax.set_xscale("log")
		return line

	def _set_current_ax(self, index_graph: tuple[np.int64, np.int64] | str = None) -> None:
		if index_graph is None:
			if not self._mosaic:
				index_graph = [0, 0]
			else:
				raise LayoutGraphError(f"name {index_graph} not found")
		try:
			if self._mosaic:
				self.current_ax = self.axes[index_graph]
			else:
				self.current_ax = self.axes[index_graph[0], index_graph[1]]
		except IndexError:
			raise LayoutGraphError(f"graph with index/name {index_graph} not found")
	
	def draw_fit_line(self, x_data: np.ndarray, y_data: np.ndarray, estimator, index_graph: tuple[np.int64, np.int64] | str = None, title: str = "", show_legend=False, **plot_args) -> None:
		self._set_current_ax(index_graph)

		if not any(isinstance(c, collections.PathCollection) for c in self.current_ax.get_children()):
			self._draw_scatter(x_data, y_data, title)
			self.current_ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}€"))
			self.current_ax.grid(True)

		y_estimated: np.ndarray = estimator(x_data)
		self._draw_line(x_data, y_estimated, "", show_legend, is_logarithmic=False, **plot_args)
		self.current_ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}km"))
		self.current_ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}€"))
		self.current_ax = None

	def draw_rmse_line(self, X: np.ndarray, rmse_values: np.ndarray, index_graph: tuple[np.int64, np.int64] | str = None, label: str = "", is_logarithmic: bool = False) -> None:
		self._set_current_ax(index_graph)

		self._draw_line(X, rmse_values, title="RMSE", show_legend=True, is_logarithmic=is_logarithmic, label=f"RMSE depending on {label}")
		self.current_ax.set_xlabel(label)
		self.current_ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v}"))
		self.current_ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:8.2f}€"))
		self.current_ax.grid(True)
		self.current_ax = None

	def draw_rsquared_line(self, X: np.ndarray, r_squared_values: np.ndarray, index_graph: tuple[np.int64, np.int64] | str = None, label: str = "", is_logarithmic: bool = False) -> None:
		self._set_current_ax(index_graph)

		self._draw_line(X, r_squared_values, title="R squared", show_legend=True, is_logarithmic=is_logarithmic, label=f"R squared depending on {label}")
		self.current_ax.set_xlabel(label)
		self.current_ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v}"))
		self.current_ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:3.2f}"))
		self.current_ax.grid(True)
		self.current_ax = None

	def draw_correlation(self, X: np.ndarray, correlation: np.ndarray, index_graph: tuple[np.int64, np.int64] | str = None, label: str = "", is_logarithmic: bool = False) -> None:
		self._set_current_ax(index_graph)

		self._draw_line(X, correlation, title="Correlation index", show_legend=True, is_logarithmic=is_logarithmic, label=f"Correlation index depending on {label}")
		self.current_ax.set_xlabel(label)
		self.current_ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v}"))
		self.current_ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:3.2f}"))
		self.current_ax.grid(True)
		self.current_ax = None

	def add_info(self, pos_x: np.int64, pos_y: np.int64, info: str, layout: list = []) -> None:
		if len(layout):
			plt.tight_layout(rect=[0, 0, 1, 0.9])

		self.fig.text(pos_x, pos_y,
            info,
            ha="left", va="top",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.8))

	def hide_graph(self, index_graph: tuple[np.int64, np.int64] | str = None) -> None:
		self._set_current_ax(index_graph)
		self.current_ax.set_visible(False)
	
	def show(self) -> None:
		if matplotlib.get_backend().lower() == "agg":
			file_name: str = "train.pdf"
			print(f"non interactive backend for matplotlib - saving plot results in {os.getcwd()}/{file_name}")
			plt.savefig(file_name)
		else:
			plt.show()

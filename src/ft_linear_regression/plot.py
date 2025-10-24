import matplotlib
import matplotlib.pyplot as plt
from matplotlib import collections
from matplotlib.lines import Line2D
from matplotlib.container import BarContainer
from matplotlib.ticker import FuncFormatter
import numpy as np

class LayoutGraphError(Exception): pass

class Plotter:
	def __init__(self, **plot_args) -> None:
		self.fig, self.axes = plt.subplots(**plot_args)
		self.current_ax = None
		manager = plt.get_current_fig_manager()
		manager.full_screen_toggle()
		self._width_bars: np.float64 = 0.0002

	def set_current_ax(self, row: int, col: int) -> None:
		try:
			self.current_ax = self.axes[row, col]
		except IndexError:
			return

	def _draw_scatter(self, x_data: np.ndarray, y_data: np.ndarray, title="", **plot_args) -> collections.PathCollection:
		if self.current_ax is None:
			raise LayoutGraphError("current postion for graph not set")
	
		scatter: collections.PathCollection = self.current_ax.scatter(x_data, y_data, **plot_args)
		self.current_ax.set_xlim(left=0)
		self.current_ax.set_ylim(bottom=0)
		if (title):
			self.current_ax.set_title(title)

		return scatter

	def _draw_line(self, x_data: np.ndarray, y_data: np.ndarray, show_legend=False, **plot_args) -> Line2D:
		if self.current_ax is None:
			raise LayoutGraphError("current postion for graph not set")

		line: Line2D = self.current_ax.plot(x_data, y_data, **plot_args)
		if show_legend:
			self.current_ax.legend()
		return line

	def _draw_bars(self, x_data: np.ndarray, y_data: np.ndarray, title="",**plot_args) -> BarContainer:
		if self.current_ax is None:
			raise LayoutGraphError("current postion for graph not set")
		# NB insert x values vertically inside the bars
		# NB bar width squished
		bars: BarContainer = self.current_ax.bar(x_data, y_data, width=self._width_bars, **plot_args)
		if (title):
			self.current_ax.set_title(title)
		self.current_ax.set_xticks(x_data)
		self.current_ax.grid(True)
		self.current_ax.legend()
		# self.current_ax.bar_label(bars, labels=[f"{v}" for v in x_data],
        #      label_type='center',   # posizione: 'center', 'edge'
        #      rotation=90,           # testo verticale
        #      color='white',         # testo visibile sul colore
        #      fontsize=9)
		return bars
	
	def draw_fit_line(self, x_data: np.ndarray, y_data: np.ndarray, estimator, index_graph=[0, 0], title="", show_legend=False, **plot_args) -> None:
		self.set_current_ax(index_graph[0], index_graph[1])

		if not any(isinstance(c, collections.PathCollection) for c in self.current_ax.get_children()):
			self._draw_scatter(x_data, y_data, title)
			self.current_ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}€"))
			self.current_ax.grid(True)
			for (x_val, y_val) in zip(x_data, y_data):
				self.current_ax.annotate(f"{y_val:.2f}€",
					(x_val, y_val),
					textcoords="offset points",
					xytext=(0, 6),
					ha="center",
					va="bottom",
					fontsize=8,
					color="blue")

		y_estimated: np.ndarray = estimator(x_data)
		self._draw_line(x_data, y_estimated, show_legend, **plot_args)
		self.current_ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}km"))
		self.current_ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}€"))
		self.current_ax = None

	def draw_rmse(self, x_data: np.ndarray, y_data: np.ndarray, index_graph=[0, 0]) -> None:
		self.set_current_ax(index_graph[0], index_graph[1])

		bars: BarContainer = self._draw_bars(x_data, y_data, title="RMSE", label=f"RMSE depending on learning rate")
		self.current_ax.set_xlabel("learning rate")
		self.current_ax.bar_label(bars, labels=[f"{b.get_height():.2f}€" for b in bars], padding=3, rotation=45)
		self.current_ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.4f}"))
		self.current_ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}€"))
		self.current_ax = None

	def draw_rsquared(self, x_data: np.ndarray, y_data: np.ndarray, index_graph=[0, 0]) -> None:
		self.set_current_ax(index_graph[0], index_graph[1])

		bars: BarContainer = self._draw_bars(x_data, y_data, title="R squared", label=f"R squared depending on learning rate")
		self.current_ax.set_xlabel("learning rate")
		self.current_ax.bar_label(bars, labels=[f"{b.get_height():.4f}" for b in bars], padding=3, rotation=45)
		self.current_ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.4f}"))
		self.current_ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
		self.current_ax = None
	
	def show(self) -> None:
		plt.show()

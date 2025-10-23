from ft_linear_regression.train import train_model_parallel, load_csv, TrainingData
from ft_linear_regression.plot import draw_scatter, show_data, plot_fit_line, init_graph, draw_bars, plot_line
import numpy as np

def show_results(trained_data: list[TrainingData], x_data: np.ndarray) -> None:
	# plot regression line for every learning rate
	for item in trained_data:
		print(item, "\n")
		plot_fit_line(x_data, tetha_1=item.tetha_1, tetha_0=item.tetha_0, color=item.color, label=f"learning rate: {item.learning_rate:.8f}")
	# plot bars of MSE and MAE for every learning rate
	x_data_bars: np.ndarray = np.array([train.learning_rate for train in trained_data], dtype=np.float64)
	y_mse_data_bars: np.ndarray = np.array([train.rmse for train in trained_data], dtype=np.float64)
	y_mae_data_bars: np.ndarray = np.array([train.mae for train in trained_data], dtype=np.float64)
	width_bars: np.float64 = 0.001
	draw_bars(x_data_bars - width_bars / 2, y_mse_data_bars, index_graph=1, width=width_bars, label=f"RMSE depending on learning rate")
	draw_bars(x_data_bars + width_bars / 2, y_mae_data_bars, index_graph=1, width=width_bars, label=f"MAE depending on learning rate")

	show_data()

def main():
	scale_factor: np.int64 = 1000_000

	# spawn learning rate values
	n_items: np.int64 = 50
	upper_limit: np.int64 = 0.05
	lower_limit: np.int64 = 0.001
	lr_values: list[np.float64] = np.linspace(lower_limit, upper_limit, n_items)

	# load x and y as np.arrays from csv
	csv_path: str = "./data.csv"
	x_data, y_data = load_csv(csv_path, scale_factor)

	# draw scatter points
	init_graph(nrows=2, ncols=1, figsize=(14, 10), layout="constrained")
	draw_scatter(x_data, y_data, color="green", label="points")

	n_procs: np.int64 = 30
	epochs: np.int64 = 100_000
	trained_data: list[TrainingData] = train_model_parallel(n_procs, x_data, y_data, epochs, lr_values)

	show_results(trained_data, x_data)


if __name__ == "__main__":
	main()

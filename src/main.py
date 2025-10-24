from ft_linear_regression.train import train_model_parallel, load_csv, TrainingData
from ft_linear_regression.plot import Plotter
import numpy as np

def show_results(x_data: np.ndarray, y_data: np.ndarray, trained_data: list[TrainingData]) -> None:
	plotter: Plotter = Plotter(nrows=2, ncols=2, figsize=(14, 10), layout="constrained")
	lower_limit: np.float64 = min([training.learning_rate for training in trained_data])
	upper_limit: np.float64 = max([training.learning_rate for training in trained_data])
	best_fit: TrainingData = max(trained_data, key=lambda train: train.r_determination)

	# plot regression line for every learning rate
	for item in trained_data:
		# print(item)
		plotter.draw_fit_line(x_data, y_data, item.estimate, index_graph=[0, 0], title=f"Regression lines with learning rates [{lower_limit}: {upper_limit}]", color=item.color, label=f"learning rate: {item.learning_rate:.4f}")
	# plot best fit line
	plotter.draw_fit_line(x_data, y_data, best_fit.estimate, index_graph=[0, 1], title=f"Best fit regression, learning rate: {best_fit.learning_rate:.3f}", show_legend=True, color="green", label=f"points")
	# plot bars of RMSE and R squared for every learning rate
	plotter.draw_rmse([train.learning_rate for train in trained_data], [train.rmse for train in trained_data], index_graph=[1, 0])
	plotter.draw_rsquared([train.learning_rate for train in trained_data], [train.r_determination for train in trained_data], index_graph=[1, 1])
	plotter.show()

def main():
	# spawn learning rate values
	n_items: np.int64 = 5
	lower_limit: np.int64 = 0.001
	upper_limit: np.int64 = 0.44
	lr_values: list[np.float64] = np.linspace(lower_limit, upper_limit, n_items)

	# load x and y as np.arrays from csv
	csv_path: str = "./data.csv"
	x_data, y_data = load_csv(csv_path)

	n_procs: np.int64 = 30
	epochs: np.int64 = 100_000
	trained_data: list[TrainingData] = train_model_parallel(n_procs, x_data, y_data, epochs, lr_values)

	show_results(x_data, y_data, trained_data)


if __name__ == "__main__":
	main()

from ft_linear_regression.train import train_model_parallel, train_model, load_csv, TrainingData
from ft_linear_regression.plot import Plotter
import numpy as np

def show_results(x_data: np.ndarray, y_data: np.ndarray, trained_data: list[TrainingData], is_logarithmic: bool) -> None:
	plotter: Plotter = Plotter(nrows=2, ncols=2, figsize=(14, 10), layout="constrained")
	# plot regression line for every learning rate
	lower_limit: np.float64 = min([training.learning_rate for training in trained_data])
	upper_limit: np.float64 = max([training.learning_rate for training in trained_data])
	for item in trained_data:
		plotter.draw_fit_line(x_data, y_data, item.estimate, index_graph=[0, 0], title=f"Regression lines with learning rates [{lower_limit:.4f}: {upper_limit:.4f}]", color=item.color, label=f"learning rate: {item.learning_rate:.4f}")

	# plot best fit line
	best_fit: TrainingData = max(trained_data, key=lambda train: train.r_determination)
	plotter.draw_fit_line(x_data, y_data, best_fit.estimate, index_graph=[0, 1], title=f"Best fit regression, learning rate: {best_fit.learning_rate:.4f}", show_legend=True, color="green", label=f"points")

	# plot bars of RMSE and R squared for every learning rate
	learning_rates: list[TrainingData] = [train.learning_rate for train in trained_data]
	plotter.draw_rmse_line(learning_rates, [train.rmse for train in trained_data], index_graph=[1, 0], is_logarithmic=is_logarithmic)
	plotter.draw_rsquared_line(learning_rates, [train.r_determination for train in trained_data], index_graph=[1, 1], is_logarithmic=is_logarithmic)
	plotter.show()

def main():
	# spawn learning rate values
	n_items: np.int64 = 30
	is_logarithmic: bool = True
	if is_logarithmic:
		lower_limit: np.int64 = -3
		upper_limit: np.int64 = 1
		lr_values = np.logspace(lower_limit, upper_limit, n_items)  # da 0.001 a 0.1 su scala logaritmica
	else:
		lower_limit: np.int64 = 0.015
		upper_limit: np.int64 = 0.019
		lr_values: list[np.float64] = np.linspace(lower_limit, upper_limit, n_items)

	# load x and y as np.arrays from csv
	csv_path: str = "./data.csv"
	x_data, y_data = load_csv(csv_path)

	n_procs: np.int64 = 30
	epochs: np.int64 = 100_000
	trained_data: list[TrainingData] = train_model_parallel(n_procs, x_data, y_data, epochs, lr_values)

	# show results
	for data in trained_data:
		print(data)
	# plot results
	show_results(x_data, y_data, trained_data, is_logarithmic)


if __name__ == "__main__":
	main()

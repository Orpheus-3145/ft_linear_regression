from ft_linear_regression.train import train_model, load_csv, estimate_y
from ft_linear_regression.plot import draw_scatter, show_data, plot_fit_line, init_graph
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from functools import partial

def main():
	scale_factor: int = 1000_000
	epochs: int = 100_000
	n_items: int = 30	# number of learning rates to spawn
	upper_limit: int = -1	# power of 10
	lower_limit: int = -3	# power of 10
	# spawn 'n_items' values for learning rate equally (logaritmically) spread between lower_limit and upper_limit
	lr_values: list = np.logspace(lower_limit, upper_limit, n_items)

	# load x and y as np.arrays from csv
	csv_path: str = "./data.csv"
	x_data, y_data = load_csv(csv_path, scale_factor)

	# draw scatter points
	init_graph(nrows=1, ncols=1, figsize=(14, 8), layout="constrained")
	draw_scatter(x_data, y_data, color="green", label="points")

	# train model for every value of learning rate
	_train_model = partial(train_model, x_data, y_data, epochs)
	with ProcessPoolExecutor(30) as pool:
		return_data = list(pool.map(_train_model, lr_values))

	# plot results
	test_value: float = 100_000.
	for item in return_data:
		print(f"learning rate: {item["learning_rate"]} - t0: {item["tetha_0"]} - t1: {item["tetha_1"]}")
		print(f"estimated price for {test_value}km: {(estimate_y(item["tetha_1"], item["tetha_0"], test_value / scale_factor) * scale_factor):.2f}€\n")
		plot_fit_line(x_data, item)
	show_data()

if __name__ == "__main__":
    main()

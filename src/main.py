from ft_linear_regression.estimate import estimate_price
from ft_linear_regression.train import train_model, load_csv
from ft_linear_regression.plot import draw_scatter_df, show_data, plot_fit_line_df, init_graph
import pandas as pd
import ft_linear_regression.config as config

CSV_PATH = "./data.csv"

if __name__ == "__main__":
	epochs: int = 100
	test_value: int = 100000
	n_graphs: int = 2

	init_graph(nrows=1, ncols=n_graphs, figsize=(14, 8), layout="constrained")

	dataset: pd.DataFrame = load_csv(CSV_PATH)
	draw_scatter_df(dataset, color='green', label='points')
	print(f"estimating price for {test_value}km (pre-training): {estimate_price(test_value)}€")
	plot_fit_line_df(dataset, color='blue', label='reg line (pre-training)')
	
	while (True):
		tmp_t_0 = config.TETHA_0
		tmp_t_1 = config.TETHA_1
		train_model(dataset)
		if (abs(tmp_t_0 - config.TETHA_0) + abs(tmp_t_1 - config.TETHA_1)) < 2 * 1e-6:
			break
		epochs -= 1
		if (epochs == 0):
			break
		print(f"{epochs} - TETHA_1: {config.TETHA_1} TETHA_0: {config.TETHA_0}")
		# plot_fit_line_df(dataset, color='yellow', label=f'{epochs}')
	
	print(f"estimating price for {test_value}km (post-training): {estimate_price(test_value)}€")
	plot_fit_line_df(dataset, color='red', label='reg line (post-training)')
	show_data()

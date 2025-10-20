import pandas as pd
import numpy as np
from .estimate import linear_regression
from . import config
from .plot import draw_scatter

def load_csv(csv_path: str) -> pd.DataFrame:
	data = pd.read_csv(csv_path, delimiter=",", header=0)
	data.km /= 1000
	return data.sort_values(by="km", ascending=False)

def train_model(dataset: pd.DataFrame) -> None:
	learning_rate: float = config.LEARNING_RATE
	population: int = len(dataset)
	sum_tetha_0: float = .0
	sum_tetha_1: float = .0
	# diffs: np.ndarray = [0] * population

	for row in dataset.itertuples(index=True):
		if row.Index == population - 1:
			break
		estimated_price: float = linear_regression(config.TETHA_1, config.TETHA_0, row.km)
		sum_tetha_0 += row.price - estimated_price
		sum_tetha_1 += (row.price - estimated_price) * row.km
		# diffs[row.Index] = estimated_price - row.price

	# X = dataset.km.to_numpy()
	# draw_scatter(X, diffs, index_graph=1, color='orange', label='deltas', s=10)

	config.TETHA_0 -= learning_rate * sum_tetha_0 / population
	config.TETHA_1 -= learning_rate * sum_tetha_1 / population

import pandas as pd
from .estimate import linear_regression
from . import config

def load_csv(csv_path: str) -> pd.DataFrame:
	return pd.read_csv(csv_path, delimiter=",", header=0)

def train_model(csv_path: str) -> None:
	dataset: pd.DataFrame = load_csv(csv_path)
	tmp_tetha_0: float = config.TETHA_0
	tmp_tetha_1: float = config.TETHA_1
	learing_rate: float = config.LEARNING_RATE
	population: int = len(dataset)
	tmp_sum_tetha_0: float = .0
	tmp_sum_tetha_1: float = .0
	
	for row in dataset.itertuples(index=True):
		if row.index == population - 1:
			break

		tmp_sum_tetha_0 += (linear_regression(tmp_tetha_0, tmp_tetha_1, row.km) - row.price)
		tmp_sum_tetha_1 += (linear_regression(tmp_tetha_0, tmp_tetha_1, row.km) - row.price) * row.km
		# print(f"tmp_t_0: {tmp_sum_tetha_0} - tmp_t_1: {tmp_sum_tetha_1}")

	config.TETHA_0 = learing_rate * tmp_sum_tetha_0 / population
	config.TETHA_1 = learing_rate * tmp_sum_tetha_1 / population
	print(f"res_t_0: {config.TETHA_0} res_t_1: {config.TETHA_1}")

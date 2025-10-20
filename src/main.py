from ft_linear_regression.estimate import estimate_price
from ft_linear_regression.train import train_model

CSV_PATH = "./data.csv"

if __name__ == "__main__":
	print(f"estimating price for 1500km (pre-training): {estimate_price(1500)}€")
	train_model(CSV_PATH)
	print(f"estimating price for 1500km (post-training): {estimate_price(1500)}€")

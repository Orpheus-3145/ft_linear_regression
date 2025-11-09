import numpy as np
import argparse

from ft_linear_regression.train import TrainingData


def main():
	tetha_0: np.float64 = .0
	tetha_1: np.float64 = .0

	# read file created from training
	try:
		with open(".tethas") as f:
			tetha_0 = np.float64(next(f).split("=", 1)[1])
			tetha_1 = np.float64(next(f).split("=", 1)[1])
	except FileNotFoundError:
		# file doesn't exist -> model is not trained -> use default values
		pass
	except Exception as error:
		print(f"Error while reading file that contains tetha_0 and tetha_1:\n\t{error}")
		return

	# parse arguments
	try:
		parser = argparse.ArgumentParser()
		parser.add_argument("--input", type=np.float64, required=True)
		args = parser.parse_args()
	except SystemExit:
		return

	fit_data: TrainingData = TrainingData(tetha_0, tetha_1)
	estimated_value: np.float64 = fit_data.estimate(args.input)

	print(f"""
tetha0: {fit_data.tetha_0:15.4f}
tetha1: {fit_data.tetha_1:15.4f}
input feature: {args.input:.2f}km
 _________________________________________ 
|                                         |
| estimated target: {estimated_value:20.2f}€ |
|_________________________________________|
""")


if __name__ == "__main__":
	main()

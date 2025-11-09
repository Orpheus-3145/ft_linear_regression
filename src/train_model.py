import numpy as np
import argparse
import os

from ft_linear_regression.train import TrainerLR, TrainingData


def main():
	epochs: np.int64 = 1_000

	trainer: TrainerLR = TrainerLR(epochs)
	trainer.load_csv("./data.csv")

	# parse arguments
	try:
		parser = argparse.ArgumentParser()
		parser.add_argument("--learning_rate", type=np.float64, default=0.01)
		args = parser.parse_args()
	except SystemExit:
		return

	learning_rate: np.float64 = args.learning_rate
	trainer.set_learning_rate(learning_rate)

	trainer.standardize()
	# start training
	trainer.train_model(store_progression=True)

	print(f"saving best fit tetha0 and tetha1 in {os.getcwd()}/.tethas")
	best_fit: TrainingData = trainer.get_best_fit()
	with open(".tethas", "w") as f:
		f.write(f"tetha0={best_fit.tetha_0}\ntetha1={best_fit.tetha_1}\n")

	# show results
	trainer.show_data()
	# plot results
	trainer.plot_data_single(learning_rate)


if __name__ == "__main__":
	main()

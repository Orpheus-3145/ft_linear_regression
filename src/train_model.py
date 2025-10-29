import numpy as np
import os

from ft_linear_regression.train import TrainerLR, TrainingData


def main():
	epochs: np.int64 = 1_00_000

	trainer: TrainerLR = TrainerLR(epochs)
	trainer.load_csv("./data.csv")

	# spawn learning rate values
	n_items: np.int64 = 5
	min: np.float64 = -5
	max: np.float64 = -1
	trainer.generate_lr_range(min, max, n_items, linear_scale=False)

	trainer.standardize()
	# start training
	n_process: np.int64 = 40
	trainer.train_model_parallel(n_process)

	print(f"\nsaving best fit tetha0 and tetha1 in {os.getcwd()}/.tethas")
	best_fit: TrainingData = trainer.get_best_fit()
	with open(".tethas", "w") as f:
		f.write(f"tetha0={best_fit.tetha_0}\ntetha1={best_fit.tetha_1}\n")

	# show results
	trainer.show_data()
	# plot results
	trainer.plot_data()



if __name__ == "__main__":
	main()

from ft_linear_regression.train import TrainerLR
import numpy as np


def main():
	epochs: int = 100_000

	trainer: TrainerLR = TrainerLR(epochs)
	# load x and y as np.arrays from csv
	csv_path: str = "./data.csv"
	trainer.load_csv(csv_path)
	# spawn learning rate values
	min: np.float64 = .01
	max: np.float64 = .04
	n_items: int = 20
	trainer.generate_lr_range(min, max, n_items, True)
	trainer.standardize()
	# start training
	# trainer.train_model()
	n_process: int = 30
	trainer.train_model_parallel(n_process)
	# show results
	trainer.show_data()
	# plot results
	trainer.plot_data()


if __name__ == "__main__":
	main()

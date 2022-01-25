import sys
sys.path.append('./deep_learning_model/')
from solver import Solver
from data_loader import get_loader_3D
import config
import numpy as np
import warnings
from matplotlib import pyplot as plt

if __name__ == '__main__':
	config = config.config()
	warnings.filterwarnings('once')

	train_loader_3D = get_loader_3D(input_folder = config.train_input_folder,
						  GT_folder = config.train_GT_folder,
						  batch_size = config.batch_size,
						  num_workers = config.num_workers,
						  shuffle = True,
						  mode = 'train',
						  augment = config.augment,
						  augmentation_option = config.augmentation_option)

	validation_loader_3D = get_loader_3D(input_folder = config.validation_input_folder,
							   GT_folder = config.validation_GT_folder,
							   batch_size = config.batch_size,
							   num_workers = config.num_workers,
							   shuffle = True,
							   mode = 'validation',
							   augment = False)

	test_loader_3D = get_loader_3D(input_folder = config.test_input_folder,
							   GT_folder = config.test_GT_folder,
							   batch_size = config.batch_size,
							   num_workers = config.num_workers,
							   shuffle = False,
							   mode = 'test',
							   augment = False)

	solver = Solver(config, train_loader_3D, validation_loader_3D, test_loader_3D)
	#solver.load_pretrained_model(config.pretrained_model_path, True)
	train_loss_history, validation_loss_history = solver.train()
	plt.plot(train_loss_history)
	plt.plot(validation_loss_history)
	plt.show()

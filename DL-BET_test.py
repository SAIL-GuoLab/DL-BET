import sys
sys.path.append('./deep_learning_model/')
from solver import Solver
from data_loader import get_loader_3D
import config
import warnings
from matplotlib import pyplot as plt
import argparse

def main(argv, config):
	parser = argparse.ArgumentParser(description = "hahaha", add_help = True)
	opt_group = parser.add_argument_group("Optional arguments")
	opt_group.add_argument("-m", "--model", help = "Which model? default is best")
	args = parser.parse_args()

	#config = config.config()
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
	if not args.model is None:
		solver.test(which_unet = args.model)
	else:
		solver.test(which_unet='best')

if __name__ == '__main__':
	main(sys.argv[1:], config.config())
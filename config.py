from config_parser import Config_Parser

class config():
	conf_settings = Config_Parser()
	print('[TRAIN]')
	for key in conf_settings['TRAIN']:
		print(key, ':', conf_settings['TRAIN'][key])

	print('\n[GENERAL]')
	for key in conf_settings['GENERAL']:
		print(key, ':', conf_settings['GENERAL'][key])

	print('\n[MODEL]')
	for key in conf_settings['MODEL']:
		print(key, ':', conf_settings['MODEL'][key])

	print('\n[FOLDERS]')
	for key in conf_settings['FOLDERS']:
		print(key, ':', conf_settings['FOLDERS'][key])

	print('\n[RESULTS]')
	for key in conf_settings['RESULTS']:
		print(key, ':', conf_settings['RESULTS'][key])

	train = conf_settings['TRAIN']
	model = conf_settings['MODEL']
	general = conf_settings['GENERAL']
	folders = conf_settings['FOLDERS']
	results = conf_settings['RESULTS']

	img_ch = int(train['img_ch'])
	output_ch = int(train['output_ch']) 
	num_epochs = int(train['num_epochs'])
	num_workers = int(train['num_workers'])
	main_axis = int(train['main_axis'])

	model_type = model['model_type']
	initialization = model['initialization']
	optimizer_choice = model['optimizer_choice']
	corregistration = model['corregistration']
	initial_lr = model['initial_lr']
	loss_function_name = model['loss_function_name']
	loss_function_lr = model['loss_function_lr']
	batch_size = int(model['batch_size'])
	clipped_gradient = model['clipped_gradient']
	gradient_max_norm = model['gradient_max_norm']
	early_stop = model['early_stop']
	cuda_idx = int(model['cuda_idx'])
	adaptive_lr = model['adaptive_lr']

	UnetLayer = 6 # This is only implemented in ResAttU_Net
	first_layer_numKernel = 2 # How many kernels in the first convolutional layer? Will be doubled every layer downward.

	mode = general['mode']
	augment = general['augment']
	augmentation_option = general['augmentation_option']

	train_input_folder = folders['train_input_folder']
	validation_input_folder = folders['validation_input_folder']
	test_input_folder = folders['test_input_folder']
	train_GT_folder = folders['train_gt_folder']
	validation_GT_folder = folders['validation_gt_folder']
	test_GT_folder = folders['test_gt_folder']

	model_saving_path = results['model_saving_path']
	predictions_path = results['predictions_path']
	loss_histories_path = results['loss_history_path']
	test_result_comparison_path = results['test_result_comparison_path']

	current_prediction_path = predictions_path 
	current_model_saving_path = model_saving_path
	current_loss_history_path = loss_histories_path

	if optimizer_choice == 'Adam':
		beta1 = float(0.5) # momentum1 in Adam
		beta2 = float(0.999) # momentum2 in Adam
	elif optimizer_choice == 'SGD':
		momentum = float(0.9)

def get_parameter(dataset):
    if dataset == 'mnist':
        # Parameters for mnist
        parameter_dic = {
            'dataset' : 'mnist', #'CIFAR10',
            'D' : 2000,
            'stationary' : False,
            'back_propagation' : True,
            'std1' : 0.0625, 
            'std2' : 0.03125,
            'batch_size' : 32, 
            'T' : 10,
            'record_batch' : 200,
            'loss_type' : 'cross_entroy', # 'hinge' 'mse'
            'lambda_A' : 1e-4, 
            'lambda_B' : 1e-4, 
            'regular_type' : 'nuc', # 'fro'
            'learning_rate' : 1e-4,
            'validate' : True
        }
    elif dataset == 'letter':
        # Parameters for letter
        parameter_dic = {
            'dataset' : 'letter', #'CIFAR10',
            'D' : 2000,
            'stationary' : False,
            'back_propagation' : True,
            'std1' : 0.05, 
            'std2' : 0.05,
            'batch_size' : 32, 
            'T' : 10,
            'record_batch' : 100,
            'loss_type' : 'cross_entroy', # 'hinge' 'mse'
            'lambda_A' : 1e-4, 
            'lambda_B' : 1e-4, 
            'regular_type' : 'nuc', # 'fro'
            'learning_rate' : 1e-2,
            'validate' : True
        }
    elif dataset == 'usps':
        # Parameters for usps
        parameter_dic = {
            'dataset' : 'usps', 
            'D' : 2000,
            'stationary' : False,
            'back_propagation' : True,
            'std1' : 0.125, 
            'std2' : 0.125,
            'batch_size' : 32, 
            'T' : 10,
            'record_batch' : 100,
            'loss_type' : 'cross_entroy', # 'hinge' 'mse'
            'lambda_A' : 10, 
            'lambda_B' : 0.01, 
            'regular_type' : 'nuc', # 'fro'
            'learning_rate' : 1e-3,
            'validate' : True
        }
    elif dataset == 'shuttle':
        # Parameters for shuttle
        parameter_dic = {
            'dataset' : 'shuttle', 
            'D' : 2000,
            'stationary' : False,
            'back_propagation' : True,
            'std1' : 0.015625, 
            'std2' : 0.015625,
            'batch_size' : 32, 
            'T' : 10,
            'record_batch' : 100,
            'loss_type' : 'cross_entroy', # 'hinge' 'mse'
            'lambda_A' : 1e-3, 
            'lambda_B' : 1e-3, 
            'regular_type' : 'nuc', # 'fro'
            'learning_rate' : 1e-2,
            'validate' : True
        }
    elif dataset == 'pendigits':
        parameter_dic = {
            'dataset' : 'pendigits', 
            'D' : 2000,
            'stationary' : False,
            'back_propagation' : True,
            'std1' : 0.05, 
            'std2' : 0.05,
            'batch_size' : 32, 
            'T' : 10,
            'record_batch' : 100,
            'loss_type' : 'cross_entroy', # 'hinge' 'mse'
            'lambda_A' : 1e-3, 
            'lambda_B' : 1e-3, 
            'regular_type' : 'nuc', # 'fro'
            'learning_rate' : 1e-2,
            'validate' : True
        }
    elif dataset == 'segment':
        parameter_dic = {
            'dataset' : 'segment', 
            'D' : 2000,
            'stationary' : False,
            'back_propagation' : True,
            'std1' : 0.05, 
            'std2' : 0.05,
            'batch_size' : 32, 
            'T' : 10,
            'record_batch' : 10,
            'loss_type' : 'cross_entroy', # 'hinge' 'mse'
            'lambda_A' : 1e-3, 
            'lambda_B' : 1e-3, 
            'regular_type' : 'nuc', # 'fro'
            'learning_rate' : 1e-2,
            'validate' : True
        }
    elif dataset == 'poker':
        parameter_dic = {
            'dataset' : dataset, 
            'D' : 2000,
            'stationary' : False,
            'back_propagation' : True,
            'std1' : 0.125, 
            'std2' : 0.125,
            'batch_size' : 32, 
            'T' : 10,
            'record_batch' : 100,
            'loss_type' : 'cross_entroy', # 'hinge' 'mse'
            'lambda_A' : 1e-3, 
            'lambda_B' : 1e-3, 
            'regular_type' : 'nuc', # 'fro'
            'learning_rate' : 1e-2,
            'validate' : True
        }
    elif dataset == 'abalone':
        parameter_dic = {
            'dataset' : dataset, 
            'D' : 2000,
            'stationary' : False,
            'back_propagation' : True,
            'std1' : 0.125, 
            'std2' : 0.125, 
            'batch_size' : 32, 
            'T' : 10,
            'record_batch' : 100,
            'loss_type' : 'cross_entroy', # 'hinge' 'mse'
            'lambda_A' : 1e-3, 
            'lambda_B' : 1e-3, 
            'regular_type' : 'nuc', # 'fro'
            'learning_rate' : 1e-2,
            'validate' : True
        }
    elif dataset == 'space_ga':
        parameter_dic = {
            'dataset' : dataset, 
            'D' : 2000,
            'stationary' : False,
            'back_propagation' : True,
            'std1' : 0.03125, 
            'std2' : 0.03125, 
            'batch_size' : 32, 
            'T' : 10,
            'record_batch' : 100,
            'loss_type' : 'cross_entroy', # 'hinge' 'mse'
            'lambda_A' : 1e-3, 
            'lambda_B' : 1e-3, 
            'regular_type' : 'nuc', # 'fro'
            'learning_rate' : 1e-2,
            'validate' : True
        }
    elif dataset == 'cpusmall':
        parameter_dic = {
            'dataset' : dataset, 
            'D' : 2000,
            'stationary' : False,
            'back_propagation' : True,
            'std1' : 0.015625, 
            'std2' : 0.015625, 
            'batch_size' : 32, 
            'T' : 10,
            'record_batch' : 100,
            'loss_type' : 'cross_entroy', # 'hinge' 'mse'
            'lambda_A' : 1e-3, 
            'lambda_B' : 1e-3, 
            'regular_type' : 'nuc', # 'fro'
            'learning_rate' : 1e-2,
            'validate' : True
        }
    elif dataset == 'cadata':
        parameter_dic = {
            'dataset' : dataset, 
            'D' : 2000,
            'stationary' : False,
            'back_propagation' : True,
            'std1' : 0.0078125,
            'std2' : 0.0078125, 
            'batch_size' : 32, 
            'T' : 10,
            'record_batch' : 100,
            'loss_type' : 'cross_entroy', # 'hinge' 'mse'
            'lambda_A' : 1e-5, 
            'lambda_B' : 1e-4, 
            'regular_type' : 'nuc', # 'fro'
            'learning_rate' : 1e-2,
            'validate' : True
        }
    else:
        parameter_dic = {
            'dataset' : dataset, 
            'D' : 2000,
            'stationary' : False,
            'back_propagation' : True,
            'std1' : 0.05, 
            'std2' : 0.05,
            'batch_size' : 32, 
            'T' : 1,
            'record_batch' : 100,
            'loss_type' : 'cross_entroy', # 'hinge' 'mse'
            'lambda_A' : 1e-3, 
            'lambda_B' : 1e-3, 
            'regular_type' : 'nuc', # 'fro'
            'learning_rate' : 1e-2,
            'validate' : True
        }
    return parameter_dic
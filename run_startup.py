from core_functions.auto_kernel_learning import askl
import core_functions.optimal_parameters as optimal_parameters
import os
import pickle

parameter_dic = optimal_parameters.get_parameter('poker')
askl(parameter_dic.copy())

parameter_dic['stationary'] = True
parameter_dic['back_propagation'] = False
parameter_dic['regular_type'] = 'fro'
askl(parameter_dic.copy())
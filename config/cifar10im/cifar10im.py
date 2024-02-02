config = {}

active_opt = {}
active_opt['trials'] = 3 # 5
active_opt['cycles'] = 5
active_opt['budget'] = 1000
active_opt['subset_size'] = 10000

config['active_opt'] = active_opt

training_opt = {}
training_opt['dataset'] = 'cifar10im'
training_opt['num_train'] = 50000
training_opt['num_classes'] = 10
training_opt['batch_size'] = 128
training_opt['num_workers'] = 4
training_opt['num_epochs'] = 200
training_opt['num_epochs_loss'] = 120
training_opt['num_epochs_vaal'] = 100
training_opt['start_epoch'] = 0
training_opt['margin'] = 1.0
training_opt['weight'] = 1.0
training_opt['adversarial'] = 1

config['training_opt'] = training_opt

optimizer_opt = {}
optimizer_opt['type'] = 'SGD'
optimizer_opt['optim_params'] = {'lr': 0.1,
                             'momentum': 0.9,
                             'weight_decay': 5e-4}
optimizer_opt['optim_module_params'] = {'lr': 0.1,
                             'momentum': 0.9,
                             'weight_decay': 5e-4}
optimizer_opt['scheduler'] = 'step'
optimizer_opt['scheduler_params'] = {'step': [160, 240],
                          'gamma': 0.1}
config['optimizer_opt'] = optimizer_opt

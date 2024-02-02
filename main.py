
# Python
import os
import random
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as T
import torchvision.models as models

# Custom
import models.resnet as resnet
import models.vgg as vgg
from models.query_models import LossNet
from train_test import train, test, test_ema, test_swa, swa_optim, ConstAnnealingWarmRestarts
from load_dataset import load_dataset, TwoStreamBatchSampler
from selection_methods import query_samples
from utils import source_import, make_dir

parser = argparse.ArgumentParser()
parser.add_argument("-l","--lambda_loss",type=float, default=1.2,
                    help="Adjustment graph loss parameter between the labeled and unlabeled")
parser.add_argument("-s","--s_margin", type=float, default=0.1,
                    help="Confidence margin of graph")
parser.add_argument("-n","--hidden_units", type=int, default=128,
                    help="Number of hidden units of the graph")
parser.add_argument("-r","--dropout_rate", type=float, default=0.3,
                    help="Dropout rate of the graph neural network")
parser.add_argument("-d","--dataset", type=str, default="cifar10",
                    help="")
parser.add_argument("-e","--num_epochs", type=int, default=200,
                    help="Number of epochs for the active learner")
parser.add_argument("-m","--method_type", type=str, default="ST-CoNAL",
                    help="")
parser.add_argument("-c","--cycles", type=int, default=5,
                    help="Number of active learning cycles")
parser.add_argument("-b", "--budget", type=int, default=1000, help="Size of active learning budget")
parser.add_argument("--subset-size", "--subset", type=int, default=10000, help="Size of unlabeled subset")
parser.add_argument("-t","--total", type=bool, default=False, help="Training on the entire dataset")
parser.add_argument("--config", type=str, default="./config/cifar10/cifar10.py", help="Configuration root")
parser.add_argument("--margin", type=float, default=1.0, help="margin for TA-VAAL or lloss or lloss_v2")
parser.add_argument("--weight", type=float, default=1.0, help="weight for TA-VAAL or loss or lloss_v2")
parser.add_argument('--gamma', type=float, default=1.0, help="gamma for multi step learning rate schedule")
parser.add_argument('--num_iter', type=int, default=25, help="number of forward passes in CSSAL or MCdropout")

## new parser for ST-CoNAL
parser.add_argument('--log_root', default='./results')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--lr_type', default='lr_multistep', choices=['lr_multistep', 'lr_CLR_cosine', 'lr_constwarmup_CLR_cosine'])
parser.add_argument('--lr_interval', type=int, default=10)
parser.add_argument('--lr_min', type=float, default=0)
parser.add_argument('--additional_epochs', default=0, type=int, help='additional training_epochs')
parser.add_argument('--swa_start', type=int, default=160)
parser.add_argument('--sharpening_temp', default=0.7, type=float)
parser.add_argument('--use_balanced_init', action='store_true')
parser.add_argument('--use_no_subset', action='store_true')
parser.add_argument('--num_students', type=int, default=5)

# SSL
parser.add_argument('--use_semi_learning', action='store_true')
parser.add_argument('--labeled_batch_size', type=int, default=32)
parser.add_argument('--ema_decay', default=0.97)
parser.add_argument('--consistency', default=100.0)
parser.add_argument('--consistency_rampup', default=5)
parser.add_argument('--consistency_type', default='mse')


parser.add_argument('--SWA_inference', action='store_true')
parser.add_argument('--linear_gamma', action='store_true')
parser.add_argument('--exp-str', type=str, default='')
parser.add_argument("--gpu", default=0)

args = parser.parse_args()
config = source_import(args.config).config

## Main
if __name__ == '__main__':

    print(config)
    active_opt = config['active_opt']
    training_opt = config['training_opt']
    optimizer_opt = config['optimizer_opt']
    if args.gamma:
        optimizer_opt['scheduler_params']['gamma'] = args.gamma
    method = args.method_type
    methods = ['Random', 'ST-CoNAL', 'ST-CoNAL-EMA']
    datasets = ['cifar10', 'cifar10im', 'cifar10imlong', 'cifar100', 'cifar100im', 'cifar100imlong', 'fashionmnist', 'svhn', 'tiny_imagenet', 'caltech256']
    dataset = training_opt['dataset']
    args.dataset = dataset
    assert method in methods, 'No method %s! Try options %s'%(method, methods)
    assert dataset in datasets, 'No dataset %s! Try options %s'%(dataset, datasets)
    
    print("Dataset: %s"%dataset)
    print("Method type:%s"%method)

    BUDGET = args.budget
    SUBSET_SIZE = args.subset_size
    TRIALS, CYCLES = active_opt['trials'], active_opt['cycles']
    BATCH = training_opt['batch_size']
    args.num_epochs = training_opt['num_epochs']
    args.epochs_loss = training_opt['num_epochs_loss']
    if args.additional_epochs:
        assert args.additional_epochs % args.lr_interval == 0, 'please choose correct additional_epochs'
        args.num_epochs += args.additional_epochs
    args.swa_start = optimizer_opt['scheduler_params']['step'][0]
    if dataset in ['cifar10', 'cifar100']:
        args.num_features = 512
    elif dataset in ['tiny_imagenet', 'caltech', 'clothing1M']:
        args.num_features = 512
    log_root = args.log_root + '/' + dataset + '_' + str(args.method_type) + '_c' + str(CYCLES) + '_b' + str(BUDGET) +\
               '_s' + str(SUBSET_SIZE) + '_gamma' + str(optimizer_opt['scheduler_params']['gamma']) + '_' + args.exp_str
    index_root = log_root + '/indexes'
    args.log_root = log_root
    make_dir(log_root)
    make_dir(index_root)
    results = open(log_root + '/' + str(args.method_type) + '_log.txt', 'w')
    args.gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.total:
        TRIALS = 1
        CYCLES = 1
    else:
        CYCLES = args.cycles
    for trial in range(TRIALS):
        # Load training and testing dataset
        data_train, data_unlabeled, data_test, NUM_CLASSES, no_train = load_dataset(dataset, args)
        print('The entire datasize is {}'.format(len(data_train)))
        NUM_TRAIN = no_train
        args.num_classes = NUM_CLASSES
        indices = list(range(NUM_TRAIN))
        with open(os.path.join(log_root, 'args.txt'), 'w') as f:
            f.write(str(args))
        with open(os.path.join(log_root, 'config.txt'), 'w') as f:
            f.write(str(config))

        if args.total:
            labeled_set= indices
        elif args.use_balanced_init:
            labeled_set = []
            for cls in range(NUM_CLASSES):
                labeled_set.extend(list(np.random.choice(np.where(np.array(data_train.targets)==cls)[0], int(BUDGET/NUM_CLASSES), replace=False)))
            unlabeled_set = [x for x in indices if x not in labeled_set]
        else:
            random.shuffle(indices)
            labeled_set = indices[:BUDGET]
            unlabeled_set = [x for x in indices if x not in labeled_set]

            # TODO: test
            args.labeled_set = labeled_set
            args.unlabeled_set = unlabeled_set

        for cycle in range(CYCLES):
            if args.use_semi_learning:
                train_loader = DataLoader(data_train,
                                          batch_sampler=TwoStreamBatchSampler(labeled_set, unlabeled_set, BATCH,
                                                                              args.labeled_batch_size),
                                          pin_memory=True)
            else:
                train_loader = DataLoader(data_train, batch_size=BATCH,
                                          sampler=SubsetRandomSampler(labeled_set),
                                          pin_memory=True)

            test_loader = DataLoader(data_test, batch_size=BATCH)
            dataloaders = {'train': train_loader, 'test': test_loader}

            with open(index_root + '/' + dataset + '_' + str((cycle+1)*BUDGET) + '_' + str(trial+1) + '.pickle', 'wb') as f:
                pickle.dump({'labeled_set': labeled_set, 'unlabeled_idxs': unlabeled_set,
                             'num_classes': NUM_CLASSES, 'dataset': dataset,
                             'num_train': NUM_TRAIN} , f, protocol=pickle.HIGHEST_PROTOCOL)

            # Randomly sample 10000 unlabeled data points
            if args.linear_gamma:
                optimizer_opt['scheduler_params']['gamma'] = args.gamma - cycle * (args.gamma-0.1)/(CYCLES-1)

            if args.use_no_subset:
                random.shuffle(unlabeled_set)
                subset = unlabeled_set
                SUBSET_SIZE = len(unlabeled_set)
                args.subset_size = SUBSET_SIZE
            else:
                random.shuffle(unlabeled_set)
                subset = unlabeled_set[:SUBSET_SIZE]

            # Model - create new instance for every cycle so that it resets
            if dataset in ['caltech256', 'tiny_imagenet']:
                feature_sizes = [56, 28, 14, 1]
                num_channels = [64, 128, 256, 512]
            else:
                if training_opt['architecture'] == 'vgg16':
                    feature_sizes = [8, 4, 2, 1]
                    num_channels = [128, 256, 512, 512]
                else:
                    feature_sizes = [32, 16, 8, 4]
                    num_channels = [64, 128, 256, 512]

            with torch.cuda.device(args.gpu):
                if dataset == "fashionmnist":
                    task_model    = resnet.ResNet18fm(num_classes=NUM_CLASSES).cuda()
                elif dataset in ['caltech256', 'tiny_imagenet']:
                    if training_opt['architecture'] == 'vgg16':
                        task_model = vgg.vgg16(False, progress=False, num_classes=NUM_CLASSES).cuda()
                        if args.use_semi_learning:
                            task_model_ema = vgg.vgg16(False, progress=False, num_classes=NUM_CLASSES, no_grad=True).cuda()
                    else:
                        task_model = resnet.ResNet18_img(num_classes=NUM_CLASSES, pretrained=True).cuda()
                        if args.use_semi_learning:
                            task_model_ema = resnet.ResNet18_img(num_classes=NUM_CLASSES, no_grad=True).cuda()
                else:  # CIFAR
                    if training_opt['architecture'] == 'vgg16':
                        task_model = vgg.vgg16(False, progress=False, num_classes=NUM_CLASSES).cuda()
                        if args.use_semi_learning:
                            task_model_ema = vgg.vgg16(False, progress=False, num_classes=NUM_CLASSES, no_grad=True).cuda()
                    elif training_opt['architecture'] == 'resnet50':
                        task_model    = resnet.ResNet50(num_classes=NUM_CLASSES).cuda()
                    else:
                        task_model    = resnet.ResNet18(num_classes=NUM_CLASSES).cuda()
                        if args.use_semi_learning:
                            task_model_ema = resnet.ResNet18(num_classes=NUM_CLASSES, no_grad=True).cuda()

                if method == 'ST-CoNAL' or args.SWA_inference:
                    if dataset == "fashionmnist":
                        task_model_swa = resnet.ResNet18fm(num_classes=NUM_CLASSES, no_grad=True).cuda()
                    elif dataset in ['caltech256', 'tiny_imagenet']:
                        task_model_swa = resnet.ResNet18_img(num_classes=NUM_CLASSES, no_grad=True).cuda()
                    else:
                        if training_opt['architecture'] == 'vgg16':
                            task_model_swa = vgg.vgg16(False, progress=False, num_classes=NUM_CLASSES, no_grad=True).cuda()
                        elif training_opt['architecture'] == 'resnet50':
                            task_model_swa = resnet.ResNet50(num_classes=NUM_CLASSES, no_grad=True).cuda()
                        else:
                            task_model_swa = resnet.ResNet18(num_classes=NUM_CLASSES, no_grad=True).cuda()
                elif method == 'ST-CoNAL-EMA':
                    if dataset == "fashionmnist":
                        task_model_ema = resnet.ResNet18fm(num_classes=NUM_CLASSES, no_grad=True).cuda()
                    elif dataset in ['caltech256', 'tiny_imagenet']:
                        task_model_ema = resnet.ResNet18_img(num_classes=NUM_CLASSES, no_grad=True).cuda()
                    else:
                        task_model_ema = resnet.ResNet18(num_classes=NUM_CLASSES, no_grad=True).cuda()

            # models      = {'backbone': torch.nn.DataParallel(resnet18).to(args.gpu)}
            models      = {'backbone': task_model}
            if method in ['ST-CoNAL'] or args.SWA_inference:
                models['backbone_swa'] = task_model_swa
            elif method in ['ST-CoNAL-EMA'] or args.use_semi_learning:
                models['backbone_ema'] = task_model_ema
            else:
                raise ValueError('No model for method %s'%method)
            
            torch.backends.cudnn.benchmark = True

            # Loss, criterion and scheduler (re)initialization
            criterion      = nn.CrossEntropyLoss(reduction='none')
            optim_params = optimizer_opt['optim_params']
            optim_backbone = optim.SGD(models['backbone'].parameters(), lr=optim_params['lr'],
                momentum=optim_params['momentum'], weight_decay=optim_params['weight_decay'])
            if args.lr_type == 'lr_CLR_cosine':
                sched_backbone = lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optim_backbone,
                                                                          T_0=(int(len(train_loader)/BATCH)+1)*args.lr_interval,
                                                                          T_mult=1,
                                                                          eta_min=args.lr_min)
            elif args.lr_type == 'lr_multistep':
                sched_backbone = lr_scheduler.MultiStepLR(optimizer=optim_backbone,
                                                          milestones=optimizer_opt['scheduler_params']['step'],
                                                          gamma=optimizer_opt['scheduler_params']['gamma'])
            elif args.lr_type == 'lr_constwarmup_CLR_cosine':
                sched_backbone = ConstAnnealingWarmRestarts(optimizer=optim_backbone,
                                                           T_0=args.swa_start,
                                                           base_lr=optim_params['lr'],
                                                           lr_min=args.lr_min,
                                                           lr_interval=args.lr_interval,
                                                           gamma=optimizer_opt['scheduler_params']['gamma'])

            optimizers = {'backbone': optim_backbone}
            schedulers = {'backbone': sched_backbone}
            if method in ['ST-CoNAL'] or args.SWA_inference:
                swa_optim_module = swa_optim(models['backbone_swa'])
                optimizers = {'backbone': optim_backbone, 'swa_optim': swa_optim_module}
            
            # Training and testing
            train(models, method, criterion, optimizers, schedulers, dataloaders, trial, cycle, args)
            acc, _ = test(models, args.num_epochs, method, criterion, dataloaders, args.gpu, mode='test')
            if args.use_semi_learning:
                acc_ema = test_ema(models, args.num_epochs, method, dataloaders, args.gpu, mode='test')

            if method in ['ST-CoNAL'] or args.SWA_inference:
                acc_swa = test_swa(models, args.num_epochs, method, dataloaders, args.gpu, mode='test')
                np.array([method, trial + 1, TRIALS, cycle + 1, CYCLES, len(labeled_set), acc, acc_swa]).tofile(results, sep=" ")
            elif method in ['ST-CoNAL-EMA']:
                acc_ema = test_ema(models, args.num_epochs, method, dataloaders, args.gpu, mode='test')
                np.array([method, trial + 1, TRIALS, cycle + 1, CYCLES, len(labeled_set), acc, acc_ema]).tofile(results, sep=" ")
            if method in ['ST-CoNAL'] or args.SWA_inference:
                print('Trial {}/{} || Cycle {}/{} || Label set size {} || Test acc {} || Swa acc {}'.format(trial + 1, TRIALS, cycle + 1,
                                                                                          CYCLES, len(labeled_set), acc, acc_swa))
            if method in ['ST-CoNAL-EMA']:
                print('Trial {}/{} || Cycle {}/{} || Label set size {} || Test acc {} || EMA acc {}'.format(trial + 1, TRIALS, cycle + 1,
                                                                                          CYCLES, len(labeled_set), acc, acc_ema))                                                                         
            else:
                print('Trial {}/{} || Cycle {}/{} || Label set size {} || Test acc {}'.format(trial + 1, TRIALS, cycle + 1, CYCLES, len(labeled_set), acc))
            results.write("\n")

            if cycle == (CYCLES-1):
                # Reached final training cycle
                print("Finished.")
                break

            # Get the indices of the unlabeled samples to train on next cycle
            arg = query_samples(models, method, data_unlabeled, subset, SUBSET_SIZE, BUDGET, BATCH, labeled_set, trial, cycle, args)

            # Update the labeled dataset and the unlabeled dataset, respectively
            labeled_set += list(torch.tensor(subset)[arg][-BUDGET:].numpy())
            listd = list(torch.tensor(subset)[arg][:-BUDGET].numpy())
            unlabeled_set = listd + unlabeled_set[SUBSET_SIZE:]
            print(len(labeled_set), min(labeled_set), max(labeled_set))


    results.close()

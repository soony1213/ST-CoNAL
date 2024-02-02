import torch
import os
import math
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torchvision.transforms as T
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn import functional as F
import models.resnet as resnet
from utils import plot_figure
import pdb

##
# Loss Prediction Loss
def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    criterion = nn.BCELoss()
    input = (input - input.flip(0))[:len(input)//2] # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target)//2]
    target = target.detach()
    diff = torch.sigmoid(input)
    one = torch.sign(torch.clamp(target, min=0)) # 1 operation which is defined by the authors
    
    if reduction == 'mean':
        loss = criterion(diff,one)
    elif reduction == 'none':
        loss = criterion(diff,one)
    else:
        NotImplementedError()
    return loss


def LossPredLoss_v1(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    input = (input - input.flip(0))[:len(input) // 2]
    target = (target - target.flip(0))[:len(target) // 2]
    target = target.detach()
    one = torch.sign(target)  # 1 operation which is defined by the authors
    loss = torch.maximum(- one * input + margin, torch.zeros_like(input))
    if reduction == 'mean':
        loss = torch.mean(loss)
    elif reduction == 'none':
        loss = loss
    else:
        NotImplementedError()

    return loss

def test(models, epoch, method, criterion, dataloaders, gpu, mode='val'):
    assert mode == 'val' or mode == 'test'
    models['backbone'].eval()
    total = 0
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for (inputs, labels) in dataloaders[mode]:
            with torch.cuda.device(gpu):
                inputs = inputs.cuda()
                labels = labels.cuda()

            scores, _, _ = models['backbone'](inputs)
            _, preds = torch.max(scores.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            test_loss += criterion(scores, labels).sum().item()

    return 100 * correct / total, test_loss

def test_ema(models, epoch, method, dataloaders, gpu, mode='val'):
    assert mode == 'val' or mode == 'test'
    models['backbone_ema'].eval()

    total = 0
    correct = 0
    with torch.no_grad():
        for (inputs, labels) in dataloaders[mode]:
            with torch.cuda.device(gpu):
                inputs = inputs.cuda()
                labels = labels.cuda()

            scores, _, _ = models['backbone_ema'](inputs)
            _, preds = torch.max(scores.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return 100 * correct / total

def test_swa(models, epoch, method, dataloaders, gpu, mode='val'):
    assert mode == 'val' or mode == 'test'
    models['backbone_swa'].eval()

    total = 0
    correct = 0
    with torch.no_grad():
        for (inputs, labels) in dataloaders[mode]:
            with torch.cuda.device(gpu):
                inputs = inputs.cuda()
                labels = labels.cuda()

            scores, _, _ = models['backbone_swa'](inputs)
            _, preds = torch.max(scores.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return 100 * correct / total

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def get_current_consistency_weight(epoch, args):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)

def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    return F.mse_loss(input_softmax, target_softmax, reduction='sum') / num_classes


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def test_tsne(models, epoch, method, dataloaders, gpu, mode='val'):
    assert mode == 'val' or mode == 'train'
    models['backbone'].eval()
    out_vec = torch.zeros(0)
    label = torch.zeros(0).long()
    with torch.no_grad():
        for (inputs, labels) in dataloaders:
            with torch.cuda.device(gpu):
                inputs = inputs.cuda()
                labels = labels.cuda()

            scores, _, _ = models['backbone'](inputs)
            preds = scores.cpu()
            labels = labels.cpu()
            out_vec = torch.cat([out_vec,preds])
            label = torch.cat([label,labels])
        out_vec = out_vec.numpy()
        label = label.numpy()
    return out_vec, label

iters = 0
def train_epoch(models, method, criterion, optimizers, dataloaders, gpu, epoch, epoch_loss, args):
    models['backbone'].train()
    if args.use_semi_learning:
        models['backbone_ema'].train()
    
    global iters
    for data in tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train'])):
        with torch.cuda.device(gpu):
            if args.use_semi_learning:
                # inputs[:args.labeled_batch_size] = labeled data (31, )
                # inputs[args.labeled_batch_size:] = unlabeled data (97, )
                inputs = data[0][0].cuda()
                inputs_ema = data[0][1].cuda()
                labels = data[1].cuda()
                consistency_weight = get_current_consistency_weight(epoch, args)
                consistency_criterion = softmax_mse_loss
            else:
                inputs = data[0].cuda()
                labels = data[1].cuda()

        iters += 1
        optimizers['backbone'].zero_grad()
        scores, _, features = models['backbone'](inputs)
        if args.use_semi_learning:
            _, _, features = models['backbone'](inputs[:args.labeled_batch_size])
            scores_ema, _, _ = models['backbone_ema'](inputs_ema)
            consistency_loss = consistency_weight * consistency_criterion(scores, scores_ema) / len(inputs)
            target_loss = criterion(scores[:args.labeled_batch_size], labels[:args.labeled_batch_size])
        else:
            target_loss = criterion(scores, labels)

        m_backbone_loss = torch.sum(target_loss) / len(inputs)
        loss            = m_backbone_loss
        if args.use_semi_learning:
            loss += consistency_loss

        loss.backward()
        optimizers['backbone'].step()
        if args.use_semi_learning or method in ['ST-CoNAL-EMA']:
            update_ema_variables(models['backbone'], models['backbone_ema'], float(args.ema_decay), iters)
    return loss

def train(models, method, criterion, optimizers, schedulers, dataloaders, trial, cycle, args):
    print('>> Train a Model.')
    best_acc = 0.
    lr_list = []
    train_loss_list = []
    val_loss_list = []
    for epoch in range(args.num_epochs):
        loss = train_epoch(models, method, criterion, optimizers, dataloaders, args.gpu, epoch, args.epochs_loss, args)
        train_loss_list.append(loss.item())
        lr_list.append(optimizers['backbone'].param_groups[0]['lr'])
        schedulers['backbone'].step(epoch)

        if args.verbose and epoch % 20  == 0:
            acc, _ = test(models, epoch, method, criterion, dataloaders, args.gpu, mode='test')
            val_loss_list.append(acc)

            if best_acc < acc:
                best_acc = acc
                print('Val Acc: {:.3f} \t Best Acc: {:.3f}'.format(acc, best_acc))

        if method in ['ST-CoNAL', 'ST-CoNAL-Aug', 'ST-CoNAL-EMA'] or args.SWA_inference:
            if (epoch+1)%args.lr_interval == 0 and (epoch+1)>=args.swa_start:
                acc, _ = test(models, epoch, method, criterion, dataloaders, args.gpu, mode='test')
                if method in ['ST-CoNAL-EMA']:
                    optimizers['swa_optim'].update(models['backbone'])
                    update_batchnorm(models['backbone_swa'], dataloaders['train'], args)
                    acc_swa = test_swa(models, epoch, method, dataloaders, args.gpu, mode='test')
                    update_batchnorm(models['backbone_ema'], dataloaders['train'], args)
                    acc_ema = test_ema(models, epoch, method, dataloaders, args.gpu, mode='test')
                    print('current epoch: {}/acc: {}/ema acc: {}/swa acc: {}'.format(epoch+1, acc, acc_ema, acc_swa))
                    torch.save({'state_dict': models['backbone'].state_dict(),
                            'epoch': epoch, 'acc': acc, 'acc_ema': acc_ema, 'acc_swa': acc_swa}, args.log_root + '/t' + str(trial) + 'c' + str(cycle) + 'e' + str(epoch + 1) + '.ckpt')
                else:
                    optimizers['swa_optim'].update(models['backbone'])
                    update_batchnorm(models['backbone_swa'], dataloaders['train'], args)
                    acc_swa = test_swa(models, epoch, method, dataloaders, args.gpu, mode='test')
                    print('current epoch: {}/acc: {}/swa acc: {}'.format(epoch+1, acc, acc_swa))
                    torch.save({'state_dict': models['backbone'].state_dict(),
                            'epoch': epoch,
                            'acc': acc},
                           args.log_root + '/t' + str(trial) + 'c' + str(cycle) + 'e' + str(epoch + 1) + '.ckpt')

    plot_figure(args.num_epochs, train_loss_list, None, 'Train_loss', 'Train_loss', path = args.log_root)
    plot_figure(args.num_epochs, lr_list, 'Learning_rate', 'Learning_rate', path = args.log_root)
    print('>> Finished.')

class swa_optim(object):
    """
    SWA or fastSWA
    """
    def __init__(self, swa_model):
        self.num_params = 0
        self.swa_model = swa_model # assume that the parameters are to be discarded at the first update

    def update(self, student_model):
        self.num_params += 1
        print("Updating SWA. Current num_params =", self.num_params)
        if self.num_params == 1:
            print("Loading State Dict")
            self.swa_model.load_state_dict(student_model.state_dict())
        else:
            # Do weight averaging
            inv = 1./float(self.num_params)
            for swa_p, src_p in zip(self.swa_model.parameters(), student_model.parameters()):
                swa_p.data.add_(-inv*swa_p.data)
                swa_p.data.add_(inv*src_p.data)

    def reset(self):
        self.num_params = 0

def update_batchnorm(model, train_loader, args, verbose=False):
    if verbose: print("Updating Batchnorm")

    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum
    if not momenta:
        return
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    model.train()
    i = 0
    for data in tqdm(train_loader, leave=False, total=len(train_loader)):
        # if i > 100:
        #     return
        with torch.cuda.device(args.gpu):
            if args.use_semi_learning:
                inputs = data[0][0].cuda()
                labels = data[1].cuda()
            else:
                inputs = data[0].cuda()
                labels = data[1].cuda()
        model_out = model(inputs)
        # i += 1

        if verbose and i % 100 == 0:
            print('Updating BN. i = {}'.format(i))

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]


class ConstAnnealingWarmRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, base_lr=0.1, lr_min=0.0, lr_interval=10, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        # if T_mult < 1 or not isinstance(T_mult, int):
        #     raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        self.T_0 = T_0
        self.base_lr = base_lr
        self.lr_max = base_lr
        self.lr_min = lr_min
        self.lr_interval = lr_interval
        self.gamma = gamma
        self.T_cur = last_epoch
        super(ConstAnnealingWarmRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self, epoch):
        if epoch < self.T_0:
            return self.base_lr
        else:
            return self.lr_min + (self.lr_max - self.lr_min) * (1 + math.cos(math.pi * self.T_cur / self.lr_interval)) / 2

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1

        if epoch >= self.T_0:
            self.T_cur = (epoch - self.T_0) % self.lr_interval
            self.lr_max = self.gamma * self.base_lr
        else:
            self.T_cur = epoch
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.get_lr(epoch)
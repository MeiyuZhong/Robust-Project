### basic modules
import numpy as np
import time, pickle, os, sys, json, PIL, tempfile, warnings, importlib, math, copy, shutil

### torch modules
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import torch.nn.functional as F

# from advertorch.attacks import L2PGDAttack
# from advertorch.context import ctx_noparamgrad_and_eval

import argparse

def argparser(data='cifar10', model='c6f2_relux',
              batch_size=256, epochs=600, warmup=20, rampup=400,
              epsilon=36/255, epsilon_train=0.1551, starting_epsilon=0.0, 
              opt='adam', lr=0.001): 

    parser = argparse.ArgumentParser()
    # log settings
    parser.add_argument('--project', default='debug_bcp', help='Name for Wandb project')
    
    # main settings
    parser.add_argument('--rampup', type=int, default=rampup)
    parser.add_argument('--warmup', type=int, default=warmup)
    parser.add_argument('--sniter', type=int, default=1) 
    parser.add_argument('--opt_iter', type=int, default=1) 
    parser.add_argument('--no_save', action='store_true') 
    parser.add_argument('--test_pth', default=None)
    parser.add_argument('--print', action='store_true')

    # optimizer settings
    parser.add_argument('--opt', default='adam')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=epochs)
    parser.add_argument("--lr", type=float, default=lr)
    parser.add_argument("--end_lr", type=float, default=5e-6)
    parser.add_argument("--step_size", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--wd_list", nargs='*', type=int, default=None)
    parser.add_argument("--lr_scheduler", default='exp')
    parser.add_argument('--more', type=int, default=25) # more epochs with initial learning rate before exp decay
    
    # test settings during training
    parser.add_argument('--test_sniter', type=int, default=100) 
    parser.add_argument('--test_opt_iter', type=int, default=1000) 

    # pgd settings
    parser.add_argument("--epsilon_pgd", type=float, default=epsilon)
    parser.add_argument("--alpha", type=float, default=epsilon/4)
    parser.add_argument("--niter", type=float, default=100)
    
    # epsilon settings
    parser.add_argument("--epsilon", type=float, default=epsilon)
    parser.add_argument("--epsilon_train", type=float, default=epsilon_train)
    parser.add_argument("--starting_epsilon", type=float, default=starting_epsilon)
    parser.add_argument('--schedule_length', type=int, default=rampup) 
    
    # kappa settings
    parser.add_argument("--kappa", type=float, default=0.0)
    parser.add_argument("--starting_kappa", type=float, default=1.0)
    parser.add_argument('--kappa_schedule_length', type=int, default=rampup) 
    
    # use gloro loss with auxillary logit
    parser.add_argument('--gloro', action='store_true')

    # model arguments
    parser.add_argument('--model', default='large')
    parser.add_argument("--init", type=float, default=1.0)

    # other arguments
    parser.add_argument('--prefix')
    parser.add_argument('--saved_model', default=None)
    parser.add_argument('--data', default=data)
    parser.add_argument('--real_time', action='store_true')
    parser.add_argument('--seed', type=int, default=2019)
    parser.add_argument('--verbose', type=int, default=200)
    parser.add_argument('--cuda_ids', type=int, default=0)
    
    # loader arguments
    parser.add_argument('--batch_size', type=int, default=batch_size)
    parser.add_argument('--test_batch_size', type=int, default=batch_size)
    parser.add_argument('--normalization', action='store_true')
    parser.add_argument('--no_augmentation', action='store_true')
    parser.add_argument('--drop_last', action='store_true')
    parser.add_argument('--no_shuffle', action='store_true')

    parser.add_argument('-f')
    args = parser.parse_args()
    
    args.augmentation = not(args.no_augmentation)
    args.shuffle = not(args.no_shuffle)
    args.save = not(args.no_save)
    
    if args.rampup:
        args.schedule_length = args.rampup
        args.kappa_schedule_length = args.rampup 
    if args.epsilon_train is None:
        args.epsilon_train = args.epsilon 
        
    if args.starting_epsilon is None:
        args.starting_epsilon = args.epsilon
    if args.prefix:
        args.prefix = 'pretrained/'+args.prefix
        if args.model is not None: 
            args.prefix += '_'+args.model
        if args.schedule_length > args.epochs: 
            raise ValueError('Schedule length for epsilon ({}) is greater than '
                             'number of epochs ({})'.format(args.schedule_length, args.epochs))
    else: 
        args.prefix = 'pretrained/temporary'

    if args.cuda_ids is not None: 
        print('Setting CUDA_VISIBLE_DEVICES to {}'.format(args.cuda_ids))
#         torch.cuda.set_device(args.cuda_ids)

    return args



#### clean train
def train(loader, model, opt, epoch, log, verbose):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()
    
    model.train()

    end = time.time()
    for i, (X,y,idx) in enumerate(loader):
        
        X,y = X.cuda(), y.cuda()
        data_time.update(time.time() - end)
        out = model(X)
        ce = nn.CrossEntropyLoss()(out, y)
        
        err = (out.max(1)[1] != y).float().sum()  / X.size(0)
        loss = ce
        
        opt.zero_grad()
        loss.backward()
        opt.step()

        # measure accuracy and record loss
        losses.update(ce.item(), X.size(0))
        errors.update(err.item(), X.size(0))
        
        # measure elapsed time
        batch_time.update(time.time()-end)
        end = time.time()
        if i % 80 == 0:
            print(epoch, i, ce.item(),err.item(), file=log) ########
        if verbose and (i==0 or i==len(loader)-1 or (i+1) % verbose == 0): 
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                  'Data {data_time.val:.4f} ({data_time.avg:.4f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error {errors.val:.4f} ({errors.avg:.4f})'.format(
                   epoch, i+1, len(loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, errors=errors))
        log.flush()
    return

#### Evaluation code
def evaluate(loader, model, epoch, log, verbose):
    batch_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()

    model.eval()

    end = time.time()
    for i, (X,y,idx) in enumerate(loader):
        X,y = X.cuda(), y.cuda()
#         out = model(X)
        out1 = model[0](X) + torch.randn_like(model[0](X), device='cuda') * 0.25
        out = model[1](out1)
        ce = nn.CrossEntropyLoss()(out, y)
        err = (out.data.max(1)[1] != y).float().sum()  / X.size(0)

        # print to logfile
        print(epoch, i, ce.item(), err.item(), file=log)

        # measure accuracy and record loss
        losses.update(ce.data, X.size(0))
        errors.update(err, X.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if verbose and (i==0 or i==len(loader)-1 or (i+1) % verbose == 0): 
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error {error.val:.4f} ({error.avg:.4f})'.format(
                      i+1, len(loader), batch_time=batch_time, loss=losses,
                      error=errors))
        log.flush()

    print(' * Error {error.avg:.4f}'.format(error=errors))
    return errors.avg


def evaluate_pgd(loader, model, args):
    errors = AverageMeter()
    model.eval()

    adversary = L2PGDAttack(
        model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=args.epsilon, 
        nb_iter=args.niter, eps_iter=args.alpha, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)

    end = time.time()
    for i, (X,y,idx) in enumerate(loader):
        X,y = X.cuda(), y.cuda()
        with ctx_noparamgrad_and_eval(model):
            X_pgd = adversary.perturb(X, y)
            
        out = model(X_pgd)
        err = (out.data.max(1)[1] != y).float().sum() / X.size(0)
        errors.update(err, X.size(0))
    print(' * Error {error.avg:.4f}'.format(error=errors))
    return errors.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count    


# train with the same learning rate for the larest eps for some more epochs and then decay
def lr_exp(start_lr, end_lr, epoch, max_epoch, more=25):
    if epoch >= (max_epoch//2 + more):
        scalar = (end_lr/start_lr)**((float(epoch)-(max_epoch//2 +more-1))/(max_epoch//2 - more))
    else:
        scalar = 1.0
    return scalar   
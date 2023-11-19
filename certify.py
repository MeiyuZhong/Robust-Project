# evaluate a smoothed classifier on a dataset

import argparse
import os
import setGPU
from datasets import get_dataset, DATASETS, get_num_classes
from core import Smooth
from time import time
import torch
import datetime
from architectures import get_architecture
import pandas as pd
import sys
sys.path.insert(1, '../code')
import train
from archs.cifar_resnet import Identity, ReLU_x_imanet, ReLU_x, ReLU_x_imanet25


def certify(name, sig_list):
    
    dataset = name#"cifar10"
    sigma = sig_list
    dic = {}
    # load the base classifier
    for sig in range(len(sigma)):
        if dataset=="cifar10":
#         base_classifier="mnist_results/checkpoint_r_158_best.pth.tar"
            if sigma[sig] == 25:
                base_classifier = "passedmodels/cifar/temporary_best_025_global.pth"
                lip_constant = 0.5
                N=100000
            elif sigma[sig] == 12: 
                base_classifier = "pretrained/temporary_best_005012.pth"
                lip_constant = 0.5
                N=100000
            elif sigma[sig] == 50:
                base_classifier="pretrained/temporary_best_00505.pth"
                lip_constant = 0.5
                N=100000
            elif sigma[sig] == 100:
                base_classifier="pretrained/temporary_best_0051.pth"
                lip_constant = 0.5
                N=100000 
# base_classifier="new_results/checkpoint0_16255_best.pth.tar"#../local_lipschitz/pretrained/temporary_checkpoint.pth"
                #"new_results/checkpoint_mid_1_16255_best.pth.tar"
#             else:
#                 base_classifier="../local_lipschitz/pretrained/temporary_best_1_layer1.pth"
#                 lip_constant = 0.6022
        elif dataset=="imagenet":
            if sigma[sig] == 50:
                base_classifier = "pretrained/temporary_temper_imagnetpretrained.pth"
                lip_constant = 0.5
                N=10000
            elif sigma[sig] == 25:
                base_classifier = "pretrained/temporary_temper_imagnet25pretr.pth"
                lip_constant = 0.7
                N=10000
            

        print("Start the Sig",sigma[sig])
        outfile="ablation_model/results_certify_newcifar_025" + str(sigma[sig])
        batch=400
        skip=50
        max_=-1
        split="test"
        N0=100
         
        alpha=0.001
        
        max_radius = 0 
        checkpoint = torch.load(base_classifier)
    #     base_classifier = get_architecture(checkpoint["arch"], dataset)
        if dataset == "mnist":
            base_classifier = mnist_model_large_relux().cuda()
        elif dataset == "cifar10": 
            arch = "cifar_resnet110"
            base_classifier = get_architecture(arch, dataset).cuda()
        elif dataset == "imagenet":
            arch = "resnet50"
            base_classifier = get_architecture(arch, dataset)
            base_classifier[1].bn1 = Identity().cuda()
            base_classifier[1].relu = ReLU_x_imanet(torch.Size([1, 64, 112, 112])).cuda()
        base_classifier.load_state_dict(checkpoint['state_dict'])
        smoothed_classifier = Smooth(base_classifier, get_num_classes(dataset), sigma[sig]/100 ,lip_constant)
        dataset = get_dataset(dataset,split)
        # prepare output file
        f = open(outfile, 'w')
        print("idx\tlabel\tpredict\tradius\tcorrect\tLip\ttime", file=f, flush=True)
        
    
        dic[sigma[sig]] = {'radius': [],'correct': [],'lip':[]}
        # create the smooothed classifier g
        
        # iterate through the dataset
        print(len(dataset))
        for i in range(len(dataset)):

            if i == max_:
                break

            (x, label) = dataset[i]
            before_time = time()
            # certify the prediction of g around x
            x = x.cuda()
            prediction, radius, lip_est = smoothed_classifier.certify(x, N0, N, alpha, batch)
            after_time = time()
            correct = int(prediction == label) 
            dic[sigma[sig]]['radius'].append(radius)
            dic[sigma[sig]]['correct'].append(correct)
            dic[sigma[sig]]['lip'].append(lip_est)
            dictionary = pd.DataFrame.from_dict(dic)
            time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
            print("{}\t{}\t{}\t{:.3}\t{}\t{}\t{}".format(
                i, label, prediction, radius, correct, lip_est, time_elapsed), file=f, flush=True)

    f.close()
    return dictionary

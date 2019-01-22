import argparse
import datetime
import os
import torch

def get_args():
    parser = argparse.ArgumentParser(description='face recognition')
    # Set log names
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    dataset = 'ORL'
    logname = '{}_{}_{}'.format(dataset, current_time, 'runlog.txt')
    logpath = '../log'
    log_path = os.path.join(logpath, logname)
    parser.add_argument('--log_path', default=log_path, type=str, help='log path')
    parser.add_argument('--dataset_path', default='../data/att_faces/', type=str, help='dataset path')
    # parser.add_argument('--dataset_path', default='/home/aaron/Datasets/att_faces/', type=str, help='dataset path')
    parser.add_argument('--rgb2gray', default=True, type=bool, help='use gray image or not')
    parser.add_argument('--num_components', default=50, type=int, help='number of principal components for pca ')

    parser.add_argument('--sub_region_height', default=16, type=int, help='height of sub regions for LBP')
    parser.add_argument('--sub_region_width', default=23, type=int, help='width of sub regions for LBP')
    parser.add_argument('--uniform', default=True, type=bool, help='use uniform pattern LBP or not')

    parser.add_argument('--lr', default=0.00001, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=5, type=int, help='')
    parser.add_argument('--num_epochs', default=100, type=int, help='')
    parser.add_argument('--model_root', default='/home/aaron/ORLFace/data/Spherenet_WebFace_LFW.pkl')

    args = parser.parse_args()
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return args


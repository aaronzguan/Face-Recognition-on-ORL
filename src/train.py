from ORL_loader import load_ORLdataset
from pca import pca_face_recognitioin
from lbp import lbp_face_recognition
from cnn import cnn_face_recognition
import common_args
import os



if __name__ == '__main__':
    args = common_args.get_args()

    log = args.log_path

    train_X, train_Y, test_X, test_Y = load_ORLdataset(args.dataset_path, args.rgb2gray)

    message = 'Train data shape: {}'.format(train_X.shape) + '\n' + \
    'Train labels shape: {}'.format(train_Y.shape) + '\n' + \
    'Test data shape: {}'.format(test_X.shape) + '\n' + \
    'Test labels shape: {}'.format(test_Y.shape)
    print(message)
    with open(log, "a") as log_file:
        log_file.write('\n' + message)

    pca_face_recognitioin(train_X, train_Y, test_X, test_Y)

    # lbp_face_recognition(train_X, train_Y, test_X, test_Y)

    # cnn_face_recognition(train_X, train_Y, test_X, test_Y)

















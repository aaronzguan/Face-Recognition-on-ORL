import numpy as np
import cv2
import os


def delete_ds_store(fpath):
    for root, dirs, files in os.walk(fpath):
        for file in files:
            if file.endswith('.DS_Store'):
                path = os.path.join(root, file)
                os.remove(path)

def load_ORLdataset(dataset_path, isgrayImg):
    """
    :param dataset_path: string of dataset path
    :return: X_train, Y_train, X_test, Y_test: each row corresponding to each image
    """
    data = []
    delete_ds_store(dataset_path)
    identity_list = os.listdir(dataset_path)
    for identity_folder in identity_list:
        identity_path = os.path.join(dataset_path, identity_folder)
        delete_ds_store(identity_path)
        img_list = os.listdir(identity_path)
        for img in img_list:
            img_bgr = cv2.imread(os.path.join(identity_path, img))
            if isgrayImg:
                img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                data.append(img_gray)
            else:
                data.append(img_bgr)

    data = np.asarray(data, 'float64')

    num_class = len(identity_list)

    label = np.zeros((data.shape[0], 1))
    for i in range(num_class):
        label[i * 10: (i + 1) * 10] = i

    train_data = []
    train_label = []
    test_data = []
    test_label = []

    for i in range(40):
        train_data.append(data[i * 10: i * 10 + 3])
        train_label.append(label[i * 10: i * 10 + 3])
        test_data.append(data[i * 10 + 3: i * 10 + 10])
        test_label.append(label[i * 10 + 3: i * 10 + 10])

    X_train = np.asarray(np.concatenate(train_data), 'float64')
    Y_train = np.asarray(np.concatenate(train_label), 'float64')
    X_test = np.asarray(np.concatenate(test_data), 'float64')
    Y_test = np.asarray(np.concatenate(test_label), 'float64')

    return X_train, Y_train, X_test, Y_test
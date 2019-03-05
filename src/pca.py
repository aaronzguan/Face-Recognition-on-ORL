import numpy as np
import matplotlib.pyplot as plt
import common_args

def eigen(data):
    """
    A*A.T * u = eigenValue * u (u is eigenVectors)
    The matrix A*A.T is very large
    So compute the eigenVectors v of A.T * A
    A.T * A * A.T * u = eigenvalue * A.T * u
    A.T * A * v = eigenvalue * v, where v = A.T * u
    u = A * v = A * A.T * u = u
    :param train_data: M * N^2 matrix, each row represents each image N*N
    :return:

    eigenValues: M, each eigenvalue corresponding one eigenvector, in descending order

    eigenVectors: M * N^2 matrix, each row represents each eigenface, in descending order
    """
    train_data_transpose = np.transpose(data)
    cov_matrix = np.dot(data, train_data_transpose)
    eigenValues, eigenVectors = np.linalg.eig(cov_matrix)
    eigenVectors = np.dot(eigenVectors, data)
    # Sort the eigenVectors in descending order corresponding to the eigenValues
    sort = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[sort]
    eigenVectors = eigenVectors[sort, :]

    return eigenValues, eigenVectors

def train_PCA(data, num_components):
    """
    Normalize the face by subtracting the mean image
    Calculate the eigenValue and eigenVector of the training face, in descending order
    Keep only num_components eigenvectors (corresponding to the num_components largest eigenvalues)
    Each training face is represented in this basis by a vector

    Calculate the weight vectors for training images
    Normalized training face = F - mean = w1*u1 + w2*u2 + ... + wk*uk => w = u.T * face

    :param train_data:  M * N^2, each row corresponding to each image, which is reshaped into 1-D vector
    :param num_components: The number of the largest eigenVector to be kept
    :return:

    mean_image: 1 * N^2
    eigenVectors: num_components * N^2 matrix, each row represents each eigenface, in descending order
    weiVec_train: M * K matrix, each row is the weight vectors used to represent the training face
    """
    mean_image = np.mean(data, axis=0)
    data = data - mean_image
    eigenValues, eigenVectors = eigen(data)
    eigenVectors = eigenVectors[:num_components]

    weiVec_train = np.dot(data, eigenVectors.T)

    return mean_image, eigenVectors, weiVec_train


def pca_face_recognitioin(train_data, train_label, test_data, test_label):
    """
    Face recognition by PCA.

    :param train_data: M * H * W * Channel(opt), each row corresponding to each image, which is reshaped into 1-D vector
    :param train_label: M training labels
    :param test_data:  L * H * W * Channel(opt), each row corresponding to each image, which is reshaped into 1-D vector
    :param test_label: L test labels
    :param num_components: The number of the largest eigenVector to be kept
    :return:
    """
    args = common_args.get_args()
    log = args.log_path

    message = '*' * 40 + '\n' + \
              'Face recognition by PCA ...'
    print(message)
    with open(log, "a") as log_file:
        log_file.write('\n' + message)

    if len(train_data.shape[1:]) == 3:
        isRGB = True
        HEIGHT, WIDTH, CHANNEL = train_data.shape[1], train_data.shape[2], train_data.shape[3]
    else:
        isRGB = False
        HEIGHT, WIDTH = train_data.shape[1], train_data.shape[2]

    # Reshape the image images into rows
    train_data = np.reshape(train_data, (train_data.shape[0], -1))
    test_data = np.reshape(test_data, (test_data.shape[0], -1))
    message = 'Reshape train images: {}'.format(train_data.shape) + '\n' + \
              'Reshape test images: {}'.format(test_data.shape)
    print(message)
    with open(log, "a") as log_file:
        log_file.write('\n' + message)

    mean_image, eigenVectors, weiVec_train = train_PCA(train_data, args.num_components)

    # Plot the average face
    if isRGB:
        plt.imshow(mean_image.reshape(HEIGHT, WIDTH, CHANNEL).astype('uint8'))
    else:
        plt.imshow(mean_image.reshape(HEIGHT, WIDTH).astype('uint8'))
    plt.title('The average face')
    plt.show()
    # Plot 10 most significant eigenFaces
    eigenFaces = []
    if isRGB:
        for i in range(eigenVectors.shape[0]):
            eigenFaces.append(eigenVectors[i].reshape(HEIGHT, WIDTH, CHANNEL))
    else:
        for i in range(eigenVectors.shape[0]):
            eigenFaces.append(eigenVectors[i].reshape(HEIGHT, WIDTH))
    eigenFaces = np.asarray(eigenFaces)
    if args.num_components > 10:
        for plt_idx in range(1, 11):
            plt.subplot(5, 2, plt_idx)
            plt.imshow(eigenFaces[plt_idx - 1].astype('uint8'))
            plt.axis('off')
        plt.show()

    # Verify the test faces
    test_data = test_data - mean_image
    weiVec_test = np.dot(test_data, eigenVectors.T)
    correct_count = 0
    # Caculate the L2 distance for the testing weight and training weight
    for i in range(weiVec_test.shape[0]):
        dist = np.linalg.norm((weiVec_train - weiVec_test[i]), axis=1)
        index_min = np.argmin(dist)
        if train_label[index_min] == test_label[i]:
            correct_count += 1

    pca_accuracy = correct_count / len(test_label)
    message = 'The accuracy of face recognition by PCA is {}'.format(pca_accuracy) + '\n' + \
              '*' * 40
    print(message)
    with open(log, "a") as log_file:
        log_file.write('\n' + message)
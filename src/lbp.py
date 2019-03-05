import numpy as np


def divide_image(data, sub_Height, sub_Width):
    """

    :param data: N * H * W matrix, where N is number of faces and H, W is the height and width of one face
    :param sub_Height: The height of each sub region
    :param sub_Width: The width of each sub region
    :return:
    """
    HEIGHT, WIDTH = data.shape[1], data.shape[2]
    subRegion = []
    for sub_h in range(int(HEIGHT/sub_Height)):
        for sub_w in range(int(WIDTH/sub_Width)):
            subRegion.append(data[:, sub_Height * sub_h + 0: sub_Height * sub_h + sub_Height, sub_Width * sub_w + 0: sub_Width * sub_w + sub_Width])

    subRegion = np.asarray(subRegion, dtype='uint8')
    subRegion = subRegion.reshape(subRegion.shape[1], subRegion.shape[0], subRegion.shape[2], subRegion.shape[3])

    return subRegion

def get_uniform_table():
    """
    Table for uniform lbp coding
    Suppose P is the number of bits
        0 bitwise transition: 00000000 or 11111111 : 2
        1 bitwise transition: 1 -> 0, 0 -> 1 : (P - 1) * 2
        2 bitwise transitions: 0 -> 1 -> 0, 1 -> 0 -> 1, Fix one 0/1 position and then look for the other 0/1 position.
        For one case: P-2 + P-3 + .... + 1 = (1+P-2)*(P-1)/2 = (P-1)(P-2)/2
        So 2 bit wise transitions: (P - 1)(P - 2)
    Total number of transitions:
    2 + (P - 1) * 2 + (P - 1)(P - 2) = P(P - 1) + 2
    :return: Uniform Table for 8 bits LBP
    """
    uniform_table = [0,1,2,3,4,58,5,6,7,58,58,58,8,58,9,10,11,58,58,58,58,58,58,58,12,58,58,58,13,58,
        14,15,16,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,17,58,58,58,58,58,58,58,18,
        58,58,58,19,58,20,21,22,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,
        58,58,58,58,58,58,58,58,58,58,58,58,23,58,58,58,58,58,58,58,58,58,58,58,58,58,
        58,58,24,58,58,58,58,58,58,58,25,58,58,58,26,58,27,28,29,30,58,31,58,58,58,32,58,
        58,58,58,58,58,58,33,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,34,58,58,58,58,
        58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,
        58,35,36,37,58,38,58,58,58,39,58,58,58,58,58,58,58,40,58,58,58,58,58,58,58,58,58,
        58,58,58,58,58,58,41,42,43,58,44,58,58,58,45,58,58,58,58,58,58,58,46,47,48,58,49,
        58,58,58,50,51,52,58,53,54,55,56,57]
    uniform_table = np.asarray(uniform_table)
    return uniform_table


def get_pixel(img, center, H, W):
    """
    LBP looks at points surrounding a central point and tests whether
    the surrounding points are greater than or less than the central point.
    If greater, then assign to 1, otherwise assign to 0.
    :param img: The img to calculate the LBP
    :param center: The center pixel value
    :param H: Height position of the surrounding point
    :param W: width position of the surrounding point
    :return: 1 or 0
    """
    new_value = 0
    if H < 0 or W < 0:
        pass
    else:
        try:
            if img[H][W] >= center:
                new_value = 1
        except:
            pass
    return new_value


def cal_lbp_hist(region):
    """
    Calculate the LBP histogram for the region by using Uniform 8 bits coding.
    :param region: The region of interest for calculating the LBP
    :return: LBP_hist: 59 bins ranging from [0, 58]
    """
    uniform_table = get_uniform_table()
    power_val = [128, 64, 32, 16, 8, 4, 2, 1]
    power_val = np.asarray(power_val, dtype='uint8')

    height = region.shape[0]
    width = region.shape[1]
    LBP_img = np.zeros((height, width), dtype='uint8')
    for h in range(height):
        for w in range(width):
            bi_pattern = []
            center_pixel = region[h][w]
            bi_pattern.append(get_pixel(region, center_pixel, h - 1, w - 1))
            bi_pattern.append(get_pixel(region, center_pixel, h - 1, w))
            bi_pattern.append(get_pixel(region, center_pixel, h - 1, w + 1))
            bi_pattern.append(get_pixel(region, center_pixel, h, w + 1))
            bi_pattern.append(get_pixel(region, center_pixel, h + 1, w + 1))
            bi_pattern.append(get_pixel(region, center_pixel, h + 1, w))
            bi_pattern.append(get_pixel(region, center_pixel, h + 1, w - 1))
            bi_pattern.append(get_pixel(region, center_pixel, h, w - 1))
            bi_pattern = np.asarray(bi_pattern, dtype='uint8')
            LBP_img[h, w] = uniform_table[np.sum(bi_pattern * power_val)]
    LBP_hist, _ = np.histogram(LBP_img.flatten(), 59, (0, 58))
    # hist = cv2.calcHist(lbp_val, [0], None, [59], [0, 256])
    # hist = hist / (sum(hist) + 1e-7) # whether to normalize does not affect the final performance

    return LBP_hist


def lbp_face_recognition(train_data, train_label, test_data, test_label):
    """
    Face recognition by LBP histogram. Matching between test image and training image is done by
    calculating the Chi square distance for LBP histogram
    :param train_data: train_data: M * H * W, each row corresponding to each gray level image
    :param train_label: M labels
    :param test_data: L * H * W, each row corresponding to each gray level image
    :param test_label: L labels
    :return:
    """
    assert len(train_data.shape) < 4, "The training images is not gray level image!"
    assert len(test_data.shape) < 4, "The test images is not gray level image!"
    print('*' * 40)
    print('Face recognition by Uniform LBP ...')

    subRegion_size = [16, 23]
    sub_Height, sub_Width = subRegion_size[0], subRegion_size[1]
    print('Each face is divided into regions of {} x {}'.format(sub_Height, sub_Width))
    # Get the LBP histogram for train images
    subRegion_train = divide_image(train_data, sub_Height, sub_Width)
    hist_lbp_train = []
    for i in range(subRegion_train.shape[0]):
        hist_lbp_train_sub = []
        for sub in range(subRegion_train.shape[1]):
            subRegion = subRegion_train[i][sub]
            hist = cal_lbp_hist(subRegion)
            hist_lbp_train_sub.append(hist)
        hist_lbp_train_sub = np.asarray(hist_lbp_train_sub, dtype='float32')
        hist_lbp_train.append(hist_lbp_train_sub)
    hist_lbp_train = np.asarray(hist_lbp_train, dtype='float32')
    print('Train images LBP histogram shape: ', hist_lbp_train.shape)

    # Get the LBP histogram for test images
    subRegion_test = divide_image(test_data, sub_Height, sub_Width)
    hist_lbp_test = []
    for i in range(subRegion_test.shape[0]):
        hist_lbp_test_sub = []
        for sub in range(subRegion_test.shape[1]):
            subRegion = subRegion_test[i][sub]
            hist = cal_lbp_hist(subRegion)
            hist_lbp_test_sub.append(hist)
        hist_lbp_test_sub = np.asarray(hist_lbp_test_sub, dtype='float32')
        hist_lbp_test.append(hist_lbp_test_sub)
    hist_lbp_test = np.asarray(hist_lbp_test, dtype='float32')
    print('Test images LBP histogram shape: ', hist_lbp_test.shape)

    # Calculate the Chi square distance between the test face and train images
    print('Calculate the Chi square distance ...')
    correct_count = 0
    for i in range(len(hist_lbp_test)):
        chi_dist = np.sum(np.sum((hist_lbp_train - hist_lbp_test[i]) ** 2 / (hist_lbp_train + hist_lbp_test[i] + 1e-10), axis=1), axis=1)
        index_min = np.argmin(chi_dist)
        if train_label[index_min] == test_label[i]:
            correct_count += 1

    lbp_accuracy = correct_count / len(test_label)
    print('The accuracy of face recognition by LBP is', lbp_accuracy)
    print('*' * 40)
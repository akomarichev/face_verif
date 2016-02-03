from scipy import signal
from scipy import sign
import scipy.io
import numpy as np
import random
from pylab import *
import matplotlib.pyplot as plt
import matplotlib
import PIL
import sklearn
from sklearn.datasets import fetch_lfw_pairs


def rectifier(x):
    return np.maximum(0, x)


def rectifier_prime(x):
    return sign(rectifier(x))

# weight initialization


def initialization(patch_size, small_patch, image_height, image_width):
    # calculate weights for the weights of the very first convolutional layer
    r = np.sqrt(6) / np.sqrt(patch_size + patch_size + 1)
    # weights for a upper branch (convolution layer)
    W11 = np.zeros((patch_size, patch_size))
    # weights for a lower branch (convolution layer)
    W12 = np.zeros((patch_size, patch_size))
    W11[:, :] = np.random.random((patch_size, patch_size)) * 2 * r - r
    W12[:, :] = np.random.random((patch_size, patch_size)) * 2 * r - r

    dim1 = (patch_size - small_patch + 1)**2
    dim2 = ((image_height - patch_size + 1) - patch_size + 1) * ((image_width - patch_size + 1) - patch_size + 1)
    # print "Numer of parameters to tune: ", 2 * filters * patch_size**2 + 2 * filters * dim1 * dim2 * small_patch**2

    # weights for a upper branch (lrf layer)
    W21 = np.zeros((small_patch, small_patch, dim2))
    # weights for a upper branch (lrf layer)
    W22 = np.zeros((small_patch, small_patch, dim2))
    r = np.sqrt(6) / np.sqrt(small_patch + small_patch + 1)
    for j in range(dim2):
        W21[:, :, j] = np.random.random((small_patch, small_patch)) * 2 * r - r
        W22[:, :, j] = np.random.random((small_patch, small_patch)) * 2 * r - r

    # sofmax layer
    W5 = np.zeros((dim1, dim2, 2))
    for i in range(dim2):
        for j in range(2):
            W5[:, i, j] = np.random.random((dim1)) * 2 * r - r
    print "W5.shape: ", W5.shape

    theta = np.concatenate(
        (W11.flatten(), W12.flatten(), W21.flatten(), W22.flatten(), W5.flatten()))

    return theta

# Local Receptive fields close to convolution but without sharing
# parameters across different locations in the image. Our assumption that
# stride = 1.
# Second Layer


def lrf(images1, images2, W, small_patch, patch_size, image_height, image_width, N):
    # z2_convolved = np.zeros((hidden_size, number_of_images), dtype=np.float64)
    number_of_patches = (image_height - patch_size + 1) * (image_width - patch_size + 1)
    hidden_size = (patch_size - small_patch + 1) ** 2
    z3_convolved = np.zeros((hidden_size, number_of_patches, N), dtype=np.float64)
    for i in range(N):
        patches_bigger, patches_smaller = extract_patches(images1[:, :, i], images2[:, :, i], image_height, image_width, small_patch, patch_size, number_of_patches)
        for t in range(number_of_patches):
            patches = extract_patches2(patches_bigger[:, :, t], patch_size, small_patch, hidden_size)
            for j in range(hidden_size):
                z3_convolved[j, t, i] = scipy.signal.convolve2d(patches[:, :, j] * patches_smaller[:, :, t], np.flipud(np.fliplr(W[:, :, t])), mode='valid')
    print "z3_convolved.shape (before): ", z3_convolved.shape
    z3_convolved = z3_convolved.reshape(hidden_size * number_of_patches, N)
    print "z3_convolved.shape (after): ", z3_convolved.shape
    a3_convolved = rectifier(z3_convolved)
    return a3_convolved, z3_convolved

# BP for lrf


def lrf_bp(images1, images2, delta, W, small_patch, patch_size, image_height, image_width, N, dim1, dim2):
    delta2 = np.zeros((patch_size, patch_size, dim2, N))
    patches_bigger = np.zeros((patch_size, patch_size, dim2, N))
    for i in range(N):
        patches_bigger[:, :, :, i], patches_smaller = extract_patches(images1[:, :, i], images2[:, :, i], image_height, image_width, small_patch, patch_size, dim2)
        for t in range(dim2):
            # patches = extract_patches2(patches_bigger[:, :, t, i], patch_size, small_patch, hidden_size)
            delta2[:, :, t, i] = scipy.signal.convolve2d(delta[:, :, t, i], np.flipud(np.fliplr(W[:, :, t] * patches_smaller[:, :, t])))
    return delta2 * patches_bigger


def generate_patch_pattern(patch_size, image_height, image_width):
    p = np.zeros((image_height, image_width))
    for i in range(image_height):
        if i < patch_size:
            x = i+1
        elif (i >= patch_size) and (i <= image_height - patch_size - 1):
            x = patch_size
        elif (i > image_height - patch_size - 1):
            x = image_height - i
        for j in range(image_width):
            if j < patch_size:
                y = j+1
            elif (j >= patch_size) and (j <= image_width - patch_size - 1):
                y = patch_size
            elif (j > image_width - patch_size - 1):
                y = image_width - j
            p[i, j] = x * y
            # print "x = ", x, ", y = ", y, ", i = ", i, ", j = ", j
    print "p: ", p
    # # print "flipped p: ", np.flipud(np.fliplr(p))
    # # print "flipped p2: ", np.fliplr(p)
    # # print "flipped p3: ", np.flipud(p)
    # print "concatenated: ", np.concatenate((p, np.fliplr(p)), axis=1)
    return p


def combine_all_patches_in_one_image(patches, patch_size, image_height, image_width, N):
    delta_combined = np.zeros((image_height, image_width, N))
    p = generate_patch_pattern(patch_size, image_height, image_width)

    for i in range(N):
        patchNumber = 0
        for x in range(image_height - patch_size + 1):
            for y in range(image_width - patch_size + 1):
                delta_combined[x:x + patch_size, y:y + patch_size, i] = delta_combined[x:x + patch_size, y:y + patch_size, i] + patches[:, :, patchNumber, i]
                patchNumber = patchNumber + 1
        delta_combined[:, :, i] = delta_combined[:, :, i] / p

    print "delta_combined: ", delta_combined
    return delta_combined


# a2_convolved, z2_convolved = lrf(images, W1, b1, patch_size, hidden_size, image_size)

# Extracting patches from image
# bigger_patches - patches (8x8) extracted from image1
# smaller_patches - patches (4x4) extracted from image2 where each of these patches
# correspond to patch from bigger_patches set and located at the center of corresponding 8x8 patch.


def extract_patches(image1, image2, image_height, image_width, small_patch, patch_size, number_of_patches):
    patches_bigger = np.zeros((patch_size, patch_size, number_of_patches))
    patches_smaller = np.zeros((small_patch, small_patch, number_of_patches))
    patchNumber = 0

    delta = (patch_size - small_patch)/2
    for x in range(image_height - patch_size + 1):
        for y in range(image_width - patch_size + 1):
            patches_bigger[:, :, patchNumber] = image1[x:x + patch_size, y:y + patch_size]
            patches_smaller[:, :, patchNumber] = image2[x + delta:x + delta + small_patch, y + delta:y + delta + small_patch]
            patchNumber = patchNumber + 1

    return patches_bigger, patches_smaller


def extract_patches2(image, image_size, patch_size, number_of_patches):
    patches = np.zeros((patch_size, patch_size, number_of_patches))
    patchNumber = 0
    for x in range(image_size - patch_size + 1):
        for y in range(image_size - patch_size + 1):
            patches[:, :, patchNumber] = image[
                x:x + patch_size, y:y + patch_size]
            patchNumber = patchNumber + 1

    return patches

# Maxout among filters


def maxout_layer(arr):
    print "arr.shape: ", arr.shape
    M = arr.shape[0]
    N = arr.shape[2]
    print "Number of features: ", M
    print "arr.shape: ", arr.shape

    maxout = np.zeros(shape=(M, 1, N))
    mask = np.zeros(shape=(arr.shape))

    for i in range(M):
        for j in range(N):
            max_index = np.argmax(arr[i, :, j])
            mask[i, max_index, j] = 1
            maxout[i, 0, j] = arr[i, max_index, j]

    return mask, maxout


def cost_and_grad(theta, images, targets, patch_size, small_patch, image_height, image_width, N, positive_pattern, negative_pattern):
    W11 = theta[0:patch_size ** 2].reshape(patch_size, patch_size)
    W12 = theta[patch_size ** 2:2 * patch_size ** 2].reshape(patch_size, patch_size)

    dim1 = (patch_size - small_patch + 1)**2
    dim2 = ((image_height - patch_size + 1) - patch_size + 1) * ((image_width - patch_size + 1) - patch_size + 1)

    W21 = theta[2 * patch_size ** 2:2 * patch_size ** 2 + dim2 * small_patch ** 2].reshape(small_patch, small_patch, dim2)
    W22 = theta[2 * patch_size ** 2 + dim2 * small_patch ** 2:2 * patch_size ** 2 + 2 * dim2 * small_patch ** 2].reshape(small_patch, small_patch, dim2)

    # print "W21.shape: ", W21.shape
    # print "W22.shape: ", W22.shape
    W5 = theta[2 * patch_size ** 2 + 2 * dim2 * small_patch ** 2:].reshape(dim1, dim2, 2)
    # print "W5.shape: ", W5.shape

    # print "theta.shape: ", theta.shape
    # print "W11.flatten.shape: ", W11.flatten().shape
    # print "W12.flatten.shape: ", W12.flatten().shape
    # print "W21.flatten.shape: ", W21.flatten().shape
    # print "W22.flatten.shape: ", W22.flatten().shape
    # print "sum dimension: ", W11.flatten().shape[0] + W12.flatten().shape[0] + W21.flatten().shape[0] + W22.flatten().shape[0] + W5.flatten().shape[0]

    # Feedforward

    # print "Original size images: ", image_height, "x", image_width

    # Convolution (1st layer)
    convolved_images11 = np.zeros(shape=(image_height - patch_size + 1, image_width - patch_size + 1, N), dtype=np.float64)
    convolved_images12 = np.zeros(shape=(image_height - patch_size + 1, image_width - patch_size + 1, N), dtype=np.float64)

    for j in range(N):
        convolved_images11[:, :, j] = scipy.signal.convolve2d(images[j, 0, :, :], np.flipud(np.fliplr(W11)), mode='valid')
        convolved_images12[:, :, j] = scipy.signal.convolve2d(images[j, 1, :, :], np.flipud(np.fliplr(W12)), mode='valid')

    z2_convolved11 = convolved_images11
    z2_convolved12 = convolved_images12
    a2_convolved11 = rectifier(convolved_images11)
    a2_convolved12 = rectifier(convolved_images12)

    # print "After convolution (1st layer): ", image_height - patch_size + 1, "x", image_width - patch_size + 1

    # print "z2_convolved11.shape: ", z2_convolved11.shape
    # print "a2_convolved11: ", a2_convolved11

    # LRF (2nd layer)

    a3_convolved21, z3_convolved21 = lrf(a2_convolved11, a2_convolved12, W21, small_patch, patch_size, image_height - patch_size + 1, image_width - patch_size + 1, N)
    a3_convolved22, z3_convolved22 = lrf(a2_convolved12, a2_convolved11, W22, small_patch, patch_size, image_height - patch_size + 1, image_width - patch_size + 1, N)

    print "Done most expensive part!"
    # print "After LRF (2nd layer): ", (image_height - patch_size + 1) - patch_size + 1, "x", (image_width - patch_size + 1) - patch_size + 1
    # mask31 = np.zeros(shape=(a3_convolved21.shape))
    # mask32 = np.zeros(shape=(a3_convolved21.shape))

    # - Maxout layer (3rd layer)

    # mask31, maxout31 = maxout_layer(a3_convolved21)
    # mask32, maxout32 = maxout_layer(a3_convolved22)

    a3_convolved21_3D = a3_convolved21.reshape(dim1 * dim2, 1, N)
    a3_convolved22_3D = a3_convolved22.reshape(dim1 * dim2, 1, N)

    combined_maxout = np.concatenate((a3_convolved21_3D, a3_convolved22_3D), axis=1)

    # Maxout layer (3rd layer)

    mask3, maxout3 = maxout_layer(combined_maxout)
    # print "maxout before: ", maxout4.shape
    maxout3 = maxout3.reshape(dim1, dim2, 1, N)

    pred = np.zeros((dim2, 2, N))
    cost = np.zeros((dim2, N))

    # print "(maxout4[:, i, j, 0].T).dot(W5[:, i, k]): ", (maxout4[:, 0, 0, 0].T).dot(W5[:, 0, 0])

    # Softmax layer (5th layer):
    for i in range(dim2):
        for j in range(N):
            for k in range(2):
                pred[i, k, j] = (maxout3[:, i, 0, j].T).dot(W5[:, i, k])

    denomin = np.sum(np.exp(pred), axis=1)
    for i in range(dim2):
        for j in range(N):
            for k in range(2):
                pred[i, k, j] = np.exp(pred[i, k, j])/denomin[i, j]
    # pred = np.exp(pred) / np.sum(np.exp(pred), axis=2)

    # print "pred: ", pred

    # print "maxout after: ", maxout4.shape

    # print "cost: ", cost

    # cost = np.sum(-, axis=2)

    print "cost.shape: ", cost.shape
    print "maxout4.shape: ", maxout3.shape

    # Calculating cost functions for each pixel
    for i in range(N):
        if targets[i] == 0:
            cost[:, i] = np.sum(np.log(pred[:, :, i]) * (-negative_pattern), axis=1)
        else:
            cost[:, i] = np.sum(np.log(pred[:, :, i]) * (-positive_pattern), axis=1)

    # Backpropagation!

    # First let's find a difference between prediction and randomly generated pattern
    diff = np.zeros(shape=(pred.shape))
    for i in range(N):
        if targets[i] == 0:
            diff[:, :, i] = - (pred[:, :, i] - negative_pattern)
        else:
            diff[:, :, i] = - (pred[:, :, i] - positive_pattern)

    print "diff.shape: ", diff.shape

    W5_d = np.zeros(shape=(W5.shape))
    for i in range(dim2):
        W5_d[:, i, :] = maxout3[:, i, 0, :].dot(diff[i, :, :].T)

    print "W5_d.shape: ", W5_d.shape
    print "mask4.shape: ", mask3.shape

    delta5 = np.zeros(shape=(maxout3.shape))
    for i in range(dim2):
        for j in range(N):
            delta5[:, i, 0, j] = W5[:, i, :].dot(diff[i, :, j].T)

    print "cost: ", cost

    print "average cost among all pixels: ", np.sum(np.sum(cost, axis=1)/N)/dim2

    print "delta5: ", delta5
    print "delta5.shape: ", delta5.shape

    delta3 = np.zeros(shape=(mask3.shape))
    delta5 = delta5.reshape(dim1 * dim2, 1, N)

    # print "mask4[:, 0, :].shape: ", mask4[:, 0, :].shape
    # print "delta5.shape: ", delta5.shape
    # print "(delta5 * mask4[:, 0, :]).shape: " (delta5[:, 0, :] * mask4[:, 0, :]).shape
    print "BP maxout 3!"
    for fltr in range(2):
        print "mask3.shape: ", mask3.shape
        print "delta5.shape: ", delta5.shape
        delta3[:, fltr, :] = delta5[:, 0, :] * mask3[:, fltr, :]

    delta31 = np.zeros(shape=(dim1 * dim2, N))
    delta32 = np.zeros(shape=(dim1 * dim2, N))
    delta31 = delta3[:, 0, :] * rectifier_prime(z3_convolved21)
    delta32 = delta3[:, 1, :] * rectifier_prime(z3_convolved22)
    # print "BP maxout 31!"
    # delta31 = np.zeros(shape=(mask31.shape))
    # for fltr in range(filters):
    #     delta31[:, fltr, :] = delta4[:, 0, :] * mask31[:, fltr, :]
    # print "BP maxout 32!"
    # delta32 = np.zeros(shape=(mask32.shape))
    # for fltr in range(filters):
    #     delta32[:, fltr, :] = delta4[:, 1, :] * mask32[:, fltr, :]

    # print "delta31: ", delta31
    # print "delta32: ", delta32

    # a3_convolved21, z3_convolved21 = lrf(a2_convolved11, a2_convolved12, W21, small_patch, patch_size, image_height - patch_size + 1, image_width - patch_size + 1, N)
    # a3_convolved22, z3_convolved22 = lrf(a2_convolved12, a2_convolved11, W22, small_patch, patch_size, image_height - patch_size + 1, image_width - patch_size + 1, N)

    delta31 = delta31.reshape(np.sqrt(dim1), np.sqrt(dim1), dim2, N)
    delta32 = delta32.reshape(np.sqrt(dim1), np.sqrt(dim1), dim2, N)
    delta21 = lrf_bp(a2_convolved11, a2_convolved12, delta31, W21, small_patch, patch_size, image_height - patch_size + 1, image_width - patch_size + 1, N, dim1, dim2)
    delta22 = lrf_bp(a2_convolved12, a2_convolved11, delta32, W22, small_patch, patch_size, image_height - patch_size + 1, image_width - patch_size + 1, N, dim1, dim2)

    print "delta21.shape: ", delta21.shape
    print "delta22.shape: ", delta22.shape
    # print "delta21: ", delta21

    print "height: ", image_height - patch_size + 1
    print "width: ", image_width - patch_size + 1

    delta21_combined = combine_all_patches_in_one_image(delta21, patch_size, image_height - patch_size + 1, image_width - patch_size + 1, N)
    delta22_combined = combine_all_patches_in_one_image(delta22, patch_size, image_height - patch_size + 1, image_width - patch_size + 1, N)

    delta21_combined = delta21_combined * z2_convolved11
    delta22_combined = delta22_combined * z2_convolved12
    # delta31 = delta31 * rectifier_prime(z3_convolved21)
    # delta32 = delta32 * rectifier_prime(z3_convolved22)

    # print "cost_sum: ", np.sum(cost, axis=2)

    # print "positive_pattern: ", positive_pattern.T
    # print "negative_pattern: ", negative_pattern.T

    # print "combined_maxout.shape: ", combined_maxout.shape
    # print "combined_maxout: ", combined_maxout

    # print "maxout4.shape: ", maxout4.shape
    # print "maxout4: ", maxout4

    # print "mask31: ", mask31

    # print "maxout31: ", maxout31

    # print "mask1.shape: ", mask31.shape
    # print "mask1[2, :]: ", mask31[2, :]
    # t = mask31[2, :]
    # t[5] = 1
    # print "mask1[2, :][5] = 1: ", t

    return 0, 0


def generate_random_patterns(length):
    # 0 for "same" pixels, 1 for "different" pixels
    # in the positive_pattern should be 90% same pixels
    # in the negative_pattern <10%

    positive_pattern = np.zeros((length, 2))
    negative_pattern = np.zeros((length, 2))

    pattern = np.random.choice([0, 1], length, p=[0.1, 0.9])
    positive_pattern[:, 0] = pattern
    positive_pattern[:, 1] = np.absolute(pattern-1)
    negative_pattern[:, 0] = np.absolute(pattern-1)
    negative_pattern[:, 1] = pattern

    # print "positive: ", positive_pattern
    # print "negative: ", negative_pattern
    return positive_pattern, negative_pattern


def prepare_data():
    lfw_pairs_train = fetch_lfw_pairs(subset='train')
    images = lfw_pairs_train.pairs
    targets = lfw_pairs_train.target
    # print "lfw_pairs_train.pairs.shape: ", lfw_pairs_train.pairs.shape
    # print "images.shape: ", images.shape
    # print "targets.shape: ", targets.shape

    # for i in range(2):
    #   subplot(1, 2, i + 1), imshow(images[202,i,:,:], cmap=cm.gray)

    # show()

    return images, targets


def run_model():
    patch_size = 8
    small_patch = 4
    # filters = 10

    images, targets = prepare_data()

    # choose only 6 samples from our original dataset; 3 of each type
    # (different people or same)
    start = 1099
    end = 1101
    images = images[start:end, :, :, :] / 255.0
    targets = targets[start:end]

    # print images[0,0,:,:]
    print targets

    N, pair, image_height, image_width = images.shape
    theta = initialization(patch_size, small_patch, image_height, image_width)

    # print N, pair, image_height, image_width

    # k = 1
    # for j in range(6):
    #   for i in range(2):
    #       subplot(6, 2, k), imshow(images[j, i, :, :], cmap=cm.gray)
    #       k = k + 1
    # show()

    positive_pattern, negative_pattern = generate_random_patterns(((image_height - patch_size + 1) - patch_size + 1) * ((image_width - patch_size + 1) - patch_size + 1))

    l_cost, l_grad = cost_and_grad(
        theta, images, targets, patch_size, small_patch, image_height, image_width, N, positive_pattern, negative_pattern)

run_model()

# generate_random_patterns(100)

import numpy as np
import fv_wtht_filters_and_lrf as fv
import scipy.optimize


def compute_grad(J, theta, l_grad, dim2):
    EPSILON = 0.0001
    patch_size = 8
    small_patch = 4
    # dim1 = (patch_size - small_patch + 1)**2

    grad = np.zeros(theta.shape, dtype=np.float64)

    print "grad.shape: ", grad.shape
    print "theta.shape: ", theta.shape

    # W5 = theta[2 * patch_size ** 2 + 2 * dim2 * small_patch ** 2:].reshape(dim1, dim2, 2)

    # start = 2 * patch_size ** 2 + 2 * dim2 * small_patch ** 2
    # start = 2 * patch_size ** 2
    # start = 2 * patch_size ** 2 + dim2 * small_patch ** 2
    start = 0

    # theta.shape[0]
    # for i in range(theta.shape[0]):
    # for i in range(start, start + 11):
    for i in range(start, start+11):
        theta_epsilon_plus = np.array(theta, dtype=np.float64)
        theta_epsilon_plus[i] = theta[i] + EPSILON
        theta_epsilon_minus = np.array(theta, dtype=np.float64)
        theta_epsilon_minus[i] = theta[i] - EPSILON
        # print "J(theta_epsilon_plus): ", J(theta_epsilon_plus)[0]
        # print "J(theta_epsilon_minus): ", J(theta_epsilon_minus)[0]
        # print "(J(theta_epsilon_plus) - J(theta_epsilon_minus)): ", (J(theta_epsilon_plus)[0] - J(theta_epsilon_minus)[0])
        # print "(J(theta_epsilon_plus) - J(theta_epsilon_minus)) / (2 * EPSILON): ", (J(theta_epsilon_plus)[0] - J(theta_epsilon_minus)[0]) / (2 * EPSILON)
        grad[i] = (J(theta_epsilon_plus)[0] - J(theta_epsilon_minus)[0]) / (2 * EPSILON)
        if i % 5 == 0 and i != 0:
            print "Computing gradient for input:", i
            # print "Diff: ", np.linalg.norm(grad[start:i] - l_grad[start:i]) / np.linalg.norm(grad[start:i] + l_grad[start:i])
            # print "l_grad[:i]: ", l_grad[start:i]
            # print "grad[:i]: ", grad[start:i]
            print "Diff: ", np.linalg.norm(grad[start:i] - l_grad[start:i]) / np.linalg.norm(grad[start:i] + l_grad[start:i])
            print "l_grad[:i]: ", l_grad[start:i]
            print "grad[:i]: ", grad[start:i]
    print "grad[:20]: ", grad
    return grad


def check_grad_in_our_dnn():
    patch_size = 8
    small_patch = 4
    # filters = 10

    images, targets = fv.prepare_data()

    # choose only 6 samples from our original dataset; 3 of each type
    # (different people or same)
    start = 1099
    end = 1101
    images = images[start:end, :, :, :] / 255.0
    targets = targets[start:end]

    # print images[0,0,:,:]
    # print targets

    N, pair, image_height, image_width = images.shape
    theta = fv.initialization(patch_size, small_patch, image_height, image_width)

    dim2 = ((image_height - patch_size + 1) - patch_size + 1) * ((image_width - patch_size + 1) - patch_size + 1)

    # print N, pair, image_height, image_width

    # k = 1
    # for j in range(6):
    #   for i in range(2):
    #       subplot(6, 2, k), imshow(images[j, i, :, :], cmap=cm.gray)
    #       k = k + 1
    # show()

    positive_pattern, negative_pattern = fv.generate_random_patterns(((image_height - patch_size + 1) - patch_size + 1) * ((image_width - patch_size + 1) - patch_size + 1))

    l_cost, l_grad = fv.cost_and_grad(theta, images, targets, patch_size, small_patch, image_height, image_width, N, positive_pattern, negative_pattern)

    J = lambda x: fv.cost_and_grad(x, images, targets, patch_size, small_patch, image_height, image_width, N, positive_pattern, negative_pattern)

    computed_grad = compute_grad(J, theta, l_grad, dim2)
    # options_ = {'maxiter': 50, 'disp': True}
    # result = scipy.optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options=options_)
    # theta = result.x

    print np.linalg.norm(computed_grad - l_grad[:20]) / np.linalg.norm(computed_grad + l_grad[:20])

    # print "computed_grad: ", compute_grad
    print "l_grad: ", l_grad[:20]


check_grad_in_our_dnn()

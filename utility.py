import random
import os

import cv2 as cv
import matplotlib.pyplot as plt

from Agent import Agent
from PSO import *
import skimage.measure
from random import randint


# shows histogram of all 3 channels
def color_hist(img):
    y = np.linspace(0, 256)
    fig, ax = plt.subplots(3, 1)
    ax[0].hist(img[:, :, 0].flatten().ravel(), color='blue', bins=256)
    ax[1].hist(img[:, :, 1].flatten().ravel(), color='green', bins=256)
    ax[2].hist(img[:, :, 2].flatten().ravel(), color='red', bins=256)

    plt.show()


def plot_hist(img):
    plt.hist(img.flatten(), bins=150)
    plt.show()


# stacking BGR channels in order after computation
def image(input):
    val = list(input)

    for p in range(len(val)):
        if val[p][1] == "B":
            b = val[p][0]
        elif val[p][1] == "G":
            g = val[p][0]
        if val[p][1] == "R":
            r = val[p][0]
    img = np.dstack([b, g, r])
    img = np.array(img, dtype=np.uint8)

    return img


# Indicating superior, inferior and intermediate channels based on mean of pixels in channel
def superior_inferior_split(img):
    B, G, R = cv.split(img)

    pixel = {"B": np.mean(B), "G": np.mean(G), "R": np.mean(R)}
    pixel_ordered = dict(sorted(pixel.items(), key=lambda x: x[1], reverse=True))

    # Classifying Maximum, Minimum and Intermediate channels of image
    label = ["Pmax", "Pint", "Pmin"]
    chanel = {}

    for i, j in zip(range(len(label)), pixel_ordered.keys()):
        if j == "B":
            chanel[label[i]] = list([B, j])

        elif j == "G":
            chanel[label[i]] = list([G, j])

        else:
            chanel[label[i]] = list([R, j])

    return chanel


def neutralize_image(img):
    track = superior_inferior_split(img)

    Pmax = track["Pmax"][0]
    Pint = track["Pint"][0]
    Pmin = track["Pmin"][0]

    # gain_factor Pint
    J = (np.sum(Pmax) - np.sum(Pint)) / (np.sum(Pmax) + np.sum(Pint))

    # gain_factor Pmin
    K = (np.sum(Pmax) - np.sum(Pmin)) / (np.sum(Pmax) + np.sum(Pmin))

    track["Pint"][0] = Pint + (J * Pmax)
    track["Pmin"][0] = Pmin + (K * Pmax)

    # neutralised image
    neu_img = image(track.values())

    return neu_img


def Stretching(image):
    LSR_img = []  # for lower stretched image
    USR_img = []  # for upper stretched image
    height, width = image.shape[:2]

    for i in range(image.shape[2]):
        img_hist = image[:, :, i]
        max_P = np.max(img_hist)
        min_P = np.min(img_hist)

        mean_P = np.mean(img_hist)
        median_P = np.median(img_hist)

        avg_point = (mean_P + median_P) / 2

        LS_img = np.zeros((height, width))
        US_img = np.zeros((height, width))

        for i in range(0, height):
            for j in range(0, width):
                if img_hist[i][j] < avg_point:
                    LS_img[i][j] = int(((img_hist[i][j] - min_P) * ((255 - min_P) / (avg_point - min_P)) + min_P))
                    US_img[i][j] = 0
                    # array_upper_histogram_stretching[i][j] = p_out
                else:
                    LS_img[i][j] = 255
                    US_img[i][j] = int(((img_hist[i][j] - avg_point) * ((255) / (max_P - avg_point))))

        LSR_img.append(LS_img)
        USR_img.append(US_img)

    LS = np.array(np.dstack(LSR_img), dtype=np.uint8)
    US = np.array(np.dstack(USR_img), dtype=np.uint8)

    return LS, US


def enhanced_image(img1, img2):
    # integration of dual intensity images to get Enhanced-constrast output image
    b1, g1, r1 = cv.split(img1)
    b2, g2, r2 = cv.split(img2)

    height, width = img1.shape[:2]
    dual_img = np.zeros((height, width, 3), dtype=np.uint8)

    dual_img[:, :, 0] = np.array(np.add(b1 / 2, b2 / 2), dtype=np.uint8)
    dual_img[:, :, 1] = np.array(np.add(g1 / 2, g2 / 2), dtype=np.uint8)
    dual_img[:, :, 2] = np.array(np.add(r1 / 2, r2 / 2), dtype=np.uint8)

    return dual_img


def pso_image(img, params, iteration):
    group = superior_inferior_split(img)

    maxi = np.mean(group["Pmax"][0])
    inte = np.mean(group["Pint"][0])
    mini = np.mean(group["Pmin"][0])

    # Defining hyperparameters
    n = 50  # number of particles
    max_iteration = iteration

    x = np.array([inte, mini])

    def func(X, P_sup=maxi):
        return np.square(P_sup - X[0]) + np.square(P_sup - X[1])

    nVar = 2  # number of variables to optimize
    VarMin = 0  # lower bound of variables , you can use np.array() for different variables
    VarMax = 255  # upper bound of variables, you can use np.array() for different variables

    gbest = pso(func, max_iter=max_iteration, num_particles=n, dim=2, vmin=VarMin, vmax=VarMax, params=params)

    # gamma correction for inferior color channels
    mean_colors = gbest['position']
    gamma = np.log(mean_colors / 255) / np.log(x / 255)

    group["Pint"][0] = np.array(255 * np.power(group["Pint"][0] / 255, gamma[0]))
    group["Pmin"][0] = np.array(255 * np.power(group["Pmin"][0] / 255, gamma[1]))

    pso_res = image(group.values())

    return pso_res


def unsharp_masking(img):
    alpha = 0.2
    beta = 1 - alpha
    img_blur = cv.GaussianBlur(img, (1, 1), sigmaX=1)
    unsharp_img = cv.addWeighted(img, alpha, img_blur, beta, 0.0)

    return unsharp_img


def NUCE(img, params, iteration):
    # superior based underwater color cast neutralization
    neu_img = neutralize_image(img)

    # Dual-intensity images fusion based on average of mean and median values
    img1, img2 = Stretching(neu_img)
    dual_img = enhanced_image(img1, img2)

    # Swarm-intelligence based mean equalization
    pso_res = pso_image(dual_img, params, iteration)

    # Unsharp masking
    nuce_img = unsharp_masking(pso_res)

    return nuce_img


def SDS_NUCE(orginalImage, iteration, agentNumber, pso_iteration):
    it = 0

    wmax_init = 0.9
    wmin_init = 0.4
    c1_init = 2
    c2_init = 2

    params = {"wmax": wmax_init, "wmin": wmin_init, "c1": c1_init, "c2": c2_init}

    pso_image = NUCE(orginalImage, params, pso_iteration)
    best_entropy = entropy(pso_image)

    wmax_range = np.arange(0.9, 1, 0.001);
    wmin_range = np.arange(0, 0.1, 0.001);
    c1_range = np.arange(1, 5, 1);
    c2_range = np.arange(1, 5, 1);

    agents = [Agent(i.__index__(), 0.00, False, False) for i in range(agentNumber)]

    for i in params:

        best_agent = None

        while it < iteration:

            for agent in agents:

                rand_number = randint(0, (2 * agentNumber) - 1) / (2 * agentNumber)

                if i == 'wmax':
                    if not agent.IsHappy and not agent.IsFollowed:
                        min_value = (wmax_range.max() - wmax_range.min()) * rand_number + wmax_range.min()
                        max_value = ((1 / (2 * agentNumber)) * (wmax_range.max() - wmax_range.min())) + min_value
                        agent.Values = [value for value in wmax_range if min_value < value <= max_value]

                    agent.Value = randint(0, len(agent.Values)) * (max(agent.Values) - min(agent.Values)) + min(
                        agent.Values)

                    params = {"wmax": agent.Value, "wmin": wmin_init, "c1": c1_init, "c2": c2_init}

                if i == 'wmin':
                    if not agent.IsHappy and not agent.IsFollowed:
                        min_value = (wmin_range.max() - wmin_range.min()) * rand_number + wmin_range.min()
                        max_value = ((1 / (2 * agentNumber)) * (wmin_range.max() - wmin_range.min())) + min_value
                        agent.Values = [value for value in wmin_range if min_value < value <= max_value]

                    agent.Value = randint(0, len(agent.Values)) * (max(agent.Values) - min(agent.Values)) + min(
                        agent.Values)

                    params = {"wmax": wmax_init, "wmin": agent.Value, "c1": c1_init, "c2": c2_init}

                if i == 'c1':
                    if not agent.IsHappy and not agent.IsFollowed:
                        min_value = (c1_range.max() - c1_range.min()) * rand_number + c1_range.min()
                        max_value = ((1 / (2 * agentNumber)) * (c1_range.max() - c1_range.min())) + min_value
                        agent.Values = [value for value in c1_range if min_value < value <= max_value]

                    agent.Value = randint(0, len(agent.Values)) * (max(agent.Values) - min(agent.Values)) + min(
                        agent.Values)

                    params = {"wmax": wmax_init, "wmin": wmin_init, "c1": agent.Value, "c2": c2_init}

                if i == 'c2':
                    if not agent.IsHappy and not agent.IsFollowed:
                        min_value = (c2_range.max() - c2_range.min()) * rand_number + c2_range.min()
                        max_value = ((1 / (2 * agentNumber)) * (c2_range.max() - c2_range.min())) + min_value
                        agent.Values = [value for value in c2_range if min_value < value <= max_value]

                    agent.Value = randint(0, len(agent.Values)) * (max(agent.Values) - min(agent.Values)) + min(
                        agent.Values)

                    params = {"wmax": wmax_init, "wmin": wmin_init, "c1": c1_init, "c2": agent.Value}

                nuce_img = NUCE(orginalImage, params, pso_iteration)
                agent.Entropy = entropy(nuce_img)
                agent.IsHappy = agent.Entropy > best_entropy

            best_entropy = max(max(agent.Entropy for agent in agents), best_entropy)

            for agent in agents:
                if not agent.IsHappy:
                    other_agents = [other_agent for other_agent in agents if other_agent != agent]
                    random_agent = random.choice(other_agents)

                    if random_agent.IsHappy:
                        agent.Values = random_agent.Values
                        agent.IsFollowed = True

                if agent.Entropy == best_entropy:
                    best_agent = agent

            it += 1

        if i == 'wmax' and best_agent is not None:
            wmax_init = best_agent.Value

        if i == 'wmin' and best_agent is not None:
            wmin_init = best_agent.Value

        if i == 'c1' and best_agent is not None:
            c1_init = best_agent.Value

        if i == 'c2' and best_agent is not None:
            c2_init = best_agent.Value

    params = {"wmax": wmax_init, "wmin": wmin_init, "c1": c1_init, "c2": c2_init}

    sds_pso_image = NUCE(orginalImage, params, pso_iteration)
    return sds_pso_image


def entropy(img):
    return skimage.measure.shannon_entropy(img)


def execute_method(sds_iteration, sds_agent_number, it, create_image):
    original_entropies = []
    nuce_entropies = []
    sds_nuce_entropies = []

    original_images = []
    NUCE_images = []

    img_w = 350  # image width
    img_h = 350  # image height

    entropies = []

    dir_path = "./images/"

    for im in os.listdir(dir_path):
        img = cv.imread(dir_path + im, 1)
        img = cv.resize(img, (img_w, img_h))
        original_images.append(img)

        sds_nuce_img = SDS_NUCE(img, sds_iteration, sds_agent_number, it)

        if create_image:
            cv.imwrite("./results_new/" + im.split('/')[-1], sds_nuce_img)

        params = {"wmax": 0.9, "wmin": 0.4, "c1": 2, "c2": 2}
        nuce_img = NUCE(img, params, it)
        NUCE_images.append(nuce_img)

        if create_image:
            cv.imwrite("./results/" + im.split('/')[-1], nuce_img)

        original_entropy = entropy(img)
        nuce_entropy = entropy(nuce_img)
        sds_nuce_entropy = entropy(sds_nuce_img)

        entropies.append("For " + im.split('/')[-1] + ": " + "Original Entropy = " + str(
            original_entropy) + " , NUCE Entropy = " + str(nuce_entropy) + " , SDS NUCE Entropy = " + str(
            sds_nuce_entropy))

        original_entropies.append(original_entropy)
        nuce_entropies.append(nuce_entropy)
        sds_nuce_entropies.append(sds_nuce_entropy)

    if create_image:
        fig, ax = plt.subplots(4, 2, figsize=(6, 9), constrained_layout=False)
        ax[0][0].set_title("Original Image")
        ax[0][1].set_title("NUCE Image")

        for i in range(4):
            ax[i][0].imshow(cv.cvtColor(original_images[i], cv.COLOR_BGR2RGB), cmap='gray')
            ax[i][0].axis('off')

            ax[i][1].imshow(cv.cvtColor(NUCE_images[i], cv.COLOR_BGR2RGB), cmap='gray')
            ax[i][1].axis('off')

        fig.tight_layout()
        plt.savefig("./results/output.jpg")
        plt.show()
    #
    # for item in entropies:
    #     print(item)

    print('iteration number ' + str(it) + ' finished.')
    print('original_entropies: ' + str(np.average(original_entropies)) + 'nuce_entropies: ' + str(np.average(nuce_entropies)) + 'sds_nuce_entropies: ' + str(np.average(sds_nuce_entropies)))
    return np.average(original_entropies), np.average(nuce_entropies), np.average(sds_nuce_entropies)

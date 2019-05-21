import numpy as np
import matplotlib.pyplot as plt

def Alphas(layers=5, step_epochs=500, padding_epochs=1500):
    """
    :param layers:
    :param step_epochs:
    :param padding_epochs:
    :return: interchangeably turn on first N layers connections [0-1], keep Nth layer on 1
    """
    ALPHAS = layers
    EPOCHS_STEP = step_epochs
    PADDING = padding_epochs
    block = np.zeros(shape=(3 * EPOCHS_STEP), dtype=np.float32)
    ls_dsc = np.linspace(1.0, 0.0, EPOCHS_STEP, endpoint=True, dtype=np.float32, axis=0)
    ls_asc = np.linspace(0.0, 1.0, EPOCHS_STEP, endpoint=True, dtype=np.float32, axis=0)
    block[0:EPOCHS_STEP] = ls_asc
    block[EPOCHS_STEP:2 * EPOCHS_STEP] = 1.0
    block[2 * EPOCHS_STEP:3 * EPOCHS_STEP] = ls_dsc
    array = np.zeros(shape=(EPOCHS_STEP * 2 * ALPHAS + PADDING + int(2.5 * EPOCHS_STEP), ALPHAS), dtype=np.float32)
    for i in range(ALPHAS):
        array[(2 * i + 1) * EPOCHS_STEP:(2 * i + 4) * EPOCHS_STEP, i] = block
        if i == ALPHAS - 1:
            array[(2 * i + 3) * EPOCHS_STEP:, i] = 1.0
    array = array[int(2.5 * EPOCHS_STEP):]
    return array


def Betas(layers=5, step_epochs=500, padding_epochs=1500):
    """
    :param layers:
    :param step_epochs:
    :param padding_epochs:
    :return: Progressively fading across N layers, starting with 0th, but never fading Nth
    """
    ALPHAS = layers
    EPOCHS_STEP = step_epochs
    PADDING = padding_epochs

    betas = np.ones(shape=(EPOCHS_STEP * 2 * ALPHAS + PADDING + int(EPOCHS_STEP * 0.5), ALPHAS), dtype=np.float32)
    ls = np.linspace(1.0, 0.0, EPOCHS_STEP, endpoint=True, dtype=np.float32, axis=0)
    for i in range(ALPHAS):
        betas[(2 * i + 1) * EPOCHS_STEP:(2 * i + 2) * EPOCHS_STEP, i] = ls
        betas[(2 * i + 2) * EPOCHS_STEP:, i] = 0.0
    betas[:, ALPHAS - 1] = 1.0
    betas = betas[int(EPOCHS_STEP * 0.5):]
    return betas


plt.plot(Betas())
plt.show()
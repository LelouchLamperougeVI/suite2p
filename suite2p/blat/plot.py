import matplotlib.pyplot as plt
import numpy as np

def stack(analysis, pc_only=True):
    analysis = analysis['analysis']
    if pc_only:
        ispc = analysis['ispc']
    else:
        ispc = np.ones_like(analysis['ispc'].shape[0]).astype(bool)
    stack = analysis['smooth']['stack'][ispc, :].T
    stack = stack[:, np.sum(stack, axis=0) > 0]
    stack = (stack - np.min(stack, axis=0)) / np.ptp(stack, axis=0)
    idx = np.argmax(stack, axis=0)
    idx = np.argsort(idx)
    
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(stack[:, idx].T)
    axs[1].imshow(np.corrcoef(stack))

    return fig, axs
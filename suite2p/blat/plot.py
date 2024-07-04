import matplotlib.pyplot as plt
import numpy as np
from suite2p.blat import utils

def bayes(analysis):
    plt.rcParams['figure.figsize'] = [11, 8]
    length = np.max(analysis.behaviour['position'][analysis.behaviour['movement']])
    t = np.linspace(0, analysis.bayes['real'].shape[0] / analysis.behaviour['fs'], analysis.bayes['real'].shape[0])
    sigma = analysis.ops['bayes']['sigma'] * analysis.ops['bayes']['bins'] / length
    rewards = analysis.behaviour['position'][analysis.behaviour['reward']]
    analysis = analysis.bayes

    fig, axs = plt.subplots(2, 2)
    gs = axs[0, 0].get_gridspec()
    for ax in axs[0, :]:
        ax.remove()
    ax = fig.add_subplot(gs[0, :])
    ax.plot(t, analysis['real'])
    ax.plot(t, analysis['decoded'])
    ax.set_xlabel('time (sec)')
    ax.set_ylabel('position (cm)')
    ax.set_yticks([0, length])
    axs[1, 0].imshow(analysis['cm'], extent=[0, length, length, 0])
    axs[1, 0].set_title('confusion matrix')
    axs[1, 0].set_xlabel('decoded position (cm)')
    axs[1, 0].set_ylabel('real position (cm)')
    axs[1, 0].set_xticks([0, length])
    axs[1, 0].set_yticks([length, 0])
    error = utils.fast_smooth(analysis['error']['error'], sigma)
    axs[1, 1].plot(np.linspace(0, length, analysis['error']['error'].shape[0]), error)
    axs[1, 1].fill_between(np.linspace(0, length, analysis['error']['error'].shape[0]), \
                           error - utils.fast_smooth(analysis['error']['sem'], sigma), \
                           error + utils.fast_smooth(analysis['error']['sem'], sigma), alpha=.5)
    axs[1, 1].vlines(rewards, np.min(error), np.max(error))
    axs[1, 1].set_xlabel('position (cm)')
    axs[1, 1].set_ylabel('absolute decoding error (cm)')
    axs[1, 1].set_xticks([0, length])
    # axs[1, 1].set_yticks([length, 0])
    

def stack(analysis, pc_only=True, evenodd=True):
    length = np.max(analysis.behaviour['position'][analysis.behaviour['movement']])
    analysis = analysis.analysis
    if pc_only:
        ispc = analysis['ispc']
    else:
        ispc = np.ones_like(analysis['ispc'].shape[0]).astype(bool)

    if evenodd:
        plt.rcParams['figure.figsize'] = [5, 8]
        stack = analysis['smooth']['stack'][ispc, :].T
        order = np.argsort(np.argmax(stack, axis=0))
        stack = analysis['smooth']['rasters'][ispc, :, :]
        even = np.mean(stack[:, :, ::2], axis=2).T
        even = (even - np.min(even, axis=0)) / np.ptp(even, axis=0)
        odd = np.mean(stack[:, :, 1::2], axis=2).T
        odd = (odd - np.min(odd, axis=0)) / np.ptp(odd, axis=0)
        
        fig, axs = plt.subplots(2, 2, height_ratios=[2, 1])
        axs[0, 0].imshow(even[:, order].T, aspect='auto', extent=[0, length, even.shape[1], 1])
        axs[0, 0].set_box_aspect(2)
        axs[0, 0].set_ylabel('neurons')
        axs[0, 0].set_xticks([0, length])
        axs[0, 0].set_yticks([even.shape[1], 1])
        axs[0, 0].set_title('even laps')
        axs[1, 0].imshow(utils.corr(even, odd, axis=1), extent=[0, length, length, 0])
        axs[1, 0].set_ylabel('even')
        axs[1, 0].set_xlabel('odd')
        axs[1, 0].set_xticks([0, length])
        axs[1, 0].set_yticks([length, 0])
        axs[1, 0].set_title('PV correlation')
        axs[0, 1].imshow(odd[:, order].T, aspect='auto', extent=[0, length, even.shape[1], 1])
        axs[0, 1].set_box_aspect(2)
        axs[0, 1].set_xlabel('position (cm)')
        axs[0, 1].set_xticks([0, length])
        axs[0, 1].set_yticks([])
        axs[0, 1].set_title('odd laps')
        axs[1, 1].remove()
        
    else:
        plt.rcParams['figure.figsize'] = [5, 5]
        stack = analysis['smooth']['stack'][ispc, :].T
        stack = (stack - np.min(stack, axis=0)) / np.ptp(stack, axis=0)
        order = np.argsort(np.argmax(stack, axis=0))
        
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(stack[:, order].T, aspect='auto', extent=[0, length, stack.shape[1], 1])
        axs[0].set_box_aspect(2)
        axs[0].set_ylabel('neurons')
        axs[0].set_xlabel('position (cm)')
        axs[0].set_xticks([0, length])
        axs[0].set_yticks([stack.shape[1], 1])
        axs[1].imshow(utils.corr(stack, axis=1), extent=[0, length, length, 0])
        axs[1].set_xlabel('position (cm)')
        axs[1].set_xticks([0, length])
        axs[1].set_yticks([])
        axs[1].set_title('PV correlation')
        

def rasters(analysis, k=8, pc_only=True):
    plt.rcParams['figure.figsize'] = [6.5, 6.5]
    
    length = np.max(analysis.behaviour['position'][analysis.behaviour['movement']])
    laps = analysis.behaviour['trial'].shape[0] - 1
    analysis = analysis.analysis
    if pc_only:
        ispc = analysis['ispc']
    else:
        ispc = np.ones_like(analysis['ispc'].shape[0]).astype(bool)

    rasters = analysis['smooth']['rasters'][ispc, :, :]
    stack = analysis['smooth']['stack'][ispc, :]
    SI = analysis['SI'][ispc]
    idx = np.argsort(SI)
    idx = idx[-1:-(k**2+1):-1]
    order = np.argsort(np.argmax(stack[idx, :], axis=1))
    order = idx[order]

    fig, axs = plt.subplots(k, k)
    for i, ax in zip(order, axs.flat):
        ax.imshow(-rasters[i, :, :].T, cmap='gray', aspect='auto', \
                  extent=[0, length, laps, 1])
        if ax is axs.flat[-1]:
            ax.set_xlabel('position (cm)')
            ax.set_ylabel('lap')
            ax.set_xticks([0, length])
            ax.set_yticks([laps, 1])
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
        else:
            ax.set_xticks([])
            ax.set_yticks([])
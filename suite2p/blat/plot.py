import matplotlib.pyplot as plt
import numpy as np
from suite2p.blat import utils
from skimage import exposure

def crossdays(planes, reg):
    plt.rcParams['figure.figsize'] = [4, 8]
    length = np.max(planes[0].behaviour['position'][planes[0].behaviour['movement'] & (planes[0].behaviour['epochs'] == 2)])
    
    fig1, axs1 = plt.subplots(len(planes), len(planes))
    # fig2, axs2 = plt.subplots(len(planes), len(planes))
    pv = []
    stab = []
    for i in range(len(planes)):
        idx = list({k for k in range(len(planes))}.difference({i}))
        stable = np.array([r[i] for r in reg])
        stable = np.any(~np.isnan(stable[idx, :]), axis=0)
        
        rstack = planes[i].analysis['smooth']['stack'].T
        rstack = (rstack - np.min(rstack, axis=0)) / np.ptp(rstack, axis=0)
        order = np.argsort(np.argmax(rstack, axis=0))
        stable = stable[order]

        rstack = rstack[:, order]
        rstack = rstack[:, stable]
        pv.append([])
        stab.append([])
        for j in range(len(planes)):
            stack = planes[j].analysis['smooth']['stack'].T
            stack = (stack - np.min(stack, axis=0)) / np.ptp(stack, axis=0)
            idx = reg[j][i][order]
            sstack = np.empty((stack.shape[0], idx.shape[0]))
            sstack.fill(np.nan)
            sstack[:, ~np.isnan(idx)] = stack[:, idx[~np.isnan(idx)].astype(int)]
            sstack = sstack[:, stable]
            sstack[np.isnan(sstack)] = 0
            axs1[i, j].imshow(sstack.T, aspect='auto', interpolation='none', extent=[0, length, sstack.shape[1], 1])
            axs1[i, j].set_xticks([])
            axs1[i, j].set_yticks([])
            if i == 0:
                axs1[i, j].set_title('day ' + str(j))
            if j == 0:
                axs1[i, j].set_yticks([1, sstack.shape[1]])
                if i == (len(planes)-1):
                    axs1[i, j].set_xticks([0, length])
                    axs1[i, j].set_xlabel('position (cm)')
                    axs1[i, j].set_ylabel('neurons')
            # axs2[i, j].imshow(utils.corr(rstack, sstack), interpolation='none')
            pv[i].append([])
            stab[i].append([])
            pv[i][j] = [np.diag(utils.corr(rstack, sstack, axis=0))]
            stab[i][j] = [np.diag(utils.corr(rstack, sstack, axis=1))]

    return pv, stab
            

def mimg(analysis):
    plt.rcParams['figure.figsize'] = [8, 8]
    mimg = analysis.mimg.copy()
    # mimg[:, :, 0] = exposure.adjust_gamma(mimg[:, :, 0])
    fig, ax = plt.subplots()
    ax.imshow(mimg)
    ax.set_xticks([])
    ax.set_yticks([])

def bayes(analysis):
    plt.rcParams['figure.figsize'] = [11, 8]
    length = np.max(analysis.behaviour['position'][analysis.behaviour['movement'] & (analysis.behaviour['epochs'] == 2)])
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
    

def stack(analysis, pc_only=True, evenodd=True, ispc=None, length=180.0):
    if ispc is None:
        if pc_only:
            ispc = analysis['ispc']
        else:
            ispc = np.ones_like(analysis['ispc']).astype(bool)

    if evenodd:
        plt.rcParams['figure.figsize'] = [5, 8]
        # stack = analysis['smooth']['stack'][ispc, :].T
        # order = np.argsort(np.argmax(stack, axis=0))
        stack = analysis['smooth']['rasters'][ispc, :, :]
        even = np.mean(stack[:, :, ::2], axis=2).T
        even = (even - np.min(even, axis=0)) / np.ptp(even, axis=0)
        odd = np.mean(stack[:, :, 1::2], axis=2).T
        odd = (odd - np.min(odd, axis=0)) / np.ptp(odd, axis=0)

        order = np.argsort(np.argmax(even, axis=0))
        
        fig, axs = plt.subplots(2, 2, height_ratios=[2, 1])
        axs[0, 0].imshow(even[:, order].T, aspect='auto', extent=[0, length, even.shape[1], 1], interpolation='none')
        axs[0, 0].set_box_aspect(2)
        axs[0, 0].set_ylabel('neurons')
        axs[0, 0].set_xticks([0, length])
        axs[0, 0].set_yticks([even.shape[1], 1])
        axs[0, 0].set_title('even laps')
        axs[1, 0].imshow(utils.corr(even, odd, axis=1), extent=[0, length, length, 0], interpolation='none')
        axs[1, 0].set_ylabel('even')
        axs[1, 0].set_xlabel('odd')
        axs[1, 0].set_xticks([0, length])
        axs[1, 0].set_yticks([length, 0])
        axs[1, 0].set_title('PV correlation')
        axs[0, 1].imshow(odd[:, order].T, aspect='auto', extent=[0, length, even.shape[1], 1], interpolation='none')
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
        axs[0].imshow(stack[:, order].T, aspect='auto', extent=[0, length, stack.shape[1], 1], interpolation='none')
        axs[0].set_box_aspect(2)
        axs[0].set_ylabel('neurons')
        axs[0].set_xlabel('position (cm)')
        axs[0].set_xticks([0, length])
        axs[0].set_yticks([stack.shape[1], 1])
        axs[1].imshow(utils.corr(stack, axis=1), extent=[0, length, length, 0], interpolation='none')
        axs[1].set_xlabel('position (cm)')
        axs[1].set_xticks([0, length])
        axs[1].set_yticks([])
        axs[1].set_title('PV correlation')
        

def rasters(analysis, k=8, pc_only=True, ispc=None, length=180.0, sort=True, hline=None, contex=None):
    plt.rcParams['figure.figsize'] = [6.5, 6.5]
    
    # length = np.max(analysis.behaviour['position'][analysis.behaviour['movement']])
    laps = analysis['smooth']['rasters'].shape[2]
    # analysis = analysis.analysis
    if ispc is None:
        if pc_only:
            ispc = analysis['ispc']
        else:
            ispc = np.ones_like(analysis['ispc']).astype(bool)

    rasters = analysis['smooth']['rasters'][ispc, :, :]
    stack = analysis['smooth']['stack'][ispc, :]
    if sort:
        SI = analysis['SI'][ispc]
        idx = np.argsort(SI)
        idx = idx[-1:-(k**2+1):-1]
        order = np.argsort(np.argmax(stack[idx, :], axis=1))
        order = idx[order]
    else:
        order = np.flatnonzero(ispc)

    if type(ispc[0]) is np.dtype('bool') or bool:
        ispc = np.flatnonzero(ispc)

    fig, axs = plt.subplots(k, k)
    for i, ax in zip(order, axs.flat):
        ax.imshow(-rasters[i, :, :].T, cmap='gray', aspect='auto', \
                  extent=[0, length, laps, 1], interpolation='none')
        ax.set_title(str(ispc[i]), fontdict={'fontsize': 8})
        if hline is not None:
            ax.hlines(hline, 0, length, colors='black')
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
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2014, Nicolas P. Rougier
# Distributed under the (new) BSD License.
#
# Contributors: Nicolas P. Rougier (Nicolas.Rougier@inria.fr)
# -----------------------------------------------------------------------------
# References:
#
# * Interaction between cognitive and motor cortico-basal ganglia loops during
#   decision making: a computational study. M. Guthrie, A. Leblois, A. Garenne,
#   and T. Boraud. Journal of Neurophysiology, 109:3025–3040, 2013.
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt


def display_ctx(sample_history, duration=3.0, filename=None):
    fig = plt.figure(figsize=(12,8))
    plt.subplots_adjust(bottom=0.15)
    (ax_begin, ax_middle, ax_end) = fig.subplots(3, 1, sharex=True)

    fig.patch.set_facecolor('.9')
    colors = ['r', 'b', 'g', 'y', 'k', 'c']
    cues = ['red_pillar', 'blue_pillar', 'brown_pillar', 'yellow_pillar']
    directions = ['right_c', 'left_c', 'right_f', 'left_c']
    plotAxis(duration, ax_begin, sample_history[0], colors, cues=cues, directions=directions)
    plotAxis(duration, ax_middle, sample_history[1], colors)
    plotAxis(duration, ax_end, sample_history[2], colors)

    ax_end.set_xlabel("Time (seconds)")
    ax_begin.set_ylabel("Activity (Hz)")
    ax_middle.set_ylabel("Activity (Hz)")
    ax_end.set_ylabel("Activity (Hz)")
    ax_begin.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
              ncol=4, fancybox=True, shadow=True)
    ax_begin.set_ylim(0.0,80)
    ax_end.set_xlim(0.0,duration)
#    plt.ylim(0.0,300.0)

#    plt.xticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
#               ['0.0','0.5\n(Trial start)','1.0','1.5', '2.0','2.5\n(Trial stop)','3.0'])

    if filename is not None:
        plt.savefig(filename)
    plt.show()

def display_choice_stays(mf, mb, num_trials, cumuRews) :
    labels = ['rewarded', 'unrewarded']

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig = plt.figure(figsize=(12,4))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2, sharey=ax1)
    ax3 = fig.add_subplot(1, 3, 3)
    (common, rare) = mf
    rects1 = ax1.bar(x - width / 2, common, width, label='Common')
    rects2 = ax1.bar(x + width / 2, rare, width, label='Rare')
    (common, rare) = mb
    rects1 = ax2.bar(x - width / 2, common, width, label='Common')
    rects2 = ax2.bar(x + width / 2, rare, width, label='Rare')
    timesteps = np.linspace(0, num_trials, num_trials)
    ax3.plot(timesteps, cumuRews[0].mean(axis=0), color='r', label='MF')
    ax3.plot(timesteps, cumuRews[1].mean(axis=0), color='b', label='MB')
    ax3.legend(frameon=False, loc='upper left')
    ax3.set_ylim(0, 100)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    #fig.set_title('Scores by transition and reward')
    ax1.set_ylim(0.5, 1)
    ax1.set_ylabel('Scores')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()
    ax1.set_title('Model-free')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_title('Model-based')
    ax3.set_title('Cumulative Reward')

    def autolabel(ax, rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

#    autolabel(rects1)
#    autolabel(rects2)

    fig.tight_layout()


def display_ctx_2step(sample_history, duration=3.0, filename=None):
    fig = plt.figure(figsize=(12,8))
    plt.subplots_adjust(bottom=0.15)
    (ax_begin, ax_middle, ax_end) = fig.subplots(3, 1, sharex=True)

    fig.patch.set_facecolor('.9')
    colors = ['k', 'c', 'r', 'b', 'g', 'y']
    plotAxis(duration, ax_begin, sample_history[0], colors)
    plotAxis(duration, ax_middle, sample_history[2], colors)
    plotAxis(duration, ax_end, sample_history[4], colors)

    colors = ['r', 'b', 'g', 'y', 'k', 'c']
    plotAxis(duration, ax_begin, sample_history[1], colors)
    plotAxis(duration, ax_middle, sample_history[3], colors)
    plotAxis(duration, ax_end, sample_history[5], colors)

    ax_end.set_xlabel("Time (seconds)")
    ax_begin.set_ylabel("Activity (Hz)")
    ax_middle.set_ylabel("Activity (Hz)")
    ax_end.set_ylabel("Activity (Hz)")
    ax_begin.legend(frameon=False, loc='upper left')
    ax_end.set_xlim(0.0,duration)
#    plt.ylim(0.0,300.0)

#    plt.xticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
#               ['0.0','0.5\n(Trial start)','1.0','1.5', '2.0','2.5\n(Trial stop)','3.0'])

    if filename is not None:
        plt.savefig(filename)
    #plt.show()

def plotProgression(num_trials, perf_data, ax):
    timesteps = np.linspace(0, num_trials, num_trials)
    ax.plot(timesteps, perf_data.mean(axis=0), c='b', lw=2)
    ax.plot(timesteps, perf_data.mean(axis=0) + perf_data.var(axis=0), c='b', lw=.5)
    ax.plot(timesteps, perf_data.mean(axis=0) - perf_data.var(axis=0), c='b', lw=.5)
    ax.fill_between(timesteps, perf_data.mean(axis=0) + perf_data.var(axis=0),
                    perf_data.mean(axis=0) - perf_data.var(axis=0), color='b', alpha=.1)
    #ax.set_xlabel("Trial number", fontsize=16)
    #ax.set_ylabel("Performance", fontsize=16)
    #ax.set_ylim(0, 1.0)
    #ax.set_xlim(1, num_trials)


def plotWeights(num_trials, WW, ax):
    timesteps = np.linspace(0, num_trials, num_trials)
    colors = ['r', 'b', 'g', 'y', 'k', 'c']
    n = WW[0].size
    for i in range(n - 1):
        ax.plot(timesteps, WW[:, i], c=colors[i])
    ax.plot(timesteps, WW[:, i + 1], c=colors[i + 1], label="Cognitive Cortex")

def plotAxis(duration, ax, history, colors, cues=[], directions=[]):
    if cues == [] : cues = colors
    if directions == []: directions = colors
    timesteps = np.linspace(0,duration, len(history))
    n = history["CTX"]["cog"][0].size
    for i in range(n):
        ax.plot(timesteps, history["CTX"]["cog"][:, i], c=colors[i], label=cues[i])
#    ax.plot(timesteps, history["CTX"]["cog"][:, i + 1], c=colors[i + 1], label="Cognitive Cortex")
    n = history["REW"]["mot"][0].size
    for i in range(n):
       ax.plot(timesteps, history["REW"]["mot"][:, i], '--', c=colors[i], label=directions[i])

    n = history["CTX"]["mot"][0].size
    for i in range(n):
        ax.plot(timesteps, history["CTX"]["mot"][:, i], ':', c=colors[i])


#    ax.plot(timesteps, history["CTX"]["mot"][:, i + 1], '--', c=colors[i + 1], label="Motor Cortex")


def display_performance(progression, num_trials, filename=None):
    fig = plt.figure(figsize=(12,8))
    plt.subplots_adjust(bottom=0.15)
    (ax_weights, ax_performance, ax_value) = fig.subplots(3, 1, sharex=True)

    fig.patch.set_facecolor('.9')

    plotWeights(num_trials, progression[0], ax_weights)
    plotProgression(num_trials, progression[1], ax_performance)
    plotProgression(num_trials, progression[2], ax_value)
    ax_weights.set_xlabel("Trials")
    ax_weights.set_ylabel("Weights")
    ax_weights.legend(frameon=False, loc='upper left')
    ax_weights.set_xlim(0.0,num_trials)
    ax_weights.set_ylim(0.40,0.60)

    if filename is not None:
        plt.savefig(filename)

    plt.show()

def displayNoonan(filename=None):
    num_trials = 120
    SP_V2HIGH_ctrl_1 = np.load('V2HIGH-ctrl-1-bkp.npy')
    SP_V2HIGH_ctrl_2 = np.load('V2HIGH-ctrl-2-bkp.npy')
    SP_V2MID_ctrl_1 = np.load('V2HIGH-ctrl-1-bkp.npy')
    SP_V2MID_ctrl_2 = np.load('V2HIGH-ctrl-2-bkp.npy')
    SP_V2LOW_ctrl_1 = np.load('V2HIGH-ctrl-1-bkp.npy')
    SP_V2LOW_ctrl_2 = np.load('V2HIGH-ctrl-2-bkp.npy')
    SP_V2HIGH_mOFC = np.load('V2HIGH-mOFC.npy')
    SP_V2HIGH_lOFC = np.load('V2HIGH-lOFC-bkp.npy')
    SP_V2MID_mOFC = np.load('V2MID-mOFC.npy')
    SP_V2MID_lOFC = np.load('V2MID-lOFC.npy')
    SP_V2LOW_mOFC = np.load('V2LOW-mOFC.npy')
    SP_V2LOW_lOFC = np.load('V2LOW-lOFC.npy')

    fig, axarr = plt.subplots(3, 3)
    axarr[0, 0].set_title('A')
    axarr[0, 1].set_title('B')
    axarr[0, 2].set_title('C')
    axarr[1, 0].set_title('D')
    axarr[1, 1].set_title('E')
    axarr[1, 2].set_title('F')
    # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
    plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
    plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
    plotProgression(num_trials, SP_V2HIGH_ctrl_1, axarr[1, 0])
    plotProgression(num_trials, SP_V2HIGH_mOFC, axarr[1, 0])
    plotProgression(num_trials, SP_V2HIGH_ctrl_1, axarr[2, 0])
    plotProgression(num_trials, SP_V2HIGH_lOFC, axarr[2, 0])

    plt.show()

def display_cog_ctx(history, duration=3.0, filename=None):
    fig = plt.figure(figsize=(12,5))
    plt.subplots_adjust(bottom=0.15)

    timesteps = np.linspace(0,duration, len(history))
    colors = ['r', 'b', 'g', 'y']
    fig.patch.set_facecolor('.9')
    ax = plt.subplot(1,1,1)

    n = history["CTX"]["cog"][0].size
    for i in range(n-1):
        plt.plot(timesteps, history["CTX"]["cog"][:,i],c=colors[i])
    plt.plot(timesteps, history["CTX"]["cog"][:,i+1],c=colors[i+1], label="Cognitive Cortex")

    plt.xlabel("Time (seconds)")
    plt.ylabel("Activity (Hz)")
    plt.legend(frameon=False, loc='upper left')
    plt.xlim(0.0,duration)
#    plt.ylim(0.0,300.0)

#    plt.xticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
#               ['0.0','0.5\n(Trial start)','1.0','1.5', '2.0','2.5\n(Trial stop)','3.0'])

    if filename is not None:
        plt.savefig(filename)
    plt.show()


def display_all(history, duration=3.0, filename=None):
    fig = plt.figure(figsize=(18,12))
    fig.patch.set_facecolor('1.0')

    timesteps = np.linspace(0,duration, len(history))

    def subplot(rows,cols,n, alpha=0.0):
        ax = plt.subplot(rows,cols,n)
        ax.patch.set_facecolor("k")
        ax.patch.set_alpha(alpha)

        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.yaxis.set_ticks_position('left')
#        ax.yaxis.set_tick_params(direction="outward")
        return ax

    ax = subplot(5,3,1)
    ax.set_title("Motor", fontsize=24)
    ax.set_ylabel("STN", fontsize=24)
    for i in range(4):
        plt.plot(timesteps, history["STN"]["mot"][:,i], c='k', lw=.5)
    ax.set_xticks([])

    ax = subplot(5,3,2)
    ax.set_title("Cognitive", fontsize=24)
    for i in range(4):
        plt.plot(timesteps, history["STN"]["cog"][:,i], c='k', lw=.5)
    ax.set_xticks([])

    ax = subplot(5,3,3,alpha=0)
    ax.set_title("Associative", fontsize=24)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['left'].set_color('none')


    ax = subplot(5,3,4)
    ax.set_ylabel("Cortex", fontsize=24)
    for i in range(4):
        plt.plot(timesteps, history["CTX"]["mot"][:,i], c='k', lw=.5)
    ax.set_xticks([])

    ax = subplot(5,3,5)
    for i in range(4):
        plt.plot(timesteps, history["CTX"]["cog"][:,i], c='k', lw=.5)
    ax.set_xticks([])

    ax = subplot(5,3,6)
    for i in range(16):
        plt.plot(timesteps, history["CTX"]["ass"][:,i], c='k', lw=.5)
    ax.set_xticks([])

    ax = subplot(5,3,7)
    ax.set_ylabel("Striatum", fontsize=24)
    for i in range(4):
        plt.plot(timesteps, history["STR"]["mot"][:,i], c='k', lw=.5)
    ax.set_xticks([])

    ax = subplot(5,3,8)
    for i in range(4):
        plt.plot(timesteps, history["STR"]["cog"][:,i], c='k', lw=.5)
    ax.set_xticks([])

    ax = subplot(5,3,9)
    for i in range(16):
        plt.plot(timesteps, history["STR"]["ass"][:,i], c='k', lw=.5)
    ax.set_xticks([])

    ax = subplot(5,3,10)
    ax.set_ylabel("GPi", fontsize=24)
    for i in range(4):
        plt.plot(timesteps, history["GPI"]["mot"][:,i], c='k', lw=.5)
    ax.set_xticks([])

    ax = subplot(5,3,11)
    for i in range(4):
        plt.plot(timesteps, history["GPI"]["cog"][:,i], c='k', lw=.5)
    ax.set_xticks([])

    ax = subplot(5,3,13)
    ax.set_ylabel("Thalamus", fontsize=24)
    for i in range(4):
        plt.plot(timesteps, history["THL"]["mot"][:,i], c='k', lw=.5)
    ax.set_xticks([])

    ax = subplot(5,3,14)
    for i in range(4):
        plt.plot(timesteps, history["THL"]["cog"][:,i], c='k', lw=.5)
    ax.set_xticks([])

    if filename is not None:
        plt.savefig(filename)
    plt.show()

#####################  DISPLAY METHODS  ########################################"
def plot_weights(fignum, figpos, W_arr, WM_arr, num_trials, title):
    # Plot the variation of weights over each trial as learning happens
    plt.figure(fignum)
    pos = 220 + (2*(figpos/2)+1)
    plt.subplot(pos)
    plt.xlabel('Number of trials')
    plt.ylabel('Synaptic weight')
    trials_set = 1+np.arange(num_trials)
    colors = ['r','b','g','c']
    for i in range(4):
        plt.plot(trials_set, W_arr[i], color=colors[i], label='S'+str(i))
    #plt.title(title + '-COG Wts')
    plt.ylim(0.48,0.60)
    plt.legend(loc=2)
    plt.subplot(pos + 1)
    plt.xlabel('Number of trials')
    plt.ylabel('Synaptic weight')
    for i in range(4):
        plt.plot(trials_set, WM_arr[i], color=colors[i], label='D'+str(i))
    #plt.title(title + '-MOT Wts')
    plt.ylim(0.48,0.60)
    plt.legend(loc=2)

#####################  DISPLAY METHODS  ########################################"
def plot_lines(fignum, figpos, data, trials_set, labels, title=''):
    # Plot the variation of weights over each trial as learning happens
    plt.figure(fignum)
    pos = 220 + figpos
    plt.subplot(pos)
    colors = ['r','b','g','c','m','y']
    for i in range(np.size(data)/np.size(trials_set)):
        if np.size(data) == np.size(trials_set):
            plt.plot(trials_set, data, color=colors[i], label=str(labels))
        else:
            plt.plot(trials_set, data[i], color=colors[i], label=str(labels[i]))
    plt.title(title)
    plt.ylim(0,2)
    plt.yticks([0.0,0.5,1.0,1.5,2.0],['0.0','0.5','1.0','',''])
    plt.legend(loc=2)

def plot_performance(fignum, figpos, num_trials, TP, title):
    # Plot the mean performance for each trial over all sessions
    plt.figure(fignum)
    pos = 210 + figpos
    ax = plt.subplot(pos)
    ax.patch.set_facecolor("w")
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.yaxis.set_ticks_position('left')
    ax.yaxis.set_tick_params(direction="in")
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_tick_params(direction="in")

    X = 1+np.arange(num_trials)
    plt.plot(X, TP.mean(axis=0), c='r', lw=2)
    plt.plot(X, TP.mean(axis=0)+TP.var(axis=0), c='r',lw=.5)
    plt.plot(X, TP.mean(axis=0)-TP.var(axis=0), c='r',lw=.5)
    plt.fill_between(X, TP.mean(axis=0)+TP.var(axis=0),
                        TP.mean(axis=0)-TP.var(axis=0), color='r', alpha=.1)
    plt.xlabel("Trial number", fontsize=16)
    plt.ylabel("Performance", fontsize=16)
    plt.ylim(0,1.0)
    plt.xlim(1,num_trials)
    #plt.title(title)


def autolabel(ax, rects):
    # attach some text labels
    for rect1, rect2 in zip(rects[0],rects[1]):
        height1 = rect1.get_height()
        height2 = rect2.get_height()
        ax.text(rect2.get_x()+rect2.get_width()/2., 1.05*height2, '%d'%(height2 - height1),
                ha='center', va='bottom')

def plot_diff_decision_times(fignum, figpos, num_trials, DTCOG, DTMOT, trial, title):
    # Plot mean decision times for COG and MOT over each trial
    plt.figure(fignum)
    pos = 210 + figpos
    ax = plt.subplot(pos)
    ind = np.arange(10)
    width = 0.35
    cog_var = DTCOG.var(axis=0)
    mot_var = DTMOT.var(axis=0)
    rects1 = ax.bar(ind, DTCOG.mean(axis=0)[-10:], width, color='r')
    rects2 = ax.bar(ind+width, DTMOT.mean(axis=0)[-10:], width, color='b')
    ax.set_ylabel('Decision time')
    if figpos > 1: ax.set_xlabel('Trial Number', fontsize=16)
    ax.set_title('Decision times - ' + title)
    ax.set_xticks(ind+width)
    ax.set_xticklabels(np.arange(num_trials-9,num_trials+1))
    ax.legend( (rects1[0], rects2[0]), ('COG', 'MOTOR') )
    plt.ylim(0,trial*1000)
    autolabel(ax, [rects1, rects2])

def plot_decision_times(fignum, figpos, num_trials, DTCOG, DTMOT, trial, title):
    # Plot the decision times of each trial over all sessions
    plt.figure(fignum)
    pos = 210 + figpos
    ax = plt.subplot(pos)
    ax.patch.set_facecolor("w")
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.yaxis.set_ticks_position('left')
    ax.yaxis.set_tick_params(direction="in")
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_tick_params(direction="in")

    X = 1+np.arange(num_trials)
    cog_times_mean = DTCOG.mean(axis=0)
    tot_cog_mean = np.mean(cog_times_mean[-20])
    mot_times_mean = DTMOT.mean(axis=0)
    tot_mot_mean = np.mean(mot_times_mean[-20])
    plt.plot(X, cog_times_mean, c='b', lw=2, label='Cognitive')
    plt.plot([X[0],X[num_trials-1]],[tot_cog_mean,tot_cog_mean], 'b--', lw=2)
    plt.plot(X, mot_times_mean, c='r', lw=2, label='Motor')
    plt.plot([X[0],X[num_trials-1]],[tot_mot_mean,tot_mot_mean], 'r--', lw=2)
    plt.ylabel("Decision Time", fontsize=16)
    plt.xlabel("Trial Number", fontsize=16)
    plt.ylim(0,0.75*trial*1000)
    plt.xlim(1,num_trials)
    plt.legend(loc=2)
    #plt.title(title)

#####################  END OF DISPLAY METHODS  ########################################"
#displayNoonan()
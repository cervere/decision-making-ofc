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
import sys
import os
import numpy as np
from model import *

if sys.version_info[0] == 2:
    # Workaround for https://github.com/PythonCharmers/python-future/issues/262
    from Tkinter import *
else:
    from tkinter import *

from datetime import datetime
NUM_CHANNELS = 4

tau     = 0.01
clamp   = Clamp(min=0, max=1000)
sigmoid = Sigmoid(Vmin=0, Vmax=20, Vh=16, Vc=3)

dtype = [ ("CTX", [("mot", float, NUM_CHANNELS), ("cog", float, NUM_CHANNELS), ("ass", float, NUM_CHANNELS*NUM_CHANNELS)]),
          ("STR", [("mot", float, NUM_CHANNELS), ("cog", float, NUM_CHANNELS), ("ass", float, NUM_CHANNELS*NUM_CHANNELS)]),
          ("REW", [("mot", float, NUM_CHANNELS), ("cog", float, NUM_CHANNELS)]),
          ("GPI", [("mot", float, NUM_CHANNELS), ("cog", float, NUM_CHANNELS)]),
          ("THL", [("mot", float, NUM_CHANNELS), ("cog", float, NUM_CHANNELS)]),
          ("STN", [("mot", float, NUM_CHANNELS), ("cog", float, NUM_CHANNELS)])]

def getDateTimeStr():
    now = datetime.now()
    date_time = now.strftime("%Y%m%d%H%M")
    return date_time


def weights(shape):
    Wmin, Wmax = 0.25, 0.75
    N = np.random.normal(0.5, 0.005, shape)
    N = np.minimum(np.maximum(N, 0.0),1.0)
    return (Wmin+(Wmax-Wmin)*N)

Wmain = weights(NUM_CHANNELS)

class BasicTCBGLoop:

    def __init__(self, allchannels=True):
        self.REW = AssociativeStructure(
                         num=[NUM_CHANNELS,NUM_CHANNELS],tau=tau, rest=- 3.0, noise=0.020, activation=clamp )
        self.CTX = AssociativeStructure(
                         num=[NUM_CHANNELS,NUM_CHANNELS],tau=tau, rest=- 3.0, noise=0.010, activation=clamp )
        self.STR = AssociativeStructure(
                         num=NUM_CHANNELS,tau=tau, rest=  0.0, noise=0.001, activation=sigmoid )
        self.STN = Structure( num=NUM_CHANNELS,tau=tau, rest=-10.0, noise=0.001, activation=clamp )
        self.GPI = Structure( num=NUM_CHANNELS,tau=tau, rest=+10.0, noise=0.030, activation=clamp )
        self.THL = Structure( num=NUM_CHANNELS,tau=tau, rest=-40.0, noise=0.001, activation=clamp )
        self.structures = (self.REW, self.CTX, self.STR, self.STN, self.GPI, self.THL)
        self.VC_weight = .075*8
        self.VC_decay = .015*8
        self.cogchoiceHistory = None
        self.rewardHistory = None

        if allchannels : ctxToStrCogGain = 1.0
        else : ctxToStrCogGain = 1.10
        self.ctxStrCogConnection = OneToOne( self.CTX.cog.V, self.STR.cog.Isyn, Wmain,   gain=ctxToStrCogGain )
        self.connections = [
        #    OneToOne( CTX.cog.V, STR.cog.Isyn, np.array([ .55, .51, .49, .47]),   gain=+1.0 ), #gain=1 only works for the normal case - all connections, for only cog, gain=1.25 works
            self.ctxStrCogConnection, #gain=1 only works for the normal case - all connections, for only cog, gain=1.25 works
            #OneToOne( self.REW.cog.Iext, self.CTX.cog.Isyn, np.ones(NUM_CHANNELS),   gain=+0.5 ),
            # REW.mot : mOFC
            OneToAll(self.REW.mot.V, self.REW.mot.Isyn, np.ones(NUM_CHANNELS), gain=-self.VC_weight),
            OneToOne(self.REW.mot.V, self.REW.mot.Isyn, np.ones(NUM_CHANNELS), gain=(2*self.VC_weight - self.VC_decay)),
            OneToOne(self.REW.mot.V, self.CTX.cog.Isyn, np.ones(NUM_CHANNELS), gain=+0.4),
            OneToOne(self.CTX.cog.V, self.STN.cog.Isyn, np.ones(NUM_CHANNELS), gain=+1.0),
            OneToOne( self.STR.cog.V, self.GPI.cog.Isyn, np.ones(NUM_CHANNELS),   gain=-2.0 ),
            OneToAll( self.STN.cog.V, self.GPI.cog.Isyn, np.ones(NUM_CHANNELS),   gain=+1.0 ),
            OneToOne( self.GPI.cog.V, self.THL.cog.Isyn, np.ones(NUM_CHANNELS),   gain=-0.5 ),
            OneToOne( self.THL.cog.V, self.CTX.cog.Isyn, np.ones(NUM_CHANNELS),   gain=+1.0 ),
            OneToOne( self.CTX.cog.V, self.THL.cog.Isyn, np.ones(NUM_CHANNELS),   gain=+0.4 ),
        ]
        if allchannels : self.getOtherConnections()

        '''
        Learning stuff
        '''
        self.threshold = 40
        self.alpha_c = 0.025
        self.alpha_LTP = 0.004
        self.alpha_LTD = 0.002
        self.Wmin, self.Wmax = 0.25, 0.75
        self.cues_value = np.ones(NUM_CHANNELS) * 0.5
        self.value_diff = 0
        self.lastMotTime = -1
        self.lastCogTime = -1
        self.lastmOFCTime = -1

    def getOtherConnections(self):
        other_connections = [
            CogToAss(self.CTX.cog.V, self.STR.ass.Isyn, weights(NUM_CHANNELS), gain=+0.2),
            CogToAss(self.REW.mot.V, self.STR.ass.Isyn, weights(NUM_CHANNELS), gain=+0.),
            OneToOne( self.CTX.mot.V, self.STR.mot.Isyn, weights(NUM_CHANNELS),   gain=+1.0 ),
            OneToOne( self.CTX.ass.V, self.STR.ass.Isyn, weights(NUM_CHANNELS*NUM_CHANNELS), gain=+1.0 ),
            MotToAss( self.CTX.mot.V, self.STR.ass.Isyn, weights(NUM_CHANNELS),   gain=+0.2 ),
            OneToOne( self.CTX.mot.V, self.STN.mot.Isyn, np.ones(NUM_CHANNELS),   gain=+1.0 ),
            OneToOne( self.STR.mot.V, self.GPI.mot.Isyn, np.ones(NUM_CHANNELS),   gain=-2.0 ),
            AssToCog( self.STR.ass.V, self.GPI.cog.Isyn, np.ones(NUM_CHANNELS),   gain=-2.0 ),
            AssToMot( self.STR.ass.V, self.GPI.mot.Isyn, np.ones(NUM_CHANNELS),   gain=-2.0 ),
            OneToAll( self.STN.mot.V, self.GPI.mot.Isyn, np.ones(NUM_CHANNELS),   gain=+1.0 ),
            OneToOne( self.GPI.mot.V, self.THL.mot.Isyn, np.ones(NUM_CHANNELS),   gain=-0.5 ),
            OneToOne( self.THL.mot.V, self.CTX.mot.Isyn, np.ones(NUM_CHANNELS),   gain=+1.0 ),
            OneToOne( self.CTX.mot.V, self.THL.mot.Isyn, np.ones(NUM_CHANNELS),   gain=+0.4 ),
        ]
        for oc in other_connections :
            self.connections.append(oc)

    def resetLoop(self):
        for structure in self.structures:
            structure.reset()
        self.lastMotTime = -1
        self.lastCogTime = -1
        self.lastmOFCTime = -1

    def resetBrain(self):
        self.resetLoop()
        self.cogchoiceHistory = None
        self.cues_value = np.ones(NUM_CHANNELS) * 0.5
        Wmain[...] = weights(NUM_CHANNELS)


    def resetInputs(self):
        self.CTX.cog.Iext = 0
        self.CTX.mot.Iext = 0
        self.CTX.ass.Iext = 0

    def setmOFCInput(self, c):
        v = 7
        noise = 0.05
        self.REW.mot.Iext[c]  = v + 2*self.cues_value[c] + np.random.normal(0,v*noise)

    def setInput(self,c, trialNum, m=-1):
        v = 7
        noise = 0.02
        self.REW.mot.Iext[c]  = v + 2*self.cues_value[c] + np.random.normal(0,v*noise)
        self.CTX.cog.Iext[c]  = v + np.random.normal(0,v*noise)
        prev=0
        if trialNum > 1:
            prev = trialNum-2
        #int(self.cogchoiceHistory[prev])
#        self.CTX.cog.Iext[c] += self.cues_value[c] + self.rewardHistory[prev]
        if m > -1 :
            self.CTX.mot.Iext[m]  = v + np.random.normal(0,v*noise)
            self.CTX.ass.Iext[c*NUM_CHANNELS+m] = v + np.random.normal(0,v*noise)

    def iterate(self,dt):
        # Flush connections
        for connection in self.connections:
            connection.flush()
        # Propagate activities
        for connection in self.connections:
            connection.propagate()
        # Compute new activities
        for structure in self.structures:
            structure.evaluate(dt)


    def isMotorDecided(self):
        return self.CTX.mot.delta > self.threshold

    def ismOFCDecided(self):
        return self.REW.mot.delta > 10

    def isCogDecided(self):
        return self.CTX.cog.delta > self.threshold

    def getHistory(self, duration):
        history = np.zeros(duration, dtype=dtype)
        history["REW"]["mot"] = self.REW.mot.history[:duration]
        history["REW"]["cog"] = self.REW.cog.history[:duration]
        history["CTX"]["mot"] = self.CTX.mot.history[:duration]
        history["CTX"]["cog"] = self.CTX.cog.history[:duration]
        history["CTX"]["ass"] = self.CTX.ass.history[:duration]
        history["STR"]["mot"] = self.STR.mot.history[:duration]
        history["STR"]["cog"] = self.STR.cog.history[:duration]
        history["STR"]["ass"] = self.STR.ass.history[:duration]
        history["STN"]["mot"] = self.STN.mot.history[:duration]
        history["STN"]["cog"] = self.STN.cog.history[:duration]
        history["GPI"]["mot"] = self.GPI.mot.history[:duration]
        history["GPI"]["cog"] = self.GPI.cog.history[:duration]
        history["THL"]["mot"] = self.THL.mot.history[:duration]
        history["THL"]["cog"] = self.THL.cog.history[:duration]
        return history


    def reflect(self):
        mot_choice = -1
        if self.CTX.mot.delta > self.threshold :
            mot_choice = np.argmax(self.CTX.mot.V)
        return mot_choice

    def learn(self, reward, cogchoice, debug=True):
        ## For now, let the cogchoice be provided
        ## Ideally, if the brain remembers presented state,
        ## it can correctly reinforce the cogcue corresponding to executed action
        # Compute prediction error
        #error = cues_reward[choice] - cues_value[choice]
        error = reward - self.cues_value[cogchoice]
        # Update cues values
        self.cues_value[cogchoice] += error * self.alpha_c
        differror = (np.sort(self.cues_value)[-1] - np.sort(self.cues_value)[-2]) - self.value_diff
        self.value_diff += differror * .05 * (.75 - self.value_diff)
        # Learn
        lrate = self.alpha_LTP if error > 0 else self.alpha_LTD
        dw = error * lrate * self.STR.cog.V[cogchoice]
        Wmain[cogchoice] = Wmain[cogchoice] + dw * (Wmain[cogchoice] - self.Wmin) * (self.Wmax - Wmain[cogchoice])

        if not debug: return

        # Just for displaying ordered cue
        oc1,oc2 = min(c1,c2), max(c1,c2)
        if cogchoice == oc1:
            print("Choice:          [%d] / %d  (good)" %(oc1,oc2))
        else:
            print("Choice:           %d / [%d] (bad)"%(oc1,oc2))
        print("Reward (%3d%%) :   %d"%(int(100 * cues_reward[cogchoice]), reward) )
        print("Mean performance: %.3f" % np.array(P).mean())
        print("Mean reward:      %.3f" % np.array(R).mean())
        #print "Response time:    %d ms" % (time)


exLoop = BasicTCBGLoop(allchannels=True)

cues_reward = np.array([3.0,2.0,1.0,0.0])/3.0
#cues_reward = np.array([2.1,1.9,0.5,0.0])/3.0
num_choices = 2

def start_trial(cues_cog, cues_mot, trialNum):
    for c, m in zip(cues_cog, cues_mot) :
        exLoop.setInput(c, trialNum, m)

def saveResults(SP, SWW, SDE, DT, sample_history, MVS) :
    # Performance, Weights, ValueDiff, DecisionTimes["cog", "mot"]
    DATESTR = getDateTimeStr()
    os.makedirs(DATESTR, exist_ok=True)
    np.save(DATESTR+'/'+'performance-by-session.npy', SP)
    np.save(DATESTR+'/'+'weights-by-session.npy', SWW)
    np.save(DATESTR+'/'+'valuediff-by-session.npy', SDE)
    np.save(DATESTR+'/'+'decisiontime-by-session.npy', DT)
    np.save(DATESTR+'/'+'sample-by-trial.npy', sample_history)
    np.save(DATESTR+'/'+'session-mean-value.npy', MVS)


def iteratefor(start, end):
    for i in range(start, end):
        exLoop.iterate(dt)
        if exLoop.lastMotTime == -1 and exLoop.isMotorDecided():
            exLoop.lastMotTime = i
        if exLoop.lastCogTime == -1 and exLoop.isCogDecided() :
            exLoop.lastCogTime = i
        if exLoop.lastmOFCTime == -1 and exLoop.ismOFCDecided() :
            exLoop.lastmOFCTime = i


dt = 0.001
prepare = 500
settle = 500
duration = 3000
num_trials = 120

test_combos = 6
Z = [[0, 1], [0, 2], [1, 2], [0, 3], [1, 3], [2, 3]]
Z1 = [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]
# 20 x all cues combinations
C = np.repeat(np.arange(test_combos), num_trials/test_combos)
# 20 x all cues positions
M = np.repeat(np.arange(test_combos), num_trials/test_combos)


sample_history = []
for j in range(num_trials):
    exLoop.resetLoop()
    iteratefor(0, prepare)
    cues_cog = Z[C[j]]
    cues_mot = Z1[M[j]]
    cues_cog = [0, 1]
    start_trial(cues_cog, cues_mot, j)
    iteratefor(prepare,prepare+50)
    for c in cues_cog :
        exLoop.setmOFCInput(c)
    iteratefor(prepare + 50, duration-settle)
    mot_choice = exLoop.reflect()
    if mot_choice > -1:
        c1, c2 = cues_cog
        m1, m2 = cues_mot
        # The actual cognitive choice may differ from the cognitive choice
        # Only the motor decision can designate the chosen cue
        if mot_choice == m1:
            cogchoice = c1
        else:
            cogchoice = c2
        if cogchoice == min(c1, c2):
            perf = 1
        else :
            perf = 0

        # Compute reward

        reward = np.random.uniform(0,1) < cues_reward[cogchoice]
#        exLoop.cogchoiceHistory[j] = cogchoice
#        exLoop.rewardHistory[j] = reward
#        exLoop.learn(reward, cogchoice)

    exLoop.resetInputs()
    iteratefor(duration-settle,duration)
    if j in [0, num_trials/2, (num_trials-1)] :
        sample_history.append(exLoop.getHistory(duration))

meta = {"num_trials" : num_trials, "type" : 'STD'}
# -----------------------------------------------------------------------------
from display import *

if 1:
    display_ctx(sample_history, 3.0, "figure-1.svg")
if 0:
    display_all(history, 3.0, "figure-1bis.pdf")

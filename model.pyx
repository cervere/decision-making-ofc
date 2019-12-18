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
cimport numpy as np
from libc.math cimport exp
from libc.stdlib cimport rand, srand, RAND_MAX


# ---------------------------------------------------------------- Function ---
cdef class Function:
    cdef double call(self, double x) except *:
        return x


# --- Identity ---
cdef class Identity(Function):

    cdef double call(self, double x) except *:
        if x < 0.0: return 0.0
        return x


# --- Clamp ---
cdef class Clamp(Function):
    cdef public double min, max

    def __init__(self, double min=0, double max=1e9):
        self.min = min
        self.max = max

    cdef double call(self, double x) except *:
        if x < self.min: return self.min
        if x > self.max: return self.max
        return x


# --- Noise ---
cdef class UniformNoise(Function):
    cdef public double amount

    def __init__(self, double amount):
        self.amount = amount

    cdef double call(self, double x) except *:
        return x + self.amount*(rand()/float(RAND_MAX) - 0.5)


# --- Sigmoid ---
cdef class Sigmoid(Function):
    cdef public double Vmin, Vmax, Vh, Vc

    def __init__(self, Vmin=0.0, Vmax=20.0, Vh=16., Vc=3.0):
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.Vh = Vh
        self.Vc = Vc

    cdef double call(self, double V) except *:
        return self.Vmin + (self.Vmax-self.Vmin)/(1.0+exp((self.Vh-V)/self.Vc))



# ------------------------------------------------------------------- Group ---
# Python group type (dtype)
dtype = [("V",  float),
         ("U",  float),
         ("Isyn",  float),
         ("Iext", float)]

# C group type (ctype)
cdef packed struct ctype:
    np.float64_t V
    np.float64_t U
    np.float64_t Isyn
    np.float64_t Iext


cdef class Group:
    """  """

    cdef double      _tau
    cdef double      _rest
    cdef double      _noise
    cdef double      _delta
    cdef ctype[:]    _units
    cdef Function    _activation
    cdef int         _history_index
    cdef double[:,:] _history
    cdef double[:,:] _history_input

    def __init__(self, units, tau=0.01, rest=0.0, noise=0.0, activation = Identity()):
        self._tau = tau
        self._rest = rest
        self._noise = noise
        self._units = units
        self._delta = 0
        self._activation = activation
        self._history_index = 0
        self._history = np.zeros((10000, len(self._units)))
        self._history_input = np.zeros((10000, len(self._units)))

    property history:
        """ Activity history (firing rate) """
        def __get__(self):
            return np.asarray(self._history)

    property history_input:
        """ Ext. input history (Iext) """
        def __get__(self):
            return np.asarray(self._history_input)

    property delta:
        """ Difference of activity between the first two maximum activites """
        def __get__(self):
            return self._delta

    property tau:
        """ Membrane time constant """
        def __get__(self):
            return self._tau
        def __set__(self, value):
            self._tau = value

    property rest:
        """ Membrane resting potential """
        def __get__(self):
            return self._rest
        def __set__(self, value):
            self._rest = value

    property V:
        """ Firing rate """
        def __get__(self):
            return np.asarray(self._units)["V"]

    property U:
        """ Membrane potential """
        def __get__(self):
            return np.asarray(self._units)["U"]

    property Isyn:
        """ Input current from external synapses """
        def __get__(self):
            return np.asarray(self._units)["Isyn"]

    property Iext:
        """ Input current from external sources """
        def __get__(self):
            return np.asarray(self._units)["Iext"]
        def __set__(self, value):
            np.asarray(self._units)["Iext"] = value

    def evaluate(self, double dt):
        """ Compute activities (Forward Euler method) """

        cdef int i
        cdef noise
        cdef ctype * unit
        cdef double max1=0, max2=0

        if self._history_index >= 10000 :
            self._history_index = 0
            self._history = np.zeros((10000, len(self._units)))
            self._history_input = np.zeros((10000, len(self._units)))

        for i in range(len(self._units)):
            unit = & self._units[i]
            # Compute white noise
            noise = self._noise*(rand()/float(RAND_MAX)) - self._noise/2.0
            # Update membrane potential
            unit.U += dt/self._tau*(-unit.U + unit.Isyn + unit.Iext - self._rest )
            # Update firing rate
            unit.V = self._activation.call(unit.U + noise)
            # Store firing rate activity
            self._history[self._history_index,i] = unit.V
            # Store external input
            self._history_input[self._history_index, i] = unit.Iext

            # Here we record the max activities to store their difference
            # This is used later to decide if a motor decision has been made
            if unit.V > max1:   max1 = unit.V
            elif unit.V > max2: max2 = unit.V

        self._delta = max1 - max2
        self._history_index +=1


    def reset(self):
        """ Reset all activities and history index """

        cdef int i
        self._history_index = 0
        for i in range(len(self._units)):
            self._units[i].V = 0
            self._units[i].U = 0
            self._units[i].Isyn = 0
            self._units[i].Iext = 0

    def softreset(self):
        """ Reset all activities and history index """

        cdef int i
        for i in range(len(self._units)):
            self._units[i].V = 0
            self._units[i].U = 0
            self._units[i].Isyn = 0
            self._units[i].Iext = 0


    def __getitem__(self, key):
        return np.asarray(self._units)[key]


    def __setitem__(self, key, value):
        np.asarray(self._units)[key] = value


# C group type (ctype)
cdef packed struct catype:
    np.float64_t V
    np.float64_t U
    np.float64_t Isyn
    np.float64_t Iext
    np.float64_t adap

cdef class GroupCon:
    """  """

    cdef double      _tauadap
    cdef double      _tau
    cdef double      _rest
    cdef double      _noise
    cdef double      _delta
    cdef double[:]   _eachmax
    cdef catype[:]    _units
    cdef Function    _activation
    cdef int         _history_index
    cdef double[:,:] _history

    def __init__(self, units, tauadap=0.01, tau=0.01, rest=0.0, noise=0.0, activation = Identity()):
        self._tauadap = tauadap
        self._tau = tau
        self._rest = rest
        self._noise = noise
        self._delta = 0
        self._units = units
        self._eachmax = np.zeros(len(self._units))
        self._activation = activation
        self._history_index = 0
        self._history = np.zeros((10000, len(self._units)))

    property history:
        """ Activity history (firing rate) """
        def __get__(self):
            return np.asarray(self._history)

    property tauadap:
        """ Membrane time constant """
        def __get__(self):
            return self._tauadap
        def __set__(self, value):
            self._tauadap = value

    property tau:
        """ Membrane time constant """
        def __get__(self):
            return self._tau
        def __set__(self, value):
            self._tau = value

    property delta:
        """ Difference of activity between the first two maximum activites """
        def __get__(self):
            return self._delta

    property eachmax:
        """ Maximum activity of each of the units """
        def __get__(self):
            return self._eachmax

    property rest:
        """ Membrane resting potential """
        def __get__(self):
            return self._rest
        def __set__(self, value):
            self._rest = value

    property V:
        """ Firing rate """
        def __get__(self):
            return np.asarray(self._units)["V"]

    property U:
        """ Membrane potential """
        def __get__(self):
            return np.asarray(self._units)["U"]

    property Isyn:
        """ Input current from external synapses """
        def __get__(self):
            return np.asarray(self._units)["Isyn"]

    property Iext:
        """ Input current from external sources """
        def __get__(self):
            return np.asarray(self._units)["Iext"]
        def __set__(self, value):
            np.asarray(self._units)["Iext"] = value

    def evaluate(self, double dt):
        """ Compute activities (Forward Euler method) """

        cdef int i
        cdef noise
        cdef catype * unit
        cdef double max1=0, max2=0, sumV=0

        if self._history_index >= 10000 :
            self._history_index = 0
            self._history = np.zeros((10000, len(self._units)))

        for i in range(len(self._units)):
            unit = & self._units[i]
            # Compute white noise
            noise = self._noise*(rand()/float(RAND_MAX)) - self._noise/2.0
            # Update adaptation
            unit.adap += dt/self.tauadap*(-unit.adap + self._activation.call(unit.Isyn))
            # Update membrane potential
            unit.U += dt/self._tau*(-unit.U + unit.Isyn - unit.adap + noise )
            # Update firing rate
            unit.V = self._activation.call(unit.U + noise) + self._rest
            # Store firing rate activity
            self._history[self._history_index, i] = unit.V

            # Here we record the max activities to store their difference
            # This is used later to decide if a motor decision has been made
            if unit.V > max1:   max1 = unit.V
            elif unit.V > max2: max2 = unit.V

            if unit.V > self._eachmax[i] : self._eachmax[i] = unit.V

        self._delta = max1 - max2
        self._history_index +=1


    def reset(self):
        """ Reset all activities and history index """

        cdef int i
        self._history_index = 0
        for i in range(len(self._units)):
            self._units[i].V = 0
            self._units[i].U = 0
            self._units[i].Isyn = 0
            self._units[i].Iext = 0

    def softreset(self):
        """ Reset all activities and history index """

        cdef int i
        for i in range(len(self._units)):
            self._units[i].V = 0
            self._units[i].U = 0
            self._units[i].Isyn = 0
            self._units[i].Iext = 0


    def __getitem__(self, key):
        return np.asarray(self._units)[key]


    def __setitem__(self, key, value):
        np.asarray(self._units)[key] = value


# -------------------- Simple  Structure -----------------------
cdef class SimpleStructure:
    cdef Group _pop
    cdef Group _desired

    def __init__(self, num=4, tau=0.01, rest=0, noise=0, activation=Identity()):
        self._pop = Group(np.zeros(num,dtype=dtype), tau=tau, rest=rest,
                           noise=noise, activation=activation)
        self._desired = Group(np.zeros(num,dtype=dtype), tau=tau, rest=rest,
                           noise=noise, activation=activation)

    property pop:
        """ The population """
        def __get__(self):
            return self._pop

    property desired:
        """ The desired activity population """
        def __get__(self):
            return self._desired

    def evaluate(self, double dt):
        self._pop.evaluate(dt)
        self._desired.V[:] = self._pop.V

    def reset(self):
        self._pop.reset()

    def softreset(self):
        self._pop.softreset()




# ---------------------------------------- Structure --------------------------
cdef class Structure:
    cdef Group _mot
    cdef Group _cog

    def __init__(self, num=4, tau=0.01, rest=0, noise=0, activation=Identity()):
        if np.size(num) == 2 :
            self._cog = Group(np.zeros(num[0],dtype=dtype), tau=tau, rest=rest,
                             noise=noise, activation=activation)
            self._mot = Group(np.zeros(num[1],dtype=dtype), tau=tau, rest=rest,
                               noise=noise, activation=activation)
        else :
            self._mot = Group(np.zeros(num,dtype=dtype), tau=tau, rest=rest,
                               noise=noise, activation=activation)
            self._cog = Group(np.zeros(num,dtype=dtype), tau=tau, rest=rest,
                             noise=noise, activation=activation)

    property mot:
        """ The motor group """
        def __get__(self):
            return self._mot

    property cog:
        """ The cognitive group """
        def __get__(self):
            return self._cog

    def evaluate(self, double dt):
        self._mot.evaluate(dt)
        self._cog.evaluate(dt)

    def reset(self):
        self._mot.reset()
        self._cog.reset()

    def softreset(self):
        self._mot.softreset()
        self._cog.softreset()



# ---------------------------------------------------- AssociativeStructure ---
cdef class AssociativeStructure(Structure):
    cdef public Group _ass

    def __init__(self, num=4, tau=0.01, rest=0, noise=0, activation=Identity()):
        Structure.__init__(self, num, tau, rest, noise, activation)
        if np.size(num) == 2 :
            self._ass = Group(np.zeros(num[0]*num[1],dtype=dtype), tau=tau, rest=rest,
                              noise=noise, activation=activation)
        else :
            self._ass = Group(np.zeros(num*num,dtype=dtype), tau=tau, rest=rest,
                              noise=noise, activation=activation)

    def evaluate(self, double dt):
        Structure.evaluate(self, dt)
        self._ass.evaluate(dt)

    def reset(self):
        Structure.reset(self)
        self._ass.reset()

    def softreset(self):
        Structure.softreset(self)
        self._ass.softreset()

    property ass:
        """ The associative group """
        def __get__(self):
            return self._ass


# ------------------------------------------------------------- Connections ---
cdef class Connection:
    cdef double[:] _source
    cdef double[:] _target
    cdef double[:] _weights
    cdef double    _gain
    cdef str _name
    cdef int         _history_index
    cdef double[:,:] _history


    def __init__(self, source, target, weights, gain, name=""):
        self._gain = gain
        self._source = source
        self._target = target
        self._weights = weights
        self._name = name
        self._history_index = 0
        self._history = np.zeros((10000, len(self._weights)))

    property history:
        """ Activity history (firing rate) """
        def __get__(self):
            return np.asarray(self._history)

    property gain:
        """Gain of the connection"""
        def __get__(self):
            return self._gain
        def __set__(self, value):
            self._gain = value

    property source:
        """Source of the connection """
        def __get__(self):
            return np.asarray(self._source)

    property target:
        """Target of the connection (numpy array)"""
        def __get__(self):
            return np.asarray(self._target)

    property weights:
        """Weights matrix (numpy array)"""
        def __get__(self):
            return np.asarray(self._weights)
        def __set__(self, weights):
            self._weights = weights

    property name:
        """Name of the connection"""
        def __get__(self):
            return self._name
        def __set__(self, name):
            self._name = name

    def flush(self):
        cdef int i
        for i in range(self._target.shape[0]):
            self._target[i] = 0.0
        if self._history_index >= 10000 :
            self._history_index = 0
            self._history = np.zeros((10000, len(self._weights)))
        self._history[self._history_index,:] = self._weights
        self._history_index +=1


    def printCTXSTRWeights(self):
        if self._name == "CTXSTR" :
            print(np.asarray(self._weights))



# --- OneToOne---
cdef class OneToOne(Connection):
    def propagate(self):
        cdef int i
        for i in range(self._target.shape[0]):
            self._target[i] += self._source[i] * self._weights[i] * self._gain
    def printWeights(self):
        print(np.asarray(self._weights))



# --- OneToAll ---
cdef class OneToAll(Connection):
    def propagate(self):
        cdef int i,j
        for i in range(self._source.shape[0]):
            v = self._source[i] * self._weights[i] * self._gain
            for j in range(self._target.shape[0]):
                self._target[j] += v

# --- AllToOne ---
cdef class AllToOne(Connection):
    def propagate(self):
        cdef int i,j
        for j in range(self._target.shape[0]):
            v = 0
            for i in range(self._source.shape[0]):
                v += self._source[i] * self._weights[i]
            self._target[j] += v * self._gain

# --- AssToMot ---
cdef class AssToMot(Connection):
    def propagate(self):
        cdef int i,j,pop
        pop = self._source.shape[0] / self._target.shape[0]
        for i in range(self._target.shape[0]):
            v = 0
            for j in range(pop):
                v += self._source[i+j*pop] * self._weights[i]
            self._target[i] += v * self._gain

# --- AssToCog ---
cdef class AssToCog(Connection):
    def propagate(self):
        cdef int i,j,pop
        pop = self._source.shape[0] / self._target.shape[0]
        for i in range(self._target.shape[0]):
            v = 0
            for j in range(pop):
                v += self._source[j+i*pop] * self._weights[i]
            self._target[i] += v * self._gain


# --- MotToAss ---
cdef class MotToAss(Connection):
    def propagate(self):
        cdef int i,j,pop
        pop = self._target.shape[0] / self._source.shape[0]
        cdef double v
        for i in range(self._source.shape[0]):
            v = self._source[i] * self._weights[i] * self._gain
            for j in range(pop):
                self._target[j*pop +i] += v

# --- CogToAss ---
cdef class CogToAss(Connection):
    def propagate(self):
        cdef int i,j,pop
        pop = self._target.shape[0] / self._source.shape[0]
        cdef double v
        for i in range(self._source.shape[0]):
            v = self._source[i] * self._weights[i] * self._gain
            for j in range(pop):
                self._target[i*pop+j] += v

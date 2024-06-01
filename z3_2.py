import numpy as np
from numpy.fft import fft, fftshift
import matplotlib.pyplot as plt

class FieldDisplay:
    def __init__(self, maxSize_m, dx, y_min, y_max, probePos, sourcePos):
        plt.ion()
        self.probePos = probePos
        self.sourcePos = sourcePos
        self.fig, self.ax = plt.subplots()
        self.line = self.ax.plot(np.arange(0, maxSize_m, dx), [0]*int(maxSize_m/dx))[0]
        self.ax.plot(probePos*dx, 0, 'xr')
        self.ax.plot(sourcePos*dx, 0, 'ok')
        self.ax.set_xlim(0, maxSize_m)
        self.ax.set_ylim(y_min, y_max)
        self.ax.set_xlabel('x, м')
        self.ax.set_ylabel('Ez, В/м')
        self.ax.grid()
        
    def updateData(self, data):
        self.line.set_ydata(data)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def stop(self):
        plt.ioff()

class Probe:
    def __init__(self, probePos, Nt, dt):
        self.Nt = Nt
        self.dt = dt
        self.probePos = probePos
        self.t = 0
        self.E = np.zeros(self.Nt)
        
    def addData(self, data):
        self.E[self.t] = data[self.probePos]
        self.t += 1

def showProbeSignal(probe):
        fig, ax = plt.subplots(1,2)
        ax[0].plot(np.arange(0, probe.Nt*probe.dt, probe.dt), probe.E)
        ax[0].set_xlabel('t, c')
        ax[0].set_ylabel('Ez, В/м')
        ax[0].set_xlim(0, probe.Nt*probe.dt)
        ax[0].grid()
        sp = np.abs(fft(probe.E))
        sp = fftshift(sp)
        df = 1/(probe.Nt*probe.dt)
        freq = np.arange(-probe.Nt*df /2, probe.Nt*df/2, df)
        ax[1].plot(freq, sp/max(sp))
        ax[1].set_xlabel('f, Гц')
        ax[1].set_ylabel('|S|/|Smax|')
        ax[1].set_xlim(0, 5e9)
        ax[1].grid()
        plt.subplots_adjust(wspace = 0.4)
        plt.show()

eps = 1.5
W0 = 120*np.pi
Nt = 1100
Nx = 1200
maxSize_m = 4.5
dx = maxSize_m/Nx
maxSize = int(maxSize_m/dx)
probePos = int(maxSize_m*7/8/dx)
sourcePos = int(maxSize_m/2/dx)
Sc = 1
dt = dx*np.sqrt(eps)*Sc/3e8
probe = Probe(probePos, Nt, dt)
display = FieldDisplay(maxSize_m, dx, -1.5, 1.5, probePos, sourcePos)
Ez = np.zeros(maxSize)
Hy = np.zeros(maxSize)
Ez_old = Ez[-2]
Sc1 = Sc/np.sqrt(eps)
k = (Sc1-1)/(Sc1+1)
fp = 1e9
Md = 1.7
Np = Sc/fp/dt
for q in range(1, Nt):
    Hy[1:] = Hy[1:] +(Ez[:-1]-Ez[1:])*Sc/W0
    Ez[:-1] = Ez[:-1] + (Hy[:-1] - Hy[1:])*Sc*W0/eps
    Ez[sourcePos] += ((1 - 2 * (np.pi *(Sc*q/Np - Md)) ** 2) *
               np.exp(-(np.pi *(Sc*q/Np - Md)) ** 2))
    Ez[-1] = Ez_old + k*(Ez[-2]-Ez[-1])
    Ez_old = Ez[-2]
    probe.addData(Ez)
    if q % 10 == 0:
        display.updateData(Ez)

display.stop()
showProbeSignal(probe)




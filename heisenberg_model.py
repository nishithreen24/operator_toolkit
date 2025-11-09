from operator_toolkit import Kets, Bras, Operators
import numpy as np
import scipy.linalg as sp
import matplotlib.pyplot as plt

J = 1
N = 15
g = 0.5
T = 2

IS, sigz, sigx, sigy = Operators.pauli_matrices()
HS = J*(sigz*sigz*IS*IS + sigx*sigx*IS*IS + sigy*sigy*IS*IS + IS*sigz*sigz*IS + IS*sigx*sigx*IS + IS*sigy*sigy*IS + IS*IS*sigz*sigz + IS*IS*sigx*sigx + IS*IS*sigy*sigy)

IB = Operators(np.eye(N))
HB = np.zeros((N, N))
for n in range(N):
    HB[n][n] = n+(1/2)#hbar = omega = 1
HB = Operators(HB)

Hint = g*(sigz*IS*IS*IS * HB)

#print(HS.operator)
#print(HB.operator)
#print(sigz.operator)

H = (HS * IB) + (IS*IS*IS*IS * HB) + Hint

ketup = Kets(([1, 0]))
ketdown = Kets(([0, 1]))

psiS = ketup * ketdown * ketup * ketdown

rhoS = psiS @ psiS.dagger()

ZB = np.trace(sp.expm(((-1/T)*HB).operator))
#print(ZB)

rhoB = Operators(sp.expm(((-1/T)*HB).operator))
rhoB = (1/ZB)*rhoB

#print(rhoS.operator)
#print(rhoB.operator)

rho = rhoS * rhoB
#print(rho.operator)

t = np.arange(0.1, 40, 0.1)
mag_decay1 = np.zeros((len(t)))
mag_decay2 = np.zeros((len(t)))
mag_decay3 = np.zeros((len(t)))
mag_decay4 = np.zeros((len(t)))
S = np.zeros((len(t)))
for i in range(len(t)):
    G = -1j*t[i]*H.operator
    U = Operators(sp.expm(G))
    rhot = (U @ rho @ U.dagger())
    partial_traceS = rhot.partial_trace(dims = [16, N], trace_out=1)
    mag_vector1 = partial_traceS @ (sigz * IS * IS * IS)
    mag_decay1[i] = mag_vector1.trace()
    mag_vector2 = partial_traceS @ (IS * sigz * IS * IS)
    mag_decay2[i] = mag_vector2.trace()
    mag_vector3 = partial_traceS @ (IS * IS * sigz * IS)
    mag_decay3[i] = mag_vector3.trace()
    mag_vector4 = partial_traceS @ (IS * IS * IS * sigz)
    mag_decay4[i] = mag_vector4.trace()
    S[i] = partial_traceS.von_neumann_entropy()

#print(mag_decay1)
#print(S)

plt.plot(t, mag_decay1)
#plt.plot(t, mag_decay2)
#plt.plot(t, mag_decay3)
#plt.plot(t, mag_decay4)
#plt.ylim(-2, 2)
#plt.legend(["site 1", "site 2", "site 3", "site 4"])
plt.grid()
plt.title("Magnetization Decay")
plt.show()

plt.plot(t, S)
plt.grid()
plt.title("Entropy Growth")
plt.show()

from matplotlib.animation import FuncAnimation, FFMpegWriter

mag = np.vstack([mag_decay1, mag_decay2, mag_decay3, mag_decay4]).T


fig, ax = plt.subplots(figsize=(6,4))
sites = np.arange(1, 5)
bars = ax.bar(sites, mag[0])

ax.set_ylim(-1.1, 1.1)
ax.set_xlabel("Sites")
ax.set_ylabel("⟨σᶻᵢ(t)⟩")
ax.set_title("Time evolution of site magnetizations")

def update(frame):
    for bar, height in zip(bars, mag[frame]):
        bar.set_height(height)
        bar.set_color(plt.cm.coolwarm((height + 1) / 2))
    ax.set_title(f"Time evolution of site magnetizations (t = {t[frame]:.1f})")
    return bars

# Animate
ani = FuncAnimation(fig, update, frames=len(t), interval=50, blit=False)
writer = FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)
ani.save('my_animation.mp4', writer=writer)
plt.tight_layout()
#plt.show()
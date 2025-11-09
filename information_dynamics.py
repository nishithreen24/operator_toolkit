"""
import numpy as np
import scipy.linalg as sp
import matplotlib.pyplot as plt
from operator_toolkit import *

IS, sigz, sigx, sigy = Operators.pauli_matrices()
def XXX_spin_chain_hamiltonian(N, J):
    HS = Operators(np.zeros((np.pow(2, N), np.pow(2, N))))
    for i in range(N-1):
        if i == 0:
            termx = sigx
            termy = sigy
            termz = sigz
        else:
            termx = IS
            termy = IS
            termz = IS
        for j in range(1, N):
            if j == i or j == (i+1):
                termx = termx * sigx
                termy = termy * sigy
                termz = termz * sigz
            else:
                termx = termx * IS
                termy = termy * IS
                termz = termz * IS
        HS = HS + termx + termy + termz
    return J*HS

H_S = XXX_spin_chain_hamiltonian(4, J=1)
N = 15
g = 0.5
T = 2

IB = Operators(np.eye(N))
HB = np.zeros((N, N))
for n in range(N):
    HB[n][n] = n+(1/2)#hbar = omega = 1
HB = Operators(HB)

Hint = g*(sigz*IS*IS*IS * HB)

H = (H_S * IB) + (IS*IS*IS*IS * HB) + Hint

t = np.arange(0.1, 40, 0.1)
S_12_t = np.zeros(len(t))
C_14_t = np.zeros(len(t))
ketup = Kets(([1, 0]))
ketdown = Kets(([0, 1]))
psi_0 = ketup * ketdown * ketup * ketdown
rho_0_S = psi_0 @ psi_0.dagger()

ZB = np.trace(sp.expm(((-1/T)*HB).operator))
#print(ZB)

rhoB = Operators(sp.expm(((-1/T)*HB).operator))
rhoB = (1/ZB)*rhoB

#print(rhoS.operator)
#print(rhoB.operator)

rho = rho_0_S * rhoB
for i in range(len(t)):
    U_t = Operators(sp.expm(-1j*t[i]*H.operator))
    rho_t = U_t @ rho @ U_t.dagger()
    rho_12_t = rho_t.partial_trace(dims = [4, 4], trace_out = 1)
    S_12_t[i] = rho_12_t.von_neumann_entropy()

    C_14_t[i] = (rho_t.partial_trace(dims = [16, N], trace_out = 1) @ (sigz * IS * IS * sigz)).trace()

plt.grid()
plt.title("Entropy growth of subsystem spin 1-2")
plt.plot(t, S_12_t)
plt.show()
plt.grid()
plt.title("Two point correlation function between spins 1 and 4")
plt.plot(t, C_14_t)
plt.show()
"""
from operator_toolkit import *
import numpy as np
import matplotlib.pyplot as plt

#from .heisenberg_model import H, rhoS
N = 5
g = 0.5
J = 1
T = 2
n = 4
IS, sigz, sigx, sigy = Operators.pauli_matrices()
def spin_hamiltonian(n, J):
    H = Operators((np.zeros((np.pow(2, n), np.pow(2, n)))))
    for i in range(n-1):
        if i==0:
            termx = sigx
            termy = sigy
            termz = sigz
        else:
            termx = IS
            termy = IS
            termz = IS
        for j in range(1, n):
            if j==i or j==(i+1):
                termx *= sigx
                termy *= sigy
                termz *= sigz
            else:
                termx *= IS
                termy *= IS
                termz *= IS
        H = H + termx + termy + termz
    return (J)*H
HS = spin_hamiltonian(n = n, J = J)
IB = Operators(np.eye(N))
HB = np.zeros((N, N))
for n in range(N):
    HB[n][n] = n+(1/2)#hbar = omega = 1
HB = Operators(HB)
Hint = Hint = g*(sigz*IS*IS*IS * HB)
H = (HS * IB) + (IS*IS*IS*IS * HB) + Hint
ketup = Kets(([1, 0]))
ketdown = Kets(([0, 1]))
psiS = ketup * ketdown * ketup * ketdown
rhoS = psiS @ psiS.dagger()
ZB = ((-1/T)*HB).exp().trace()
rhoB = ((-1/T)*HB).exp()
rhoB = rhoB * (1/ZB)
rho = rhoS * rhoB



t = np.arange(0.001, 100, 0.1)
S_12 = np.zeros((len(t)))
C_14 = np.zeros((len(t)))
for i in range(len(t)):
    #print(t[i])
    U = ((-1j*t[i])*H).exp()
    #print("U computed")
    rhot = U @ rho @ U.dagger()
    #print("rhot computed")
    rhoSt = rhot.partial_trace(dims = [np.pow(2, n), N], trace_out = 1)
    #print("rhoSt computed")
    rho12t = rhoSt.partial_trace(dims = [4, 4], trace_out = 1)
    #print("rho12t computed")
    S_12[i] = rho12t.von_neumann_entropy()

    C_14[i] = (rhoSt @ (sigz * IS * IS * sigz)).trace()

plt.figure()

plt.subplot(1, 2, 1)
plt.title("Bipartite Entanglement Entropy")
plt.grid(True)
plt.plot(t, S_12)
plt.xlabel("Time")
plt.subplot(1, 2, 2)
plt.title("Two-Point Correlation Function (spins 1 and 4)")
plt.grid(True)
plt.plot(t, C_14)
plt.xlabel("Time")
plt.show()
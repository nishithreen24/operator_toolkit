"""
from operator_toolkit import Kets, Bras, Operators
import numpy as np
import scipy.linalg as sp
import matplotlib.pyplot as plt

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
def disordered_hamiltonian(N, J, W):
    H_S = XXX_spin_chain_hamiltonian(N, J)
    for i in range(N):
        if i == 0:
            term = sigz
        else:
            term = IS
        for j in range(1, N):
            if i==j:
                term = term * sigz
            else:
                term = term * IS
        H_S = H_S + np.random.randint(low = -W, high = W + 1)*term
    return H_S

H_S_prime = disordered_hamiltonian(N = 4, J = 1, W = 0)
H_S_weak = disordered_hamiltonian(N = 4, J = 1, W = 1)
H_S_strong = disordered_hamiltonian(N = 4, J = 1, W = 8)
"""
from operator_toolkit import *
import numpy as np
import matplotlib.pyplot as plt

N = 15
g = 0.5
J = 1
T = 2
n = 4
W = 0
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

def disorder(HS, W):
    HS_prime = Operators(HS.operator)
    for i in range(n):
        if i==0:
            term = sigz
        else:
            term = IS
        for j in range(1,n):
            if i==j:
                term *= sigz
            else:
                term *= IS
        HS_prime += np.random.uniform(-W, W)*term
    return HS_prime

HS_prime = disorder(HS, W = W)
#print(np.array_equal(HS.operator, HS_prime.operator))

IB = Operators(np.eye(N))
HB = np.zeros((N, N))
for n in range(N):
    HB[n][n] = n+(1/2)#hbar = omega = 1
HB = Operators(HB)
Hint = g*(sigz*IS*IS*IS * HB)
H = (HS_prime * IB) + (IS*IS*IS*IS * HB) + Hint
ketup = Kets(([1, 0]))
ketdown = Kets(([0, 1]))
psiS = ketup * ketdown * ketup * ketdown
rhoS = psiS @ psiS.dagger()
ZB = ((-1/T)*HB).exp().trace()
rhoB = ((-1/T)*HB).exp()
rhoB = rhoB * (1/ZB)
rho = rhoS * rhoB

t = np.arange(0.001, 100, 0.1)
mag_decay1 = np.zeros((len(t)))
mag_decay2 = np.zeros((len(t)))
mag_decay3 = np.zeros((len(t)))
mag_decay4 = np.zeros((len(t)))
S = np.zeros((len(t)))
for i in range(len(t)):
    U =  ((-1j*t[i])*H).exp()
    rhot = U @ rho @ U.dagger()
    rhoSt = rhot.partial_trace(dims = [np.pow(2, n), N], trace_out = 1)
    mag_decay1[i] = (rhoSt @ (sigz * IS * IS * IS)).trace()
    mag_decay2[i] = (rhoSt @ (IS * sigz * IS * IS)).trace()
    mag_decay3[i] = (rhoSt @ (IS * IS * sigz * IS)).trace()
    mag_decay4[i] = (rhoSt @ (IS * IS * IS * sigz)).trace()
    S[i] = rhoSt.von_neumann_entropy()


plt.figure()
plt.suptitle(f"W = {W}")
plt.subplot(1, 2, 1)
plt.title("Site magnetizations")
plt.grid(True)
plt.plot(t, mag_decay1, label='Site 1')
plt.plot(t, mag_decay2, label='Site 2')
plt.plot(t, mag_decay3, label='Site 3')
plt.plot(t, mag_decay4, label='Site 4')
plt.ylim(-1.1, 1.1)
plt.xlabel("Time")
plt.legend()
plt.subplot(1, 2, 2)
plt.title("Entropy")
plt.grid(True)
plt.plot(t, S)
plt.xlabel("Time")
plt.show()
"""
from operator_toolkit import *
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
J=1

H_S = XXX_spin_chain_hamiltonian(4, J)

D, P = H_S.spectral_decomposition()

n = 8

lam = D.operator[n][n]
psi = Kets(P.operator[:, n])

print(n, "th excited state")
print(lam)
print(psi.ket)

rho = psi @ psi.dagger()
print(rho.operator)

rho_12 = rho.partial_trace(dims = [4, 4], trace_out = 1)
print(rho_12.operator)
S_rho_12 = rho_12.von_neumann_entropy()
print(S_rho_12)

H_12 = XXX_spin_chain_hamiltonian(2, 1)
#print(H_12.operator)

T = 1.567
rho_thermal = Operators(sp.expm(((-1/T)*H_12).operator))
#print(rho_thermal.operator)
rho_thermal = (1/rho_thermal.trace())*rho_thermal
print(rho_thermal.operator)
print(rho_thermal.von_neumann_entropy())
"""

from operator_toolkit import *
import numpy as np
import scipy.linalg as sp
import matplotlib.pyplot as plt

I_S, sigz, sigx, sigy = Operators.pauli_matrices()

def spin_hamiltonian(n, J):
    H = Operators((np.zeros((np.pow(2, n), np.pow(2, n)))))
    for i in range(n-1):
        if i==0:
            termx = sigx
            termy = sigy
            termz = sigz
        else:
            termx = I_S
            termy = I_S
            termz = I_S
        for j in range(1, n):
            if j==i or j==(i+1):
                termx *= sigx
                termy *= sigy
                termz *= sigz
            else:
                termx *= I_S
                termy *= I_S
                termz *= I_S
        H = H + termx + termy + termz
    return (J)*H

H_S = spin_hamiltonian(n = 4, J = 1)
E, psi = H_S.spectral_decomposition()

n = 6
E_n = E.operator[n][n]
psi_n = Kets(psi.operator[:, n])
rho_n = psi_n @ psi_n.dagger()
rho_12 = rho_n.partial_trace(dims = [4, 4], trace_out = 1)

def average_energy(T):
    G = (H_S*(-1/T)).exp()
    Z = G.trace()
    return (H_S@G).trace()/Z


"""def average_energy(H, T):
    beta = 1 / T
    expH = sp.expm(-beta * H)
    Z = np.trace(expH)
    HexpH = H @ expH
    return np.trace(HexpH) / Z
"""

print(E_n)
print(average_energy(T = 9.77))
T = 9.77

H_12 = spin_hamiltonian(2, 1)
Z = (((-1/T)*H_12).exp()).trace()
rho_thermal = (((-1/T)*H_12).exp())*(1/Z)
print(f"rho_12 for {n}th excited state:")
print(rho_12.operator)
print(rho_12.von_neumann_entropy())
print("rho_thermal")
print(rho_thermal.operator)
print(rho_thermal.von_neumann_entropy())
"""
t = np.arange(0.1, 40, 0.1)
S_12 = np.zeros((len(t)))
S_thermal = np.zeros((len(t)))

for i in range(len(t)):
    U = ((-1j/t[i])*H_12).exp()
    rho_12_t = U @ rho_12 @ U.dagger()
    rho_thermal_t = U @ rho_thermal @ U.dagger()
    S_12[i] = rho_12_t.von_neumann_entropy()
    S_thermal[i] = rho_thermal_t.von_neumann_entropy()

plt.ylim(0, 5)
plt.plot(t, S_12)
plt.plot(t, S_thermal)
plt.title("Entanglement Entropy")
plt.legend(['Heisenberg State', 'Canonical Thermal State'])
plt.grid()
plt.show()
"""
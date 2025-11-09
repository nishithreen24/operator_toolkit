from operator_toolkit import *
from numpy import sqrt, exp, array_equal, random, pi

class gates(Operators):

    I, sigz, sigx, sigy = Operators.pauli_matrices()
    ket0 = Kets([1, 0])
    ket1 = Kets([0, 1])

    plusx = (1/sqrt(2))*(ket0+ket1)
    minusx = (1/sqrt(2))*(ket0-ket1)

    q0 = Kets(ket0.ket)
    q1 = Kets(ket0.ket)
    q3 = Kets(ket0.ket)
    #def CNOT(q0, q1):   

    CNOT = ((ket0 @ ket0.dagger())*I) + ((ket1 @ ket1.dagger())*sigx)
    #print((CNOT @ (ket1 * ket0)).ket)
    #print(CNOT(ket1, ket1))

    CPHASE = ((ket0 @ ket0.dagger())*I) + ((ket1 @ ket1.dagger())*sigz)
    #print((CPHASE @ (ket1 * ket1)).ket)

    SWAP = ((ket0*ket0)@(ket0*ket0).dagger()) + ((ket0*ket1)@(ket1*ket0).dagger()) + ((ket1*ket0)@(ket0*ket1).dagger()) + ((ket1*ket1)@(ket1*ket1).dagger())
    #print((SWAP @ (ket0 *ket1)).ket)

    Fredkin = ((ket0 @ ket0.dagger()) * I * I) + (ket1 @ ket1.dagger()) * SWAP
    #print((Fredkin @ (ket1*ket0*ket1)).ket)

    #Hadamard = (ket0 @ plusx.dagger()) + (ket1 @ minusx.dagger())
    Hadamard = (1/sqrt(2))*(sigx + sigz)

    #print((Hadamard @ ket0).ket)
    def X(q):
        return gates.sigx @ q
    def Y(q):
        return gates.sigy @ q
    def Z(q):
        return gates.sigz @ q
    def hadamard(q):
        return gates.Hadamard @ q
    def phase(q, theta):
        return ((gates.ket0 @ gates.ket0.dagger()) + exp(1j*theta)*(gates.ket1 @ gates.ket1.dagger())) @ q
#    def phase_gate(theta):
#        return ((gates.ket0 @ gates.ket0.dagger()) + exp(1j*theta)*(gates.ket1 @ gates.ket1.dagger()))
    def cnot(c, t):
        return ((gates.ket0 @ gates.ket0.dagger())*gates.I) + ((gates.ket1 @ gates.ket1.dagger())*gates.sigx) @ (c*t)
    def cnot_target(c, t):
        if array_equal(c.ket, gates.ket0.ket):
            return t
        elif array_equal(c.ket, gates.ket1.ket):
            return gates.sigx @ t
    def cphase(c, t, theta):
        return ((gates.ket0 @ gates.ket0.dagger())*gates.I) + ((gates.ket1 @ gates.ket1.dagger())*gates.sigz) @ (c*t)
    def swap(q1, q2):
        return gates.SWAP @ (q1 * q2)
    def fredkin(c, t1, t2):
        return ((gates.ket0 @ gates.ket0.dagger()) * gates.I * gates.I) + ((gates.ket0 @ gates.ket0.dagger()) * gates.SWAP) @ (c*t1*t2)
    def toffoli(c, t1, t2):
        ket00 = gates.ket0*gates.ket0
        ket01 = gates.ket0*gates.ket1
        ket10 = gates.ket1 * gates.ket0
        ket11 = gates.ket1 * gates.ket1
        return (((ket00 @ ket00.dagger() + ket01 @ ket01.dagger() + ket10 @ ket10.dagger()) * gates.I) + (ket11 @ ket11.dagger())*gates.sigx) @ (c * t1 * t2)
    def bell_states(q1, q2):
        return (gates.CNOT(gates.hadamard(q1), q2))
    def GHZ_states(q1, q2, q3):
        return gates.CNOT(gates.CNOT(gates.hadamard(q1), q2), q3)
    def measure(q1):
        probabilities = []
        for c in q1.ket:
            probabilities.append(abs(c)**2)
        return random.choice([gates.ket0, gates.ket1], probabilities)
    def teleport(q1, q2, q3):
        q2 = gates.hadamard(q2)
        q3 = gates.cnot(q2, q3)
        q2 = gates.cnot(q1, q2)
        q1 = gates.hadamard(q1)
        q1 = gates.measure(q1)
        q2 = gates.measure(q2)
        q3 = gates.cnot(q2, q3)
        q3 = gates.cphase(q1, q3, pi)
        return q3

#   def qaoa(betak, gammak):
        
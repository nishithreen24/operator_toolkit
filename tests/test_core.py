import pytest
from operator_toolkit import Kets, Bras, Operators
import numpy as np

ket0 = Kets([1, 0])
ket1 = Kets([0, 1])
a = np.sqrt(0.5)

def test_Kets_add():
    #print(a*ket0+a*ket1)
    assert np.array_equal((a*ket0+a*ket1).ket, np.array(([np.sqrt(0.5)], [np.sqrt(0.5)])))

sigx = Operators(data = np.array(([0, 1], [1, 0])), m = 2, n = 2)
sigy = Operators(data = np.array(([0, -1j], [1j, 0])), m = 2, n = 2)
sigz = Operators(data = np.array(([1, 0], [0, -1])), m = 2, n = 2)

def test_commutator():
    assert np.array_equal(Operators.commutator(sigx, sigy), 2j*sigz)
import numpy as np
import scipy as sp
np.set_printoptions(precision=3, suppress=True)
class Kets:
    def __init__(self, data = None, n = None):
        if data is not None:
            self.ket = np.reshape(data, (-1, 1))
        elif n is not None:
            self.ket = np.zeros((n, 1))
        else:
            raise ValueError("Either dimensions or data must be provided")
    def hermitian_conjugate(self):
        return Bras(np.conjugate(self.ket))
    adjoint = hermitian_conjugate
    dagger = adjoint
    def __add__(self, other):
        if len(self.ket) != len(other.ket):
            raise ValueError("Incompatible dimensions")
        else:
            return Kets(self.ket+other.ket)
    def __sub__(self, other):
        if len(self.ket) != len(other.ket):
            raise ValueError("Incompatible dimensions")
        else:
            return Kets(self.ket-other.ket)
    def __rmul__(self, other):
        if isinstance(other, (int, float, complex)):
            return Kets(other*self.ket)
        else:
            raise TypeError("Incompatible Objects")
    def __rmatmul__(self, other):
        if isinstance(other, Operators):
            if(other.operator.shape[1]==len(self.ket)):
                return Kets(other.operator @ self.ket)
            else:
                raise ValueError("Incompatible dimensions")
        else:
            return TypeError("Incompatible Objects")
    def __matmul__(self, other):
        if isinstance(other, Bras):
            return Operators(data = np.outer(self.ket, other.bra), m=other.bra.shape[1], n=len(self.ket))
        elif isinstance(other, Kets):
            if(len(self.ket)==len(other.ket)):
                return Kets.dagger(self).bra @ other.ket
            else:
                raise ValueError("Incompatible dimensions")
        else:
            raise TypeError("Incompatible Objects")
    def __mul__(self, other):
        if isinstance(other, (int, float, complex)):
            return Kets(other*self.ket)
        elif isinstance(other, Kets):
            return Kets(np.outer(self.ket, other.ket))
        else:
            raise TypeError("Incompatible Objects")
    def rho(self):
        return Operators(self.ket @ self.dagger().bra)
    density_matrix = rho
    def tensor_product(*args):
        out = args[0]
        for i in range(1, len(args)):
            out = out * args[i]
        return out
        

class Bras:
    def __init__(self, data = None, n = None):
        if data is not None:
            self.bra = np.reshape(data, (1, -1))
        elif n is not None:
            self.bra = np.zeros((1, n))
        else:
            raise ValueError("Either dimensions or data must be provided")
    def hermitian_conjugate(self):
        return Kets(np.conjugate(self.bra))
    adjoint = hermitian_conjugate
    dagger = adjoint
    def __add__(self, other):
        if self.bra.shape[1] != other.bra.shape[1]:
            raise ValueError("Incompatible dimensions")
        else:
            return Bras(self.bra+other.bra)
    def __sub__(self, other):
        if self.bra.shape[1] != other.bra.shape[1]:
            raise ValueError("Incompatible dimensions")
        else:
            return Bras(self.bra-other.bra)
    def __rmul__(self, other):
        if isinstance(other, (int, float, complex)):
            return Bras(other*self.bra)
        else:
            raise TypeError("Incompatible Objects")
    def __matmul__(self, other):
        if isinstance(other, Kets):
            if(self.bra.shape[1]==len(other.ket)):
                return (self.bra @ other.ket)
            else:
                raise ValueError("Incompatible dimensions")
        elif isinstance(other, Operators):
            if(self.bra.shape[1]==len(other.operator)):
                return Bras(self.bra @ other.operator)
            else:
                raise ValueError("Incompatible dimensions")
        elif isinstance(other, Bras):
            if(self.bra.shape[1]==other.bra.shape[1]):
                return self.bra @ Bras.dagger(other).ket
            else:
                raise ValueError("Incompatible dimensions")
        else:
            raise TypeError("Incompatible Objects")
    def __mul__(self, other):
        if isinstance(other, Bras):
            return Bras(np.outer(self.bra, other.bra))
        elif isinstance(other, (int, float, complex)):
            return Bras(other*self.bra)
        else:
            raise TypeError("Incompatible Objects")
    def rho(self):
        return Operators(self.dagger().ket @ self.bra)
    density_matrix = rho
    def tensor_product(*args):
        out = args[0]
        for i in range(1, len(args)):
            out = out * args[i]
        return out
        

class Operators:
    def __init__(self, data = None, m = None, n = None): #A: \mathbb{C}^m \to \mathbb{C}^n
        if data is not None:
            if (m is not None and n is not None):
                self.operator = np.asarray(np.reshape(data, (n, m)))
            elif m is not None:
                self.operator = np.asarray(np.reshape(data, (m, m)))
            elif n is not None:
                self.operator = np.asarray(np.reshape(data, (n, n)))
            elif np.sqrt(np.array(data).size)%1==0:
                self.operator = np.asarray(np.reshape(data, ((int)(np.sqrt(np.array(data).size)), (int)(np.sqrt(np.array(data).size)))))
            else:
                raise ValueError("Dimension(s) must be provided")
        else:
            if (m is not None and n is not None):
                self.operator = np.asarray(np.zeros((n, m)))
            elif m is not None:
                self.operator = np.asarray(np.zeros((m, m)))
            elif n is not None:
                self.operator = np.asarray(np.zeros((n, n)))
            else:
                raise ValueError("Dimension(s) and/or data must be provided")
    def hermitian_conjugate(self):
        return Operators(data = self.operator.conj().T, m = self.operator.shape[0], n = self.operator.shape[1])
    adjoint = hermitian_conjugate
    dagger = adjoint
    def __add__(self, other):
        if (len(self.operator) == len(other.operator)) and (self.operator.shape[1]==other.operator.shape[1]):
            return Operators(data = self.operator+other.operator, m = self.operator.shape[1], n = len(self.operator))
        else:
            raise ValueError("Incompatible dimensions")
    def __sub__(self, other):
        if (len(self.operator) == len(other.operator)) and (self.operator.shape[1]==other.operator.shape[1]):
            return Operators(data = self.operator-other.operator, m = self.operator.shape[1], n = len(self.operator))
        else:
            raise ValueError("Incompatible dimensions")
    def __matmul__(self, other):
        if isinstance(other, Operators):
            if(self.operator.shape[1]==len(other.operator)):
                return Operators(data = self.operator @ other.operator, m = other.operator.shape[1], n = len(self.operator))
            else:
                raise ValueError("Incompatible dimensions")
        elif isinstance(other, Kets):
            if self.operator.shape[1]==other.ket.shape[0]:
                return Kets(self.operator @ other.ket)
            else:
                raise ValueError("Incompatible dimensions")
        else:
            raise TypeError("Incompatible Objects")
    def __mul__(self, other):
        if isinstance(other, (int, float, complex)):
            return Operators(data = other*self.operator, m = self.operator.shape[1], n = len(self.operator))
        elif isinstance(other, Operators):
            return Operators(data = np.kron(self.operator, other.operator), m = self.operator.shape[1]*other.operator.shape[1], n = len(self.operator)*len(other.operator))
        else:
            raise TypeError("Incompatible Objects")
    def __rmul__(self, other):
        if isinstance(other, (int, float, complex)):
            return Operators(data = other*self.operator, m = self.operator.shape[1], n = len(self.operator))
        else:
            raise TypeError("Incompatible Objects")
    def isHermitian(self):
        if(self.operator.shape[0]==self.operator.shape[1]):
            return np.array_equal(self.operator, Operators.dagger(self).operator)
        else:
            raise ValueError("Only square matrices can be hermitian")
    def isUnitary(self):
        if(len(self.operator)==self.operator.shape[1]):
            return np.array_equal(self.operator @ Operators.dagger(self).operator, np.eye(len(self.operator)))
        else:
            raise ValueError("Only square matrices can be Unitary")
    def isNormal(self):
        if(len(self.operator)==self.operator.shape[1]):
            return np.array_equal(self.operator @ Operators.dagger(self).operator, Operators.dagger(self).operator @ self.operator)
        else:
            raise ValueError("Only square matrices can be Normal")
    def isPositive(self):
        eigenvalues = np.linalg.eigvalsh(self.operator)
        return np.all(eigenvalues >= 0)
    def commutator(A, B):
        return (A.operator @ B.operator) - (B.operator @ A.operator) 
    def anticommutator(A, B):
        return (A.operator @ B.operator) + (B.operator @ A.operator)
    def spectral_decomposition(self):
        if(self.isHermitian()):
            eigenvalues, eigenvectors = np.linalg.eigh(self.operator)
            #values = Operators(np.diag(eigenvalues))
            #vectors = Operators(eigenvectors)
            #print("D =\n", values.operator)
            #print("P =\n", vectors.operator)
            return (Operators(data = np.diag(eigenvalues)), Operators(data = eigenvectors))
        else:
            raise ValueError("Input must be a Hermitian Matrix")
    def tensor_product(*args):
        out = args[0]
        for i in range(1, len(args)):
            out = out * args[i]
        return out
    def trace(self):
        return np.round((np.trace(self.operator)), 3)
    def partial_trace(self, dims, trace_out):
        if len(dims)!=2:
            raise ValueError("This function only handles density matrices of 2 subsystems")
        else:
            dimA, dimB = dims
            if(trace_out == 0):#rhoB = partial trace over A of rhoAB
                rhoB = np.zeros((dimB, dimB), dtype=complex)
                for i in range(dimA):
                    rhoB += self.operator[i*dimB:(i+1)*dimB, i*dimB:(i+1)*dimB]#takes the sum of dimB by dimB blocks along the diagonal of the composite system density matrix
                return Operators(rhoB)
            elif(trace_out == 1):#rhoA = partial trace over B of rhoAB
                rhoA = np.zeros((dimA, dimA))
                for i in range(dimA):
                    for j in range(dimA):
                        block = self.operator[i*dimB:(i+1)*dimB, j*dimB:(j+1)*dimB]#each element of rhoA is the trace of a dimBxdimB block, as there are dimAxdimA such blocks
                        rhoA[i, j] = np.trace(block)
                return Operators(rhoA)
            else:
                raise ValueError("Trace_out must be over A(0) or B(1)")

    def von_neumann_entropy(self):
        if (True):#(self.isHermitian()) and (self.trace()==1) and (self.isPositive):
            eigs = np.linalg.eigvalsh(self.operator)
            eigs = eigs[eigs>0]
            return np.round(np.sum(-1*eigs*np.log2(eigs)), 3)
        else:
            raise ValueError("Von Neumann Entropy can only be calculated for Hermitian, Positive-definite matrices with trace 1")
    def pauli_matrices():
        I = Operators(data = np.eye(2))
        sigz = Operators(data = np.array(([1, 0], [0, -1])))
        sigx = Operators(data = np.array(([0, 1], [1, 0])))
        sigy = Operators(data = np.array(([0, -1j], [1j, 0])))
        return I, sigz, sigx, sigy
    def exp(self):
        return Operators(sp.linalg.expm(self.operator))
    
    class DensityMatrix:
        def __init__(self, data):
            if isinstance(data, Operators):
                self.rho = data
            elif isinstance(data, Kets):
                self.rho = data @ Bras(data)
            else:
                print("Incorrect data")
        def purity(self):
            rho_squared = self.rho @ self.rho
            return rho_squared.trace()
        def entropy(self):
            eigenvalues, eigenvectors = self.rho.spectral_decomposition()
            eigenvalues = eigenvalues[eigenvalues > 1e-10]
            logrho_eigvals = np.log2(eigenvalues)
            logrho = eigenvectors @ logrho_eigvals @ eigenvectors.dagger()
            return (-1 * (self.rho @ logrho).trace())
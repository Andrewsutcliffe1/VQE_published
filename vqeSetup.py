import pennylane as qml
import numpy as np
from pennylane.pauli import PauliWord
import networkx as nx
import jax.numpy as jnp
import jax.debug

def TFHamiltonianRevised(J,h,disorder,dims):
    
    x = 2 if disorder else 1 #used to skip wires when disorder is true
    coeffs = jnp.concatenate([h,-J])
    ops = [qml.X(x*i) for i in range(len(h))]
    ops.extend([(qml.Z(x*i) @ qml.Z((x*(i + 1)))) for i in range(len(J)-1)])
    ops.extend([(qml.Z(x*(len(J)-1)) @ qml.Z(0))])

    hamiltonian = qml.Hamiltonian(coeffs, ops)
    return hamiltonian

def exactSolution(HMatrix):
    evals, evecs = np.linalg.eigh(HMatrix)
    posOfMinEval = np.flatnonzero(evals == evals.min())
    GrStateE = evals[posOfMinEval[0]]
    GrStates = evecs[:,posOfMinEval]
    return evals, GrStateE, GrStates

##########

def initialState(J, n, disorder):
    state0s = np.repeat(0, n)
    
    if disorder:
        stateJs = ((-J + 1) / 2).astype(int) # J being 1 means ancilla in state 0, J being -1 means ancilla in state 1
        # Weaving physical and ancilla qubits
        combined_state = jnp.empty(n * 2, dtype=int)
        combined_state = combined_state.at[0::2].set(state0s)
        combined_state = combined_state.at[1::2].set(stateJs)
        combined_wires = list(range(n * 2))
    else:
        combined_state = state0s
        combined_wires = list(range(n))
    return combined_state, combined_wires


def ansatz(J,n,variationalParams,numLayers,disorder):
    x = 2 if disorder else 1 # this is used to skip wires when disorder is true
    state, wires = initialState(J,n,disorder)
    
    qml.BasisState(state=state, wires=wires) # I do indeed need these 2 lines

    for i in range(numLayers):
        for j in range(n):
            qml.RX(variationalParams[j,0,i],x*j)
            qml.RY(variationalParams[j,1,i],x*j)
            qml.RX(variationalParams[j,2,i],x*j)
            if disorder:
                qml.RX(variationalParams[j,4,i], wires=[2*j+1])
                qml.RY(variationalParams[j,5,i], wires=[2*j+1])
                qml.RX(variationalParams[j,6,i], wires=[2*j+1])

        for k in range(n-1):
            qml.IsingZZ(variationalParams[k,3,i], wires=[x*k,(x*(k+1))])

        qml.IsingZZ(variationalParams[n-1,3,i], wires=[x*(n-1),0])

        if disorder: # Ancilla entangling gates, logic could be simplified
            for k in range(n):
                qml.IsingZZ(variationalParams[k,7,i], wires=[2*k,2*k+1])
                qml.IsingZZ(variationalParams[k,8,i], wires=[2*k+1,(2*k+2)%(2*n)])
    
    
    

#######
def list_indiv_costs(variationalParams, n, numLayers, disorder, disorderCases):
     # note this func doesnt need to know current J, it treats all disorder cases equally
    def helper(variationalParams, J, n, numLayers, disorder):
        h = jnp.repeat(1, n)
        ham = TFHamiltonianRevised(J, h, disorder, (1, n))
        return local_cost_fn(variationalParams, J, n, numLayers, ham, disorder)

    vmappedHelper = jax.vmap(helper, in_axes=[None, 0, None, None, None]) #speeds up in parallel
    eachExpVal = vmappedHelper(variationalParams, disorderCases, n, numLayers, disorder)
    return eachExpVal # the local cost_fn for each disorder case

# This is the one that is jitted
def global_cost_mean(variationalParams, n, numLayers, disorder, disorderCases):
    return jnp.mean(list_indiv_costs(variationalParams, n, numLayers, disorder, disorderCases))


def local_cost_fn(variationalParams,J,n,numLayers,ham,disorder):
    wires = n if not disorder else 2 * n
    @qml.qnode(qml.device("default.qubit", wires),interface = 'jax')
    def expVal(J,n,variationalParams,disorder):
        ansatz(J,n,variationalParams,numLayers,disorder)
        return qml.expval(ham)
    return expVal(J,n,variationalParams,disorder)


# Used for fidelity calucations after algorithm runs
def getRho(variationalParams,J,n,numLayers,ham,disorder=False):
    wires = n if not disorder else 2 * n
    @qml.qnode(qml.device("default.qubit", wires),interface = 'jax')
    def state(J,n, variationalParams, disorder):
        ansatz(J, n, variationalParams, numLayers, disorder)
        return qml.density_matrix(wires = list(range(n)) if not disorder else list(range(0, 2 * n, 2))) #this is the density matrix of the physical qubits only
    return state(J,n, variationalParams, disorder)
    
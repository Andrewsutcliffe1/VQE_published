import numpy as np

import optax
import jax
import jax.numpy as jnp
from jax import value_and_grad
from jax import config
config.update("jax_enable_x64", True)

from vqeSetup import local_cost_fn, global_cost_mean, list_indiv_costs


def optLoopAdamW(n, numLayers, J, h, ham, lrs, numSeeds, disorder, disorderCases, prevLayersParams=None, seed=42, returnOnlyEf=True):
    max_iterations = 7000
    conv_tol = 1e-10
    lrs = [0.001] if lrs is None else lrs
    varParamsPerQbitPerLayer = 9 if disorder else 4 # See circuit drawing: 4 params per layer per qubit if no disorder, 9 if disorder (2 ancilla params for entangling, and 3 more for the arbiratry ancilla rotation. ie ancilla below current qubit "belongs" to current qubit)

    energyMatrix = np.empty(shape=(numSeeds,len(lrs),),dtype=object)
    variationalParamsMatrix = jnp.zeros(shape=(n,varParamsPerQbitPerLayer,numLayers,numSeeds,len(lrs))) # in format of circuit drawing, with 4th/5th dim to allow seeds/lrs handled in parallel
    convergedMatrix = np.full((numSeeds, len(lrs)), False, dtype=bool) # this is a boolean matrix to check if the run converged or not.

    if disorderCases is None:
        jittedGradlocal = jax.jit(value_and_grad(local_cost_fn), static_argnames = ["n","numLayers","ham","disorder"]) #only take gradient wrt the angles (doesnt work to add "J" as a static param, but appears not to matter)
    else: # we are training on multiple cases
        eachJsDescent = np.empty((len(disorderCases),len(lrs)), dtype=object)
        for i in range(len(disorderCases)):
            for j in range(len(lrs)):
                eachJsDescent[i, j] = []
        jittedGradGlobal = jax.jit(value_and_grad(global_cost_mean), static_argnames = ["n","numLayers","disorder"]) #breaks if you tell it disorderCases is static. still though, the shape of the grad show its not actually doing it wrt disorderCases 
        def full_call(variationalParams, n, numLayers, disorder, disorderCases): # I need to split into 2 fns to get the gradient as well as each individual local energy
            proposedGlobalEnergy, gradient = jittedGradGlobal(variationalParams, n, numLayers, disorder, disorderCases)
            proposedLocalEnergies = list_indiv_costs(variationalParams, n, numLayers, disorder, disorderCases)
            return proposedGlobalEnergy, proposedLocalEnergies, gradient
        full_call = jax.jit(full_call, static_argnames = ["n", "numLayers", "disorder"])

    for lridx,lr in enumerate(lrs):
        opt = optax.adamw(lr, b1 = 0.9, b2=0.95) # classical optimiser

        for seedidx in range(numSeeds):
            key = jax.random.PRNGKey(seedidx+seed)
            seedEnergies = []
            
            if prevLayersParams is not None: # here we initialise the previous layers with the good params from the last run, if this is not None.
                variationalParams = jnp.zeros(shape=(n,varParamsPerQbitPerLayer,numLayers),dtype=jnp.float64) #this is in the physical format of the drawing, with 3rd dim being layers
                variationalParams = variationalParams.at[:,:,0:prevLayersParams.shape[2]].set(prevLayersParams[:,:,:,seedidx,lridx]) #selects the correct seed. Ths is Jax lingo for setting the first few layers to what we want
            else:
                variationalParams = jax.random.uniform(key, shape=(n,varParamsPerQbitPerLayer,numLayers), minval=0, maxval=2 * jnp.pi, dtype=jnp.float64)
            
            variationalParams = jnp.array(variationalParams) # unsure if i need this line
            opt_state = opt.init(variationalParams)
                       
            diffBefore = conv_tol+1 #bigger than conv_tol just to ensure we dont stop immediately
            for j in range(max_iterations):
                if disorderCases is None:
                    proposedEnergy, gradient = jittedGradlocal(variationalParams,J,n,numLayers,ham,disorder)   #note, gradient is only taken wrt variationalParams, ie gradient.shape = variationalParams.shape, even though i never told it to exclude J
                else: 
                    proposedEnergy, proposedLocalEnergies, gradient = full_call(variationalParams,n,numLayers,disorder,disorderCases)
                    for dc_idx in range(disorderCases.shape[0]):
                        eachJsDescent[dc_idx, lridx].append(float(proposedLocalEnergies[dc_idx]))
                if j == max_iterations//2:
                    print("Half way")
                updates, opt_state = opt.update(gradient, opt_state, variationalParams) # opt_state is just being tracked to allow for momentum based gradients; right now it does nothing
                variationalParams = optax.apply_updates(variationalParams, updates)
    
                diff = 0 if j == 0 else proposedEnergy-seedEnergies[-1]
                converged = np.abs(diff) < conv_tol and np.abs(diffBefore) < conv_tol
    
                seedEnergies.append(float(proposedEnergy))
                diffBefore = diff
                if converged:
                    print("Tolerance reached! Stopping after {} iterations".format(j))
                    energyMatrix[seedidx,lridx,] = seedEnergies
                    convergedMatrix[seedidx,lridx] = True
                    variationalParamsMatrix = variationalParamsMatrix.at[:,:,:,seedidx,lridx].set(variationalParams)
                    break
            energyMatrix[seedidx,lridx,] = seedEnergies
            variationalParamsMatrix = variationalParamsMatrix.at[:,:,:,seedidx,lridx].set(variationalParams)


    if disorderCases is not None:
        combinedEnergies = np.empty(shape=(disorderCases.shape[0]+1,len(lrs),),dtype=object)
        combinedEnergies[0,:] = energyMatrix[0,:] # energyMatrix is the global descent
        for idx in range(len(eachJsDescent)):
            combinedEnergies[idx + 1, :] = eachJsDescent[idx,:]
    else:
        combinedEnergies = energyMatrix  # already shape (num_seeds, num_lrs, descent_array)

    if returnOnlyEf: # default, last E value only
        return np.array([[E[-1] for E in row] for row in combinedEnergies]), variationalParamsMatrix, convergedMatrix
    else: # if you want to check the full descent. here it is assumed that there's only 1 seed and 1 lr
        return combinedEnergies, variationalParamsMatrix, converged
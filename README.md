# Variational Quantum Eigensolver Training for Disordered Spin Hamiltonian

## Project with Gian and Friederike under CQSL

The goal: Parallel Training
The goal of this project is to, given a family of Hamiltonians, investigate whether the parameters learnt by the circuit on a subset of Hamiltonians can generalise to the unseen cases. I refer to this notion as \textit{parallel training}. Finding all ground states individually for an exponentially large family of Hamiltonians is computationally intensive;  this work addresses this challenge. The family of Hamiltonians in question is the disordered TFIM,
\begin{equation} \label{eq:1}
\hat {H}=-\sum _{j=1}^{n} J_i \sigma _{j}^{z}\sigma _{j+1}^{z}-h\sum _{j=1}^{n}\sigma _{j}^{x} 
\end{equation}
where the disorder is encoded in the interaction term $J_i \in \pm 1$. The external field is kept constant at $h = 1$; This setup introduces quantum fluctuations, and avoids trivial ferromagnetic or paramagnetic limits. 


VQE.ipynb contains the functions for producing the figures in the report. These are roughly in order as seen in the report.
vqeSetup.py contains everything to do with the circuit, Hamiltonian, and cost functions
optimise.py contains only optLoopAdamW
Test.py is collecting data and producing graphs for report
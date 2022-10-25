# Topological Slepians
This repo contains the code used for implementing the numerical results in the paper: 

**"Topological Slepians: maximally localized representations of signals over simplicial complexes"**

*C. Battiloro, P. Di Lorenzo, S. Barbarossa*

DIET Department, Sapienza University of Rome, Rome, Italy 


<p align="center">
	<img src="https://github.com/clabat9/Topological-Slepians/blob/main/slep-num-2-cropped1024_1.jpg?raw=true" alt="drawing" width="400"/>
</p>

## Abstract
This paper introduces topological Slepians, i.e., a novel class of signals defined over topological spaces (e.g., simplicial complexes) that are maximally concentrated on the topological domain (e.g., over a set of nodes, edges, triangles, etc.) and perfectly localized on the dual domain (e.g., a set of frequencies). These signals are obtained as the principal eigenvectors of a matrix built from proper localization operators acting over topology and frequency domains. Then, we suggest a principled procedure to build dictionaries of topological Slepians, which theoretically provide non-degenerate frames. Finally, we evaluate the effectiveness of the proposed topological Slepian dictionary in two applications, i.e., sparse signal representation and denoising of edge flows.

## Summary
The code  is ready to run  (with saving directory to be specified on local machines). For any questions, comments or suggestions, please e-mail Claudio Battiloro at claudio.battiloro@uniroma1.it. The results showed in the sparse representation curve are in sparsity.csv (that it the output of `simplicial_slepians_experiments.py`. The NMSE vs SNS curve of the denoising task can be obtained via the `denoising_curve_script_vs_SNR.m` that takes as inputs the denoising results (i.e. the outputs of `simplicial_slepians_experiments_denois.py`). 
Thanks to Mitch Roddenberry (Rice ECE) for sharing the code of his Hodgelets paper; this code is built on top of it.


## Important files description

1. __`simplicial_slepians_experiments.py`__: This python script computes the results of the representation task.

2. __`simplicial_slepians_experiments_denois.py`__: This python script computes the results of the denoising task.
  
3. __`lib.py`__: 
	This python script contains the implementation of Topological Slepians and competitors.



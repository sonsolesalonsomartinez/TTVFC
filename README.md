# Targeted Time-Varying FC (TTVFC)

TTVFC is a novel approach for characterising the temporal dynamics of specific connections in neuroimaging data.

This model uses a variant of the HMM (i.e., regression-based HMM) and is implemented within the Python toolbox introduced in [Vidaurre et al. (2025)](https://direct.mit.edu/imag/article/doi/10.1162/imag_a_00460/127499/The-Gaussian-Linear-Hidden-Markov-model-a-Python). Th [GLHMM toolbox](https://github.com/vidaurre/glhmm/tree/main) offers extensive functionality for data preprocessing, model estimation, and result visualization. Detailed documentation and usage instructions are available in the [GLHMM Documentation](https://glhmm.readthedocs.io/en/latest/index.html)

This repository provides code to assess the performance of the regression-based HMM using synthetic data. The simulations were designed such that the task-relevant information was contained in specific connections only. We corroborate that in this case the targeted-TVFC better captures the task-relevant information than the conventional all-pairs approach (covariance-based HMM), which considers the joint statistical properties across all regions indistinctly (i.e. including variances and all cross-covariances). 

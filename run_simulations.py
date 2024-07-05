# -*- coding: utf-8 -*-
# Author: Sonsoles Alonso <sonsoles.alonso@cfin.au.dk>

import numpy as np
import pickle
from glhmm import glhmm, auxiliary
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def generate_simulated_sequence(nTime, nTrial, K):
    """
    Generates simulated state timecourses.

    Parameters:
    - nTime (int): Number of time points in each trial.
    - nTrial (int): Number of trials.
    - K (int): Number of states.

    Returns:
    - gamma_gm (ndarray): Gamma matrix representing state sequences.
    """
        
    # Calculate state changes
    state_changes = np.arange(nTime//K, nTime, nTime//K)
    simulated_sequence = []  # List to store state sequences for each trial
    
    # Generate random state sequences for each trial
    for _ in range(nTrial):
         # Initialize state sequence for current trial
        state_sequence = np.zeros(nTime, dtype=int) 
        for i, change_time in enumerate(state_changes):
            # Calculate random change within range of +/- 4
            random_change = np.random.randint(-4, 5)
            new_change_time = change_time + random_change
            # Ensure change does not exceed time limits
            new_change_time = np.clip(new_change_time, 0, nTime - 1)
            # Assign new state change to sequence
            state_sequence[new_change_time:] = i + 1  # States start from 1
        # Add state sequence of current trial to the list
        simulated_sequence.append(state_sequence)
    
    # Convert list to numpy array and reshape for gamma_gm matrix
    simulated_sequence = np.array(simulated_sequence).reshape(-1, nTime * nTrial)
    
    # Initialize gamma_gm matrix
    gamma_gm = np.zeros((nTime * nTrial, K), dtype=int)
    
    # Populate gamma_gm matrix based on state sequence
    for k in range(K):
        idx = np.where(simulated_sequence == k)[1]
        gamma_gm[idx, k] = 1
    
    return gamma_gm

def sample_covariance_matrix(dim, samples):
    """
    Generates sample covariance matrices.

    Parameters:
    - dim (int): Dimension of the covariance matrices.
    - samples (int): Number of covariance matrices to generate.

    Returns:
    - ndarray: Array of sample covariance matrices.
    """

    covariance_matrices = []
    for _ in range(samples):
        # Generate random matrix
        random_matrix = np.random.randn(dim, dim)

        # Compute covariance matrix
        covariance_matrix = np.dot(random_matrix, random_matrix.T)
        
        # Regularize covariance matrix
        covariance_matrix += np.eye(dim) * 0.01  # Regularization parameter
        
        covariance_matrices.append(covariance_matrix)

    return np.array(covariance_matrices)

def plot_simulation_results(filename):
    """
    Plots simulation results from a saved file and saves the figure as a PNG.

    Parameters:
    - filename (str): Path to the file containing simulation results.

    """
        
    import matplotlib.pyplot as plt
    import numpy as np
    import pickle

    # Load results
    with open(filename, "rb") as f:
            corresponding_values, acc_fc, acc_ct, tp_k_cond_indices = pickle.load(f)
            
    # Convert lists to numpy arrays for easier manipulation
    acc_fc = np.array(acc_fc)
    acc_ct = np.array(acc_ct)
    corresponding_values = np.array(corresponding_values)

    sigma_range = np.unique([val[0] for val in corresponding_values])
    p_range = np.unique([val[1] for val in corresponding_values])

    labels_sigma_range = [f'{rate:.2f}' for rate in sigma_range]
    labels_p_range = [int(label) for label in p_range]

    # Compute average accuracy over this period
    m_acc_fc = np.mean(acc_fc[:,tp_k_cond_indices], axis = 1)
    m_acc_ct = np.mean(acc_ct[:,tp_k_cond_indices], axis = 1)

    # Prepare data for matrix representation
    acc_fc_mat = []
    acc_ct_mat = []
    for rate in sigma_range:
        acc_fc_row = []
        acc_ct_row = []
        for p in p_range:
            # Find indices where sr and p match
            indices = np.where((corresponding_values[:, 0] == rate) & 
                            (corresponding_values[:, 1] == p))[0]
            # Compute mean across repetitions for acc_fc and acc_ct
            acc_fc_mean = np.mean(m_acc_fc[indices])
            acc_ct_mean = np.mean(m_acc_ct[indices])
            acc_fc_row.append(acc_fc_mean)
            acc_ct_row.append(acc_ct_mean)
        acc_fc_mat.append(acc_fc_row)
        acc_ct_mat.append(acc_ct_row)
    acc_fc_mat = np.array(acc_fc_mat)
    acc_ct_mat = np.array(acc_ct_mat)



    # Define figure and subplots with adjusted layout 
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, 
                                            figsize=(12, 2.5), 
                                            subplot_kw={'aspect': 'auto'}, 
                                            constrained_layout=True)

    #set limitis
    vmin = 0.5
    vmax = 1

    # Plot example ACC for a specific setting (ax0)
    rate = sigma_range[5]
    p = 5
    indices = np.where((corresponding_values[:, 0] == rate) & (corresponding_values[:, 1] == p))[0]
    y_fc = acc_fc[indices, :]
    y_ct = acc_ct[indices, :]
    mY_fc = np.mean(y_fc, axis=0)
    eY_fc = np.std(y_fc, axis=0) / np.sqrt(y_fc.shape[0])
    mY_ct = np.mean(y_ct, axis=0)
    eY_ct = np.std(y_ct, axis=0) / np.sqrt(y_ct.shape[0])

    # Plot shaded error bar for Cov HMM (y_fc) and Reg HMM (y_ct)
    ax0.plot(np.arange(len(mY_ct)), mY_ct, color='orange', label='Reg HMM')
    ax0.fill_between(np.arange(len(mY_ct)), mY_ct - eY_ct, mY_ct + eY_ct, alpha=0.3, color='orange')
    ax0.plot(np.arange(len(mY_fc)), mY_fc, color='blue', label='Cov HMM')
    ax0.fill_between(np.arange(len(mY_fc)), mY_fc - eY_fc, mY_fc + eY_fc, alpha=0.3, color='blue')

    # Shade before tp_start
    ax0.axvspan(0,  tp_k_cond_indices[0], alpha=0.3, color='gray')
    ax0.axvspan( tp_k_cond_indices[-1], len(mY_fc), alpha=0.3, color='gray')

    # Set labels and legend 
    ax0.set_xlabel('Time Index')
    ax0.set_ylabel('Acc (Mean ± SE)')
    ax0.set_title(f'Example ($p$:{p}, $σ$:{np.round(rate,2)})')
    ax0.legend() 

    # Plot heatmap for Reg HMM (ax1)
    im1 = ax1.imshow(acc_ct_mat, vmin=vmin, vmax=vmax)
    ax1.set_xlabel('No. variables in X')
    ax1.set_ylabel('Sigma')
    ax1.set_title('Reg HMM')
    ax1.set_yticks(range(len(sigma_range)), labels_sigma_range)
    ax1.set_xticks(range(0, len(p_range), 2)) 
    ax1.set_xticklabels(labels_p_range[0:len(p_range):2]) 

    # Plot heatmap for Cov HMM (ax2)
    im2 = ax2.imshow(acc_fc_mat, vmin=vmin, vmax=vmax)
    ax2.set_xlabel('No. variables in X')
    ax2.set_title('Cov HMM')
    ax2.set_yticks(range(len(sigma_range)), [])
    cbar2 = fig.colorbar(im2) 
    cbar2.set_label('Acc')
    ax2.set_xticks(range(0, len(p_range), 2)) 
    ax2.set_xticklabels(labels_p_range[0:len(p_range):2]) 

    # Plot heatmap of the difference between acc_ct_mat and acc_fc_mat (ax3)
    diff_mat = acc_ct_mat - acc_fc_mat
    li = np.max(np.maximum(np.abs(diff_mat), np.abs(diff_mat)))
    im3 = ax3.imshow(diff_mat, cmap="RdBu_r", vmin=-li, vmax=li)
    ax3.set_xlabel('No. variables in X')
    ax3.set_title('Difference (Reg - Cov)')
    ax3.set_yticks(range(len(sigma_range)), labels_sigma_range)
    ax3.set_ylabel('Sigma')
    ax3.set_xticks(range(0, len(p_range), 2)) 
    ax3.set_xticklabels(labels_p_range[0:len(p_range):2]) 
    cbar3 = fig.colorbar(im3)
    cbar3.set_label('Acc Diff')   

    plt.show()


# Define the filename for saving and loading simulation results
filename = "acc_results_sim1.pkl"

# Plot the simulation results directly from the saved file
plot_simulation_results(filename)

# Alternatively, users can run their own simulations
# USER: Define simulation parameters
nTrial = 1000  # Number of trials
nTime = 100  # Number of time points
q = 1  # Number of variables in Y
K = 5  # Number of states
k_cond = 2  # Task-dependent state index
sigma_range = np.arange(0.001, 0.2, 0.02)  # Range of sigma values
p_range = np.arange(1, 26, 2)  # Range of variables in X
nrep = 100  # Number of repetitions
dd = 1000  # Dirichlet diagonal

# Calculate trial indices
size  = np.full((nTrial,), nTime)
T = auxiliary.make_indices_from_T(size) 

# Randomly asign trials to conditions
trial_labels = np.zeros((nTrial, 1))
random_trials = np.random.choice(T.shape[0], size=T.shape[0] // 2, replace=False)
trial_labels[random_trials] = 1

# Simulate state time courses for X simulations
hmm = glhmm.glhmm(K=K)
hmm.Pi = np.full((1, K), 1/(K))
P = np.abs(np.random.randn(K, K))
np.fill_diagonal(P, np.diag(P) + 0.5)
P = P / np.sum(P, axis=1)[:, np.newaxis]
hmm.P = P
gamma_x = glhmm.glhmm.sample_Gamma(hmm,size)

# Ground truth State time courses for Y simulations
gamma_gm = generate_simulated_sequence(nTime, nTrial, K)

# Period where k_cond was active
g_gm = gamma_gm[:,k_cond].reshape(nTrial,nTime)
m_g_gm = np.mean(g_gm, axis = 0)
tp_k_cond_indices = np.where(m_g_gm == 1)[0]

# Generate Gammas to initiliaze both the covariance and regression-based models
hmm=[]
hmm = glhmm.glhmm(K=K+2)
hmm.Pi = np.full((1, K+2), 1/(K+2))
P = np.abs(np.random.randn(K+2, K+2))
np.fill_diagonal(P, np.diag(P) + 0.5)
P = P / np.sum(P, axis=1)[:, np.newaxis]
hmm.P = P
gamma_init = glhmm.glhmm.sample_Gamma(hmm,size)

# Initialize empty lists to accumulate values of ACC
acc_ct = []
acc_fc = []
corresponding_values = []

# Run simulations
for rep in range(nrep):
    for p in p_range:
        # Simulate X from HMM with state-dependent covariances
        X = np.zeros((np.sum(size),p)) 

        # Random covariance matrices   
        cov_mat = sample_covariance_matrix(p,K)    

        for k in range(K):
            X += np.random.multivariate_normal(mean=np.zeros(p),cov=cov_mat[k, :, :],size=sum(size)) \
                * np.expand_dims(gamma_x[:,k],axis=1)

        # Simulate Y from HMM with state-dependent regressors
        Ya = np.zeros((len(X),q))
        Yb = np.zeros((len(X),q))

        # Generate random betas
        betas = np.random.uniform(low=-1, high=1, size=(p, q, K))
        betas_a = np.copy(betas)
        betas_b = np.copy(betas)
        betas_a[:,:,k_cond] += np.random.normal(loc=0, scale=0.1, size=[p,q])
        betas_b[:,:,k_cond] += np.random.normal(loc=0, scale=0.1, size=[p,q])

        for k in range(K):
                Ya += (X @ betas_a[:,:,k]) * np.expand_dims(gamma_gm[:,k],axis=1)
                Yb += (X @ betas_b[:,:,k]) * np.expand_dims(gamma_gm[:,k],axis=1)
        
        # Y contains half trials from cond a and half from cond b
        Y0 = np.copy(Ya) 
        
        # Replace selected trials from Yb
        for i in random_trials:
            trial_range = np.arange(T[i, 0], T[i, 1])
            Y0[trial_range] = Yb[trial_range] 

        # Compute the standard deviation of Y
        std_Y = np.std(Y0)

        # Generate random noise from a normal distribution
        normal_noise = np.random.normal(loc=0, scale=1, size=Y0.shape)

        for rate in sigma_range:
            # Scale the noise by the standard deviation of Y multiplied by the factor rate
            scaled_noise = std_Y * rate * normal_noise
            
            # Add the scaled noise to Y
            Y = Y0 + scaled_noise

            # Define Models
            hmm_ct = glhmm.glhmm(model_beta='state',
                                model_mean='no',
                                covtype='shareddiag', 
                                K=K+2, 
                                dirichlet_diag=dd
                                )
            hmm_fc = glhmm.glhmm(model_beta='no', 
                                model_mean='no', 
                                covtype='full', 
                                K=K+2, 
                                dirichlet_diag=dd
                                )
            
            # Train Models
            gamma_ct, _, _ = hmm_ct.train(X=X, Y=Y,indices=T, 
                                            Gamma=gamma_init
                                            )
            gamma_fc, _, _ = hmm_fc.train(X=None, Y=np.concatenate((Y, X), axis=1),
                                            indices=T,
                                            Gamma=gamma_init
                                            )

            # Evaluate models
            g_ct=gamma_ct.reshape(nTrial,nTime,-1)
            g_fc=gamma_fc.reshape(nTrial,nTime,-1)

            yct = []
            yfc = []

            for t in range(nTime):
                # Extract gamma values at the current timepoint
                Xt = g_ct[:, t, :]  # Assuming g_ct has shape (nTime, nTrial, K)
                
                # Train Logistic regression model
                log_reg = LogisticRegression()
                log_reg.fit(Xt, trial_labels.ravel())
                
                # Predict trial conditions for the current timepoint
                predictions = log_reg.predict(Xt)
                
                # Calculate accuracy for the current timepoint
                yct.append(accuracy_score(trial_labels, predictions))
            acc_ct.append(yct)
                        
            for t in range(nTime):
                # Extract gamma values at the current timepoint
                Xt = g_fc[:, t, :]  # Assuming g_ct has shape (nTime, nTrial, K)
                
                # Train Logistic regression model
                log_reg = LogisticRegression()
                log_reg.fit(Xt, trial_labels.ravel())
                
                # Predict trial conditions for the current timepoint
                predictions = log_reg.predict(Xt)
                
                # Calculate accuracy for the current timepoint
                yfc.append(accuracy_score(trial_labels, predictions))
            acc_fc.append(yfc)
            
            # Store corresponding ACC values
            corresponding_values.append((rate, p, rep))

            del hmm_ct
            del hmm_fc

# Save results        
with open(filename, "wb") as f:
        corresponding_values, acc_fc, acc_ct, tp_k_cond_indices = pickle.dump(f)


# PLOT RESULTS
plot_simulation_results(filename)

import numpy as np
from scipy.stats import invgamma, norm
from sklearn.mixture import GaussianMixture

def create_gmm(y_pred, num_samples=3):
    """
    Sample pairs of (mean, variance) and create a Gaussian Mixture Model (GMM).
    """
    gamma, v, alpha, beta = tf.split(y_pred, 4, axis=-1)
    gamma = gamma.numpy()
    v = v.numpy()
    alpha = alpha.numpy()
    beta = beta.numpy()
    
    gaussians = [norm(loc=g, scale=np.sqrt(b / v)) for g, v, b in zip(gamma, v, beta)]
    inv_gammas = [invgamma(a=a, scale=b) for a, b in zip(alpha, beta)]
    
    means = []
    variances = []
    
    for _ in range(num_samples):
        for gaussian, inv_gamma in zip(gaussians, inv_gammas):
            variance = inv_gamma.rvs()
            variances.append(variance)
            mean = gaussian.rvs()
            means.append(mean)
    
    gmm = GaussianMixture(n_components=num_samples)
    gmm.means_ = np.array(means).reshape(-1, 1)
    gmm.covariances_ = np.array(variances).reshape(-1, 1, 1)
    gmm.weights_ = np.ones(num_samples) / num_samples
    gmm.precisions_cholesky_ = np.array([np.linalg.cholesky(np.linalg.inv(cov)) for cov in gmm.covariances_]) # For efficient computation
    
    return gmm

import numpy as np
from scipy.stats import norm, invgamma
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

class DistributionallyRobustCVaR:
    def __init__(self, gmm):
        self.gmm = gmm

    def calculate_var(self, mu, sigma, alpha=0.95):
        """
        Calculate Value at Risk (VaR) for a normal distribution.
        """
        var = mu + sigma * norm.ppf(alpha)
        return var

    def calculate_cvar(self, mu, sigma, alpha=0.95):
        """
        Calculate Conditional Value at Risk (CVaR) for a normal distribution.
        """
        var = self.calculate_var(mu, sigma, alpha)
        cvar = mu + sigma * (norm.pdf(norm.ppf(alpha)) / (1 - alpha))
        return cvar

    def compute_infimum_cvar(self, alpha=0.95):
        """
        Compute the infimum of CVaR values from the GMM components.
        """
        cvar_values = []
        for mean, cov in zip(self.gmm.means_, self.gmm.covariances_):
            mu = mean[0]
            sigma = np.sqrt(cov[0, 0])
            cvar = self.calculate_cvar(mu, sigma, alpha)
            cvar_values.append(cvar)
        infimum_cvar = np.min(cvar_values)
        return infimum_cvar

    def is_within_boundary(self, boundary, alpha=0.95):
        """
        Check if the infimum CVaR is within the specified boundary.
        """
        infimum_cvar = self.compute_infimum_cvar(alpha)
        return infimum_cvar <= boundary

# Example usage
if __name__ == "__main__":
    # Assume gmm is already created with create_gmm function from previous code
    gmm = ...  # your GMM object here

    cvar_filter = DistributionallyRobustCVaR(gmm)
    boundary = 10  # example boundary
    infimum_cvar = cvar_filter.compute_infimum_cvar(alpha=0.95)
    within_boundary = cvar_filter.is_within_boundary(boundary, alpha=0.95)

    print(f"Infimum CVaR: {infimum_cvar}")
    print(f"Within Boundary: {within_boundary}")

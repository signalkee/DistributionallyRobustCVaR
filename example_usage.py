import numpy as np
from distributionally_robust_cvar import DistributionallyRobustCVaR
from gmm_creation import create_gmm

# Example data, replace with actual y_pred from your model
y_pred = np.random.rand(1, 4)

# Create GMM
gmm = create_gmm(y_pred, num_samples=3)

# Initialize CVaR filter
cvar_filter = DistributionallyRobustCVaR(gmm)

# Define boundary
boundary = 10

# Compute infimum CVaR
infimum_cvar = cvar_filter.compute_infimum_cvar(alpha=0.95)

# Check if within boundary
within_boundary = cvar_filter.is_within_boundary(boundary, alpha=0.95)

print(f"Infimum CVaR: {infimum_cvar}")
print(f"Within Boundary: {within_boundary}")

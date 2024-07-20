# Distributionally Robust CVaR Filter

This repository contains an implementation of a Distributionally Robust Conditional Value at Risk (CVaR) filter using Gaussian Mixture Models (GMM). The goal is to evaluate and manage risk in uncertain environments.

## Features
- Calculate Value at Risk (VaR) and Conditional Value at Risk (CVaR) for each Gaussian component of the GMM.
- Compute the Distributionally Robust CVaR from the GMM components.
- Check if the Distributionally Robust CVaR is within a specified boundary.

## Equations

### Notations
- $x$: Random variable representing the system state.
- $P^*$: True underlying distribution of $x$.
- $P$: Ambiguity set containing possible distributions of $x$.
- $\mu$: Mean of the normal distribution.
- $\sigma$: Standard deviation of the normal distribution.
- $\alpha$: Confidence level.
- $l(x)$: Loss function, representing any general loss or cost associated with $x$.
- $\gamma$: Threshold value for the VaR.


### Value at Risk (VaR)
The Value at Risk (VaR) at a confidence level $\alpha$ for a normal distribution with mean $\mu$ and standard deviation $\sigma$ is given by:

$$ \text{VaR}_{\alpha}(l(x)) := \inf\{\gamma \in \mathbb{R} \mid \text{Prob}(l(x) > \gamma) \leq \alpha\} $$

where $l(x)$ is the loss function.


### Conditional Value at Risk (CVaR)
The Conditional Value at Risk (CVaR) at a confidence level $\alpha$ is the expected loss given that the loss is beyond the VaR threshold. For a normal distribution, it is calculated as:

```math
\text{CVaR}_{\alpha}(l(x)) := \inf_{\beta \in \mathbb{R}}\left\{\beta + \frac{1}{\alpha} \mathbb{E}\left[(l(x) - \beta)^{+}\right]\right\}
```

where $(\cdot)^{+}$ denotes the positive part.


### Distributionally Robust CVaR (DR-CVaR)
The Distributionally Robust CVaR from the GMM components is computed over an ambiguity set $\mathcal{P}$ of distributions. It is defined as:

```math
\sup_{P \in \mathcal{P}} \text{CVaR}_{\alpha}^{P}(l(x)) \leq 0 \Rightarrow \inf_{P \in \mathcal{P}} \text{Prob}_{P}(l(x) \leq 0) \geq 1 - \alpha
```

where $\mathcal{P}$ is the ambiguity set containing all possible distributions that are consistent with the known first and second-order moments of the data.



## Installation

To install the required packages, run:
```bash
pip install -r requirements.txt
```


## Example plot

### GMM Plot
![GMM Plot](images/gmm_plot.png)

### GMM with DR-CVaR Boundaries
![GMM with CVaR Boundaries](images/gmm_cvar_plot.png)


## Running the Example
To generate the plots and see the Distributionally Robust CVaR in action, run:
```bash
python example_usage.py
```
This script will:

1. Create a Gaussian Mixture Model (GMM) with predefined Gaussian and Inverse-Gamma distributions.
Plot the GMM.
2. Initialize the Distributionally Robust CVaR filter.
3. Compute and print the Distributionally Robust CVaR.
4. Check if the CVaR is within a specified boundary.
5. Plot the GMM with individual components and CVaR boundaries.







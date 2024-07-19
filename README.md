# Distributionally Robust CVaR Filter

This repository contains an implementation of a Distributionally Robust Conditional Value at Risk (CVaR) filter using Gaussian Mixture Models (GMM). The goal is to evaluate and manage risk in uncertain environments.

## Features
- Calculate Value at Risk (VaR) and Conditional Value at Risk (CVaR) for each Gaussian component of the GMM.
- Compute the Distributionally Robust CVaR from the GMM components.
- Check if the Distributionally Robust CVaR is within a specified boundary.

## Equations

### Value at Risk (VaR)
The Value at Risk at a confidence level \( \alpha \) for a normal distribution with mean \( \mu \) and standard deviation \( \sigma \) is given by:

\[ \text{VaR}_{\alpha} = \mu + \sigma \cdot \Phi^{-1}(\alpha) \]

where \( \Phi^{-1} \) is the inverse cumulative distribution function (quantile function) of the standard normal distribution.

### Conditional Value at Risk (CVaR)
The Conditional Value at Risk at a confidence level \( \alpha \) is the expected loss given that the loss is beyond the VaR threshold. For a normal distribution, it is calculated as:

\[ \text{CVaR}_{\alpha} = \mu + \sigma \cdot \frac{\phi(\Phi^{-1}(\alpha))}{1 - \alpha} \]

where \( \phi \) is the probability density function of the standard normal distribution and \( \Phi^{-1} \) is as defined above.

### Distributionally Robust CVaR
The Distributionally Robust CVaR from the GMM components is the greatest lower bound of the calculated CVaR values for each component:

\[ \inf_{\alpha} \text{CVaR} = \min \left\{ \text{CVaR}_{\alpha,1}, \text{CVaR}_{\alpha,2}, \ldots, \text{CVaR}_{\alpha,N} \right\} \]

where \( N \) is the number of components in the GMM.

## Installation

To install the required packages, run:
```bash
pip install -r requirements.txt

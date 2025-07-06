import math
from datetime import date

try:
    from scipy.stats import norm
except ImportError:
    raise ImportError("scipy is required for greece calculations: pip install scipy")


def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Helper: computes d1 in Black-Scholes formulas."""
    return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))


def _d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Helper: computes d2 in Black-Scholes formulas."""
    return _d1(S, K, T, r, sigma) - sigma * math.sqrt(T)


def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes price for a European call option."""
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes price for a European put option."""
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def delta(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    """Option Delta: sensitivity of price to underlying"""
    d1 = _d1(S, K, T, r, sigma)
    if option_type.lower() == "call":
        return float(norm.cdf(d1))
    elif option_type.lower() == "put":
        return float(norm.cdf(d1) - 1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Option Gamma: rate of change of Delta"""
    d1 = _d1(S, K, T, r, sigma)
    return float(norm.pdf(d1) / (S * sigma * math.sqrt(T)))


def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Option Vega: sensitivity of price to volatility (per 1 vol)"""
    d1 = _d1(S, K, T, r, sigma)
    return float(S * norm.pdf(d1) * math.sqrt(T))


def theta(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    """Option Theta: time decay (per year)"""
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    pdf_d1 = norm.pdf(d1)
    term1 = -S * pdf_d1 * sigma / (2 * math.sqrt(T))
    if option_type.lower() == "call":
        term2 = -r * K * math.exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == "put":
        term2 = r * K * math.exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    return float(term1 + term2)


def rho(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    """Option Rho: sensitivity of price to interest rate"""
    d2 = _d2(S, K, T, r, sigma)
    if option_type.lower() == "call":
        return float(K * T * math.exp(-r * T) * norm.cdf(d2))
    elif option_type.lower() == "put":
        return float(-K * T * math.exp(-r * T) * norm.cdf(-d2))
    else:
        raise ValueError("option_type must be 'call' or 'put'")

# Example usage:
if __name__ == "__main__":
    # Parameters: S=100, K=100, T=30 days → 30/252 years, r=0.08, sigma=0.2
    T = 30 / 252
    S, K, r, sigma = 100, 100, 0.08, 0.2
    print("Call Delta:", delta(S, K, T, r, sigma, "call"))
    print("Put Delta:", delta(S, K, T, r, sigma, "put"))
    print("Gamma:", gamma(S, K, T, r, sigma))
    print("Vega:", vega(S, K, T, r, sigma))
    print("Theta call:", theta(S, K, T, r, sigma, "call"))
    print("Theta put:", theta(S, K, T, r, sigma, "put"))
    print("Rho call:", rho(S, K, T, r, sigma, "call"))
    print("Rho put:", rho(S, K, T, r, sigma, "put"))

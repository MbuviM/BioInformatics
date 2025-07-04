import math
from scipy.stats import poisson
def poisson_probability(k, lam):
	"""
	Calculate the probability of observing exactly k events in a fixed interval,
	given the mean rate of events lam, using the Poisson distribution formula.
	:param k: Number of events (non-negative integer)
	:param lam: The average rate (mean) of occurrences in a fixed interval
	"""
	val = poisson.pmf(k, lam)
	return round(val, 5)

prob = poisson_probability(k=3, lam=2.5)
print(f"The probability of observing exactly 3 events when the average rate is 2.5 is: {prob}")

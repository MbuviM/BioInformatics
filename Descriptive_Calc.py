"""
Write a Python function to calculate various descriptive statistics metrics for a given dataset. 
The function should take a list or NumPy array of numerical values 
and return a dictionary containing mean, median, mode, variance, standard deviation, percentiles (25th, 50th, 75th), and interquartile range (IQR).
item() helps to convert numpy scalar to python scalar
tolist() helps to convert numpy array to python list
Round the variance and standard deviation to four decimal places.
"""
import numpy as np 
from scipy import stats
def descriptive_statistics(data):
    # Your code here
    data = np.array(data)

    mean = np.mean(data).item()
    median = np.median(data).item()
    mode = stats.mode(data, keepdims=True).mode[0].item()
    variance = np.round(np.var(data), 4).item()
    std_dev = np.round(np.std(data), 4).item()
    percentiles = np.percentile(data, [25, 50, 75]).tolist()
    iqr = (percentiles[2] - percentiles[0])


    stats_dict = {
        "mean": mean,
        "median": median,
        "mode": mode,
        "variance": variance,
        "standard_deviation": std_dev,
        "25th_percentile": percentiles[0],
        "50th_percentile": percentiles[1],
        "75th_percentile": percentiles[2],
        "interquartile_range": iqr
    }
    return stats_dict

ans = descriptive_statistics([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(ans)

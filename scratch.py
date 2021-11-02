import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt


n = 101
sigma_c = 10
sigma_s = 30

def gauss(n, sigma):
    r = np.arange(-int(n/2), int(n/2)+1)
    return 1 / (sigma * np.sqrt(2*np.pi)) * np.exp(-r**2/(2*sigma**2))


def rectify(x, thresh=0):
    output = x.copy()
    output[output < thresh] = thresh
    return output

xx = np.linspace(-1, 1, n)
y_center = gauss(n, sigma_c)
y_surround = gauss(n, sigma_s)

y_linear = y_center - y_surround


fh, ax = plt.subplots(1, 1, figsize=(6, 4))
# plt.plot(xx, y_center, 'b--', alpha=0.2)
# plt.plot(xx, y_surround, 'r--', alpha=0.2)

# ax.plot(xx, y_linear, 'k--', label='Linear')
ax.plot(xx, rectify(y_linear, thresh=-.007), 'k-', label='Original')

ax.plot(xx, rectify(y_linear*0.5, thresh=-.007), 'r', label='Decrease gain')

ax.plot(xx, rectify(y_linear, thresh=0), 'b', label='Increase threshold')

fh.legend()

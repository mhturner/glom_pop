import numpy as np
import matplotlib.pyplot as plt

stim_time = 20 # sec
mean_intersaccade_time = 0.5  # sec
refractory = 0.25  # sec

saccade_duration = 0.2  # sec
saccade_amplitude = 70 # deg.

inter_saccade_times = np.random.exponential(mean_intersaccade_time, size=4*int(stim_time / mean_intersaccade_time))

# remove inter-saccade times that are faster than the refractory period
inter_saccade_times = inter_saccade_times[inter_saccade_times>refractory]

saccade_times = np.cumsum(inter_saccade_times)
saccade_times = saccade_times[saccade_times<(stim_time-saccade_duration)]
saccade_steps = np.cumsum(np.random.choice([-saccade_amplitude, +saccade_amplitude], size=len(saccade_times)))


timepoints = np.zeros(shape=2*len(saccade_times))
timepoints[0::2] = saccade_times
timepoints[1::2] = saccade_times + saccade_duration
timepoints = np.insert(timepoints, 0, 0)
timepoints = np.append(timepoints, stim_time)

theta = np.zeros_like(timepoints)
theta[1::2] = np.insert(saccade_steps, 0, 0)
theta[2::2] = saccade_steps

fh, ax = plt.subplots(1, 1, figsize=(12,4))
plt.plot(saccade_times, np.zeros_like(saccade_times), 'rx')

timepoints
theta
plt.plot(timepoints, theta, 'k-')

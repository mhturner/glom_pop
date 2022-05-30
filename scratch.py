import numpy as np


np.arange(0, 4, 0.25)

stim_time = 3
saccade_sample_period = 0.25
saccade_times = np.arange(0, stim_time, saccade_sample_period)
saccade_times
image_index = [5, 10, 15]
parameter_list = (image_index, saccade_times)

np.array(np.meshgrid(*parameter_list)).T.reshape(np.prod(list(len(x) for x in parameter_list)), len(parameter_list))

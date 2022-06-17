import os
import matplotlib.pyplot as plt
import numpy as np
from glom_pop import WalkingData
import os
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm, Normalize
from glom_pop import dataio


data_dir = '/Users/mhturner/Dropbox/ClandininLab/Analysis/WalkingTrajectories'
save_directory = dataio.get_config_file()['save_directory']

# From Ryan pre-processed data...
time_res = 0.01  # sec
fn = 'wt_mel_virgin_mated_velocities_for_max_110119.csv'
df = pd.read_csv(os.path.join(data_dir, fn))

trans_vel = df['translational_velocity']
ang_vel = df['angular_velocity']

sns.jointplot(x=trans_vel, y=ang_vel,
              kind='hex', norm=LogNorm(), cmap='viridis')


# %% Histogram of ang. vel, with turn threshold
thresh = 160 # deg/sec
fh, ax = plt.subplots(1, 1, figsize=(2, 1.5))
ax.hist(ang_vel, 1000, density=True, color=[0.5, 0.5, 0.5]);
ax.axvline(x=thresh, color='r', linestyle=':', linewidth=2)
ax.axvline(x=-thresh, color='r', linestyle=':', linewidth=2)
ax.set_yscale('log')
ax.set_xlabel('Angular velocity ($\degree$/s)')
ax.set_ylabel('Probability')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fh.savefig(os.path.join(save_directory, 'walk_stats_ang_hist.svg'), transparent=True)

# %% Get turn times
# turn often immediately followed by quick rebound turn to other direction within 20
# So take +ve and -ve separately

V_orig = ang_vel.values[0:-2]
V_shift = ang_vel.values[1:-1]
ups = np.where(np.logical_and(V_orig < thresh, V_shift >= thresh))[0] + 1
downs = np.where(np.logical_and(V_orig >= -thresh, V_shift < -thresh))[0] + 1
turn_points = np.sort(np.append(ups, downs))
dturn = np.diff(turn_points)


# Snippet of rotational velocity, with threshold crossings
all_time = np.arange(len(ang_vel)) * time_res
pts = np.arange(23000, 30000)

time_vec = np.arange(len(pts)) * time_res
up_times = all_time[ups] - all_time[pts[0]]
up_times = up_times[up_times<time_vec[-1]]
up_times = up_times[up_times>=time_vec[0]]
down_times = all_time[downs] - all_time[pts[0]]
down_times = down_times[down_times<time_vec[-1]]
down_times = down_times[down_times>=time_vec[0]]
fh, ax = plt.subplots(1, 1, figsize=(4, 1.5))
ax.plot(time_vec, ang_vel[pts], 'k-')
ax.plot(up_times, thresh*np.ones_like(up_times), 'rv')
ax.plot(down_times, -thresh*np.ones_like(down_times), 'r^')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Angular velocity ($\degree$/s)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fh.savefig(os.path.join(save_directory, 'walk_stats_ang_snippet.svg'), transparent=True)


# %% Histogram of inter-turn times

from scipy.stats import poisson


inter_turn_interval = np.diff(turn_points) * time_res  # sec
cutoff = 8  # sec
inter_turn_interval = inter_turn_interval[inter_turn_interval <= cutoff]
fh, ax = plt.subplots(1, 1, figsize=(4, 1.5))
bin_width=0.25
ct, bin_edge=np.histogram(inter_turn_interval, bins=np.arange(0, cutoff, bin_width))
bin_ctr = 0.5 * (bin_edge[1:] + bin_edge[:-1])
ct_norm = ct / np.sum(ct)

ax.bar(bin_ctr, ct_norm, color='k', width=bin_width, edgecolor='w')
ax.set_xlabel('Inter-turn interval (sec)')
ax.set_ylabel('Probability')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fh.savefig(os.path.join(save_directory, 'walk_stats_iti.svg'), transparent=True)


# %% Raw data from Avery...
data_dir = '/Users/mhturner/Dropbox/ClandininLab/Analysis/WalkingTrajectories/data_from_avery'
#
# dir_names = ['exp-20190521-183908',
#              'exp-20190613-174032',
#              'exp-20190614-133817',
#              'exp-20190625-161754']

dir_names = ['exp-20190613-174032']
sigma = 0.1 # sec

# %% load all trials

flies = []
for d in dir_names:
    trial_dir = os.listdir(os.path.join(data_dir, d))
    trial_dir = [x for x in trial_dir if 'trial' in x]
    new_trials = [WalkingData.Trial(os.path.join(data_dir, d, x)) for x in trial_dir]
    new_flies = [WalkingData.Fly(x, time_res=time_res, sigma=sigma/time_res) for x in new_trials]
    flies.append(new_flies)

flies = np.hstack(flies)

# %% plot example trajectory
quiver_stride = 40
eg_fly_ind = 2
fly = flies[eg_fly_ind]
fh, ax = plt.subplots(1, 1, figsize=(2, 2))

duration = len(fly.t)
# center traj and scale to cm
x = (fly.x - np.mean(fly.x)) * 100
y = (fly.y - np.mean(fly.y)) * 100
ax.plot(x, y, linewidth=2, color='k', alpha=0.5)
ax.plot(x[0], y[0], 'go')
ax.plot(x[-1], y[-1], 'ro')
ax.quiver(x[::quiver_stride],
          y[::quiver_stride],
          np.cos(np.radians(fly.a[::quiver_stride])),
          np.sin(np.radians(fly.a[::quiver_stride])),
          linewidth=2,
          scale=7, width=0.01, color='k', zorder=10)

ax.set_axis_off()
window_width = 6  # cm
dx = 1  # cm
ax.plot([-2, -2+dx], [-2, -2],
         color='k',
         linewidth=2)
ax.set_xlim([-2.5, -2.5+window_width])
ax.set_ylim([-3, -3+window_width])

fh.savefig(os.path.join(save_directory, 'walk_stats_trajectory.svg'), transparent=True)


# %%
fh, ax = plt.subplots(1, 1, figsize=(6, 2))
ax.set_ylim([-400, 400])
# ax.set_xlim([50, 100])
ax.plot(fly.t[:-1], fly.ang_vel)




# %%

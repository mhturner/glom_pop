import os
import matplotlib.pyplot as plt
import numpy as np
from glom_pop import WalkingData
import os
import pandas as pd
import seaborn as sns
from glom_pop import dataio


save_directory = dataio.get_config_file()['save_directory']
# %% # From Ryan pre-processed data...
data_dir = '/Users/mhturner/Dropbox/ClandininLab/Analysis/WalkingTrajectories'


time_res = 0.01  # sec
fn = 'wt_mel_virgin_mated_velocities_for_max_110119.csv'
df = pd.read_csv(os.path.join(data_dir, fn))

trans_vel = df['translational_velocity']
ang_vel = df['angular_velocity']



# %% Histogram of ang. vel, with turn threshold
thresh = 160 # deg/sec
fh, ax = plt.subplots(1, 1, figsize=(2, 1.5))
ct, bin_edge = np.histogram(ang_vel, bins=1000)
ct_norm = ct / np.sum(ct)
bin_ctr = 0.5 * (bin_edge[1:] + bin_edge[:-1])
bin_width = np.diff(bin_ctr)[0]
ax.bar(bin_ctr, ct_norm, width=bin_width, color=[0.5, 0.5, 0.5])
ax.axvline(x=thresh, color='r', linestyle=':', linewidth=2)
ax.axvline(x=-thresh, color='r', linestyle=':', linewidth=2)
# ax.set_yscale('log')
ax.set_ylim([0, 0.002])
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


# %%
def getTurnPoints(angular_velocity, thresh):
    V_orig = angular_velocity[0:-2]
    V_shift = angular_velocity[1:-1]
    ups = np.where(np.logical_and(V_orig < thresh, V_shift >= thresh))[0] + 1
    downs = np.where(np.logical_and(V_orig >= -thresh, V_shift < -thresh))[0] + 1
    turn_points = np.sort(np.append(ups, downs))
    return turn_points, ups, downs


# %% Raw data from Avery...
data_dir = '/Users/mhturner/Dropbox/ClandininLab/Analysis/WalkingTrajectories/data_from_avery'

dir_names = ['exp-20190521-183908',
             'exp-20190613-174032',
             'exp-20190614-133817',
             'exp-20190625-161754']

time_res = 0.033  # sec
sigma = 0.1
# %% load all trials

flies = []
for d in dir_names:
    trial_dir = os.listdir(os.path.join(data_dir, d))
    trial_dir = [x for x in trial_dir if 'trial' in x]
    new_trials = [WalkingData.Trial(os.path.join(data_dir, d, x)) for x in trial_dir]
    new_flies = [WalkingData.Fly(x, time_res=time_res, sigma=sigma/time_res) for x in new_trials]
    flies.append(new_flies)

flies = np.hstack(flies)
print('Loaded {} flies'.format(len(flies)))

# %% Histogram of ang. vel, with turn threshold

thresh = 100  # deg/sec, 100

trim_thresh = 350
all_ang_vel = np.hstack([fly.ang_vel for fly in flies])
frac_under_cutoff = np.sum(np.abs(all_ang_vel) < trim_thresh) / all_ang_vel.size
print('Included {:.3f} of ang vel'.format(frac_under_cutoff))
all_ang_vel_trimmed = all_ang_vel[np.abs(all_ang_vel) < trim_thresh]

fh, ax = plt.subplots(1, 1, figsize=(2, 1.5))
ct, bin_edge = np.histogram(all_ang_vel_trimmed, bins=np.arange(-trim_thresh, trim_thresh, 10))
ct_norm = ct / np.sum(ct)
bin_ctr = 0.5 * (bin_edge[1:] + bin_edge[:-1])
bin_width = np.diff(bin_ctr)[0]
ax.bar(bin_ctr, ct_norm, width=bin_width, color=[0.5, 0.5, 0.5], edgecolor=[0.5, 0.5, 0.5])
ax.axvline(x=thresh, color='r', linestyle=':', linewidth=2)
ax.axvline(x=-thresh, color='r', linestyle=':', linewidth=2)
ax.set_yscale('log')
ax.set_xlabel('Angular velocity ($\degree$/s)')
ax.set_ylabel('Probability')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Hist of inter-turn interval
turn_points, _, _ = getTurnPoints(all_ang_vel_trimmed, thresh=thresh)
inter_turn_interval = np.diff(turn_points) * time_res  # sec
cutoff = 8  # sec
frac_under_cutoff = np.sum(inter_turn_interval <= cutoff) / len(inter_turn_interval)
inter_turn_interval = inter_turn_interval[inter_turn_interval <= cutoff]
fh1, ax1 = plt.subplots(1, 1, figsize=(4, 1.5))
bin_width = 0.25
ct, bin_edge = np.histogram(inter_turn_interval, bins=np.arange(0, cutoff, bin_width), density=True)
bin_ctr = 0.5 * (bin_edge[1:] + bin_edge[:-1])
ct_norm = ct / np.sum(ct)
ax1.bar(bin_ctr, ct_norm, color='k', width=bin_width, edgecolor='w')
ax1.set_xlabel('Inter-turn interval (sec)')
ax1.set_ylabel('Probability')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

print('Mean ITI = {:.2f}'.format(np.mean(inter_turn_interval)))
fh.savefig(os.path.join(save_directory, 'walk_stats_ang_hist.svg'), transparent=True)
fh1.savefig(os.path.join(save_directory, 'walk_stats_iti.svg'), transparent=True)

# %% plot example trajectory and angular velocity trace

quiver_stride = 100
eg_fly_ind = 3  # 3
fly = flies[eg_fly_ind]
fh, ax = plt.subplots(1, 1, figsize=(2, 2))
duration = fly.t[-1]  # sec
print('Duration = {:.1f} sec'.format(duration))

# center traj and scale to cm
x = (fly.x - np.mean(fly.x)) * 100
y = (fly.y - np.mean(fly.y)) * 100
ax.plot(x, y, linewidth=2, color='k', linestyle='-')
ax.plot(x[0], y[0], 'go')
ax.quiver(x[::quiver_stride],
          y[::quiver_stride],
          np.cos(np.radians(fly.a[::quiver_stride])),
          np.sin(np.radians(fly.a[::quiver_stride])),
          linewidth=2,
          scale=10, width=0.01, color='r', zorder=10)

ax.set_axis_off()
window_width = 15  # cm
dx = 2  # cm
ax.plot([-5, -5+dx], [4, 4],
        color='k',
        linewidth=2)
ax.set_xlim([-7, -7+window_width])
ax.set_ylim([-9, -9+window_width])

turn_points, ups, downs = getTurnPoints(fly.ang_vel, thresh=thresh)
fh1, ax1 = plt.subplots(1, 1, figsize=(4, 1.5))
ax1.set_xlim([0, 20])
ax1.plot(fly.t[:-1], fly.ang_vel, 'k-')
ax1.plot(fly.t[ups], thresh*np.ones_like(ups), 'rv')
ax1.plot(fly.t[downs], -thresh*np.ones_like(downs), 'r^')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Angular velocity ($\degree$/s)')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

fh.savefig(os.path.join(save_directory, 'walk_stats_trajectory.svg'), transparent=True)
fh1.savefig(os.path.join(save_directory, 'walk_stats_ang_snippet.svg'), transparent=True)

# %% ITI vs threshold
trim_thresh = 350 # deg
cutoff = 8  # sec
thresholds = np.arange(25, 200, 25)
iti = []
for thresh in thresholds:

    all_ang_vel = np.hstack([fly.ang_vel for fly in flies])
    all_ang_vel_trimmed = all_ang_vel[np.abs(all_ang_vel) < trim_thresh]


    turn_points, _, _ = getTurnPoints(all_ang_vel_trimmed, thresh=thresh)
    inter_turn_interval = np.diff(turn_points) * time_res  # sec
    inter_turn_interval = inter_turn_interval[inter_turn_interval <= cutoff]
    iti.append(np.median(inter_turn_interval))

fh, ax = plt.subplots(1, 1, figsize=(2, 1))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.plot(thresholds, iti, 'ko')
# twax = ax.twinx()
ct, bin_edge = np.histogram(np.abs(all_ang_vel_trimmed), bins=np.arange(0, trim_thresh, 10))
ct_norm = ct / np.sum(ct)
ct_int = np.cumsum(ct_norm)
bin_ctr = 0.5 * (bin_edge[1:] + bin_edge[:-1])
# twax.plot(bin_ctr, ct_int, 'k-')

ax.set_xlabel(r'Saccade $\theta$ ($\degree$/s)')
ax.set_ylabel('Median ITI (sec)')
ax.set_ylim([0, 2.5])
iti
fh.savefig(os.path.join(save_directory, 'walk_stats_iti_vs_thresh.svg'), transparent=True)



# %% Get turn times and inter-turn interval


# %%
# # Snippet of rotational velocity, with threshold crossings
# all_time = np.arange(len(ang_vel)) * time_res
# pts = np.arange(23000, 30000)
#
# time_vec = np.arange(len(pts)) * time_res
# up_times = all_time[ups] - all_time[pts[0]]
# up_times = up_times[up_times < time_vec[-1]]
# up_times = up_times[up_times >= time_vec[0]]
# down_times = all_time[downs] - all_time[pts[0]]
# down_times = down_times[down_times < time_vec[-1]]
# down_times = down_times[down_times >= time_vec[0]]
# fh, ax = plt.subplots(1, 1, figsize=(4, 1.5))
# ax.plot(time_vec, ang_vel[pts], 'k-')
# ax.plot(up_times, thresh*np.ones_like(up_times), 'rv')
# ax.plot(down_times, -thresh*np.ones_like(down_times), 'r^')
# ax.set_xlabel('Time (s)')
# ax.set_ylabel('Angular velocity ($\degree$/s)')
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
#
# fh.savefig(os.path.join(save_directory, 'walk_stats_ang_snippet.svg'), transparent=True)





# %%

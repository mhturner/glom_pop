import os
import matplotlib.pyplot as plt
import numpy as np
from glom_pop import WalkingData
import pandas as pd
from glom_pop import dataio
from matplotlib.colors import LogNorm


save_directory = dataio.get_config_file()['save_directory']


def getTurnPoints(angular_velocity, thresh):
    # turn often immediately followed by quick rebound turn to other direction
    # So take +ve and -ve separately
    V_orig = angular_velocity[0:-2]
    V_shift = angular_velocity[1:-1]
    ups = np.where(np.logical_and(V_orig < thresh, V_shift >= thresh))[0] + 1
    downs = np.where(np.logical_and(V_orig >= -thresh, V_shift < -thresh))[0] + 1
    turn_points = np.sort(np.append(ups, downs))
    return turn_points, ups, downs
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

# %% Histogram of fwd vel
fh, ax = plt.subplots(1, 1, figsize=(2, 1.5))
ct, bin_edge = np.histogram(trans_vel, bins=1000)
ct_norm = ct / np.sum(ct)
bin_ctr = 0.5 * (bin_edge[1:] + bin_edge[:-1])
bin_width = np.diff(bin_ctr)[0]
ax.bar(bin_ctr, ct_norm, width=bin_width, color=[0.5, 0.5, 0.5])
ax.set_ylim([0, 0.003])
ax.set_xlabel('Forward velocity (cm/s)')
ax.set_ylabel('Probability')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fh.savefig(os.path.join(save_directory, 'walk_stats_fwd_hist.svg'), transparent=True)

# %%
# Get turn times
turn_points, ups, downs = getTurnPoints(ang_vel.values, thresh=160)

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
fh, ax = plt.subplots(2, 1, figsize=(4, 3))
ax[0].plot(time_vec, ang_vel[pts], 'k-')
ax[0].plot(up_times, thresh*np.ones_like(up_times), 'rv')
ax[0].plot(down_times, -thresh*np.ones_like(down_times), 'r^')
ax[0].set_ylabel('Rotational \nvelocity ($\degree$/s)')

ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)

ax[1].plot(time_vec, trans_vel[pts], 'k-')
# ax.plot(up_times, thresh*np.ones_like(up_times), 'rv')
# ax.plot(down_times, -thresh*np.ones_like(down_times), 'r^')
ax[1].set_xlabel('Time (s)')
ax[1].set_ylabel('Forward \nvelocity (cm/sec)')
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)


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

# %% ITI vs threshold
trim_thresh = 350 # deg
cutoff = 8  # sec
thresholds = np.arange(25, 300, 25)
iti = []
for thresh in thresholds:

    all_ang_vel = ang_vel.copy().values
    all_ang_vel_trimmed = all_ang_vel[np.abs(all_ang_vel) < trim_thresh]


    turn_points, _, _ = getTurnPoints(all_ang_vel_trimmed, thresh=thresh)
    print('thresh, = {} / num saccades = {}'.format(thresh, len(turn_points)))
    inter_turn_interval = np.diff(turn_points) * time_res  # sec
    inter_turn_interval = inter_turn_interval[inter_turn_interval <= cutoff]
    iti.append(np.median(inter_turn_interval))

fh, ax = plt.subplots(1, 1, figsize=(2, 1))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.plot(thresholds, iti, 'ko')

ax.set_xlabel(r'Saccade $\theta$ ($\degree$/s)')
ax.set_ylabel('Median ITI (sec)')
ax.set_ylim([0, 2.5])

fh.savefig(os.path.join(save_directory, 'walk_stats_iti_vs_thresh.svg'), transparent=True)

# %% Histogram of ang and fwd vels in saccade and inter-saccade times

saccade_window = 0.4 # sec
saccade_window_radius = int((saccade_window / time_res) / 2)  # points

saccade_snippets_ang = []
saccade_snippets_fwd = []

intersaccade_ang = []
intersaccade_fwd = []
for ti, tp in enumerate(turn_points[:-1]):
    new_snippet_ang = ang_vel[tp-saccade_window_radius:tp+saccade_window_radius]
    new_snippet_fwd = trans_vel[tp-saccade_window_radius:tp+saccade_window_radius]
    if len(new_snippet_ang) == (2*saccade_window_radius):
        saccade_snippets_ang.append(new_snippet_ang.values)
        saccade_snippets_fwd.append(new_snippet_fwd.values)

    if (turn_points[ti+1] - tp) > (saccade_window_radius*2):
        new_intersaccade_ang = ang_vel[tp+saccade_window_radius:turn_points[ti+1]-saccade_window_radius]
        new_intersaccade_fwd = trans_vel[tp+saccade_window_radius:turn_points[ti+1]-saccade_window_radius]
        intersaccade_ang.append(new_intersaccade_ang.values)
        intersaccade_fwd.append(new_intersaccade_fwd.values)



saccade_snippets_ang = np.vstack(saccade_snippets_ang)
saccade_snippets_fwd = np.vstack(saccade_snippets_fwd)

mean_saccade_angle = np.abs(np.mean(saccade_snippets_ang, axis=1))
mean_saccade_fwd = np.mean(saccade_snippets_fwd, axis=1)

mean_intersacccade_angle = np.abs([np.mean(x) for x in intersaccade_ang])
mean_intersacccade_fwd = [np.mean(x) for x in intersaccade_fwd]

fh, ax = plt.subplots(2, 1, figsize=(3.5, 4.5), tight_layout=True)
[x.spines['top'].set_visible(False) for x in ax]
[x.spines['right'].set_visible(False) for x in ax]

# Ang velocity histograms
bins = np.linspace(0, 360, 20)
ct, _ = np.histogram(mean_intersacccade_angle, bins=bins, density=True)
bin_ctr = 0.5 * (bins[1:] + bins[:-1])
bin_width=np.diff(bin_ctr)[0]
ct_norm = ct / np.sum(ct)
ax[0].bar(bin_ctr, ct_norm, color='k', width=bin_width, edgecolor='w', alpha=0.5, label='Intersaccade')

ct, _ = np.histogram(mean_saccade_angle, bins=bins, density=True)
bin_ctr = 0.5 * (bins[1:] + bins[:-1])
bin_width=np.diff(bin_ctr)[0]
ct_norm = ct / np.sum(ct)
ax[0].bar(bin_ctr, ct_norm, color='r', width=bin_width, edgecolor='w', alpha=0.5, label='Saccade')
ax[0].set_xlabel('Angular speed ($\degree$/sec)')

# Fwd velocity histograms
bins = np.linspace(0, 2.1, 20)
ct, _ = np.histogram(mean_intersacccade_fwd, bins=bins, density=True)
bin_ctr = 0.5 * (bins[1:] + bins[:-1])
bin_width=np.diff(bin_ctr)[0]
ct_norm = ct / np.sum(ct)
ax[1].bar(bin_ctr, ct_norm, color='k', width=bin_width, edgecolor='w', alpha=0.5)


# ax[1].hist(mean_intersacccade_fwd, bins=bins, color='k', alpha=0.5, density=True)
ct, _ = np.histogram(mean_saccade_fwd, bins=bins, density=True)
bin_ctr = 0.5 * (bins[1:] + bins[:-1])
bin_width=np.diff(bin_ctr)[0]
ct_norm = ct / np.sum(ct)
ax[1].bar(bin_ctr, ct_norm, color='r', width=bin_width, edgecolor='w', alpha=0.5)

# ax[1].hist(mean_saccade_fwd, bins=bins, color='r', alpha=0.5, density=True)
ax[1].set_xlabel('Forward velocity (cm/sec)')
fh.supylabel('Probability')
fh.legend()

fh.savefig(os.path.join(save_directory, 'walk_stats_sac_conditioned_histograms.svg'), transparent=True)
# %%

all_fwd_vel = trans_vel.copy()
all_ang_vel = ang_vel.copy()
all_ang_vel = np.abs(all_ang_vel)


log_norm = LogNorm()
fh2, ax2 = plt.subplots(1, 1, figsize=(2.5, 2.25))
hb = ax2.hexbin(all_fwd_vel, all_ang_vel, gridsize=35, cmap="Reds", norm=log_norm)
cb = fh2.colorbar(hb, ax=ax2, label='Count')
ax2.set_xlabel('Forward velocity (cm/sec)')
ax2.set_ylabel('Angular speed ($\degree$/sec)')

fh2.savefig(os.path.join(save_directory, 'walk_stats_heatmap.svg'), transparent=True)



# %% Eg trajectory trace
data_dir = '/Users/mhturner/Dropbox/ClandininLab/Analysis/WalkingTrajectories/data_from_avery'
dir_name = 'exp-20190521-183908'
# trial_dir = 'trial-5-20190521-184532'
trial_dir = 'trial-7-20190521-184834'
time_res = 0.033  # sec
sigma = 0.025
quiver_stride = 100

trial = WalkingData.Trial(os.path.join(data_dir, dir_name, trial_dir))
fly = WalkingData.Fly(trial, time_res=time_res, sigma=sigma/time_res)

fh, ax = plt.subplots(1, 1, figsize=(2, 2))
duration = fly.t[-1]  # sec
print('Total Duration = {:.1f} sec'.format(duration))

# center traj and scale to cm
x = (fly.x - np.mean(fly.x)) * 100
y = (fly.y - np.mean(fly.y)) * 100

# Trim to subset of trajectory
start_ind = 100
end_ind = 280  # 100-280 about 6 sec
x = x[start_ind:end_ind]
y = y[start_ind:end_ind]
a = fly.a[start_ind:end_ind]
t = fly.t[start_ind:end_ind]
ang_vel = fly.ang_vel[start_ind:end_ind]
vel = fly.vel[start_ind:end_ind]

duration = t[-1] - t[0]  # sec
print('Total Duration = {:.1f} sec'.format(duration))

thresh = 100
turn_points, ups, downs = getTurnPoints(ang_vel, thresh=thresh)

ax.plot(x, y, linewidth=2, color='k', linestyle='-')
ax.plot(x[turn_points],
        y[turn_points],
        'ro')
# ax.quiver(x[turn_points],
#           y[turn_points],
#           np.cos(np.radians(a[turn_points+15])),
#           np.sin(np.radians(a[turn_points+15])),
#           linewidth=2,
#           scale=10, width=0.01, color='r', zorder=10)

ax.set_axis_off()
window_buffer = 0.2  # cm
dx = 0.25  # cm
ax.plot([x.min()-window_buffer, x.min()-window_buffer+dx], [y.min(), y.min()],
        color='k',
        linewidth=2)
ax.set_xlim([x.min()-window_buffer, x.max()+window_buffer])
ax.set_ylim([y.min()-window_buffer, y.max()+window_buffer])

fh.savefig(os.path.join(save_directory, 'walk_stats_eg_trajectory.svg'), transparent=True)

# %%

start_ind = 250
end_ind = 900
x = x[start_ind:end_ind]
y = y[start_ind:end_ind]
a = fly.a[start_ind:end_ind]
t = fly.t[start_ind:end_ind] - fly.t[start_ind]
ang_vel = fly.ang_vel[start_ind:end_ind]
vel = fly.vel[start_ind:end_ind]

thresh = 100
turn_points, ups, downs = getTurnPoints(ang_vel, thresh=thresh)


y_lev = 300
# Snippet of rotational velocity, with threshold crossings
fh, ax = plt.subplots(2, 1, figsize=(4, 3), tight_layout=True)
ax[0].plot(t, ang_vel, 'k-')
ax[0].plot(t[ups], y_lev*np.ones_like(ups), 'rv', alpha=0.5)
ax[0].plot(t[downs], -y_lev*np.ones_like(downs), 'r^', alpha=0.5)
ax[0].set_ylabel('Rotational \nvelocity ($\degree$/s)')

ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)
# ax[0].set_xlim([0, 20])
ax[1].plot(t, vel, 'k-')
# ax.plot(up_times, thresh*np.ones_like(up_times), 'rv')
# ax.plot(down_times, -thresh*np.ones_like(down_times), 'r^')
ax[1].set_xlabel('Time (s)')
ax[1].set_ylabel('Forward \nvelocity (cm/sec)')
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)

fh.savefig(os.path.join(save_directory, 'walk_stats_ang_talk_snippet.svg'), transparent=True)



# %%

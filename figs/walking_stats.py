import os
import matplotlib.pyplot as plt
import numpy as np
from glom_pop import WalkingData
import os
import pandas as pd
import seaborn as sns

data_dir = '/Users/mhturner/Dropbox/ClandininLab/Analysis/WalkingTrajectories/data_from_avery'
output_dir = '/Users/mhturner/Dropbox/ClandininLab/Analysis/WalkingTrajectories/plots'
#
# dir_names = ['exp-20190521-183908',
#              'exp-20190613-174032',
#              'exp-20190614-133817',
#              'exp-20190625-161754']

dir_names = ['exp-20190613-174032']


time_res = 0.01
sigma = 0.02 # sec
minimum_trial_length = 20  # sec

# %%

flies = []
for d in dir_names:
    trial_dir = os.listdir(os.path.join(data_dir, d))
    trial_dir = [x for x in trial_dir if 'trial' in x]
    new_trials = [WalkingData.Trial(os.path.join(data_dir, d, x)) for x in trial_dir]
    new_flies = [WalkingData.Fly(x, time_res=time_res, sigma=sigma/time_res) for x in new_trials if x.trial_length > minimum_trial_length]
    flies.append(new_flies)

flies = np.hstack(flies)
print('{} flies loaded'.format(len(flies)))

# %% eg fly
eg_fly_ind = 12  # 12
fly = flies[eg_fly_ind]
fh, ax = plt.subplots(2, 1, figsize=(6, 4))
# [x.set_xlim([10, 20]) for x in ax.ravel()]
ax[0].plot(fly.t[:-1], fly.vel)
ax[0].set_ylabel('Forward\nvelocity (cm/sec)')
ax[0].set_ylim([0, 3])

ax[1].axhline(y=0, color='k', linewidth=2)
ax[1].plot(fly.t[:-1], fly.ang_vel)
ax[1].set_ylim([-250, 250])
ax[1].set_ylabel('Angular\nvelocity ($\degree$/sec)')
ax[1].set_xlabel('Time (s)')




# %%


# %% From Ryan processed data...
from matplotlib.colors import LogNorm, Normalize
time_res = 0.01
data_dir = '/Users/mhturner/Dropbox/ClandininLab/Analysis/WalkingTrajectories'
fn = 'wt_mel_virgin_mated_velocities_for_max_110119.csv'
df = pd.read_csv(os.path.join(data_dir, fn))

trans_vel = df['translational_velocity']
ang_vel = df['angular_velocity']

sns.jointplot(x=trans_vel, y=ang_vel,
              kind='hex', norm=LogNorm(), cmap='viridis')

# %%

pts = np.arange(1000, 20000)
time_vec = np.arange(len(pts)) * time_res
fh, ax = plt.subplots(2, 1, figsize=(12, 3))
ax[0].plot(time_vec, trans_vel[pts])
ax[1].plot(time_vec, ang_vel[pts])
ax[1].set_xlabel('Time (s)')

# %%
thresh = 160 # deg/sec
fh, ax = plt.subplots(1, 1, figsize=(4, 2))
ax.hist(ang_vel, 1000, density=True, color=[0.5, 0.5, 0.5]);
ax.axvline(x=thresh, color='r', linestyle='--')
ax.axvline(x=-thresh, color='r', linestyle='--')
ax.set_yscale('log')

# %%

V_orig = ang_vel.values[0:-2]
V_shift = ang_vel.values[1:-1]
ups = np.where(np.logical_and(V_orig < thresh, V_shift >= thresh))[0] + 1
downs = np.where(np.logical_and(V_orig >= thresh, V_shift < thresh))[0] + 1
turn_times = np.sort(np.append(ups, downs))


# %%
all_time = np.arange(len(ang_vel)) * time_res

pts = np.arange(20000, 25000)
time_vec = np.arange(len(pts)) * time_res
turn_times = all_time[ups] - all_time[pts[0]]
turn_times = turn_times[turn_times<time_vec[-1]]
turn_times = turn_times[turn_times>=time_vec[0]]

fh, ax = plt.subplots(1, 1, figsize=(6, 3))
ax.plot(turn_times, 300*np.ones_like(turn_times), 'rx')
ax.plot(time_vec, ang_vel[pts])
ax.set_xlabel('Time (s)')


# %%
inter_turn_interval = np.diff(ups) * time_res  # sec
cutoff = 10  # sec
inter_turn_interval = inter_turn_interval[inter_turn_interval <= cutoff]
fh, ax = plt.subplots(1, 1, figsize=(12, 3))
ax.hist(inter_turn_interval, np.arange(0, 10, 0.1), density=True);
ax.set_yscale('log')
ax.set_xlabel('Inter-turn interval (sec)')
ax.set_ylabel('Probability')

# %% load all trials

flies = []
for d in dir_names:
    trial_dir = os.listdir(os.path.join(data_dir, d))
    trial_dir = [x for x in trial_dir if 'trial' in x]
    new_trials = [WalkingData.Trial(os.path.join(data_dir, d, x)) for x in trial_dir]
    new_flies = [WalkingData.Fly(x, time_res=time_res, sigma=sigma/time_res) for x in new_trials]
    flies.append(new_flies)

flies = np.hstack(flies)

# %% plot all trajectories

fh, ax = plt.subplots(12, 7, figsize=(20, 12))
axes = ax.ravel()
[x.set_axis_off() for x in axes];

for ind, fly in enumerate(flies):
    duration = len(fly.t)
    axes[ind].plot(fly.x, fly.y, 'k')
    axes[ind].axis('equal')
    axes[ind].set_title(duration)

# %% plot single fly traj
quiver_stride = 100


for fly in flies:
    # fly_ind = 13
    # fly = flies[fly_ind]
    fh, ax = plt.subplots(1, 1, figsize=(8, 8))

    duration = len(fly.t)
    ax.plot(fly.x, fly.y, 'k')
    ax.plot(fly.x[0], fly.y[0], 'go')
    ax.plot(fly.x[-1], fly.y[-1], 'ro')
    ax.quiver(fly.x[::quiver_stride], fly.y[::quiver_stride], np.cos(np.radians(fly.a[::quiver_stride])), np.sin(np.radians(fly.a[::quiver_stride])))
    ax.axis('equal')
    ax.set_axis_off()

    fn = '{}.png'.format(fly.trialId.split('/')[-1])

    # fh.savefig(os.path.join(output_dir, fn))


# %%
fly = flies[2]
fh, ax = plt.subplots(3, 1, figsize=(8, 8))
duration = len(fly.t)
ax[0].plot(fly.x, fly.y, 'k')
ax[0].plot(fly.x[0], fly.y[0], 'go')
ax[0].plot(fly.x[-1], fly.y[-1], 'ro')
ax[0].quiver(fly.x[::quiver_stride], fly.y[::quiver_stride], np.cos(np.radians(fly.a[::quiver_stride])), np.sin(np.radians(fly.a[::quiver_stride])))

ax[1].quiver(0, 0, np.cos(np.radians(90+0)), np.sin(np.radians(90+0)))

import os
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter1d


class Trial:
    def __init__(self, dirName, do_angle_jump_correct=True):
        self.cam = Cam(os.path.join(dirName, 'cam.txt'), do_angle_jump_correct=do_angle_jump_correct)
        self.cnc = Cnc(os.path.join(dirName, 'cnc.txt'))
        self.dirName = dirName
        self.trial_length = self.cam.tvec[-1] - self.cam.tvec[0]  # sec


class Cam:
    def __init__(self, fname, do_angle_jump_correct=True):
        self.tvec = np.genfromtxt(fname, delimiter=',', skip_header=1, usecols=(0,))
        self.xvec = np.genfromtxt(fname, delimiter=',', skip_header=1, usecols=(1,))
        self.yvec = np.genfromtxt(fname, delimiter=',', skip_header=1, usecols=(2,))
        self.avec = np.rad2deg(np.genfromtxt(fname, delimiter=',', skip_header=1, usecols=(3,)))

        if do_angle_jump_correct:
            self.jumpCorrect(jump_threshold=150)
            self.correctForSwappedHeadAndButt()

    def jumpCorrect(self, jump_threshold=150):
        """
        -Heading tracking data comes from ellipse-fitting using camera input.
        -Frame-by-frame heading estimate can sometimes swap head and butt of fly
        -This correction looks for large, sudden jumps in apparent heading between
        frames and corrects them by applying a 180-degree shift

        :jump_threshold: in degrees
        """
        a_adj = self.avec.copy()
        for ind in range(1, len(a_adj)):
            dtheta = a_adj[ind] - a_adj[ind-1]
            while np.abs(dtheta) > jump_threshold:
                a_adj[ind] = a_adj[ind] - np.sign(dtheta) * 180
                dtheta = a_adj[ind] - a_adj[ind-1]
        self.avec = a_adj

    def correctForSwappedHeadAndButt(self):
        """
        -jumpCorrect above can introduce an artifact where every heading vector
        is pointed opposite to the direction of motion, since the first apparent
        heading is taken to be the baseline "head" direction. If the first heading
        vector actually points towards the tail jumpCorrect() will align
        each subsequent heading vector 180-deg offset from the real head direction
        -This correction ensures that the first heading vector points along
        the direction of motion, if not, it switches each heading vector by 180
        """
        first_vector = [self.xvec[1]-self.xvec[0], self.yvec[1]-self.yvec[0]]
        first_direction = np.arctan(first_vector[1] / first_vector[0])
        first_heading = np.radians(self.avec[0])
        opposite_heading = np.radians(self.avec[0]+180)
        if np.abs(first_direction-opposite_heading) < np.abs(first_direction-first_heading):
            self.avec = np.rad2deg(np.radians(self.avec + 180))


class Cnc:
    def __init__(self, fname):
        self.tvec = np.genfromtxt(fname, delimiter=',', skip_header=1, usecols=(0,))
        self.xvec = np.genfromtxt(fname, delimiter=',', skip_header=1, usecols=(1,))
        self.yvec = np.genfromtxt(fname, delimiter=',', skip_header=1, usecols=(2,))


class Fly:
    def __init__(self, trial, time_res=0.033, sigma=0):
        """
        trial: Trial object, defined above
        time_res: in seconds
        sigma: st-dev for gaussian smoothing of x, y, angle
        """
        self.trialId = trial.dirName
        tmin = max(trial.cam.tvec[0], trial.cnc.tvec[0])
        tmax = min(trial.cam.tvec[-1], trial.cnc.tvec[-1])
        self.t = np.arange(tmin, tmax, time_res)
        cama = interp1d(trial.cam.tvec, trial.cam.avec)
        camx = interp1d(trial.cam.tvec, trial.cam.xvec)
        camy = interp1d(trial.cam.tvec, trial.cam.yvec)

        cncx = interp1d(trial.cnc.tvec, trial.cnc.xvec)
        cncy = interp1d(trial.cnc.tvec, trial.cnc.yvec)

        self.x = camx(self.t) + cncx(self.t)
        self.y = camy(self.t) + cncy(self.t)
        self.a = cama(self.t)

        # coordinate frame offset
        self.a = 180-self.a

        # Time vector start at 0
        self.t = self.t - self.t[0]

        # Trim last 10% of fly, when tracking is lost
        keep_len = np.int(0.9 * len(self.t))
        self.t = self.t[:keep_len]
        self.a = self.a[:keep_len]
        self.x = self.x[:keep_len]
        self.y = self.y[:keep_len]


        if sigma != 0:  # smooth x, y, angle trajectory with gaussian(sigma)
            self.x = gaussian_filter1d(self.x, sigma)
            self.y = gaussian_filter1d(self.y, sigma)
            self.a = gaussian_filter1d(self.a, sigma)


        # Compute instantaneous velocities
        d_xy = np.sqrt(np.diff(self.x)**2 + np.diff(self.y)**2)  # m per time point
        self.vel = 100 * d_xy / time_res  # cm/sec
        self.ang_vel = np.diff(self.a) / time_res  # deg/sec

        # Decompose velocity into forward vs. sideslip

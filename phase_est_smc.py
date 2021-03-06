import numpy as np
from smc_memory import smc_memory
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from scipy.stats import gaussian_kde
#from sklearn.model_selection import LeaveOneOut

class phase_est_smc:

    def __init__(self, omega_star, t0, max_iters, bandwidth=5e-3, heuristic=None):
        self.omega_star = omega_star
        self.t = t0
        self.break_flag = False
        self.counter = 0
        self.data = []
        self.max_iters = max_iters
        self.curr_omega_est = 0 # best current estimate of omega
        self.memory = smc_memory() # memory to track statistics of each run
        self.rng = np.random.default_rng(10)

        if heuristic is None:
            self.heuristic = 'exponential'
        elif heuristic == 'adaptive':
            self.heuristic = 'adaptive'

        self.bandwidth = bandwidth

    def init_particles(self, num_particles):
        """
        Initializes the particles for SMC.

        Args:
            num_particles: number of particles in the SMC algorithm
        """
        self.num_particles = num_particles
        self.particle_pos = self.rng.uniform(-np.pi, np.pi, size=self.num_particles)
        self.particle_wgts = np.ones(num_particles) * 1/num_particles # uniform weight initialization

    def particles(self, num_measurements=1, threshold=None):
        """
        Runs the SMC algorithm for current experiment until the threshold exceeded

        Args:
            num_measurements: number of measurements per update
            threshold: threshold for n_eff. defaults to self.num_particles/10

        Returns:
            array of particle positions and their corresponding weights
        """
        
        if threshold is None:
            threshold = self.num_particles/10

        n_eff = None # init None so it will be calculated on first iteration of while loop

        while n_eff is None or n_eff >= threshold:

            # phi_k = self.rng.uniform() * 2 * np.pi
            phi_k = self.rng.uniform(low=-1, high=1) * np.pi
            # phi_k = 0

            measure_list = []
            for _ in range(num_measurements):
                r = self.rng.uniform()
                if r <= prob_zero(self.omega_star, phi_k, self.t):
                    measure_list.append(0)
                else:
                    measure_list.append(1)

            particle_prob = np.ones(shape=self.num_particles)
            for i in range(num_measurements):
                if measure_list[i] == 0:
                    particle_prob = np.multiply(particle_prob, prob_zero(self.particle_pos, phi_k, self.t))
                else:
                    particle_prob = np.multiply(particle_prob, prob_one(self.particle_pos, phi_k, self.t))

            # bayes update of weights
            self.particle_wgts = np.multiply(self.particle_wgts, particle_prob) # numerator
            norm = np.sum(self.particle_wgts) # denominator
            self.particle_wgts /= norm
            
            # recalculate n_eff
            n_eff = 1/(np.sum(self.particle_wgts**2))

            self.counter += 1

            # if (self.counter+1)%20 == 0:
            #     print(self.counter)

            self.data.append(np.average(self.particle_pos, weights = self.particle_wgts))
 
            if self.counter == self.max_iters:
                # self.curr_omega_est = self.particle_pos[np.argmax(self.particle_wgts)]
                self.curr_omega_est = np.average(self.particle_pos, weights = self.particle_wgts)
                self.break_flag=True
                break

            self.update_t()

        # update memory with statistics before resampling
        self.memory.upd_pos_wgt_bef_res(self.particle_pos, self.particle_wgts)

        return self.particle_pos, self.particle_wgts

    def get_bins(self, num_bins, num_samples=10000):
        """
        Draw samples from current posterior for binning so that we can
        pass into NN for resampling
        """
        data = self.rng.choice(self.particle_pos, size = num_samples, p=self.particle_wgts)

        # if std deviation of posterior is 0, it means the distribution is sharply peaked
        # and will not change anymore. we can exit the algorithm
        # if np.std(data) == 0 or self.break_flag:
        #     self.curr_omega_est = self.particle_pos[np.argmax(self.particle_wgts)]
        #     self.break_flag=True
        #     return None, None
        
        bins, edges = np.histogram(data, num_bins)
        edges = (edges[1:] + edges[:-1]) / 2 # take midpoint of bin edges
        bins = bins/num_samples

        # update memory
        self.memory.upd_bins_edges_bef_res(bins, edges)

        return bins, edges

    def kde_resample(self, num_samples=10000, kernel='gaussian', method=None):

        if method == 1:
            data = self.rng.choice(self.particle_pos, size=num_samples, p=self.particle_wgts)
            data = data[:, np.newaxis]
            #bandwidths = 10 ** np.linspace(-3, -2, 10)
            # grid = GridSearchCV(KernelDensity(kernel='gaussian'),
            #                     {'bandwidth': bandwidths})
            # grid.fit(data)
            # bandwidth = grid.best_params_['bandwidth']
            # print(bandwidth)
            
            kernel = KernelDensity(bandwidth = self.bandwidth, kernel = kernel)
            kernel.fit(data)
            self.particle_pos = kernel.sample(n_samples=self.num_particles).squeeze()
            self.particle_pos += self.rng.normal(scale=self.bandwidth, size=self.particle_pos.shape)
            self.particle_wgts = np.ones(self.num_particles) * 1/self.num_particles

        elif method == 2:
            kernel = gaussian_kde(self.particle_pos, weights=self.particle_wgts)
            self.particle_pos = kernel.resample(size=self.num_particles).squeeze()
            self.particle_wgts = np.ones(self.num_particles) * 1/self.num_particles

        self.memory.upd_kernel(kernel)
        self.memory.upd_pos_wgt_aft_res(self.particle_pos, self.particle_wgts)
    def update_t(self):
        """
        Updates time 
        """

        if self.heuristic == 'exponential':
            self.t = self.t * 9/8
        elif self.heuristic == 'adaptive':
            two_particles = self.rng.choice(self.particle_pos, size=2, replace=False, p=self.particle_wgts)
            self.t = 1 / np.abs(two_particles[0] - two_particles[1])

    def bootstrap_resample(self):
        """
        Simple bootstrap resampler
        """
        self.particle_pos = self.rng.choice(self.particle_pos, size = self.num_particles, p=self.particle_wgts)
        self.particle_wgts = np.ones(self.num_particles) * 1/self.num_particles
        
    def nn_bins_to_particles(self, bins, edges):
        """
        Convert NN bins to particles.

        Args:
            bins:  [1 x n] np array of bins
            edges: [n, ] np array of bin edges
        """
        

        bins = bins[0]
        self.memory.upd_bins_edges_aft_res(bins, edges)
        particle_pos = edges

        ## method 2: sample from binned distribution
        ## but take every particle position to be middle of bin
        # self.particle_pos = self.rng.choice(particle_pos, size=self.num_particles, p=bins)
        # self.particle_wgts = np.ones(self.num_particles) * 1/self.num_particles

        ## every bin defines a gaussian with mean the bin center, and std 1/2 bin width
        ## we sample as many particles in each bin when converting from bins to particles

        # first convert bins from probabilities to number of particles
        # then check if there is a discrepancy with total number of particles SMC is supposed to have
        # if there is then correct for it            
        bins = bins * self.num_particles
        bins = np.rint(bins).astype(int)
        bins_sum = bins.sum()

        if bins_sum != self.num_particles:
            n = abs(self.num_particles - bins_sum) ## difference in number of particles
            choices = self.rng.choice(np.arange(len(bins)), size=n, p=bins/bins_sum)
            for i in choices:
                if bins_sum < self.num_particles:
                    bins[i] += 1
                else:
                    bins[i] -= 1

        particle_pos = []
        edge_width = edges[1] - edges[0]
        std = edge_width/4
        for i in range(len(edges)):
            n_part_from_bin = bins[i]
            pos_from_bin = self.rng.normal(edges[i], std, size=n_part_from_bin).tolist()
            particle_pos.extend(pos_from_bin)

        self.particle_pos = np.array(particle_pos).copy()

        if len(self.particle_pos) != self.num_particles:
            print("Error in conversion from bins to particles. Particle numbers don't match!")
            print("Resample {:d} particles, but supposed to have {:d} particles".format(len(self.particle_pos), self.num_particles))
            exit()

        self.particle_wgts = np.ones(self.num_particles) * 1/self.num_particles
        self.memory.upd_pos_wgt_aft_res(self.particle_pos, self.particle_wgts)

    def liu_west_resample(self, a=0.98):
        
        mu = np.average(self.particle_pos, weights=self.particle_wgts)
        e_x2 = np.average(self.particle_pos**2, weights=self.particle_wgts)
        var = (1-a**2) * ( e_x2 - mu**2 ) # var = E(X^2) - E(X)^2

        if var < 0:
            self.curr_omega_est = np.average(self.particle_pos, weights = self.particle_wgts)
           #self.curr_omega_est = self.particle_pos[np.argmax(self.particle_wgts)]
            self.break_flag = True
            return

        new_particle_pos = self.rng.choice(self.particle_pos, size=self.num_particles, p=self.particle_wgts)
        for i in range(len(new_particle_pos)):
            mu_i = a * new_particle_pos[i] + (1-a) * mu
            new_particle_pos[i] = self.rng.normal(loc=mu_i, scale=np.sqrt(var))  ## scale is standard deviation

        self.particle_pos = np.copy(new_particle_pos)
        self.particle_wgts = np.ones(self.num_particles) * 1/self.num_particles ## set all weights to 1/N again

        self.memory.upd_pos_wgt_aft_res(self.particle_pos, self.particle_wgts)
 
def prob_zero(omega, phi, t):
    return np.cos((omega+phi)*t/2) **2

def prob_one(omega, phi, t):
    return np.sin((omega+phi)*t/2) **2
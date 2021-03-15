import numpy as np

class phase_est_smc:

    def __init__(self, omega_star, t0):
        self.omega_star = omega_star
        self.t = t0
        self.lw_break_flag = False

    def init_particles(self, num_particles):
        """
        Initializes the particles for SMC.

        Args:
            num_particles: number of particles in the SMC algorithm
        """
        self.num_particles = num_particles
        self.particle_pos = np.linspace(0, 2*np.pi, self.num_particles)
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

        counter = 0 
        while n_eff is None or n_eff >= threshold:
            
            # get measurement
            phi_k = np.random.uniform() * 2 * np.pi
            measure_list = []
            for _ in range(num_measurements):
                r = np.random.uniform()
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

            counter += 1
            self.update_t(factor=51/50)

            # if counter % 20 == 0:
            #     print("current iteration {:d}, n_eff = {:f} vs threshold {:f} and t is {:f}".format(counter, n_eff, threshold, self.t))

        return self.particle_pos, self.particle_wgts

    def update_t(self, factor=9/8):
        """
        Updates time by given factor
        """
        self.t = self.t * factor

    def bootstrap_resample(self):
        """
        Simple bootstrap resampler
        """
        self.particle_pos = np.random.choice(self.particle_pos, size = self.num_particles, p=self.particle_wgts)
        self.particle_wgts = np.ones(self.num_particles) * 1/self.num_particles
        
    def nn_resample(self, bins, edges, mean, std):
        """
        Convert NN bins to particles.

        Args:
            bins:  [1 x n] np array of bins
            edges: [n+1] np array of bin edges
            mean: mean of data before resampling
            std: std of data before resampling
        """
        
        bins = bins[0]
        particle_pos = (edges[1:] + edges[:-1]) / 2 # mid point of each bin is the position
        self.particle_pos = particle_pos * std + mean # undo normalization
        self.particle_wgts = bins

    def liu_west_resample(self, a=0.98):
        
        mu = np.average(self.particle_pos, weights=self.particle_wgts)
        e_x2 = np.average(self.particle_pos**2, weights=self.particle_wgts)
        var = np.sqrt(1-a**2) * ( e_x2 - mu**2 ) # var = E(X^2) - E(X)^2

        if var < 0:
            self.lw_break_flag = True
            return

        new_particle_pos = np.random.choice(self.particle_pos, size=self.num_particles, p=self.particle_wgts)
        for i in range(len(new_particle_pos)):
            mu_i = a * new_particle_pos[i] + (1-a) * mu
            new_particle_pos[i] = np.random.normal(loc=mu_i, scale=np.sqrt(var))  ## scale is standard deviation

        self.particle_pos = np.copy(new_particle_pos)
        self.particle_wgts = np.ones(self.num_particles) * 1/self.num_particles ## set all weights to 1/N again
 
def prob_zero(omega, phi, t):
    return (np.cos((omega-phi)*t/2)) **2

def prob_one(omega, phi, t):
    return (np.sin((omega-phi)*t/2)) **2
import numpy as np

class phase_est_smc:

    def __init__(self, omega_star, t0):
        self.omega_star = omega_star
        self.t = t0

    def init_particles(self, num_particles):
        """
        Initializes the particles for SMC.

        Args:
            num_particles: number of particles in the SMC algorithm
        """
        self.num_particles = num_particles
        self.particle_pos = np.linspace(0, 2*np.pi, self.num_particles)
        self.particle_wgts = np.ones(num_particles) * 1/num_particles # uniform weight initialization

    def update(self, threshold=None):
        """
        Runs one time step of the SMC update

        Args:
            threshold: threshold for n_eff. defaults to self.num_particles/10

        Returns:
            array of particle positions and their corresponding weights
        """

        if threshold is None:
            threshold = self.num_particles/10

        phi_k = np.random.uniform() * 2 * np.pi 
        n_eff = 1/(np.sum(self.particle_wgts**2))

        # get measurement
        r = np.random.uniform()
        if r <= prob_zero(self.omega_star, self.t, phi_k):
            measurement = 0
        else:
            measurement = 1

        while n_eff >= threshold:
            
            particle_prob = np.empty(shape=0)
            
            for i in range(self.num_particles):
                
                omega = self.particle_pos[i]
                
                if measurement == 0:
                    particle_prob = np.append(particle_prob, prob_zero(omega, phi_k, self.t))
                else:
                    particle_prob  = np.append(particle_prob, prob_one(omega, phi_k, self.t))
            
            # bayes update of weights
            self.particle_wgts = np.multiply(self.particle_wgts, particle_prob) # numerator
            norm = np.sum(self.particle_wgts) # denominator
            self.particle_wgts /= norm 
            
            # recalculate n_eff
            n_eff = 1/(np.sum(self.particle_wgts**2))
        
        return self.particle_pos, self.particle_wgts

    def update_t(self, factor=9/8):
        self.t = self.t * 9/8

def prob_zero(omega, phi, t):
    return (np.cos((omega-phi)*t/2)) **2

def prob_one(omega, phi, t):
    return (np.sin((omega-phi)*t/2)) **2
import numpy as np

class FPE():

    def __init__(self, k, t):
        self.k_star = k
        self.t = t

    def prob_zero(self, k, M, theta):
        return (1 + np.cos(2 * np.pi * M * k/self.t + theta))/2

    def prob_one(self, k, M, theta):
        return (1 - np.cos(2 * np.pi * M * k/self.t + theta))/2

    def get_measurements(self, s):

        M_array = np.random.randint(self.t, size = s)
        theta_array = np.random.randint(low = 0, high = 2*np.pi, size = s)
        measure_results = []

        for i in range(s):
            M_i = M_array[i]
            theta_i = theta_array[i]
            
            r = np.random.random()
            if r <= self.prob_zero(self.k_star, M_i, theta_i):
                measure_results.append(0)
            else:
                measure_results.append(1)
        
        return measure_results, M_array, theta_array

    def get_log_likelihood(self, measure_results, M_array, theta_array):

        log_prob_array = []

        for k in range(self.t):
            
            log_prob = 0
            
            for i in range(len(measure_results)):
                M_i = M_array[i]
                theta_i = theta_array[i]
                if measure_results[i] == 0:
                    log_prob += np.log(self.prob_zero(k, M_i, theta_i))
                else:
                    log_prob += np.log(self.prob_one(k, M_i, theta_i))
                    
            log_prob_array.append(log_prob)

        return log_prob_array
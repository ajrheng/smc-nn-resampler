class smc_memory:

    def __init__(self):

        # particle positions and weights before resampling
        self.pos_wgt_bef_res = []

        # bins and edges before passing into NN
        self.bins_edges_bef_res = []

        # bins mean and std before passing into NN
        self.mean_std_bef_res = []

        # bins and edges after resampling from NN
        self.bins_edges_aft_res = []

        # particle positions and weights after resampling
        self.pos_wgt_aft_res = []

    def upd_pos_wgt_bef_res(self, pos, wgt):
        self.pos_wgt_bef_res.append((pos, wgt))

    def upd_bins_edges_bef_res(self, bins, edges):
        self.bins_edges_bef_res.append((bins, edges))

    def upd_mean_std_bef_res(self, mean, std):
        self.mean_std_bef_res.append((mean, std))

    def upd_bins_edges_aft_res(self, bins, edges):
        self.bins_edges_aft_res.append((bins, edges))

    def upd_pos_wgt_aft_res(self, pos, wgt):
        self.pos_wgt_aft_res.append((pos, wgt))



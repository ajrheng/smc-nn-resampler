class smc_memory:

    def __init__(self):

        # particle positions and weights before resampling
        self.pos_wgt_bef_res = []

        # bins and edges before passing into resampler
        self.bins_edges_bef_res = []

        # bins and edges after resampling
        self.bins_edges_aft_res = []

        # particle positions and weights after resampling
        self.pos_wgt_aft_res = []

        self.kernel = []

    def upd_pos_wgt_bef_res(self, pos, wgt):
        self.pos_wgt_bef_res.append((pos, wgt))

    def upd_bins_edges_bef_res(self, bins, edges):
        self.bins_edges_bef_res.append((bins, edges))

    def upd_bins_edges_aft_res(self, bins, edges):
        self.bins_edges_aft_res.append((bins, edges))

    def upd_pos_wgt_aft_res(self, pos, wgt):
        self.pos_wgt_aft_res.append((pos, wgt))

    def upd_kernel(self, kernel):
        self.kernel.append(kernel)

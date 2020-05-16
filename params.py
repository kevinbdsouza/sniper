import os
from scipy.io import loadmat


class Params:
    def __init__(self):
        self.hic_path = "/data2/hic_lstm/data/"
        self.downstream_dir = "/data2/hic_lstm/downstream"
        self.sizes_file = "chr_cum_sizes.npy"
        self.start_end_file = "starts.npy"

        self.input_file = self.hic_path + "GM12878/GM12878.hic"
        self.target_file = self.input_file
        self.label_file = None
        self.juicer_tools_path = "/data2/hic_lstm/softwares/juicer_tools.jar"
        self.cropMap = loadmat('/home/kevindsouza/Documents/projects/sniper/crop_map/cropMap.mat')
        self.cropIndices = loadmat(
            '/home/kevindsouza/Documents/projects/sniper/crop_map/cropIndices.mat')

        self.inter_chromosomal = False
        self.save_matrix = True
        self.autoremove = True
        self.overwrite = True
        self.usemat = False

        self.chr = None
        self.mode = None
        self.output_txt_path = None
        self.output_mat_path = None
        self.exp = "pca"
        self.num_features = 16

        if self.exp == "pca":
            self.dump_dir = self.hic_path + "pca"
        else:
            self.dump_dir = self.hic_path + "sniper"

import os
import numpy as np

from scipy.io import savemat
from utilities.interchromosome_matrix import construct_chr
from sklearn.decomposition import PCA

class DataProcessing:
    def __init__(self, params):
        self.params = params

    def chrom_sizes(self, f, length=np.inf):
        data = open(f, 'r')

        sizes = {}

        for line in data:
            ldata = line.split()

            if len(ldata[0]) > length:
                continue

            sizes[ldata[0]] = int(ldata[1])

        return sizes

    def get_pca_encodings(self, inputM):
        pca = PCA(n_components=self.params.num_features)
        pca_encodings = pca.fit_transform(inputM)

        return pca_encodings

    def constructAndSave(self, tmp_dir, prefix):
        M = construct_chr(self.params, self.chrom_sizes, tmp_dir, prefix=prefix)

        if self.params.exp == "pca":
            M = M
        else:
            output_mat_path = os.path.join(tmp_dir, '%s_matrix.mat' % prefix)
            self.params.output_mat_path = output_mat_path
            savemat(self.params.output_mat_path, {'inter_matrix': M}, do_compression=True)

        return M

    def hicToMat(self, params, prefix='hic'):
        """ Calls juicer_tools to extract hic data into txt files """

        M = None
        if params.inter_chromosomal:
            for chrm1 in range(1, 3, 2):
                for chrm2 in range(2, 4, 2):
                    output_txt_path = os.path.join(params.dump_dir,
                                                   '{2}_chrm{0}_chrm{1}.txt'.format(chrm1, chrm2, prefix))

                    os.system("java -jar {0} dump observed KR {1} {2} {3} BP 100000 {4} > tmp_juicer_log".format(
                        params.juicer_tools_path, params.input_file,
                        chrm1, chrm2,
                        output_txt_path))

        else:
            output_txt_path = os.path.join(params.dump_dir, '{1}_chrm{0}_chrm{0}.txt'.format(params.chr, prefix))
            self.params.output_txt_path = output_txt_path
            os.system(
                "java -jar {0} dump observed KR {1} {2} {2} BP 10000 {3}".format(params.juicer_tools_path,
                                                                                 params.input_file, params.chr,
                                                                                 output_txt_path))

        if params.save_matrix:
            M = self.constructAndSave(params.dump_dir, prefix)

        return M

    """ Trims sparse, NA, and B4 regions """

    def trimMat(self, M, indices):
        row_indices = indices['odd_indices'].flatten()
        col_indices = indices['even_indices'].flatten()

        M = M[row_indices, :]
        M = M[:, col_indices]

        return M

    """Set delta to avoid dividing by zero"""

    def contactProbabilities(self, M, delta=1e-10):
        coeff = np.nan_to_num(1 / (M + delta))
        PM = np.power(1 / np.exp(1), coeff)

        return PM

    def RandomSample(self, data):
        N = len(data)
        return data[np.random.randint(N)]

    def bootstrap(self, data, labels, samplesPerClass=None):
        Nsamples = samplesPerClass
        classes = np.unique(labels)

        maxSamples = np.max(np.bincount(labels))

        if samplesPerClass is None or samplesPerClass < maxSamples:
            Nsamples = maxSamples

        bootstrapSamples = []
        bootstrapClasses = []

        for i, c in enumerate(classes):
            classLabel = c
            classData = data[labels == c]

            nBootstrap = Nsamples

            for n in range(nBootstrap):
                sample = self.RandomSample(classData)

                bootstrapSamples.append(sample)
                bootstrapClasses.append(c)

        bootstrapSamples = np.asarray(bootstrapSamples)
        bootstrapClasses = np.asarray(bootstrapClasses)

        bootstrapData = np.hstack((bootstrapSamples, np.array([bootstrapClasses]).T))
        np.random.shuffle(bootstrapData)

        return (bootstrapData[:, :-1], bootstrapData[:, -1])

    def Sigmoid(self, data):
        return 1 / (1 + np.exp(-data))

    def getColorString(self, n):
        subcColors = ['34,139,34', '152,251,152', '220,20,60', '255,255,0', '112,128,144']
        return subcColors[n]

    def getSubcName(self, n):
        order = ['A1', 'A2', 'B1', 'B2', 'B3']
        return order[n]

    def predictionsToBed(self, path, odds, evens, cropMap, res=100000, sizes_file='data/hg19.chrom.sizes'):
        rowMap = cropMap['rowMap'].astype(np.int)
        colMap = cropMap['colMap'].astype(np.int)

        sizes = self.chrom_sizes(sizes_file)

        file = open(path, 'w')

        for i, p in enumerate(np.argmax(odds, axis=1)):
            m = rowMap
            chrm, start = 'chr' + str(m[i, 1]), m[i, 2] * res
            end = np.min([start + res, sizes[chrm]])
            line = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(chrm, start, end, self.getSubcName(p), 0, '.', start,
                                                                 end,
                                                                 self.getColorString(p))
            file.write(line)

        for i, p in enumerate(np.argmax(evens, axis=1)):
            m = colMap
            chrm, start = 'chr' + str(m[i, 1]), m[i, 2] * res
            end = np.min([start + res, sizes[chrm]])
            line = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(chrm, start, end, self.getSubcName(p), 0, '.', start,
                                                                 end,
                                                                 self.getColorString(p))
            file.write(line)

        file.close()

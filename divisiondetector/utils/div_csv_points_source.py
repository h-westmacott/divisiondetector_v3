from gunpowder.nodes import CsvPointsSource
import numpy as np
import chardet

# CsvPointsSource from gunpowder, edited to check and use encoding of CSV

class DivCsvPointsSource(CsvPointsSource):
    def _parse_csv(self):
        '''Read one point per line. If ``ndims`` is None, all values in one line
        are considered as the location of the point. If positive, only the
        first ``ndims`` are used. If negative, all but the last ``-ndims`` are
        used.
        '''

        with open(self.filename, 'rb') as f:
            metadata = chardet.detect(f.read(10000))

        with open(self.filename, "r", encoding=metadata['encoding']) as f:
            self.data = np.array(
                [[float(t.strip(",")) for t in line.split()] for line in f],
                dtype=np.float32,
            )

        if self.ndims is None:
            self.ndims = self.data.shape[1]

        if self.scale is not None:
            self.data[:,:self.ndims] *= self.scale

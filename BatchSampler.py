

from torch.utils.data import Sampler
import random


class BucketBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, bucket_size, shuffle=True):
        self.data_source = data_source
        self.batch_size = batch_size
        self.bucket_size = bucket_size
        self.shuffle = shuffle

        # Precompute lengths for bucketing
        self.indices = list(range(len(data_source)))
        self.lengths = [len(data_source[i][0]) for i in self.indices]  # Use source length

    def __iter__(self):
        if self.shuffle:
            zipped = list(zip(self.indices, self.lengths))
            random.shuffle(zipped)
            self.indices, self.lengths = zip(*zipped)

        # Sort within buckets
        buckets = [self.indices[i:i+self.bucket_size] for i in range(0, len(self.indices), self.bucket_size)]

        for bucket in buckets:
            bucket = sorted(bucket, key=lambda i: self.lengths[i])
            for i in range(0, len(bucket), self.batch_size):
                yield bucket[i:i + self.batch_size]

    def __len__(self):
        return len(self.indices) // self.batch_size

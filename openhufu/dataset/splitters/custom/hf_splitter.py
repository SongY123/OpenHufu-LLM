from datasets import Dataset

from openhufu.dataset.splitters.generic import IIDSplitter, LDASplitter
from openhufu.dataset.splitters.splitter_factory import SplitterFactory

@SplitterFactory.register(Dataset, 'iid')
class HFIIDSplitter(IIDSplitter):
    def __init__(self, n_clients, seed=None):
        super(HFIIDSplitter, self).__init__(n_clients, seed)

    def subset(self, dataset: Dataset, idxs: list):
        return dataset.select(idxs)

@SplitterFactory.register(Dataset, 'lda')
class HFLDASplitter(LDASplitter):
    def __init__(self, n_clients, seed=None):
        super(HFLDASplitter, self).__init__(n_clients, seed)

    def subset(self, dataset: Dataset, idxs: list):
        return dataset.select(idxs)

from torch.utils.data import Dataset, Subset

from openhufu.dataset.splitters.generic import IIDSplitter, LDASplitter
from openhufu.dataset.splitters.splitter_factory import SplitterFactory

@SplitterFactory.register(Dataset, 'iid')
class TorchIIDSplitter(IIDSplitter):
    def __init__(self, n_clients, seed=None):
        super(TorchIIDSplitter, self).__init__(n_clients, seed)

    def subset(self, dataset: Dataset, idxs: list):
        return Subset(dataset, idxs)

@SplitterFactory.register(Dataset, 'lda')
class TorchLDASplitter(LDASplitter):
    def __init__(self, n_clients, seed=None):
        super(TorchLDASplitter, self).__init__(n_clients, seed)

    def subset(self, dataset: Dataset, idxs: list):
        return Subset(dataset, idxs)
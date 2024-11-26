from openhufu.dataset.splitters.generic import IIDSplitter, LDASplitter
from openhufu.dataset.splitters.splitter_factory import SplitterFactory

@SplitterFactory.register(list, 'iid')
class ListIIDSplitter(IIDSplitter):
    def __init__(self, n_clients, seed=None):
        super(ListIIDSplitter, self).__init__(n_clients, seed)

    def subset(self, dataset: list, idxs: list):
        return [dataset[idx] for idx in idxs]

@SplitterFactory.register(list, 'lda')
class ListLDASplitter(LDASplitter):
    def __init__(self, n_clients, seed=None):
        super(ListLDASplitter, self).__init__(n_clients, seed)

    def subset(self, dataset: list, idxs: list):
        return [dataset[idx] for idx in idxs]
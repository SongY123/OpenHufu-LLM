import abc
import inspect


class BaseSplitter(abc.ABC):
    def __init__(self, n_clients, seed=None):
        self.n_clients = n_clients
        self.seed = seed

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        """
        Returns: Meta information for `Splitter`.
        """
        sign = inspect.signature(self.__init__).parameters.values()
        meta_info = tuple([(val.name, getattr(self, val.name))
                           for val in sign])
        return f'{self.__class__.__name__}{meta_info}'

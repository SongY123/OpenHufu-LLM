
class SplitterFactory:
    registry_splitters = {}

    @classmethod
    def register(cls, dataset_type, splitter_type):
        def decorator(splitter_class):
            cls.registry_splitters[(dataset_type, splitter_type)] = splitter_class
            return splitter_class

        return decorator

    @classmethod
    def get_splitter(cls, dataset_type: type, splitter_type, n_clients, **kwargs):
        splitter_class = cls.registry_splitters.get((dataset_type, splitter_type))
        if not splitter_class:
            raise ValueError(f"Splitter not found for dataset type: {dataset_type} and splitter type: {splitter_type}")
        return splitter_class(n_clients, **kwargs)
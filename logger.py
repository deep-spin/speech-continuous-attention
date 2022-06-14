from speechbrain.utils.train_logger import FileTrainLogger as BaseFileTrainLogger


class FileTrainLogger(BaseFileTrainLogger):

    def _item_to_string(self, key, value, dataset=None):
        """Convert one item to string, handling floats"""
        if isinstance(value, float) and 1.0 < value < 100.0:
            value = f"{value:.{self.precision}f}"
        elif isinstance(value, float):
            value = f"{value:.{self.precision}f}"
            # value = f"{value:.{self.precision}e}"
        if dataset is not None:
            key = f"{dataset} {key}"
        return f"{key}: {value}"

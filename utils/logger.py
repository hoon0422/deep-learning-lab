from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, filename=None, mode='a'):
        self._filename = Path(filename) if filename is not None else None
        self._mode = mode
        self._print_func = print

    @property
    def filename(self):
        return self._filename

    @property
    def mode(self):
        return self._mode

    @property
    def print_func(self):
        return self._print_func

    @mode.setter
    def mode(self, new_mode):
        self._mode = new_mode

    @print_func.setter
    def print_func(self, new_print_func):
        self._print_func = new_print_func

    def log(self, msg, mode=None):
        if self._filename:
            with open(self._filename, mode=self._mode if mode is None else mode) as file:
                file.write(msg)
        else:
            self._print_func(msg)


class TensorboardWriter:
    def __init__(self, log_dir=None):
        self.writer = SummaryWriter(Path(log_dir))

    def print_image(self, title, image):
        image = image.detach().cpu().numpy()
        self.writer.add_image(title, image)
        del image

    def print_network(self, network, dummy_input_data):
        self.writer.add_graph(network, dummy_input_data)

    def add_point(self, title, x_val, y_val):
        self.writer.add_scalar(title, y_val, x_val)

    def add_points(self, title, x_val, y_val_dict):
        self.writer.add_scalar(title, y_val_dict, x_val)

    @classmethod
    def open(cls, log_dir=None):
        return cls(log_dir)

    def close(self):
        self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

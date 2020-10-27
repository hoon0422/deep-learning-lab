import logging
from pathlib import Path
from types import MethodType

from torch.utils.tensorboard import SummaryWriter

DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL


class Logger:
    def __init__(self, filename=None, condition=None, mode='w', level=INFO, old_formatting_style=False):
        self._filename = Path(filename) if filename is not None else None
        self._condition = condition if condition is not None else lambda: True
        self._mode = mode
        self._level = level
        self._formats = {}
        self._old_formatting_style = old_formatting_style
        self._print_func = print

        levels = {
            'debug': DEBUG,
            'info': INFO,
            'warning': WARNING,
            'error': ERROR,
            'critical': CRITICAL
        }

        for name, lev in levels.items():
            setattr(self, name, MethodType(lambda s, msg, m=None: s.log(msg, lev, m), self))
            setattr(self, name + '_format', MethodType(
                lambda s, value_dict, fm, m=None, ofs=None:
                s.log_format(value_dict, fm, lev, m, ofs), self
            ))
            setattr(self, name + '_console', MethodType(
                lambda s, msg: s.log_console(msg, lev), self
            ))
            setattr(self, name + '_console_format', MethodType(
                lambda s, value_dict, fm, ofs=None:
                s.log_console_format(value_dict, fm, lev, ofs), self
            ))

    @property
    def filename(self):
        return self._filename

    @property
    def condition(self):
        return self._condition

    @property
    def mode(self):
        return self._mode

    @property
    def level(self):
        return self._level

    @property
    def formats(self):
        return self._formats

    @property
    def old_formatting_style(self):
        return self._old_formatting_style

    @property
    def print_func(self):
        return self._print_func

    @mode.setter
    def mode(self, new_mode):
        self._mode = new_mode

    @level.setter
    def level(self, new_level):
        self._level = new_level

    @print_func.setter
    def print_func(self, new_print_func):
        self._print_func = new_print_func

    def add_format(self, format_name, format):
        self._formats[format_name] = format

    def log(self, msg, level, mode=None):
        if not self._condition() or self._level > level:
            return

        if self._filename:
            with open(self._filename, mode=self._mode if mode is None else mode) as file:
                file.write(msg)
        else:
            self._print_func(msg)

    def log_format(self, value_dict, format_name, level, mode=None, old_formatting_style=None):
        if not self._condition() or self._level > level:
            return

        old_formatting_style = self._old_formatting_style if old_formatting_style is None else old_formatting_style
        format = self._formats[format_name]
        if old_formatting_style:
            msg = format % value_dict
        else:
            msg = format.format(**value_dict)

        self.log(msg, level, mode)

    def log_console(self, msg, level):
        if not self._condition() or self._level > level:
            return

        self._print_func(msg)

    def log_console_format(self, value_dict, format_name, level, old_formatting_style=None):
        if not self._condition() or self._level > level:
            return

        old_formatting_style = self._old_formatting_style if old_formatting_style is None else old_formatting_style
        format = self._formats[format_name]
        if old_formatting_style:
            msg = format % value_dict
        else:
            msg = format.format(**value_dict)

        self.log_console(msg, level)


class TensorboardWriter:
    def __init__(self, log_dir=None, condition=None):
        self.condition = condition if condition is not None else lambda: True

        if not self.condition():
            return
        self.writer = SummaryWriter(Path(log_dir))

    def print_image(self, title, image):
        if not self.condition():
            return
        image = image.detach().cpu().numpy()
        self.writer.add_image(title, image)
        del image

    def print_network(self, network, dummy_input_data):
        if not self.condition():
            return
        self.writer.add_graph(network, dummy_input_data)

    def add_point(self, title, x_val, y_val):
        if not self.condition():
            return
        self.writer.add_scalar(title, y_val, x_val)

    def add_points(self, title, x_val, y_val_dict):
        if not self.condition():
            return
        self.writer.add_scalar(title, y_val_dict, x_val)

    def close(self):
        if not self.condition():
            return
        self.writer.close()

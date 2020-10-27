import argparse

__options = None
__parser = argparse.ArgumentParser(conflict_handler='resolve')


class OptionParser:
    @staticmethod
    def add_options(parser: argparse.ArgumentParser):
        return parser


def get_parser():
    return __parser


def get_options():
    return __options


def parse_options(*args, **kwargs):
    global __options
    __options = __parser.parse_args(*args, **kwargs)


def is_option_parsed():
    return __options is not None


def params_to_argv(params):
    argv = []
    for key, val in params.items():
        if isinstance(val, bool):
            if val:
                argv.append(f'--{key}')
            continue
        else:
            argv.append(f'--{key}')

        if isinstance(val, list):
            for v in val:
                argv.append(str(v))
        else:
            argv.append(str(val))

    return argv

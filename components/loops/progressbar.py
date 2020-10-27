from typing import Callable
from components.loops import BaseLoop
from tqdm import tqdm


class ProgressbarLoop(BaseLoop):
    progressbar: tqdm
    write: Callable = tqdm.write

    def __init__(self, description: str = None, position: int = 0):
        super(ProgressbarLoop, self).__init__()
        self.description = description if description is not None else description
        self.position = position

    def before_loop(self, condition=True):
        if condition:
            self.progressbar = tqdm(range(1, 1 + self.length), dynamic_ncols=True, position=self.position)
            self.progressbar.set_description_str(self.description)

    def before_iteration(self, *args, **kwargs):
        pass

    def after_iteration(self, condition=True):
        if condition:
            self.progressbar.update()

    def after_loop(self, condition=True):
        if condition:
            self.progressbar.close()

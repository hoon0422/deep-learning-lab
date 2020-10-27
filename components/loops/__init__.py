import abc
from enum import Enum
from pprint import pprint
from typing import Callable, Iterable, List


class CLoop:
    __leaf__ = None

    class CB(Enum):
        BEFORE_LOOP = 'BEFORE_LOOP'
        BEFORE_ITERATION = 'BEFORE_ITERATION'
        AFTER_ITERATION = 'AFTER_ITERATION'
        AFTER_LOOP = 'AFTER_LOOP'

    def __init__(self):
        self._callback = dict((cb_type, lambda cloop, *args, **kwargs: None) for cb_type in CLoop.CB)
        self._pass_args = dict((cb_type, lambda *args, **kwargs: None) for cb_type in CLoop.CB)
        self._get_args = dict((cb_type, lambda: (tuple(), dict())) for cb_type in CLoop.CB)
        self._iterable = None
        self._saved_length = None
        self._element = None
        self._index = None
        self._parent = None
        self._args_containers = dict()
        self._kwargs_containers = dict()

    @property
    def element(self):
        return self._element

    @element.setter
    def element(self, el):
        self._element = el

    @property
    def index(self) -> int:
        return self._index

    @index.setter
    def index(self, idx):
        self._index = idx

    @property
    def iterable(self) -> Iterable:
        return self._iterable

    @property
    def length(self) -> int:
        return len(self._iterable) if self._saved_length is None else self._saved_length

    def set_iterable(self, iterable: Iterable, length: int = None):
        if not isinstance(iterable, Iterable):
            raise TypeError("'set_iterable' must have iterable object as an argument")
        self._iterable = iterable
        if length is not None:
            self._saved_length = length

    def parent(self, level=1) -> 'CLoop':
        parent = self
        for i in range(level):
            parent = parent._parent
        return parent

    def get_callback(self, cb_type: 'CLoop.CB') -> Callable[..., None]:
        if not isinstance(cb_type, CLoop.CB):
            raise TypeError("'cb_type' must be instance of 'CLoop.CB'")
        return self._callback[cb_type]

    def set_callback(self, cb_type: 'CLoop.CB', callback: Callable[..., None]) -> 'CLoop':
        if not isinstance(cb_type, CLoop.CB):
            raise TypeError("'cb_type' must be instance of 'CLoop.CB'")
        if not hasattr(callback, '__call__'):
            raise TypeError("'callback' must be callable")

        self._args_containers[callback] = tuple()
        self._kwargs_containers[callback] = dict()

        def pass_args(cloop, *args, **kwargs):
            cloop._args_containers[callback] = args
            cloop._kwargs_containers[callback] = kwargs

        def get_args(cloop):
            args = cloop._args_containers[callback]
            kwargs = cloop._kwargs_containers[callback]
            cloop._args_containers[callback] = tuple()
            cloop._kwargs_containers[callback] = dict()
            return args, kwargs

        self._callback[cb_type] = callback
        self._pass_args[cb_type] = lambda *args, **kwargs: pass_args(self)
        self._get_args[cb_type] = lambda *args, **kwargs: get_args(self)

        return self

    def get_args_passer(self, cb_type: 'CLoop.CB') -> Callable[..., None]:
        if not isinstance(cb_type, CLoop.CB):
            raise TypeError("'cb_type' must be instance of 'CLoop.CB'")
        return self._pass_args[cb_type]

    def __iter__(self):
        if self._iterable is None:
            raise ValueError("Iterable object has not been set yet")

        with self:
            args, kwargs = self._get_args[CLoop.CB.BEFORE_LOOP]()
            self.get_callback(CLoop.CB.BEFORE_LOOP)(self, *args, **kwargs)
            for element in self._iterable:
                self.element = element

                args, kwargs = self._get_args[CLoop.CB.BEFORE_ITERATION]()
                self.get_callback(CLoop.CB.BEFORE_ITERATION)(self, *args, **kwargs)

                yield self.element

                args, kwargs = self._get_args[CLoop.CB.AFTER_ITERATION]()
                self.get_callback(CLoop.CB.AFTER_ITERATION)(self, *args, **kwargs)

                self.index += 1

            args, kwargs = self._get_args[CLoop.CB.AFTER_LOOP]()
            self.get_callback(CLoop.CB.AFTER_LOOP)(self, *args, **kwargs)

    def __call__(self, iterable: Iterable, length: int = None):
        self.set_iterable(iterable, length)
        return self

    def __enter__(self):
        self._index = 0
        self._parent = CLoop.__leaf__
        CLoop.__leaf__ = self

    def __exit__(self, exc_type, exc_val, exc_tb):
        CLoop.__leaf__ = self.parent()
        self._parent = None
        self._index = None
        self._element = None
        self._iterable = None


class BaseLoop(abc.ABC, CLoop):
    def __init__(self):
        super(BaseLoop, self).__init__()
        self.set_callback(CLoop.CB.BEFORE_LOOP, type(self).before_loop) \
            .set_callback(CLoop.CB.BEFORE_ITERATION, type(self).before_iteration) \
            .set_callback(CLoop.CB.AFTER_ITERATION, type(self).after_iteration) \
            .set_callback(CLoop.CB.AFTER_LOOP, type(self).after_loop)

    def before_loop(self, *args, **kwargs):
        raise NotImplementedError("'before_loop' is not implemented")

    def before_iteration(self, *args, **kwargs):
        raise NotImplementedError("'before_iteration is not implemented")

    def after_iteration(self, *args, **kwargs):
        raise NotImplementedError("'after_iteration' is not implemented")

    def after_loop(self, *args, **kwargs):
        raise NotImplementedError("'after_loop' is not implemented")


class ZipLoop(BaseLoop):
    def __init__(self, cloops: Iterable[BaseLoop] = ()):
        super(ZipLoop, self).__init__()
        self._cloops: List[BaseLoop] = list(cloops)

    @property
    def element(self):
        return self._element

    @element.setter
    def element(self, el):
        self._element = el
        for cloop in self._cloops:
            cloop.element = el

    @property
    def index(self) -> int:
        return self._index

    @index.setter
    def index(self, idx):
        self._index = idx
        for cloop in self._cloops:
            cloop.index = idx

    def set_iterable(self, iterable: Iterable, length: int = None):
        if not isinstance(iterable, Iterable):
            raise TypeError("'set_iterable' must have iterable object as an argument")
        self._iterable = iterable
        if length is not None:
            self._saved_length = length

        for cloop in self._cloops:
            cloop.set_iterable(iterable, length)

    def append(self, cloop: BaseLoop):
        self._cloops.append(cloop)

    def before_loop(self, *args, **kwargs):
        for cloop in self._cloops:
            cloop_args, cloop_kwargs = cloop._get_args[CLoop.CB.BEFORE_LOOP]()
            cloop.get_callback(CLoop.CB.BEFORE_LOOP)(cloop, *cloop_args, **cloop_kwargs)

    def before_iteration(self, *args, **kwargs):
        for cloop in self._cloops:
            cloop_args, cloop_kwargs = cloop._get_args[CLoop.CB.BEFORE_ITERATION]()
            cloop.get_callback(CLoop.CB.BEFORE_ITERATION)(cloop, *cloop_args, **cloop_kwargs)

    def after_iteration(self, *args, **kwargs):
        for cloop in self._cloops:
            cloop_args, cloop_kwargs = cloop._get_args[CLoop.CB.AFTER_ITERATION]()
            cloop.get_callback(CLoop.CB.AFTER_ITERATION)(cloop, *cloop_args, **cloop_kwargs)

    def after_loop(self, *args, **kwargs):
        for cloop in self._cloops:
            cloop_args, cloop_kwargs = cloop._get_args[CLoop.CB.AFTER_LOOP]()
            cloop.get_callback(CLoop.CB.AFTER_LOOP)(cloop, *cloop_args, **cloop_kwargs)

    def __getitem__(self, idx: int) -> BaseLoop:
        return self._cloops[idx]

    def __setitem__(self, idx: int, cloop: BaseLoop):
        self._cloops[idx] = cloop

    def __delitem__(self, idx):
        del self._cloops[idx]

    def __len__(self):
        return len(self._cloops)

    def __enter__(self):
        self._index = 0
        for cloop in self._cloops:
            cloop._index = 0
        self._parent = CLoop.__leaf__
        for cloop in self._cloops:
            cloop._parent = CLoop.__leaf__
        CLoop.__leaf__ = self

    def __exit__(self, exc_type, exc_val, exc_tb):
        CLoop.__leaf__ = self.parent()
        self._parent = None
        for cloop in self._cloops:
            cloop._parent = None
        self._index = None
        for cloop in self._cloops:
            cloop._index = None
        self._element = None
        for cloop in self._cloops:
            cloop._element = None
        self._iterable = None
        for cloop in self._cloops:
            cloop._iterable = None

from pathlib import Path
from typing import Dict, List, Iterable, Optional, Callable, Union, Tuple, Any

import torch
import torchvision
from components.loops import BaseLoop

MetricContainer = Dict[str, Dict[str, List[float]]]
MetricAverage = Dict[str, Dict[str, float]]


class MetricLoop(BaseLoop):
    field_and_metric_names_map: Dict[str, Iterable[str]]
    prefix: str
    metric_containers: "MetricLoop._MetricContainer"

    @staticmethod
    def format_metric_avg(metric_avg: MetricAverage):
        return '\n'.join([
            f'* {field}\n' + '\n'.join([
                f'  - {metric_name}: {metric}'
                for metric_name, metric in name_and_metrics_map.items()
            ]) for field, name_and_metrics_map in metric_avg.items()
        ])

    @staticmethod
    def format_metric_header(
            header_name_and_value_map: Dict[str, Any],
            prefix: str = '',
            suffix: str = '',
            delimiter: str = '-',
            num_delimiter: int = 15
    ):
        splitter = delimiter * num_delimiter
        return prefix + splitter + ' ' + ' // '.join(
            name + ': ' + str(value)
            for name, value in header_name_and_value_map.items()
        ) + ' ' + splitter + suffix

    class _MetricContainer:
        _container: MetricContainer
        _avg_cache: MetricAverage
        _update_required: bool

        def __init__(self, field_and_metric_names_map):
            self._container = dict(
                (field, dict(
                    (metric_name, list())
                    for metric_name in metric_names
                ))
                for field, metric_names in field_and_metric_names_map.items()
            )
            self._update_required = False

        def insert_metric(self, field: str, metric_name: str, metric: float):
            self._update_required = True
            self._container[field][metric_name].append(metric)

        def insert_metrics(self, field: str, name_and_metric_map: Dict[str, float]):
            self._update_required = True
            for metric_name, metric in name_and_metric_map.items():
                self.insert_metric(field, metric_name, float(metric))

        def insert_field_metrics(self, field_and_metric_names_and_metric_map: MetricAverage):
            for field, name_and_metric_map in field_and_metric_names_and_metric_map.items():
                self.insert_metrics(field, name_and_metric_map)

        def calc_avg(self) -> MetricAverage:
            if self._update_required:
                self._update_required = False
                self._avg_cache = dict(
                    (field, dict(
                        (metric_name, sum(metrics) / len(metrics))
                        for metric_name, metrics in name_and_metrics_map.items()
                    ))
                    for field, name_and_metrics_map in self._container.items()
                )
            return self._avg_cache

    def __init__(self, field_and_metric_names_map: Dict[str, Iterable[str]], prefix=''):
        super(MetricLoop, self).__init__()
        self.field_and_metric_names_map = field_and_metric_names_map
        self.prefix = prefix

    def before_loop(self):
        self.metric_containers = self._MetricContainer(self.field_and_metric_names_map)

    def before_iteration(self, *args, **kwargs):
        pass

    def after_iteration(
            self,
            calculated_metrics: Dict[str, Dict[str, float]],
            callback: Optional[Callable[[MetricAverage], None]] = None
    ):
        self.metric_containers.insert_field_metrics(calculated_metrics)  # insert metrics
        callback(self._add_prefix_to_field(self.metric_containers.calc_avg()))

    def after_loop(self, callback: Optional[Callable[[MetricAverage], None]] = None):
        callback(self._add_prefix_to_field(self.metric_containers.calc_avg()))

    def _add_prefix_to_field(self, avg: MetricAverage) -> MetricAverage:
        if self.prefix == '':
            return avg

        prefix = self.prefix + '_'
        return dict(
            (prefix + field, dict(
                (metric_name, metric_avg)
                for metric_name, metric_avg in name_and_avg_map.items()
            ))
            for field, name_and_avg_map in avg.items()
        )


class ImageLoop(BaseLoop):
    @staticmethod
    def save_image(image_tensor, filename=Union[str, Path]):
        torchvision.utils.save_image(image_tensor, filename)

    @staticmethod
    def format_filename(
            header_name_and_value_map: Dict[str, Any],
            directory: Union[str, Path],
            prefix: str = '',
            suffix: str = '',
            extension: str = '.png',
    ) -> Path:
        return Path(directory,
                    prefix + '__'.join(
                        name + '-' + str(value)
                        for name, value in header_name_and_value_map.items()
                    ) + suffix + extension)

    @staticmethod
    def format_tensorboard_name(
            header_name_and_value_map: Dict[str, Any],
            prefix: str = '',
            suffix: str = '',
    ):
        return prefix + '/'.join(
            name + ': ' + str(value)
            for name, value in header_name_and_value_map.items()
        ) + suffix

    def __init__(
            self,
            num_cols: int = 1,
            normalize: bool = False,
            scale_each: bool = False,
            range: bool = None,
            pad_value: int = 0
    ):
        super(ImageLoop, self).__init__()
        self.num_cols = num_cols
        self.normalize = normalize
        self.scale_each = scale_each
        self.range = range
        self.pad_value = pad_value

    def before_loop(self, *args, **kwargs):
        pass

    def before_iteration(self, *args, **kwargs):
        pass

    def after_iteration(
            self,
            image_tensors: Union[Tuple[torch.Tensor], List[torch.Tensor]],
            post_callback: Optional[Callable[[torch.Tensor], None]],
            condition: bool = True,
            preprocess: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        if not condition:
            return

        if preprocess is not None:
            image_tensors = tuple(preprocess(image) for image in image_tensors)

        num_cols = min(self.num_cols, len(image_tensors[0]))
        image_tensors = tuple(
            (img_tensor[:num_cols]
             if img_tensor.shape[1] == 3
             else img_tensor[:num_cols].repeat((1, 3, 1, 1)))
            for img_tensor in image_tensors
        )

        grid = torchvision.utils.make_grid(
            torch.cat(image_tensors, dim=0),
            nrow=num_cols,
            normalize=self.normalize,
            scale_each=self.scale_each,
            range=self.range,
            pad_value=self.pad_value
        )

        post_callback(grid)

    def after_loop(self, *args, **kwargs):
        pass

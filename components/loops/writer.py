from pathlib import Path

from typing import Dict, List

import torch
import torchvision
from tqdm import tqdm
import utils
from utils import logger, parallel
from components.loops import BaseLoop


class MetricLoop(BaseLoop):
    def __init__(self, metric_names_with_field, criterion_names, loop_name='epoch', iter_name='batch', lab_name='',
                 instance_name='', prefix='', print_on_tensorboard=True, tensorboard_writer=None, no_parent=False):
        super(MetricLoop, self).__init__()
        self.metric_names_with_field = metric_names_with_field
        self.criterion_names = criterion_names

        self.metric_containers: Dict[str, Dict[str, List[float]]] = None
        self.loss_containers: Dict[str, List[float]] = None
        self._initialize_containers()

        self.prefix = prefix
        self.loop_name = loop_name
        self.iter_name = iter_name
        self.lab_name = lab_name
        self.instance_name = instance_name

        self.print_on_tensorboard = print_on_tensorboard
        self.tensorboard_writer = tensorboard_writer

        self.log_filename = f'{self.prefix}.log'
        self.logger = None
        self.progress_bar = None
        self.no_parent = no_parent

        options = option.get_options()
        self.print_frequency = options.print_frequency
        self.result_dir = options.result_dir

    def before_loop(self):
        if parallel.is_first_process():
            self.logger = self._create_logger()
            self.progress_bar = self._create_progress_bar()
            self.logger.print_func = self.progress_bar.write
        self._initialize_containers()

    def before_iteration(self, *args, **kwargs):
        pass

    def after_iteration(self, calculated_metrics, calculated_losses):
        # insert losses
        self._insert_losses(calculated_losses)
        self._insert_metrics(calculated_metrics)

        if parallel.is_first_process() and (self.index + 1) % self.print_frequency == 0:
            # calculate metrics
            metrics_avg = self.calc_metrics_avg()
            losses_avg = self.calc_losses_avg()
            merged_avg = self._merge_avg_metrics_and_losses(metrics_avg, losses_avg)
            self._print_and_write_iter_log(merged_avg, self.index + 1, self.get_parent_index())

        if parallel.is_first_process():
            self.progress_bar.update()

    def after_loop(self):
        metrics_avg = self.calc_metrics_avg()
        losses_avg = self.calc_losses_avg()
        merged_avg = self._merge_avg_metrics_and_losses(metrics_avg, losses_avg)

        if parallel.is_distributed_initialized() and parallel.get_size() > 1:
            merged_avg = utils.reduce_all_iterable(merged_avg)

        if parallel.is_first_process():
            self._print_and_write_loop_log(merged_avg, self.get_parent_index())
            self._save_points_on_tensorboard(merged_avg)
            self.progress_bar.close()

        if parallel.is_distributed_initialized() and parallel.get_size() > 1:
            parallel.wait_for_all_processes()

    def get_parent_index(self):
        return self.parent().index

    def calc_metrics_avg(self) -> Dict[str, Dict[str, float]]:
        avgs = dict()
        for field, func_names in self.metric_containers.items():
            avgs[field] = dict()
            for func_name, metrics in func_names.items():
                metrics = self.metric_containers[field][func_name]
                avgs[field][func_name] = sum(metrics) / len(metrics)
        return avgs

    def calc_losses_avg(self) -> Dict[str, float]:
        avgs = dict()
        for criterion, losses in self.loss_containers.items():
            avgs[criterion] = sum(losses) / len(losses)
        return avgs

    def _initialize_containers(self):
        self.metric_containers = dict()
        for field, metric_names in self.metric_names_with_field.items():
            self.metric_containers[field] = dict()
            for metric_name in metric_names:
                self.metric_containers[field][metric_name] = list()
        self.loss_containers = dict((criterion, list()) for criterion in self.criterion_names)

    def _insert_metrics(self, metrics_with_name_with_field):
        for field, metrics_with_name in metrics_with_name_with_field.items():
            for name, metric in metrics_with_name.items():
                self.metric_containers[field][name].append(metric)

    def _insert_losses(self, losses_with_field):
        for field, loss in losses_with_field.items():
            self.loss_containers[field].append(loss)

    def _merge_avg_metrics_and_losses(self, metrics_avg, losses_avg):
        metric_dict = dict()
        prefix = self.prefix + "_" if self.prefix != "" else ""
        for field, metrics_with_name in metrics_avg.items():
            for metric_name, avg_metric in metrics_with_name.items():
                metric_dict[f'{prefix}{field}/{metric_name}'] = avg_metric
        for loss_name, avg_losses in losses_avg.items():
            metric_dict[f'{prefix}criterion/{loss_name}'] = avg_losses
        return metric_dict

    def _create_progress_bar(self):
        pb = tqdm(range(1, 1 + self.length), dynamic_ncols=True)
        pb.set_description(f'{self.prefix}/{self.iter_name}')
        return pb

    def _create_logger(self, loop_char='=', iter_char='-'):
        log = logger.Logger(
            Path(self.result_dir, self.log_filename),
            parallel.is_first_process,
            mode='a',
            level=logger.DEBUG
        )

        if self.lab_name != '' and self.instance_name != '':
            lab_instance_str = f'[lab: {self.lab_name} // instance: {self.instance_name}]\n'
        else:
            lab_instance_str = ''

        loop_str = f' {self.loop_name} {{loop:04}}/{self.parent().length:04} '
        loop_title = (loop_char * 25) + loop_str + (loop_char * 25) + '\n'

        if self.no_parent:
            iter_str = f' {self.iter_name} {{iter:04}}/{self.length:04} '
            iter_title = (iter_char * 25) + iter_str + (iter_char * 25) + '\n'
        else:
            iter_str = f'// {self.iter_name} {{iter:04}}/{self.length:04} '
            iter_title = (iter_char * 25) + loop_str + iter_str + (iter_char * 25) + '\n'

        metric_format = self._get_metric_print_format()
        log.add_format('loop', f'{lab_instance_str}{loop_title}{metric_format}')
        log.add_format('iter', f'{lab_instance_str}{iter_title}{metric_format}')

        return log

    def _get_metric_print_format(self):
        prefix = self.prefix + "_" if self.prefix != "" else ""
        return '\n'.join([
                             f'* {prefix}{field}\n' + '\n'.join([
                                 f'  - {metric_name}: {{{prefix}{field}/{metric_name}}}'
                                 for metric_name in metric_names
                             ]) for field, metric_names in self.metric_names_with_field.items()
                         ] + [
                             '* criterion\n' + '\n'.join([
                                 f'  - {criterion_name}: {{{prefix}criterion/{criterion_name}}}'
                                 for criterion_name in self.criterion_names
                             ])
                         ])

    def _print_and_write_iter_log(self, metric_dict, iter, loop=-1):
        md = metric_dict.copy()
        md['iter'] = iter
        if not self.no_parent:
            md['loop'] = loop
        self.logger.info_format(md, 'iter')
        self.logger.info_console_format(md, 'iter')

    def _print_and_write_loop_log(self, metric_dict, loop):
        md = metric_dict.copy()
        md['loop'] = loop
        self.logger.info_format(md, 'loop')
        self.logger.info_console_format(md, 'loop')

    def _get_tensorboard_writer(self):
        if self.tensorboard_writer is not None:
            return self.tensorboard_writer
        if self.no_parent:
            return None
        return self.parent()._get_tensorboard_writer()

    def _save_points_on_tensorboard(self, metric_dict):
        tensorboard = self._get_tensorboard_writer()
        if tensorboard is not None and self.print_on_tensorboard:
            loop_idx = self.get_parent_index()
            for key, metric in metric_dict.items():
                self._get_tensorboard_writer().add_point(key, loop_idx, metric)


class ImageLoop(BaseLoop):
    def __init__(self, loop_name='epoch', iter_name='batch', lab_name='', instance_name='', prefix='',
                 print_on_tensorboard=True, tensorboard_writer=None, no_parent=False, parent_level=2):
        super(ImageLoop, self).__init__()

        self.prefix = prefix
        self.loop_name = loop_name
        self.iter_name = iter_name
        self.lab_name = lab_name
        self.instance_name = instance_name
        self.no_parent = no_parent
        self.parent_level = parent_level

        self.print_on_tensorboard = print_on_tensorboard
        self.tensorboard_writer = tensorboard_writer

        options = option.get_options()
        self.image_save_frequency = options.image_save_frequency
        self.ncols_image = options.ncols_image
        self.result_dir = Path(options.result_dir, f'{self.prefix + "_" if self.prefix != "" else ""}images')
        self.result_dir.mkdir(parents=True, exist_ok=True)

    def get_parent_index(self):
        return self.parent().index

    def before_loop(self, *args, **kwargs):
        pass

    def before_iteration(self, *args, **kwargs):
        pass

    def after_iteration(self, image_tensors):
        if parallel.is_first_process() and (self.index + 1) % self.image_save_frequency == 0:
            grid = self._make_grid_image(image_tensors)
            self._save_image_as_file(grid)
            self._save_image_on_tensorboard(grid)

    def after_loop(self, *args, **kwargs):
        pass

    def _make_grid_image(self, image_tensors):
        ncols_image = min(self.ncols_image, len(image_tensors[0]))
        images = tuple((img_tensor[:ncols_image]
                        if img_tensor.shape[1] == 3
                        else img_tensor[:ncols_image].repeat((1, 3, 1, 1)))
                       for img_tensor in image_tensors)
        return torchvision.utils.make_grid(
            torch.cat(images, dim=0),
            nrow=ncols_image,
            normalize=True,
            scale_each=True
        )

    def _save_image_as_file(self, grid):
        self.result_dir.mkdir(parents=True, exist_ok=True)
        if self.parent(2) is not None:
            title = f'Epoch-{self.parent(2).index + 1}_{self.loop_name}-' \
                    f'{self.parent().index + 1:04}_{self.iter_name}-{self.index + 1:04d}'
        elif self.parent(1) is not None:
            title = f'{self.loop_name}-{self.parent(1).index + 1}_' \
                    f'{self.iter_name}-{self.index + 1:04d}'
        else:
            title = f'{self.loop_name}-{self.parent().index + 1:04}_{self.iter_name}-{self.index + 1:04d}'
        torchvision.utils.save_image(grid, Path(self.result_dir, title + '.png'))

    def _get_tensorboard_writer(self):
        if self.tensorboard_writer is not None:
            return self.tensorboard_writer
        if self.no_parent:
            return None
        return self.parent()._get_tensorboard_writer()

    def _save_image_on_tensorboard(self, grid):
        tensorboard = self._get_tensorboard_writer()
        if tensorboard is not None and self.print_on_tensorboard:
            prefix = (self.prefix + "_") if self.prefix != "" else ""
            if self.parent(2) is not None:
                title = f'{prefix}Epoch-{self.parent(2).index + 1}/{self.loop_name}-' \
                        f'{self.parent().index + 1:04}_{self.iter_name}-{self.index + 1:04d}'
            elif self.parent(1) is not None:
                title = f'{prefix}{self.loop_name}-{self.parent(1).index + 1}/' \
                        f'{self.iter_name}-{self.index + 1:04d}'
            else:
                title = f'{prefix}{self.loop_name}-{self.parent().index + 1:04}_{self.iter_name}-{self.index + 1:04d}'
            tensorboard.print_image(title, grid)

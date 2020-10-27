from typing import Dict, Iterable, Callable, List, Tuple
import math

import torch
import torch.nn.functional as F
from skimage.color import rgb2lab


def l1_loss(output: torch.Tensor, target: torch.Tensor) -> float:
    return torch.mean(F.l1_loss(output, target)).item()


def smooth_l1_loss(output: torch.Tensor, target: torch.Tensor) -> float:
    return torch.mean(F.smooth_l1_loss(output, target)).item()


def mse_loss(output: torch.Tensor, target: torch.Tensor) -> float:
    return torch.mean(F.mse_loss(output, target)).item()


def rmse_loss(output: torch.Tensor, target: torch.Tensor) -> float:
    return torch.mean(torch.sqrt(F.mse_loss(output, target))).item()


def rmse_loss_shadow(output: Tuple[torch.Tensor], target: Tuple[torch.Tensor]) -> float:
    output_free = output[0].detach().clone()
    target_free = target[0].detach().clone()
    target_mask = target[1].detach()

    shadow_part = (target_mask > 0.5).repeat(1, 3, 1, 1)
    ret = (output_free - target_free) ** 2
    return math.sqrt(ret[shadow_part].mean().item())


def rmse_loss_non_shadow(output: Tuple[torch.Tensor], target: Tuple[torch.Tensor]) -> float:
    output_free = output[0].detach().clone()
    target_free = target[0].detach().clone()
    target_mask = target[1].detach()

    non_shadow_part = (target_mask < 0.5).repeat(1, 3, 1, 1)
    ret = (output_free - target_free) ** 2
    return math.sqrt(ret[non_shadow_part].mean().item())


def bce_logit_loss(output: torch.Tensor, target: torch.Tensor) -> float:
    return torch.mean(F.binary_cross_entropy_with_logits(output, target)).item()


def bce_loss(output: torch.Tensor, target: torch.Tensor) -> float:
    return torch.mean(F.binary_cross_entropy(output, target)).item()


def cross_entropy_loss(output: torch.Tensor, target: torch.Tensor) -> float:
    return torch.mean(F.cross_entropy(output, target)).item()


def ber(output: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    batch_size = output.shape[0]
    target_flat = target.view(batch_size, -1) >= threshold
    output_flat = output.view(batch_size, -1) >= threshold
    true = target_flat
    false = torch.logical_not(target_flat)
    positive = output_flat
    negative = torch.logical_not(output_flat)
    TP = torch.sum(torch.as_tensor(torch.logical_and(true, positive), dtype=torch.float), dim=1)
    TN = torch.sum(torch.as_tensor(torch.logical_and(false, negative), dtype=torch.float), dim=1)
    FP = torch.sum(torch.as_tensor(torch.logical_and(false, positive), dtype=torch.float), dim=1)
    FN = torch.sum(torch.as_tensor(torch.logical_and(true, negative), dtype=torch.float), dim=1)

    fp_rate = FP / (TN + FP)
    fn_rate = FN / (FN + TP)

    fp_rate[fp_rate != fp_rate] = .0
    fn_rate[fn_rate != fn_rate] = .0

    return torch.mean(.5 * (fp_rate + fn_rate)).item()


def false_negative_rate(output: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    batch_size = output.shape[0]
    target_flat = target.view(batch_size, -1) >= threshold
    output_flat = output.view(batch_size, -1) >= threshold

    true = target_flat
    positive = output_flat
    negative = torch.logical_not(output_flat)
    TP = torch.sum(torch.as_tensor(torch.logical_and(true, positive), dtype=torch.float), dim=1)
    FN = torch.sum(torch.as_tensor(torch.logical_and(true, negative), dtype=torch.float), dim=1)

    fnr = FN / (FN + TP)
    fnr[fnr != fnr] = .0

    return torch.mean(fnr).item()


def false_positive_rate(output: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    batch_size = output.shape[0]
    target_flat = target.view(batch_size, -1) >= threshold
    output_flat = output.view(batch_size, -1) >= threshold

    false = torch.logical_not(target_flat)
    positive = output_flat
    negative = torch.logical_not(output_flat)
    TN = torch.sum(torch.as_tensor(torch.logical_and(false, negative), dtype=torch.float), dim=1)
    FP = torch.sum(torch.as_tensor(torch.logical_and(false, positive), dtype=torch.float), dim=1)

    fpr = FP / (TN + FP)
    fpr[fpr != fpr] = .0

    return torch.mean(fpr).item()


def accuracy(output: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    batch_size = output.shape[0]
    target_flat = target.view(batch_size, -1) >= threshold
    output_flat = output.view(batch_size, -1) >= threshold
    equal = torch.as_tensor(output_flat == target_flat, dtype=torch.float)

    return torch.mean(torch.mean(equal, dim=1)).item()


def denormalize_image(img, mean, std):
    demean = tuple(-m / s for m, s in zip(mean, std))
    destd = tuple(1 / s for s in std)

    demean = torch.tensor(demean, dtype=torch.float, device=img.device).view(1, -1, 1, 1)
    destd = torch.tensor(destd, dtype=torch.float, device=img.device).view(1, -1, 1, 1)

    return img.sub_(demean).div_(destd)


def rgb_to_lab_space(img):
    to_squeeze = (img.dim() == 3)
    device = img.device

    img = img.detach().cpu()
    if to_squeeze:
        img = img.unsqueeze(0)

    img = img.permute(0, 2, 3, 1).numpy()
    transformed = rgb2lab(img)
    output = torch.from_numpy(transformed).float().permute(0, 3, 1, 2)
    if to_squeeze:
        output = output.squeeze(0)

    return output.to(device)


def create_split_field_and_metric_funcs_dict(splits: Iterable[str], field_metric_funcs_dict: Dict[
    str, Dict[str, Callable[[torch.Tensor, torch.Tensor], float]]]) \
        -> Dict[str, Dict[str, Dict[str, Callable[[torch.Tensor, torch.Tensor], float]]]]:
    return dict(
        (split, dict(
            (f'{split}_{field}', metric_funcs)
            for field, metric_funcs in field_metric_funcs_dict.items()
        )) for split in splits
    )


def create_split_and_criterion_lists_dict(splits: Iterable[str], criterion_names: Iterable[str]) -> Dict[
    str, List[str]]:
    return dict(
        (split, [f'{split}_{criterion_name}' for criterion_name in criterion_names]) for split in splits
    )


def get_avg_metrics_and_losses(metric_containers, criterion_container):
    avg_metrics_and_losses = {}
    avg_losses = criterion_container.calc_averages()
    for field, metric_container in metric_containers.items():
        avg_metrics_and_losses[field] = metric_container.calc_averages()

    for criterion_field, loss in avg_losses.items():
        field = criterion_field[criterion_field.rfind('/') + 1:]
        if field not in avg_metrics_and_losses:
            avg_metrics_and_losses[field] = dict()
        avg_metrics_and_losses[field][criterion_field] = loss

    return avg_metrics_and_losses


class MetricCalculator:
    def __init__(self, metric_funcs, transform=None):
        self.metric_funcs = dict()
        self.transform = transform if transform is not None else (lambda x: x)
        self.add_metric_funcs(metric_funcs)

    def add_metric_func(self, metric_name, metric_func):
        self.metric_funcs[metric_name] = metric_func

    def add_metric_funcs(self, metric_funcs):
        if not isinstance(metric_funcs, dict):
            metric_funcs = dict((func.__name__, func) for func in metric_funcs)

        for name, func in metric_funcs.items():
            self.add_metric_func(name, func)

    def calc_metrics(self, output, target):
        output = self.transform(output)
        target = self.transform(target)
        return dict((name, func(output, target)) for name, func in self.metric_funcs.items())

    def __str__(self):
        nl = '\n'
        return f'transform: {str(self.transform)}' \
               + f'\nfunctions: {nl.join([str(key) for key in self.metric_funcs])}\n'


class MetricContainer:
    def __init__(self, field_name=None):
        self.container = dict()
        self.field_name = field_name

    def initialize(self):
        self.container = dict()

    def append_metric(self, metric_key, metric_value):
        if metric_key in self.container:
            self.container[metric_key].append(metric_value)
        else:
            self.container[metric_key] = [metric_value]

    def append_metrics(self, metrics_dict):
        for key, value in metrics_dict.items():
            self.append_metric(key, value)

    def calc_average(self, metric_key):
        return sum(self.container[metric_key]) / float(len(self.container[metric_key]))

    def calc_averages(self):
        if self.field_name is None:
            return dict((key, self.calc_average(key)) for key in self.container)
        else:
            return dict((f'{self.field_name}/{key}', self.calc_average(key)) for key in self.container)

    def __str__(self):
        return f'field name: {self.field_name}\n' + \
               '\n'.join([str(key) + ": " + str(values) for key, values in self.container]) + '\n'

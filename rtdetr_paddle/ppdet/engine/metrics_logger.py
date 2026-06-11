# Copyright (c) 2024 RT-DETR Alt PE Authors.
#
# Callback to log training losses and COCO evaluation metrics to JSON files.
# Integrates into the PaddleDetection callback system.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import paddle.distributed as dist

from .callbacks import Callback

from ppdet.utils.logger import setup_logger
logger = setup_logger('ppdet.engine')

__all__ = ['MetricsLogger']

# Names for the 12 standard COCO evaluation metrics (indices 0-11)
COCO_EVAL_METRIC_NAMES = [
    'AP',       # AP @ IoU=0.50:0.95, area=all, maxDets=100
    'AP50',     # AP @ IoU=0.50
    'AP75',     # AP @ IoU=0.75
    'APs',      # AP @ IoU=0.50:0.95, area=small
    'APm',      # AP @ IoU=0.50:0.95, area=medium
    'APl',      # AP @ IoU=0.50:0.95, area=large
    'AR1',      # AR @ IoU=0.50:0.95, maxDets=1
    'AR10',     # AR @ IoU=0.50:0.95, maxDets=10
    'AR100',    # AR @ IoU=0.50:0.95, maxDets=100
    'ARs',      # AR @ IoU=0.50:0.95, area=small
    'ARm',      # AR @ IoU=0.50:0.95, area=medium
    'ARl',      # AR @ IoU=0.50:0.95, area=large
]


class MetricsLogger(Callback):
    """
    Callback that logs all training loss components and COCO evaluation metrics
    to JSON files for later analysis and graphing.

    Directory structure created:
        results_dir/
            loss/
                loss.json              # [val1, val2, ...] per step
                loss_class.json
                loss_bbox.json
                ...
            eval/
                AP.json                # [val_epoch1, val_epoch2, ...]
                AP50.json
                ...

    Each loss JSON contains one value per logged step (every log_iter steps).
    Each eval JSON contains one value per evaluation epoch.
    """

    def __init__(self, model, results_dir):
        super(MetricsLogger, self).__init__(model)
        self.results_dir = results_dir
        self.loss_dir = os.path.join(results_dir, 'loss')
        self.eval_dir = os.path.join(results_dir, 'eval')
        os.makedirs(self.loss_dir, exist_ok=True)
        os.makedirs(self.eval_dir, exist_ok=True)

        # In-memory accumulators
        self.loss_data = {}           # {loss_name: [values]}
        self.eval_data = {}           # {metric_name: [values]}

        # Track per-epoch loss accumulation
        self._epoch_loss_sums = {}    # {loss_name: sum}
        self._epoch_loss_counts = {}  # {loss_name: count}

        # Load existing data if resuming
        self._load_existing()

        logger.info(f"MetricsLogger: saving results to {results_dir}")

    def _load_existing(self):
        """Load previously saved JSON files to allow resuming."""
        for fname in os.listdir(self.loss_dir):
            if fname.endswith('.json'):
                key = fname[:-5]  # strip .json
                fpath = os.path.join(self.loss_dir, fname)
                try:
                    with open(fpath, 'r') as f:
                        self.loss_data[key] = json.load(f)
                except (json.JSONDecodeError, IOError):
                    self.loss_data[key] = []

        for fname in os.listdir(self.eval_dir):
            if fname.endswith('.json'):
                key = fname[:-5]
                fpath = os.path.join(self.eval_dir, fname)
                try:
                    with open(fpath, 'r') as f:
                        self.eval_data[key] = json.load(f)
                except (json.JSONDecodeError, IOError):
                    self.eval_data[key] = []

    def _save_loss(self):
        """Write all loss data to JSON files."""
        for key, values in self.loss_data.items():
            fpath = os.path.join(self.loss_dir, f'{key}.json')
            with open(fpath, 'w') as f:
                json.dump(values, f)

    def _save_eval(self):
        """Write all eval data to JSON files."""
        for key, values in self.eval_data.items():
            fpath = os.path.join(self.eval_dir, f'{key}.json')
            with open(fpath, 'w') as f:
                json.dump(values, f)

    def on_step_end(self, status):
        """Accumulate loss values for the current epoch."""
        if dist.get_world_size() >= 2 and dist.get_rank() != 0:
            return

        mode = status['mode']
        if mode != 'train':
            return

        training_status = status.get('training_status', None)
        if training_status is None or training_status.meters is None:
            return

        # Accumulate raw values from SmoothedValue deques
        for key, sv in training_status.meters.items():
            if key not in self._epoch_loss_sums:
                self._epoch_loss_sums[key] = 0.0
                self._epoch_loss_counts[key] = 0
            self._epoch_loss_sums[key] += float(sv.value)
            self._epoch_loss_counts[key] += 1

    def on_epoch_end(self, status):
        """Save epoch-average losses and COCO eval metrics."""
        if dist.get_world_size() >= 2 and dist.get_rank() != 0:
            return

        mode = status['mode']

        if mode == 'train':
            # Compute epoch averages and append
            for key in self._epoch_loss_sums:
                count = self._epoch_loss_counts[key]
                if count > 0:
                    avg = self._epoch_loss_sums[key] / count
                else:
                    avg = 0.0
                if key not in self.loss_data:
                    self.loss_data[key] = []
                self.loss_data[key].append(round(avg, 6))

            # Reset accumulators
            self._epoch_loss_sums = {}
            self._epoch_loss_counts = {}

            # Write to disk
            self._save_loss()

        elif mode == 'eval':
            # Extract COCO metrics from the metric objects
            for metric in self.model._metrics:
                results = metric.get_results()
                if 'bbox' in results:
                    bbox_stats = results['bbox']  # numpy array of 12 floats
                    for i, name in enumerate(COCO_EVAL_METRIC_NAMES):
                        if i < len(bbox_stats):
                            if name not in self.eval_data:
                                self.eval_data[name] = []
                            self.eval_data[name].append(
                                round(float(bbox_stats[i]), 6))

            # Write to disk
            self._save_eval()

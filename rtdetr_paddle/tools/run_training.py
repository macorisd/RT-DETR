#!/usr/bin/env python3
"""
Run RT-DETR training with different positional encoding wave functions
and log all metrics (losses + COCO eval) to structured JSON files.

Results are saved to:
    rtdetr_paddle/results/results_[timestamp]/<wave_name>/loss/*.json
    rtdetr_paddle/results/results_[timestamp]/<wave_name>/eval/*.json

Usage examples:
    # Train all 4 waves sequentially with RT-DETR-R18 (default)
    python tools/run_training.py

    # Train only sinusoid and triangular
    python tools/run_training.py --functions sinusoid,triangular

    # Use a different model config
    python tools/run_training.py --config configs/rtdetr/rtdetr_r50vd_6x_coco.yml

    # Resume into an existing results directory
    python tools/run_training.py --resume results/results_20260219_120000

    # Train with AMP enabled and custom epochs
    python tools/run_training.py --amp --epoch 12 --functions sinusoid
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
from datetime import datetime

# Add parent path so ppdet is importable
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

import warnings
warnings.filterwarnings('ignore')

import paddle

from ppdet.core.workspace import load_config, merge_config
from ppdet.engine import Trainer, init_parallel_env, set_random_seed, init_fleet_env
from ppdet.engine.metrics_logger import MetricsLogger
from ppdet.utils.cli import merge_args
import ppdet.utils.check as check
from ppdet.utils.logger import setup_logger

logger = setup_logger('run_training')

ALL_FUNCTIONS = ['sinusoid', 'triangular', 'square', 'sawtooth']

DEFAULT_CONFIG = 'configs/rtdetr/rtdetr_r18vd_6x_coco.yml'


def parse_args():
    parser = argparse.ArgumentParser(
        description='RT-DETR training launcher with alternative positional encoding functions')

    parser.add_argument(
        '-c', '--config', type=str, default=DEFAULT_CONFIG,
        help=f'Model config file (default: {DEFAULT_CONFIG})')
    parser.add_argument(
        '--functions', type=str, default=','.join(ALL_FUNCTIONS),
        help='Comma-separated wave functions to train '
             f'(default: {",".join(ALL_FUNCTIONS)})')
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Path to an existing results_[timestamp] directory to resume into. '
             'Waves that already have eval data will be skipped.')
    parser.add_argument(
        '--epoch', type=int, default=None,
        help='Override number of training epochs')
    parser.add_argument(
        '--eval', action='store_true', default=True,
        help='Run COCO evaluation at the end of each epoch (default: True)')
    parser.add_argument(
        '--no-eval', dest='eval', action='store_false',
        help='Disable per-epoch evaluation')
    parser.add_argument(
        '--amp', action='store_true', default=False,
        help='Enable automatic mixed precision')
    parser.add_argument(
        '--fleet', action='store_true', default=False,
        help='Use fleet for distributed training')
    parser.add_argument(
        '-o', '--opt', nargs='*', default=[],
        help='Extra PaddleDetection config overrides (e.g. TrainReader.batch_size=2)')

    args = parser.parse_args()
    args.functions = [f.strip() for f in args.functions.split(',')]
    for fn in args.functions:
        if fn not in ALL_FUNCTIONS:
            parser.error(f"Unknown function '{fn}'. Choose from: {ALL_FUNCTIONS}")
    return args


def _parse_opt(opts):
    """Parse -o key=value pairs into a config dict, same logic as ArgsParser."""
    import yaml
    config = {}
    if not opts:
        return config
    for s in opts:
        s = s.strip()
        k, v = s.split('=', 1)
        if '.' not in k:
            config[k] = yaml.load(v, Loader=yaml.Loader)
        else:
            keys = k.split('.')
            if keys[0] not in config:
                config[keys[0]] = {}
            cur = config[keys[0]]
            for idx, key in enumerate(keys[1:]):
                if idx == len(keys) - 2:
                    cur[key] = yaml.load(v, Loader=yaml.Loader)
                else:
                    cur[key] = {}
                    cur = cur[key]
    return config


def wave_is_complete(wave_dir):
    """Check if a wave directory already has evaluation results."""
    eval_dir = os.path.join(wave_dir, 'eval')
    if not os.path.isdir(eval_dir):
        return False
    json_files = [f for f in os.listdir(eval_dir) if f.endswith('.json')]
    if not json_files:
        return False
    # Check that the AP.json has at least 1 entry
    ap_file = os.path.join(eval_dir, 'AP.json')
    if os.path.exists(ap_file):
        import json
        try:
            with open(ap_file) as f:
                data = json.load(f)
            return len(data) > 0
        except Exception:
            return False
    return False


def train_single_wave(args, wave_func, results_dir):
    """
    Run one full training for a specific wave function.
    """
    wave_dir = os.path.join(results_dir, wave_func)

    if args.resume and wave_is_complete(wave_dir):
        logger.info(f"[{wave_func}] Already has evaluation results, skipping.")
        return

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Starting training: {wave_func}")
    logger.info(f"Results dir: {wave_dir}")
    logger.info(f"{'=' * 60}\n")

    os.makedirs(wave_dir, exist_ok=True)

    # Load config fresh for each wave
    cfg = load_config(args.config)

    # Set wave function
    opt_wave = {'HybridEncoder': {'periodic_func': wave_func}}
    merge_config(opt_wave)

    # Apply user-specified -o overrides
    user_opt = _parse_opt(args.opt)
    if user_opt:
        merge_config(user_opt)

    # Apply epoch override
    if args.epoch is not None:
        cfg.epoch = args.epoch

    # Apply AMP
    if args.amp:
        cfg.amp = True

    # Apply fleet
    if args.fleet:
        cfg.fleet = True

    # Standard settings
    if 'use_npu' not in cfg:
        cfg.use_npu = False
    if 'use_xpu' not in cfg:
        cfg.use_xpu = False
    if 'use_gpu' not in cfg:
        cfg.use_gpu = False
    if 'use_mlu' not in cfg:
        cfg.use_mlu = False

    # Merge again to ensure wave_func sticks after all merges
    merge_config(opt_wave)

    # Device
    if cfg.use_gpu:
        paddle.set_device('gpu')
    else:
        paddle.set_device('cpu')

    check.check_config(cfg)
    check.check_gpu(cfg.use_gpu)
    check.check_version()

    # Set save_dir so model weights go into the wave directory
    cfg.save_dir = wave_dir

    # Init parallel env
    if cfg.get('fleet', False):
        init_fleet_env(cfg.get('find_unused_parameters', False))
    else:
        init_parallel_env()

    # Build trainer
    trainer = Trainer(cfg, mode='train')

    # Load pretrained weights
    if 'pretrain_weights' in cfg and cfg.pretrain_weights:
        trainer.load_weights(cfg.pretrain_weights)

    # Register our MetricsLogger callback
    metrics_cb = MetricsLogger(trainer, wave_dir)
    trainer.register_callbacks([metrics_cb])

    # Train
    trainer.train(args.eval)

    logger.info(f"\n[{wave_func}] Training complete. Results in {wave_dir}\n")


def main():
    args = parse_args()

    # Determine results directory
    if args.resume:
        results_dir = args.resume
        if not os.path.isdir(results_dir):
            logger.error(f"Resume directory does not exist: {results_dir}")
            sys.exit(1)
        logger.info(f"Resuming into: {results_dir}")
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_results = os.path.join(parent_path, 'results')
        results_dir = os.path.join(base_results, f'results_{timestamp}')
        os.makedirs(results_dir, exist_ok=True)
        logger.info(f"New results directory: {results_dir}")

    logger.info(f"Wave functions to train: {args.functions}")
    logger.info(f"Config: {args.config}")
    if args.epoch:
        logger.info(f"Epochs: {args.epoch}")
    logger.info(f"Eval each epoch: {args.eval}")
    logger.info(f"AMP: {args.amp}")
    logger.info("")

    for wave_func in args.functions:
        train_single_wave(args, wave_func, results_dir)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"All training runs complete.")
    logger.info(f"Results saved in: {results_dir}")
    logger.info(f"Run `python tools/graph.py --results-dir {results_dir}` to generate graphs.")
    logger.info(f"{'=' * 60}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3

import argparse
import logging

from digit_pair_detector.trainer import Trainer, TrainerConfig


def parse_args():
    """
    parse user arguments

    :return: parsed user arguments
    """
    parser = argparse.ArgumentParser(description='Run PyTorch segmentation model training',
                                     formatter_class=argparse.HelpFormatter)
    parser.add_argument('--cfg', '-c', type=str, required=True,
                        help='Training configuration file.')
    parser.add_argument('--output-folder', '-o', type=str, required=True,
                        help='Output destination_folder to save intermediate data and the final result.')
    parser.add_argument('--job-name', '-j', type=str, required=False, default='',
                        help='Job name of the experiment. '
                             'Will be the destination_folder with a timestamp where output is saved.')

    opts = parser.parse_args()

    return opts


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    args = parse_args()
    logging.info(args)
    cfg = TrainerConfig.from_file(args.cfg)
    trainer = Trainer(cfg=cfg, output_folder=args.output_folder, job_name=args.job_name)
    trainer.run()


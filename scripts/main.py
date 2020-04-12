import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from deephsb.engine import Trainer
from deephsb.config import get_cfg_defaults

def get_args():
    parser = argparse.ArgumentParser(description="Human Behavior Prediction")
    parser.add_argument('--config-file', type=str, help='configuration file (yaml)')
    parser.add_argument('--output', type=str, help='output path')
    parser.add_argument('--mode', type=str, help='running mode')
    args = parser.parse_args()
    return args

def main():
    # load configurations
    args = get_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    print(cfg)    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    trainer = Trainer(cfg, device, args.output)
    if args.mode == 'cross_validate':
        trainer.cross_validate()
    elif args.mode == 'inference':
        trainer.inference()
    else:
        raise NotImplementedError

if __name__ == '__main__':
    main()

import os
import argparse
from importlib import import_module
import utils.option as option

parser = argparse.ArgumentParser()
parser.add_argument('--lab', default='sequential', type=str) #, required=True)
parser.add_argument('--instance', default='sequential_heightmap', type=str) #, required=True)
parser.add_argument('--distributed', action='store_true', default=False)
parser.add_argument('--mp', type=str, required=False)
parser.add_argument('--master_port', type=int, required=False)
args = parser.parse_args()

param = import_module(f'laboratory.{args.lab}.{args.instance}.param')
args_to_pass = option.params_to_argv(param.get_params(args.lab, args.instance, args.distributed))

if not args.distributed:
    train = import_module(f'laboratory.{args.lab}.{args.instance}.train')
    train.main(args_to_pass)
else:
    mp = f'--mp "{args.mp}"' if args.mp is not None else ''
    master_port = f'--master_port {str(args.master_port)}' if args.master_port is not None else ''
    os.system(
        f'PYTHONPATH={os.getcwd()} python utils/distributed.py {mp} {master_port} '
        f'laboratory/{args.lab}/{args.instance}/train.py {" ".join(args_to_pass)}'
    )

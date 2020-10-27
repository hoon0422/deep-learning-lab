import subprocess
import os
import sys
from argparse import ArgumentParser, REMAINDER
from typing import Union, List


class Process:
    size = '0'

    def __init__(self, address: str, address_rank: Union[str, int], device: Union[str, int]):
        self.address = address if address != 'localhost' else '127.0.0.1'
        self.address_rank = str(address_rank)
        self.device = str(device)
        self.rank = ''
        self.local_rank = ''

    def set_rank(self, rank: Union[str, int]):
        if self.rank != '':
            raise ValueError(f'Rank is already set to {self.rank}. Cannot change it.')
        self.rank = str(rank)
        Process.size = str(int(Process.size) + 1)

    def set_local_rank(self, local_rank: Union[str, int]):
        if self.local_rank != '':
            raise ValueError(f'Local rank is already set to {self.local_rank}. Cannot change it.')
        self.local_rank = str(local_rank)

    def __eq__(self, other):
        if not isinstance(other, Process):
            return False
        return self.address == other.address and self.address_rank == other.address_rank and self.device == other.device

    def __hash__(self):
        return hash((self.address, self.address_rank, self.device))


class Node:
    def __init__(self, address: str, address_rank: Union[str, int]):
        self.address = address
        self.address_rank = address_rank
        self._processes: List[Process] = []

    def devices_to_processes(self, devices: List[str]):
        self._processes = [Process(self.address, self.address_rank, device) for device in devices]
        for lr, p in enumerate(self._processes):
            p.set_local_rank(lr)

    @property
    def processes(self):
        return self._processes

    def __getitem__(self, item) -> Process:
        return self._processes[item]

    def __len__(self):
        return len(self._processes)

    def __iter__(self):
        return self._processes.__iter__()

    def __eq__(self, other):
        return isinstance(other, Node) and other.address == self.address and other.address_rank == self.address_rank

    def __hash__(self):
        return hash((self.address, self.address_rank))


def parse_args():
    parser = ArgumentParser(description="PyTorch distributed training launch "
                                        "helper utility that will spawn up "
                                        "multiple distributed processes")

    # server1/0(0,1):server1/1(2):server2(0,1):server3(1):...
    parser.add_argument("--mp", type=str, default="127.0.0.1/0(0)")
    parser.add_argument("--master_port", type=int, default=29500)

    # positional
    parser.add_argument("training_script", type=str,
                        help="The full path to the single GPU training "
                             "program/script to be launched in parallel, "
                             "followed by all the arguments for the "
                             "training script")

    # rest from the training program
    parser.add_argument('training_script_args', nargs=REMAINDER)
    return parser.parse_args()


def parse_node(node_str) -> Node:
    node_str = ''.join(node_str.split())
    address, devices = node_str[:-1].split('(')
    address, address_rank = address.split('/') if '/' in address else (address, '0')

    node = Node(address, address_rank)
    node.devices_to_processes(devices.split(','))

    return node


def is_my_ip_addr(ip_addr: str) -> bool:
    f = os.popen('ifconfig | grep "inet\ "')
    for ip_info in f.readlines():
        ip_info = ip_info.strip().split(' ')
        if ip_info[1] == ip_addr:
            return True
    return False


def get_current_node(nodes: List[Node]) -> Node:
    for node in nodes:
        if is_my_ip_addr(node.address):
            return node
    raise ValueError(f'Cannot find process with current ip address')


def parse_server_and_devices(server_and_devices_str: str) -> List[Node]:
    node_str_list = server_and_devices_str.split(':')
    nodes = [parse_node(node_str) for node_str in node_str_list]

    rank = 0
    for node in nodes:
        for process in node:
            process.set_rank(rank)
            rank += 1

    return nodes


def main():
    args = parse_args()
    servers_and_devices = args.mp
    master_port = args.master_port
    nodes = parse_server_and_devices(servers_and_devices)

    # if 'OMP_NUM_THREADS' not in os.environ and args.nproc_per_node > 1:
    #     current_env["OMP_NUM_THREADS"] = str(1)
    #     print("*****************************************\n"
    #           "Setting OMP_NUM_THREADS environment variable for each process "
    #           "to be {} in default, to avoid your system being overloaded, "
    #           "please further tune the variable for optimal performance in "
    #           "your application as needed. \n"
    #           "*****************************************".format(current_env["OMP_NUM_THREADS"]))

    cmd = [sys.executable, "-u", args.training_script]
    cmd.extend(args.training_script_args)
    node = get_current_node(nodes)
    node_processes = []

    # set PyTorch distributed related environmental variables
    current_env = os.environ.copy()
    current_env["MASTER_ADDR"] = nodes[0].address
    current_env["MASTER_PORT"] = str(master_port)
    current_env["WORLD_SIZE"] = str(Process.size)
    for process in node:
        # each process's rank
        current_env["RANK"] = process.rank
        current_env["LOCAL_RANK"] = process.local_rank
        current_env["DEVICE_ID"] = process.device

        # spawn the processes
        node_processes.append(subprocess.Popen(cmd.copy(), env=current_env))

    for process in node_processes:
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd)


if __name__ == "__main__":
    main()

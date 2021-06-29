import argparse
import os
import logging
import time
import math

from datetime import datetime
from functools import partial
from multiprocessing import Pool, Process, Queue, log_to_stderr

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models

try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
    from nvidia.dali.pipeline import pipeline_def
    import nvidia.dali.types as types
    import nvidia.dali.fn as fn
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")

def parse():
    model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('data', metavar='DIR', nargs='*',
                        help='path(s) to dataset (if one path is provided, it is assumed\n' +
                       'to have subdirectories named "train" and "val"; alternatively,\n' +
                       'train and val paths can be specified directly by providing both paths as arguments)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
    parser.add_argument('--process', default=4, type=int, metavar='N',
                        help='number of data loading processes (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size per process (default: 256)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--dali_cpu', action='store_true',
                        help='Runs CPU based version of DALI pipeline.')

    parser.add_argument("--local_rank", default=0, type=int)

    parser.add_argument('-t', '--test', action='store_true',
                        help='Launch test mode with preset arguments')
    args = parser.parse_args()
    return args

@pipeline_def
def create_dali_pipeline(data_dir, crop, shard_id, num_shards, dali_cpu=True, is_training=True):
    images, labels = fn.readers.file(file_root=data_dir,
                                     shard_id=shard_id,
                                     num_shards=num_shards,
                                     random_shuffle=is_training,
                                     pad_last_batch=True,
                                     name="Reader")
    dali_device = 'cpu' if dali_cpu else 'gpu'
    decoder_device = 'cpu' if dali_cpu else 'mixed'
    device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
    host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
    print('Dali device is {}, decoder device is {}'.format(dali_device, decoder_device))
    images = fn.decoders.image_random_crop(images,
                                           device=decoder_device, output_type=types.RGB,
                                           device_memory_padding=device_memory_padding,
                                           host_memory_padding=host_memory_padding,
                                           random_aspect_ratio=[0.8, 1.25],
                                           random_area=[0.1, 1.0],
                                           num_attempts=100)
    images = fn.resize(images,
                       device=dali_device,
                       resize_x=crop,
                       resize_y=crop,
                       interp_type=types.INTERP_TRIANGULAR)
    mirror = fn.random.coin_flip(probability=0.5)

    images = fn.crop_mirror_normalize(images,
                                      device=dali_device,
                                      dtype=types.FLOAT,
                                      output_layout="CHW",
                                      crop=(crop, crop),
                                      mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                      std=[0.229 * 255,0.224 * 255,0.225 * 255],
                                      mirror=mirror)
    if dali_device == 'gpu':
        labels = labels.gpu()
    return images, labels


def main():
    global args
    args = parse()

    if not len(args.data):
        raise Exception("error: No data set provided")

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    global_rank = 0
    if 'RANK' in os.environ:
        global_rank = int(os.environ['RANK'])

    if 'MASTER_ADDR' in os.environ:
        print('Master address is {}'.format(os.environ['MASTER_ADDR']))
    if 'MASTER_PORT' in os.environ:
        print('Master port is {}'.format(os.environ['MASTER_PORT']))
    args.world_size = 1
    if args.distributed:
        print('Inits distributed process group with gloo backend')
        # Distributed information will be passed in through environment variable WORLD_SIZE and RANK
        torch.distributed.init_process_group(backend='gloo',
                                             init_method='env://')
        print('Process group inited')
        args.world_size = torch.distributed.get_world_size()
    
    if len(args.data) == 1:
        traindir = os.path.join(args.data[0], 'train')
    else:
        traindir = args.data[0]

    if(args.arch == "inception_v3"):
        raise RuntimeError("Currently, inception_v3 is not supported by this example.")
        # crop_size = 299
    else:
        crop_size = 224

    num_shards = args.world_size * args.process
    shard_id = range(global_rank * args.process, (global_rank + 1) * args.process)
    print('Parameters: world_size[{}], global_rank[{}], batch_size[{}], processes[{}], '
          'num_shards[{}], current_shard_id[{}]'.format(args.world_size, global_rank, args.batch_size,
                                                args.process, num_shards, shard_id))
    log_to_stderr(logging.DEBUG)
    pool = Pool(processes=args.process)
    dali_func = partial(dali, args.batch_size, traindir, args.print_freq, crop_size, args.dali_cpu, num_shards)
    result = pool.map(dali_func, shard_id)
    print(result)

def dali(batch_size, traindir, print_freq, crop_size, dali_cpu, num_shards, shard_id):
    print('Parameters: batch_size[{}], traindir[{}], print_freq[{}], crop_size[{}], dali_cpu[{}], '
          'num_shards[{}], current_shard_id[{}], starting at[{}]'.format(batch_size, traindir, print_freq,
                                                        crop_size, dali_cpu, num_shards, shard_id, datetime.now().time()))
    pipe = create_dali_pipeline(batch_size=batch_size,
                                num_threads=1,
                                seed=12 + args.local_rank,
                                data_dir=traindir,
                                crop=crop_size,
                                dali_cpu=dali_cpu,
                                shard_id=shard_id,
                                num_shards=num_shards,
                                device_id=-99999)
    pipe.build()

    train_loader = DALIClassificationIterator(pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)

    total_time = AverageMeter()
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        total_train_time = train(train_loader, epoch, batch_size, print_freq, shard_id)
        print('Total train time is {}'.format(total_train_time))
        total_time.update(total_train_time)
        if args.test:
            break

    train_loader.reset()
    return total_time.sum

def train(train_loader, epoch, batch_size, print_freq, shard_id):
    batch_time = AverageMeter()
    train_loader_len = int(math.ceil(train_loader._size / batch_size))
    start = time.time()
    end = time.time()

    for i, data in enumerate(train_loader):
        input = data[0]["data"]
        target = data[0]["label"].squeeze(-1).long()
        if i%print_freq == 0:
            batch_time.update((time.time() - end)/print_freq)
            end = time.time()

            if shard_id == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Batch time {3}\t'.format(
                    epoch, i, train_loader_len,
                    batch_time.avg))
        # use the time.sleep to replace the actually training logics
        time.sleep(0.3)

    duration = time.time() - start
    print('End time is {}, Train loader size is {}, total time is {}, Image/s for this node is {}'
          .format(datetime.now().time(), train_loader._size, duration, train_loader._size / duration))
    return duration

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()

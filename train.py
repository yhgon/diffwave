# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import warnings
warnings.simplefilter("ignore", UserWarning)

import argparse
from torch.cuda import device_count
from torch.multiprocessing import spawn

from learner import train, train_distributed
from params import params


def parse_args(parser):
  """
  Parse commandline arguments.
  """
  parser.add_argument('-c', '--config',       type=str, required=True, help='configure file')  
  parser.add_argument('-o', '--model_dir', type=str, default='./log', help='directory in which to store model checkpoints and training logs')  
  parser.add_argument('-w', '--wav_dir', type=str, default='./',  help='Path to dataset wav files')
  parser.add_argument('-m', '--mel_dir', type=str, default='./',  help='Path to dataset mel files')    
  parser.add_argument('--max_steps', default=None, type=int,      help='maximum number of training steps')
  parser.add_argument('--fp16', action='store_true', default=False,      help='use 16-bit floating point operations for training')
    

  training = parser.add_argument_group('training setup')
  training.add_argument('--epochs',                     type=int, required=True,        help='Number of total epochs to run')
  training.add_argument('--epochs-per-checkpoint',      type=int, required=True,         help='Number of epochs per checkpoint')
  training.add_argument('--checkpoint-path',            type=str, default=None,         help='Checkpoint path to resume training')
  training.add_argument('--resume',                               action='store_true',  help='Resume training from the last available checkpoint')
  training.add_argument('--seed',                        type=int,  default=1234,        help='Seed for PyTorch random number generators')
  training.add_argument('--amp',                                     action='store_true', help='Enable AMP')
  training.add_argument('--cuda',                                    action='store_true',   help='Run on GPU using CUDA')
  training.add_argument('--cudnn-enabled',                           action='store_true',   help='Enable cudnn')
  training.add_argument('--cudnn-benchmark',                         action='store_true',   help='Run cudnn benchmark')
  training.add_argument('--ema-decay',                   type=float, default=0, help='Discounting factor for training weights EMA')
  training.add_argument('--gradient-accumulation-steps', type=int,   default=1,   help='Training steps to accumulate gradients for')

  optimization = parser.add_argument_group('optimization setup')
  optimization.add_argument('--optimizer',                 type=str,     default='lamb',   help='Optimization algorithm')
  optimization.add_argument('-lr', '--learning-rate',      type=float,   required=True,    help='Learing rate')
  optimization.add_argument('--weight-decay',               type=float,  default=1e-6,     help='Weight decay')
  optimization.add_argument('--grad-clip-thresh',           type=float,  default=1000.0,   help='Clip threshold for gradients')
  optimization.add_argument('-bs', '--batch-size',          type=int,    required=True,    help='Batch size per GPU')
  optimization.add_argument('--warmup-steps',               type=int,    default=1000,     help='Number of steps for lr warmup')
  optimization.add_argument('--dur-predictor-loss-scale',   type=float,  default=1.0,      help='Rescale duration predictor loss')
  optimization.add_argument('--pitch-predictor-loss-scale', type=float,  default=1.0,      help='Rescale pitch predictor loss')

  #dataset = parser.add_argument_group('dataset parameters')
  #dataset.add_argument('--training-files', type=str, required=True,  help='Path to training filelist')
  #dataset.add_argument('--validation-files', type=str, required=True,  help='Path to validation filelist')

  distributed = parser.add_argument_group('distributed setup')
  distributed.add_argument('--local_rank', type=int,    default=os.getenv('LOCAL_RANK', 0),    help='Rank of the process for multiproc. Do not set manually.')
  distributed.add_argument('--world_size', type=int,    default=os.getenv('WORLD_SIZE', 1),    help='Number of processes for multiproc. Do not set manually.')
  return parser   

def _get_free_port():
  import socketserver
  with socketserver.TCPServer(('localhost', 0), None) as s:
    return s.server_address[1]


def main(args, config):
  replica_count = device_count()
  if replica_count > 1:
    if params.batch_size % replica_count != 0:
      raise ValueError(f'Batch size {params.batch_size} is not evenly divisble by # GPUs {replica_count}.')
    params.batch_size = params.batch_size // replica_count
    port = _get_free_port()
    print("DEBUG: multiGPU run")
    spawn(train_distributed, args=(replica_count, port, args, params), nprocs=replica_count, join=True)
  else:
    print("DEBUG: single GPU run")
    train(args, params)


if __name__ == '__main__': 

  parser = argparse.ArgumentParser(description='PyTorch DiffWave Training',  allow_abbrev=False) 
  parser = parse_args(parser)
  args, _ = parser.parse_known_args()

  ### additional configuration from config file 
  with open(args.config) as f:
      config = ConfigWrapper(**json.load(f))
      
  print("DEBUG", args, config)

  main(args, config)
    
  

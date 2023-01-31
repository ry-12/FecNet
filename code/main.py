import argparse
from run import Run

import warnings
warnings.filterwarnings('ignore')

def create_parser():
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--device', default='cpu', type=str, help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--res_dir', default='./results', type=str)
    parser.add_argument('--ex_name', default='Debug', type=str)
    parser.add_argument('--use_gpu', default=False, type=bool)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=1, type=int)

    # dataset parameters
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size')
    parser.add_argument('--val_batch_size', default=4, type=int, help='Batch size')
    parser.add_argument('--data_root', default=r'E:\py\.py\moving_mnist')
    parser.add_argument('--num_workers', default=0, type=int)

    # model parameters
    parser.add_argument('--num_hidden', default=[64, 64, 64, 64], type=int, nargs='*')
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--img_channel', default=1, type=int)
    parser.add_argument('--filter_size', default=5, type=int)
    parser.add_argument('--stride', default=1, type=int)
    parser.add_argument('--sr_size', default=4, type=int)
    parser.add_argument('--total_length', default=20, type=int)
    parser.add_argument('--input_length', default=10, type=int)
    parser.add_argument('--img_height', default=64, type=int)
    parser.add_argument('--img_width', default=64, type=int)

    # Training parameters
    parser.add_argument('--epochs', default=240, type=int)
    parser.add_argument('--T_max', default=260, type=int)
    parser.add_argument('--log_step', default=1, type=int)
    parser.add_argument('--lr', default=0.0005, type=float, help='Learning rate')
    parser.add_argument('--eta', default=1.0, type=float)
    parser.add_argument('--iter', default=80000, type=int)


    return parser


if __name__ == '__main__':
    args = create_parser().parse_args()
    config = args.__dict__

    run = Run(args)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>  start <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    run.train(args)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>> testing <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    mse = run.test(args)

import time
import argparse
import os
import logging
from models.TripletEmbedding import TripletNet

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def parase_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')

    parser.add_argument('--photo_root', type=str, default='/home/bc/Work/caffe2torch/data/dataset/photo-train-all', help='Training photo root')
    parser.add_argument('--sketch_root', type=str, default='/home/bc/Work/caffe2torch/data/dataset/sketch-triplet-train-all', help='Training sketch root')

    parser.add_argument('--batch_size', type=int, default=16, help='The size of batch (default :16')
    parser.add_argument('--test_batch_size', type=int, default=32, help='The size of batch (default :16')
    parser.add_argument('--device', type=str, default='0', help='The cuda device to be used (default: 0)')
    parser.add_argument('--epochs', type=int, default=10000, help='The number of epochs to run (default: 1000)')
    parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>,<save_latest_freq>+<epoch_count>...')
    parser.add_argument('--niter', type=int, default=1000, help='# of iter at starting learning rate')
    parser.add_argument('--niter_decay', type=int, default=9000, help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--lr', type=float, default=1e-5, help='The learning rate of the model')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Adm weight decay')
    parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau')
    parser.add_argument('--test_freq', type=int, default=50, help='test frequency')

    parser.add_argument('--only_save_best', type=str2bool, nargs='?', default=False, help='is only save best model')
    parser.add_argument('--photo_test', type=str, default='/home/bc/Work/caffe2torch/data/dataset/photo-test-all', help='Testing photo root')
    parser.add_argument('--sketch_test', type=str, default= '/home/bc/Work/caffe2torch/data/dataset/sketch-triplet-test-all',help='Testing sketch root')

    parser.add_argument('--save_dir', type=str, default='/home/bc/Work/caffe2torch/checkpoints', help='The folder to save the model status')
    parser.add_argument('--log_dir', type=str, default='/home/bc/Work/caffe2torch/logs', help='The folder to save the model logs')
    parser.add_argument('--log_name', type=str, default='logs.txt', help='The file to save the model logs')

    parser.add_argument('--vis', type=str2bool, nargs='?', default=True, help='Whether to visualize')
    parser.add_argument('--env', type=str, default='caffe2torch_tripletloss', help='The visualization environment')

    parser.add_argument('--fine_tune', type=str2bool, nargs='?', default=False, help='Whether to fine tune')
    parser.add_argument('--model_root', type=str, default=None, help='The model status files\'s root')

    parser.add_argument('--margin', type=float, default=0.3, help='The margin of the triplet loss')
    parser.add_argument('--p', type=int, default=2, help='The p of the triplet loss')

    parser.add_argument('--net', type=str, default='vgg16', help='The model to be used (vgg16, resnet34, resnet50)')
    parser.add_argument('--cat', type=str2bool, nargs='?', default=True, help='Whether to use category loss')
    parser.add_argument('--weight_cat', type=float, default=1.0, help='Whether to use category loss')
    parser.add_argument('--weight_tri', type=float, default=10.0, help='Whether to use triplet loss')
    parser.add_argument('--weight_mut', type=float, default=20.0, help='Whether to use mutual loss')

    parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
    parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
    parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')

    return check_args(parser.parse_args())

def check_args(args):
    args.save_dir = os.path.join(args.save_dir, args.net + '_' + args.name)
    save_photo_dir = os.path.join(args.save_dir, 'photo')
    save_sketch_dir = os.path.join(args.save_dir, 'sketch')
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
        os.mkdir(save_photo_dir)
        os.mkdir(save_sketch_dir)

    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    args.log_name = args.net + '_' + args.name + '_' + args.log_name

    str_ids = args.device.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)

    try:
        assert args.epochs >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    try:
        assert args.net in ['vgg16', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
    except:
        print('net model must be chose from [\'vgg16\', \'resnet34\', \'resnet50\', \'resnet101\', \'resnet152\']')

    if args.fine_tune:
        try:
            assert not args.model_root
        except:
            print('you should specify the model status file')

    return args

def main():

    args = parase_args()
    if args is None:
        exit()

    # First，create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # log level
    # Second, create a handler, for write to log file
    log_name = os.path.join(args.log_dir, args.log_name)
    fh = logging.FileHandler(log_name, mode='w')
    fh.setLevel(logging.INFO)  # log file log level
    # Third, define handler formatter
    formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    fh.setFormatter(formatter)
    # Fourth,
    logger.addHandler(fh)

    opts = vars(args)
    print('------------ Options -------------')
    logger.info('------------ Options -------------')
    for k, v in sorted(opts.items()):
        print('%s=%s' % (str(k), str(v)))
        logger.info('%s=%s' % (str(k), str(v)))
    print('-------------- End ----------------')
    logger.info('-------------- End ----------------')

    args.logger = logger
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    print('start time: ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    triplet_net = TripletNet(args)
    triplet_net.run()


if __name__ == '__main__':
    main()
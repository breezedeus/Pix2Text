from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os


class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # basic experiment setting
        self.parser.add_argument('task', default='ctdet',
                                 help='ctdet | ddd | multi_pose | exdet | ctdet_subfield')
        self.parser.add_argument('--dataset', default='huntie',
                                 help='coco | kitti | coco_hp | pascal | huntie | structure')
        self.parser.add_argument('--test', action='store_true')
        self.parser.add_argument('--data_src', default="default", type=str,
                                 help='The path of input data.')
        self.parser.add_argument('--exp_id', default='default', type=str)
        self.parser.add_argument('--vis_corner', type=int, default=0,
                                 help='vis corner or not'
                                      '0: do not vis corner'
                                      '1: vis corner')
        self.parser.add_argument('--convert_onnx', type=int, default=0,
                                 help='0: donot convert'
                                      '1: convert pytorch model to onnx')
        self.parser.add_argument('--onnx_path', type=str, default="auto",
                                 help='path of output onnx file.')
        self.parser.add_argument('--debug', type=int, default=0,
                                 help='level of visualization.'
                                      '1: only show the final detection results'
                                      '2: show the network output features'
                                      '3: use matplot to display'  # useful when lunching training with ipython notebook
                                      '4: save all visualizations to disk')
        self.parser.add_argument('--load_model', default='',
                                 help='path to pretrained model')
        self.parser.add_argument('--resume', action='store_true',
                                 help='resume an experiment. '
                                      'Reloaded the optimizer parameter and '
                                      'set load_model to model_last.pth '
                                      'in the exp dir if load_model is empty.')

        # system
        self.parser.add_argument('--gpus', default='-1',
                                 help='-1 for CPU, use comma for multiple gpus')
        self.parser.add_argument('--num_workers', type=int, default=16,
                                 help='dataloader threads. 0 for single-thread.')
        self.parser.add_argument('--not_cuda_benchmark', action='store_true',
                                 help='disable when the input size is not fixed.')
        self.parser.add_argument('--seed', type=int, default=317,
                                 help='random seed')  # from CornerNet

        # log
        self.parser.add_argument('--print_iter', type=int, default=0,
                                 help='disable progress bar and print to screen.')
        self.parser.add_argument('--hide_data_time', action='store_true',
                                 help='not display time during training.')
        self.parser.add_argument('--save_all', action='store_true',
                                 help='save model to disk every 5 epochs.')
        self.parser.add_argument('--metric', default='loss',
                                 help='main metric to save best model')
        self.parser.add_argument('--vis_thresh', type=float, default=0.3,
                                 help='visualization threshold.')
        self.parser.add_argument('--nms_thresh', type=float, default=0.3,
                                 help='nms threshold.')
        self.parser.add_argument('--corner_thresh', type=float, default=0.3,
                                 help='threshold for corner.')
        self.parser.add_argument('--debugger_theme', default='white',
                                 choices=['white', 'black'])

        # model
        self.parser.add_argument('--arch', default='dla_34',
                                 help='model architecture. Currently tested'
                                      'res_18 | res_101 | resdcn_18 | resdcn_101 |'
                                      'dlav0_34 | dla_34 | hourglass')
        self.parser.add_argument('--head_conv', type=int, default=-1,
                                 help='conv layer channels for output head'
                                      '0 for no conv layer'
                                      '-1 for default setting: '
                                      '64 for resnets and 256 for dla.')
        self.parser.add_argument('--down_ratio', type=int, default=4,
                                 help='output stride. Currently only supports 4.')

        # input
        self.parser.add_argument('--input_res', type=int, default=-1,
                                 help='input height and width. -1 for default from '
                                      'dataset. Will be overriden by input_h | input_w')
        self.parser.add_argument('--input_h', type=int, default=-1,
                                 help='input height. -1 for default from dataset.')
        self.parser.add_argument('--input_w', type=int, default=-1,
                                 help='input width. -1 for default from dataset.')

        # train
        self.parser.add_argument('--lr', type=float, default=1.25e-4,
                                 help='learning rate for batch size 32.')
        self.parser.add_argument('--lr_step', type=str, default='80',
                                 help='drop learning rate by 10.')
        self.parser.add_argument('--NotFixList', type=str, default='',
                                 help='not fix layer name.')
        self.parser.add_argument('--num_epochs', type=int, default=90,
                                 help='total training epochs.')
        self.parser.add_argument('--batch_size', type=int, default=32,
                                 help='batch size')
        self.parser.add_argument('--master_batch_size', type=int, default=-1,
                                 help='batch size on the master gpu.')
        self.parser.add_argument('--num_iters', type=int, default=-1,
                                 help='default: #samples / batch_size.')
        self.parser.add_argument('--val_intervals', type=int, default=5,
                                 help='number of epochs to run validation.')
        self.parser.add_argument('--trainval', action='store_true',
                                 help='include validation in training and test on test set')
        self.parser.add_argument('--negative', action='store_true',
                                 help='flip data augmentation.')
        self.parser.add_argument('--adamW', action='store_true',
                                 help='using adamW or adam.')

        # test
        self.parser.add_argument('--save_dir', default="default", type=str,
                                 help='The path of output data.')
        self.parser.add_argument('--flip_test', action='store_true',
                                 help='flip data augmentation.')
        self.parser.add_argument('--test_scales', type=str, default='1',
                                 help='multi scale test augmentation.')
        self.parser.add_argument('--nms', action='store_false',
                                 help='run nms in testing.')
        self.parser.add_argument('--K', type=int, default=100,
                                 help='max number of output objects.')
        self.parser.add_argument('--fix_res', action='store_true',
                                 help='fix testing resolution or keep the original resolution')
        self.parser.add_argument('--keep_res', action='store_true',
                                 help='keep the original resolution during validation.')

        # dataset
        self.parser.add_argument('--not_rand_crop', action='store_true',
                                 help='not use the random crop data augmentation from CornerNet.')
        self.parser.add_argument('--shift', type=float, default=0.1,
                                 help='when not using random crop apply shift augmentation.')
        self.parser.add_argument('--scale', type=float, default=0.4,
                                 help='when not using random crop apply scale augmentation.')
        self.parser.add_argument('--rotate', type=float, default=0,
                                 help='when not using random crop apply rotation augmentation.')
        self.parser.add_argument('--flip', type=float, default=0.5,
                                 help='probability of applying flip augmentation.')
        self.parser.add_argument('--maskvisual', type=float, default=0.,
                                 help='probability of masking image.')
        self.parser.add_argument('--maskgrid', type=float, default=0.,
                                 help='probability of masking grid, only available when visual is not masked.')
        self.parser.add_argument('--no_color_aug', action='store_true',
                                 help='not use the color augmenation from CornerNet')
        self.parser.add_argument('--MK', default=500,
                                 help='max corner number')
        self.parser.add_argument('--rot', action='store_false',
                                 help='rotate image')
        self.parser.add_argument('--warp', action='store_false',
                                 help='warp image')
        self.parser.add_argument('--normal_padding', action='store_false',
                                 help='normal_padding image')
        self.parser.add_argument('--extra_channel', action='store_true',
                                 help='concat edge channel to the input image')
        self.parser.add_argument('--init_emb', type=str, default='',
                                 help='embedding layer.')
        self.parser.add_argument('--grid_type', type=str, default='char_point',
                                 help='type of grid, candidates: char_point, char_box (CharGrid), line (WordGrid).')
        self.parser.add_argument('--finetune_emb', action='store_true',
                                 help='embedding finetune')
        self.parser.add_argument('--dic', type=str, default='',
                                 help='dic file for grid.')
        self.parser.add_argument('--sample_limit', type=int, default=-1,
                                 help='limit samples for training')   

        # multi_pose
        self.parser.add_argument('--aug_rot', type=float, default=0,
                                 help='probability of applying rotation augmentation.')
        # ddd
        self.parser.add_argument('--aug_ddd', type=float, default=0.5,
                                 help='probability of applying crop augmentation.')
        self.parser.add_argument('--rect_mask', action='store_true',
                                 help='for ignored object, apply mask on the '
                                      'rectangular region or just center point.')
        self.parser.add_argument('--kitti_split', default='3dop',
                                 help='different validation split for kitti: '
                                      '3dop | subcnn')

        # loss
        self.parser.add_argument('--mse_loss', action='store_true',
                                 help='use mse loss or focal loss to train keypoint heatmaps.')
        # ctdet
        self.parser.add_argument('--num_classes', type=int, default=-1,
                                 help='the number of main category. -1 means use default from dataset.')
        self.parser.add_argument('--num_secondary_classes', type=int, default=-1,
                                 help='the number of secondary category. -1 means use default from dataset.')
        self.parser.add_argument('--reg_loss', default='l1',
                                 help='regression loss: sl1 | l1 | l2')
        self.parser.add_argument('--hm_weight', type=float, default=1,
                                 help='loss weight for keypoint heatmaps.')
        self.parser.add_argument('--cls_weight', type=float, default=1,
                                 help='loss weight for keypoint heatmaps.')
        self.parser.add_argument('--ftype_weight', type=float, default=1,
                                 help='loss weight for keypoint heatmaps.')
        self.parser.add_argument('--mk_weight', type=float, default=1,
                                 help='loss weight for corner keypoint heatmaps.')
        self.parser.add_argument('--off_weight', type=float, default=1,
                                 help='loss weight for keypoint local offsets.')
        self.parser.add_argument('--wh_weight', type=float, default=1,
                                 help='loss weight for bounding box size.')
        # multi_pose
        self.parser.add_argument('--hp_weight', type=float, default=1,
                                 help='loss weight for human pose offset.')
        self.parser.add_argument('--hm_hp_weight', type=float, default=1,
                                 help='loss weight for human keypoint heatmap.')
        # ddd
        self.parser.add_argument('--dep_weight', type=float, default=1,
                                 help='loss weight for depth.')
        self.parser.add_argument('--dim_weight', type=float, default=1,
                                 help='loss weight for 3d bounding box size.')
        self.parser.add_argument('--rot_weight', type=float, default=1,
                                 help='loss weight for orientation.')
        self.parser.add_argument('--peak_thresh', type=float, default=0.1)

        # task
        # ctdet
        self.parser.add_argument('--norm_wh', action='store_true',
                                 help='L1(\hat(y) / y, 1) or L1(\hat(y), y)')
        self.parser.add_argument('--dense_wh', action='store_true',
                                 help='apply weighted regression near center or '
                                      'just apply regression on center point.')
        self.parser.add_argument('--cat_spec_wh', action='store_true',
                                 help='category specific bounding box size.')
        self.parser.add_argument('--not_reg_offset', action='store_true',
                                 help='not regress local offset.')
        # exdet
        self.parser.add_argument('--agnostic_ex', action='store_true',
                                 help='use category agnostic extreme points.')
        self.parser.add_argument('--scores_thresh', type=float, default=0.3,
                                 help='threshold for extreme point heatmap.')
        self.parser.add_argument('--center_thresh', type=float, default=0.3,
                                 help='threshold for centermap.')
        self.parser.add_argument('--aggr_weight', type=float, default=0.0,
                                 help='edge aggregation weight.')
        # multi_pose
        self.parser.add_argument('--dense_hp', action='store_true',
                                 help='apply weighted pose regression near center '
                                      'or just apply regression on center point.')
        self.parser.add_argument('--not_hm_hp', action='store_true',
                                 help='not estimate human joint heatmap, '
                                      'directly use the joint offset from center.')
        self.parser.add_argument('--not_reg_hp_offset', action='store_true',
                                 help='not regress local offset for '
                                      'human joint heatmaps.')
        self.parser.add_argument('--not_reg_bbox', action='store_true',
                                 help='not regression bounding box size.')

        # ground truth validation
        self.parser.add_argument('--eval_oracle_hm', action='store_true',
                                 help='use ground center heatmap.')
        self.parser.add_argument('--eval_oracle_mk', action='store_true',
                                 help='use ground corner heatmap.')
        self.parser.add_argument('--eval_oracle_wh', action='store_true',
                                 help='use ground truth bounding box size.')
        self.parser.add_argument('--eval_oracle_offset', action='store_true',
                                 help='use ground truth local heatmap offset.')
        self.parser.add_argument('--eval_oracle_kps', action='store_true',
                                 help='use ground truth human pose offset.')
        self.parser.add_argument('--eval_oracle_hmhp', action='store_true',
                                 help='use ground truth human joint heatmaps.')
        self.parser.add_argument('--eval_oracle_hp_offset', action='store_true',
                                 help='use ground truth human joint local offset.')
        self.parser.add_argument('--eval_oracle_dep', action='store_true',
                                 help='use ground truth depth.')

    def parse(self, args=None):
        if isinstance(args, dict):
            task_name = args.get("task", "ctdet")
            opt = self.parser.parse_args(args=[task_name])
            opt.__dict__.update(args)
        else:
            opt = self.parser.parse_args(args=args)

        # import json
        # with open("task_config.json", "w") as f:
        #     json.dump(opt.__dict__, f, ensure_ascii=False, indent=4)

        opt.gpus_str = opt.gpus
        opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
        opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >= 0 else [-1]
        opt.lr_step = [int(i) for i in opt.lr_step.split(',')]
        opt.test_scales = [float(i) for i in opt.test_scales.split(',')]

        opt.fix_res = not opt.keep_res
        print('Fix size testing.' if opt.fix_res else 'Keep resolution testing.')
        opt.reg_offset = not opt.not_reg_offset
        opt.reg_bbox = not opt.not_reg_bbox
        opt.hm_hp = not opt.not_hm_hp
        opt.reg_hp_offset = (not opt.not_reg_hp_offset) and opt.hm_hp

        if opt.head_conv == -1:  # init default head_conv
            opt.head_conv = 256 if 'dla' in opt.arch else 64
        opt.pad = 0  # opt.pad = 127 if 'hourglass' in opt.arch else 31
        opt.num_stacks = 2 if opt.arch == 'hourglass' else 1

        if opt.trainval:
            opt.val_intervals = 100000000

        if opt.debug > 0:
            opt.num_workers = 0
            opt.batch_size = 1
            opt.gpus = [opt.gpus[0]]
            opt.master_batch_size = -1

        if opt.master_batch_size == -1:
            opt.master_batch_size = opt.batch_size // len(opt.gpus)
        rest_batch_size = (opt.batch_size - opt.master_batch_size)
        opt.chunk_sizes = [opt.master_batch_size]
        for i in range(len(opt.gpus) - 1):
            slave_chunk_size = rest_batch_size // (len(opt.gpus) - 1)
            if i < rest_batch_size % (len(opt.gpus) - 1):
                slave_chunk_size += 1
            opt.chunk_sizes.append(slave_chunk_size)
        print('training chunk_sizes:', opt.chunk_sizes)

        opt.root_dir = os.path.join(os.path.dirname(__file__), '..', '..')
        opt.data_dir = os.path.join(opt.root_dir, 'data') if opt.data_src == "default" else opt.data_src
        opt.exp_dir = os.path.join(opt.root_dir, 'exp', opt.task)
        # import pdb; pdb.set_trace()
        opt.save_dir = os.path.join(opt.exp_dir, opt.exp_id) if opt.save_dir == "default" else os.path.join(opt.save_dir, opt.exp_id)
        opt.debug_dir = os.path.join(opt.save_dir, 'debug')
        print('The output will be saved to ', opt.save_dir)

        if opt.resume and opt.load_model == '':
            model_path = opt.save_dir[:-4] if opt.save_dir.endswith('TEST') \
                else opt.save_dir
            opt.load_model = os.path.join(model_path, 'model_last.pth')
        return opt

    def update_dataset_info_and_set_heads(self, opt, dataset):
        input_h, input_w = dataset.default_resolution
        opt.mean, opt.std = dataset.mean, dataset.std

        if opt.num_classes == -1:
            opt.num_classes = dataset.num_classes
        if opt.num_secondary_classes == -1:
            opt.num_secondary_classes = dataset.num_secondary_classes

        # input_h(w): opt.input_h overrides opt.input_res overrides dataset default
        input_h = opt.input_res if opt.input_res > 0 else input_h
        input_w = opt.input_res if opt.input_res > 0 else input_w
        opt.input_h = opt.input_h if opt.input_h > 0 else input_h
        opt.input_w = opt.input_w if opt.input_w > 0 else input_w
        opt.output_h = opt.input_h // opt.down_ratio
        opt.output_w = opt.input_w // opt.down_ratio
        opt.input_res = max(opt.input_h, opt.input_w)
        opt.output_res = max(opt.output_h, opt.output_w)

        if opt.task == 'exdet':
            # assert opt.dataset in ['coco']
            num_hm = 1 if opt.agnostic_ex else opt.num_classes
            opt.heads = {'hm_t': num_hm, 'hm_l': num_hm,
                         'hm_b': num_hm, 'hm_r': num_hm,
                         'hm_c': opt.num_classes}
            if opt.reg_offset:
                opt.heads.update({'reg_t': 2, 'reg_l': 2, 'reg_b': 2, 'reg_r': 2})
        elif opt.task == 'ddd':
            # assert opt.dataset in ['gta', 'kitti', 'viper']
            opt.heads = {'hm': opt.num_classes, 'dep': 1, 'rot': 8, 'dim': 3}
            if opt.reg_bbox:
                opt.heads.update(
                    {'wh': 2})
            if opt.reg_offset:
                opt.heads.update({'reg': 2})
        elif opt.task == 'ctdet':
            # assert opt.dataset in ['pascal', 'coco']
            opt.heads = {'hm': opt.num_classes, 'cls': 4, 'ftype': opt.num_secondary_classes,
                         'wh': 8 if not opt.cat_spec_wh else 8 * opt.num_classes}
            if opt.reg_offset:
                opt.heads.update({'reg': 2})
        elif opt.task == 'ctdet_dualmodal':
            # assert opt.dataset in ['pascal', 'coco']
            opt.heads = {'hm': opt.num_classes, 'cls': 4, 'ftype': opt.num_secondary_classes,
                         'wh': 8 if not opt.cat_spec_wh else 8 * opt.num_classes}
            if opt.reg_offset:
                opt.heads.update({'reg': 2})
        elif opt.task == 'multi_pose':
            # assert opt.dataset in ['coco_hp']
            opt.flip_idx = dataset.flip_idx
            opt.heads = {'hm': opt.num_classes, 'wh': 2, 'hps': 34}
            if opt.reg_offset:
                opt.heads.update({'reg': 2})
            if opt.hm_hp:
                opt.heads.update({'hm_hp': 17})
            if opt.reg_hp_offset:
                opt.heads.update({'hp_offset': 2})
        elif opt.task == 'ctdet_subfield':
            # assert opt.dataset in ['pascal', 'coco']
            opt.heads = {'hm': opt.num_classes-2, 'cls': 4, 'ftype': opt.num_secondary_classes,
                         'wh': 8 if not opt.cat_spec_wh else 8 * opt.num_classes, 'hm_sub': 2, 'wh_sub': 8 }
            if opt.reg_offset:
                opt.heads.update({'reg': 2})
                opt.heads.update({'reg_sub': 2})
        else:
            assert 0, 'task not defined!'
        print('heads', opt.heads)
        return opt


if __name__ == '__main__':
    print("Testing config ... ")
    config_dict = {"batch_size": 32, "dataset": "huntie"}
    opt = opts().parse(args=config_dict)
    print(opt.__dict__)

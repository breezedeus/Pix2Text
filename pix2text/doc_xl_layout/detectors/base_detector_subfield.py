# coding: utf-8
import os
import time

import cv2
import numpy as np
import torch
from ..models.model import create_model, load_model

# from ..utils.debugger import Debugger
from ..utils.image import get_affine_transform


class BaseDetector(object):
    def __init__(self, opt):
        # if opt.gpus[0] >= 0:
        #     opt.device = torch.device('cuda')
        # else:
        #     opt.device = torch.device('cpu')

        self.model = create_model(opt.arch, opt.heads, opt.head_conv, opt.convert_onnx, {})
        self.model = load_model(self.model, opt.load_model)
        self.model = self.model.to(opt.device)
        self.model.eval()

        self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)
        self.max_per_image = opt.K
        self.num_classes = opt.num_classes
        self.scales = opt.test_scales
        self.opt = opt
        self.pause = True

    def pre_process(self, image, scale, meta=None):
        height, width = image.shape[0:2]
        new_height = int(height * scale)
        new_width = int(width * scale)
        if self.opt.fix_res:
            inp_height, inp_width = self.opt.input_h, self.opt.input_w
            c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
            s = max(height, width) * 1.0
        else:
            inp_height = (new_height | self.opt.pad)  # + 1
            inp_width = (new_width | self.opt.pad)  # + 1
            c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
            s = np.array([inp_width, inp_height], dtype=np.float32)

        trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
        resized_image = cv2.resize(image, (new_width, new_height))
        inp_image = cv2.warpAffine(
            resized_image, trans_input, (inp_width, inp_height),
            flags=cv2.INTER_LINEAR)
        vis_image = inp_image
        # import pdb; pdb.set_trace()
        inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)

        images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
        if self.opt.flip_test:
            images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
        images = torch.from_numpy(images)
        meta = {'c': c, 's': s,
                'input_height': inp_height,
                'input_width': inp_width,
                'vis_image': vis_image,
                'out_height': inp_height // self.opt.down_ratio,
                'out_width': inp_width // self.opt.down_ratio}
        return images, meta

    def resize(self, image):
        h, w, _ = image.shape
        scale = self.opt.input_h / (max(w, h) + 1e-4)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
        image = cv2.copyMakeBorder(image, 0, self.opt.input_h - int(h * scale), 0, self.opt.input_h - int(w * scale),
                                   cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return image, scale

    def process(self, images, return_time=False):
        raise NotImplementedError

    def post_process(self, dets, meta, scale=1):
        raise NotImplementedError

    def merge_outputs(self, detections):
        raise NotImplementedError

    def debug(self, debugger, images, dets, output, scale=1):
        raise NotImplementedError

    def show_results(self, debugger, image, results):
        raise NotImplementedError

    def ps_convert_minmax(self, results):
        detection = {}
        for j in range(1, self.num_classes + 1):
            detection[j] = []
        for j in range(1, self.num_classes + 1):
            for bbox in results[j]:
                if bbox[8] < self.opt.scores_thresh:
                    continue
                minx = max(min(bbox[0], bbox[2], bbox[4], bbox[6]), 0)
                miny = max(min(bbox[1], bbox[3], bbox[5], bbox[7]), 0)
                maxx = max(bbox[0], bbox[2], bbox[4], bbox[6])
                maxy = max(bbox[1], bbox[3], bbox[5], bbox[7])
                detection[j].append([minx, miny, maxx, maxy, bbox[8], bbox[-1]])
        for j in range(1, self.num_classes + 1):
            detection[j] = np.array(detection[j])
        return detection

    def Duplicate_removal(self, results):
        bbox = []
        for box in results:
            if box[8] > self.opt.scores_thresh:
                # for i in range(8):
                #     if box[i] < 0:
                #         box[i] = 0
                #     if box[i]>self.opt.input_h:
                #      box[i]=self.opt.input_h
                bbox.append(box)
        if len(bbox) > 0:
            return np.array(bbox)
        else:
            return np.array([[0] * 12])

    def run(self, image_or_path_or_tensor, meta=None):
        load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
        merge_time, tot_time = 0, 0
        # debugger = Debugger(dataset=self.opt.dataset, ipynb=(self.opt.debug == 3), num_classes=self.opt.num_classes,
        #                     theme=self.opt.debugger_theme)
        start_time = time.time()
        pre_processed = False
        if isinstance(image_or_path_or_tensor, np.ndarray):
            image = image_or_path_or_tensor
        elif type(image_or_path_or_tensor) == type(''):
            image = cv2.imread(image_or_path_or_tensor)
        else:
            image = image_or_path_or_tensor['image'][0].numpy()
            pre_processed_images = image_or_path_or_tensor
            pre_processed = True

        loaded_time = time.time()
        load_time += (loaded_time - start_time)

        detections = []
        for scale in self.scales:
            scale_start_time = time.time()
            if not pre_processed:
                images, meta = self.pre_process(image, scale, meta)
            else:
                images = pre_processed_images['images'][scale][0]
                meta = pre_processed_images['meta'][scale]
                meta = {k: v.numpy()[0] for k, v in meta.items()}

            # import ipdb;ipdb.set_trace()
            # images = np.load('data.npy').astype(np.float32)
            # images = torch.from_numpy(images)
            
            images = images.to(self.opt.device)
            # torch.cuda.synchronize()
            pre_process_time = time.time()
            pre_time += pre_process_time - scale_start_time
            output, dets, dets_sub, corner, forward_time = self.process(images, return_time=True)
            # torch.cuda.synchronize()
            net_time += forward_time - pre_process_time
            decode_time = time.time()
            dec_time += decode_time - forward_time

            # if self.opt.debug >= 2:
            #     self.debug(debugger, images, dets, output, scale)

            dets, corner = self.post_process(dets, corner, meta, scale)
            for j in range(1, self.num_classes + 1):
                dets[j] = self.Duplicate_removal(dets[j])
                
            # add sub
            dets_sub, corner = self.post_process(dets_sub, corner, meta, scale)
            for j in range(1, self.num_classes + 1):
                dets_sub[j] = self.Duplicate_removal(dets_sub[j])
                
            # import ipdb;ipdb.set_trace()   
            # torch.cuda.synchronize()
            post_process_time = time.time()
            post_time += post_process_time - decode_time
            
            dets[12] = dets_sub[12]
            dets[13] = dets_sub[13]
            
            detections.append(dets)

        results = self.merge_outputs(detections)
        # torch.cuda.synchronize()
        end_time = time.time()
        merge_time += end_time - post_process_time
        tot_time += end_time - start_time

        # import pdb; pdb.set_trace()
        if self.opt.debug >= 1:
            if isinstance(image_or_path_or_tensor, str):
                image_name = os.path.basename(image_or_path_or_tensor)
            else:
                print("--> warning: use demo.py for a better visualization")
                image_name = "{}.jpg".format(time.time())
            # self.show_results(debugger, image, results, corner, image_name)

        return {'results': results, 'tot': tot_time, 'load': load_time,
                'pre': pre_time, 'net': net_time, 'dec': dec_time, 'corner': corner,
                'post': post_time, 'merge': merge_time, 'output': output}

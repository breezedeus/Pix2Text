# coding: utf-8
import time
import numpy as np
import torch

# from external.nms import soft_nms
from ..external.shapelyNMS import pnms
from ..models.decode import ctdet_4ps_decode, ctdet_cls_decode
from ..models.utils import flip_tensor
from ..utils.post_process import ctdet_4ps_post_process
from .base_detector_subfield import BaseDetector


class CtdetDetector_Subfield(BaseDetector):
    def __init__(self, opt):
        super(CtdetDetector_Subfield, self).__init__(opt)

    def process(self, images, return_time=False):
        # import ipdb;ipdb.set_trace()
        with torch.no_grad():
            output = self.model(images)[-1]
            if self.opt.convert_onnx == 1:
                # torch.cuda.synchronize()
                inputs = ['data']
                outputs = [
                    'hm.0.sigmoid',
                    'hm.0.maxpool',
                    'cls.0.sigmoid',
                    'ftype.0.sigmoid',
                    'wh.2',
                    'reg.2',
                    'hm_sub.0.sigmoid',
                    'hm_sub.0.maxpool',
                    'wh_sub.2',
                    'reg_sub.2',
                ]
                dynamic_axes = {
                    'data': {2: 'h', 3: 'w'},
                    'hm.0.sigmoid': {2: 'H', 3: 'W'},
                    'hm.0.maxpool': {2: 'H', 3: 'W'},
                    'cls.0.sigmoid': {2: 'H', 3: 'W'},
                    'ftype.0.sigmoid': {2: 'H', 3: 'W'},
                    'wh.2': {2: 'H', 3: 'W'},
                    'reg.2': {2: 'H', 3: 'W'},
                    'hm_sub.0.sigmoid': {2: 'H', 3: 'W'},
                    'hm_sub.0.maxpool': {2: 'H', 3: 'W'},
                    'wh_sub.2': {2: 'H', 3: 'W'},
                    'reg_sub.2': {2: 'H', 3: 'W'},
                }

                onnx_path = self.opt.onnx_path
                if self.opt.onnx_path == "auto":
                    onnx_path = "{}_{}cls_{}ftype.onnx".format(
                        self.opt.dataset,
                        self.opt.num_classes,
                        self.opt.num_secondary_classes,
                    )

                torch.onnx.export(
                    self.model,
                    images,
                    onnx_path,
                    input_names=inputs,
                    output_names=outputs,
                    dynamic_axes=dynamic_axes,
                    do_constant_folding=True,
                    opset_version=10,
                )
                print("--> info: onnx is saved at: {}".format(onnx_path))
                cls = output['cls_sigmoid']
                hm = output['hm_sigmoid']
                ftype = output['ftype_sigmoid']

                # add sub
                hm_sub = output['hm_sigmoid_sub']
            else:
                hm = output['hm'].sigmoid_()
                cls = output['cls'].sigmoid_()
                ftype = output['ftype'].sigmoid_()

                # add sub
                hm_sub = output['hm_sub'].sigmoid_()

            wh = output['wh']
            reg = output['reg'] if self.opt.reg_offset else None

            # add sub
            wh_sub = output['wh_sub']
            reg_sub = output['reg_sub'] if self.opt.reg_offset else None

            if self.opt.flip_test:
                hm = (hm[0:1] + flip_tensor(hm[1:2])) / 2
                wh = (wh[0:1] + flip_tensor(wh[1:2])) / 2
                reg = reg[0:1] if reg is not None else None
            # torch.cuda.synchronize()
            forward_time = time.time()
            # return dets [bboxes, scores, clses]
            # breakpoint()
            dets, inds = ctdet_4ps_decode(hm, wh, reg=reg, K=self.opt.K)

            # add sub
            dets_sub, inds_sub = ctdet_4ps_decode(
                hm_sub, wh_sub, reg=reg_sub, K=self.opt.K
            )

            box_cls = ctdet_cls_decode(cls, inds)
            box_ftype = ctdet_cls_decode(ftype, inds)
            clses = torch.argmax(box_cls, dim=2, keepdim=True)
            ftypes = torch.argmax(box_ftype, dim=2, keepdim=True)
            dets = np.concatenate(
                (
                    dets.detach().cpu().numpy(),
                    clses.detach().cpu().numpy(),
                    ftypes.detach().cpu().numpy(),
                ),
                axis=2,
            )
            dets = np.array(dets)

            # add subfield
            dets_sub = np.concatenate(
                (
                    dets_sub.detach().cpu().numpy(),
                    clses.detach().cpu().numpy(),
                    ftypes.detach().cpu().numpy(),
                ),
                axis=2,
            )
            dets_sub = np.array(dets_sub)
            dets_sub[:, :, -3] += 11

            corner = 0

        if return_time:
            return output, dets, dets_sub, corner, forward_time
        else:
            return output, dets, dets_sub

    def post_process(self, dets, corner, meta, scale=1):
        if self.opt.nms:
            detn = pnms(dets[0], self.opt.scores_thresh)
            if detn.shape[0] > 0:
                dets = detn.reshape(1, -1, detn.shape[1])
        k = dets.shape[2] if dets.shape[1] != 0 else 0
        if dets.shape[1] != 0:
            dets = dets.reshape(1, -1, dets.shape[2])
            # return dets is list and what in dets is dict. key of dict is classes, value of dict is [bbox,score]
            dets = ctdet_4ps_post_process(
                dets.copy(),
                [meta['c']],
                [meta['s']],
                meta['out_height'],
                meta['out_width'],
                self.opt.num_classes,
            )
            for j in range(1, self.num_classes + 1):
                dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, k)
                dets[0][j][:, :8] /= scale
        else:
            ret = {}
            dets = []
            for j in range(1, self.num_classes + 1):
                ret[j] = np.array([0] * k, dtype=np.float32)  # .reshape(-1, k)
            dets.append(ret)
        return dets[0], corner

    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0
            ).astype(np.float32)
            # if len(self.scales) > 1 or self.opt.nms:
            #  results[j] = pnms(results[j],self.opt.nms_thresh)
        shape_num = 0
        for j in range(1, self.num_classes + 1):
            shape_num = shape_num + len(results[j])
        if shape_num != 0:
            # print(np.array(results[1]))
            scores = np.hstack(
                [results[j][:, 8] for j in range(1, self.num_classes + 1)]
            )
        else:
            scores = []
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.num_classes + 1):
                keep_inds = results[j][:, 8] >= thresh
                results[j] = results[j][keep_inds]
        return results

    def debug(self, debugger, images, dets, output, scale=1):
        # detection = dets.detach().cpu().numpy().copy()
        detection = dets.copy()
        detection[:, :, :8] *= self.opt.down_ratio
        for i in range(1):
            img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
            img = ((img * self.std + self.mean) * 255).astype(np.uint8)
            pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hm_{:.1f}'.format(scale))
            debugger.add_img(img, img_id='out_pred_{:.1f}'.format(scale))
            # pdb.set_trace()
            for k in range(len(dets[i])):
                if detection[i, k, 8] > self.opt.center_thresh:
                    debugger.add_4ps_coco_bbox(
                        detection[i, k, :8],
                        detection[i, k, -1],
                        detection[i, k, 8],
                        img_id='out_pred_{:.1f}'.format(scale),
                    )

    def show_results(self, debugger, image, results, Corners, image_name):
        debugger.add_img(image, img_id='ctdet')
        count = 0
        for j in range(1, self.num_classes + 1):
            for bbox in results[j]:
                if bbox[8] > self.opt.scores_thresh:
                    count += 1
                    # print("bbox info:",j-1,  bbox.tolist())
                    # print(j-1)
                    debugger.add_4ps_coco_bbox(
                        bbox, j - 1, bbox[8], show_txt=True, img_id='ctdet'
                    )
        debugger.save_all_imgs(image_name, './outputs/')

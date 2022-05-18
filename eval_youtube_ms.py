"""
YouTubeVOS has a label structure that is more complicated than DAVIS 
Labels might not appear on the first frame (there might be no labels at all in the first frame)
Labels might not even appear on the same frame (i.e. Object 0 at frame 10, and object 1 at frame 15)
0 does not mean background -- it is simply "no-label"
and object indices might not be in order, there are missing indices somewhere in the validation set

Dealing with these makes the logic a bit convoluted here
It is not necessarily hacky but do understand that it is not as straightforward as DAVIS

Validation/test set.
"""


import os
from os import path
from argparse import ArgumentParser
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import cv2

from model.eval_network import STCN
from dataset.yv_test_dataset import YouTubeVOSTestDataset
from util.tensor_util import unpad
from inference_core_yv import InferenceCore

from progressbar import progressbar

"""
Arguments loading
"""
parser = ArgumentParser()
parser.add_argument('--model', default='saves/stcn.pth')
parser.add_argument('--yv_path', default='../YouTube')

parser.add_argument('--output_all', help=
"""
We will output all the frames if this is set to true.
Otherwise only a subset will be outputted, as determined by meta.json to save disk space.
For ensemble, all the sources must have this setting unified.
""", action='store_true')

parser.add_argument('--output')
parser.add_argument('--split', help='valid/test', default='valid')
parser.add_argument('--top', type=int, default=20)
parser.add_argument('--amp', action='store_true')
parser.add_argument('--mem_every', default=5, type=int)
parser.add_argument('--include_last', help='include last frame as temporary memory?', action='store_true')
parser.add_argument('--vis', help='visualize the outputs for analysis', action='store_true')
parser.add_argument('--vname', type=str, default='')
parser.add_argument('--memory_type', type=str, default='topk')
parser.add_argument('--sigma', type=int, default=17)


args = parser.parse_args()

yv_path = args.yv_path
out_path = args.output

# Simple setup
os.makedirs(out_path, exist_ok=True)
palette = Image.open(path.expanduser(yv_path + '/valid/Annotations/0a49f5265b/00000.png')).getpalette()

torch.autograd.set_grad_enabled(False)

# Load the json if we have to
if not args.output_all:
    with open(path.join(yv_path, args.split, 'meta.json')) as f:
        meta = json.load(f)['videos']

# Setup Dataset
test_dataset = YouTubeVOSTestDataset(data_root=yv_path, split=args.split, vname = args.vname, res = 480)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
test_iter480 = iter(test_loader)

test_dataset1 = YouTubeVOSTestDataset(data_root=yv_path, split=args.split, vname = args.vname, res = 600)
test_loader1 = DataLoader(test_dataset1, batch_size=1, shuffle=False, num_workers=4)
test_iter600 = iter(test_loader1)

# Load our checkpoint
prop_model = STCN().cuda().eval()

# Performs input mapping such that stage 0 model can be loaded
prop_saved = torch.load(args.model)
for k in list(prop_saved.keys()):
    if k == 'value_encoder.conv1.weight':
        if prop_saved[k].shape[1] == 4:
            pads = torch.zeros((64,1,7,7), device=prop_saved[k].device)
            prop_saved[k] = torch.cat([prop_saved[k], pads], 1)
prop_model.load_state_dict(prop_saved)

# Start eval
inference_num = 0
while True:
    print(inference_num)
    inference_num += 1

    ##############################
    if inference_num > 507:
        break
    ##############################

    for inference_setting in ['480o', '480f', '600o', '600f']:
        try:
            if inference_setting == '480o':
                data = next(test_iter480)
            elif inference_setting == '600o':
                data = next(test_iter600)
        except:
            break

        with torch.cuda.amp.autocast(enabled=args.amp):
            rgb = data['rgb']
            msk = data['gt'][0]
            if inference_setting in ['480f', '600f']:
                rgb = torch.flip(rgb, dims=(4,))
                msk = torch.flip(msk, dims=(4,))
            info = data['info']
            name = info['name'][0]

            num_objects = len(info['labels'][0])
            gt_obj = info['gt_obj']
            size = info['size']

            # Load the required set of frames (if we don't need all)
            req_frames = None
            if not args.output_all:
                req_frames = []
                objects = meta[name]['objects']
                for key, value in objects.items():
                    req_frames.extend(value['frames'])

                # Map the frame names to indices
                req_frames_names = set(req_frames)
                req_frames = []
                for fi in range(rgb.shape[1]):
                    frame_name = info['frames'][fi][0][:-4]
                    if frame_name in req_frames_names:
                        req_frames.append(fi)
                req_frames = sorted(req_frames)

            # Frames with labels, but they are not exhaustively labeled
            frames_with_gt = sorted(list(gt_obj.keys()))

            if inference_setting in ['480o', '480f']:
                memory_type = 'topk'
                sigma = 17
                top_k = 20
            elif inference_setting in ['600o', '600f']:
                memory_type = 'kmn'
                sigma = 17
                top_k = 20

            processor = InferenceCore(prop_model, rgb, num_objects=num_objects, top_k=top_k, 
                                        mem_every=args.mem_every, include_last=args.include_last, 
                                        req_frames=req_frames, memory_type = memory_type, sigma = sigma)
            # min_idx tells us the starting point of propagation
            # Propagating before there are labels is not useful
            min_idx = 99999
            for i, frame_idx in enumerate(frames_with_gt):
                min_idx = min(frame_idx, min_idx)
                # Note that there might be more than one label per frame
                obj_idx = gt_obj[frame_idx][0].tolist()
                # Map the possibly non-continuous labels into a continuous scheme
                obj_idx = [info['label_convert'][o].item() for o in obj_idx]

                # Append the background label
                with_bg_msk = torch.cat([
                    1 - torch.sum(msk[:,frame_idx], dim=0, keepdim=True),
                    msk[:,frame_idx],
                ], 0).cuda()

                # We perform propagation from the current frame to the next frame with label
                if i == len(frames_with_gt) - 1:
                    processor.interact(with_bg_msk, frame_idx, rgb.shape[1], obj_idx)
                else:
                    processor.interact(with_bg_msk, frame_idx, frames_with_gt[i+1]+1, obj_idx)

            # Do unpad -> upsample to original size (we made it 480p)
            out_masks = torch.zeros((processor.t, 1, *size), dtype=torch.uint8, device='cuda')

            if inference_setting == '480o':
                out_logits = torch.zeros((processor.t, num_objects + 1, 1, *size), dtype=torch.float32, device='cuda')
            for ti in range(processor.t):
                prob = unpad(processor.prob[:,ti], processor.pad)
                prob = F.interpolate(prob, size, mode='bilinear', align_corners=False)
                if inference_setting in ['480f', '600f']:
                    prob = torch.flip(prob,dims=(3,))

                out_logits[ti] = out_logits[ti] + prob
                #out_masks[ti] = torch.argmax(prob, dim=0)

            del rgb
            del msk
            del processor



    #out_masks = (out_masks.detach().cpu().numpy()[:,0]).astype(np.uint8)
    out_masks = (torch.argmax(out_logits, dim=1).detach().cpu().numpy()[:,0]).astype(np.uint8)
    # Remap the indices to the original domain
    idx_masks = np.zeros_like(out_masks)
    for i in range(1, num_objects+1):
        backward_idx = info['label_backward'][i].item()
        idx_masks[out_masks==i] = backward_idx
        
    # Save the results
    this_out_path = path.join(out_path, 'Annotations', name)
    os.makedirs(this_out_path, exist_ok=True)
    if args.vis:
        this_out_vis_path = path.join(out_path, 'Vis', name)
        os.makedirs(this_out_vis_path, exist_ok=True)
        this_out_vis_video_path = path.join(out_path, 'Video')
        os.makedirs(this_out_vis_video_path, exist_ok=True)

    saved_num = 0
    for f in range(idx_masks.shape[0]):
        if f >= min_idx:
            if args.output_all or (f in req_frames):
                img_E = Image.fromarray(idx_masks[f])
                img_E.putpalette(palette)
                img_E.save(os.path.join(this_out_path, info['frames'][f][0].replace('.jpg','.png')))

                if args.vis:
                    img_E_array = np.array(img_E.convert('RGB'))
                    img_RGB = np.array(Image.open(os.path.join(yv_path,'valid/JPEGImages',name,info['frames'][f][0])))
                    img_RGB[idx_masks[f] > 0] = 0.2 * img_RGB[idx_masks[f] > 0] + 0.8 * img_E_array[idx_masks[f] > 0]
                    img_RGB = Image.fromarray(img_RGB.astype(np.uint8))
                    img_RGB.save(os.path.join(this_out_vis_path, str(saved_num).zfill(5) + '.jpg'))
                    saved_num += 1
    if args.vis:
        os.system('ffmpeg -r 5 -f image2 -i {}/%5d.jpg {} -loglevel quiet'.format(this_out_vis_path, os.path.join(this_out_vis_video_path, name + '.mp4')))


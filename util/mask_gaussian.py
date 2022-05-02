import numpy as np
import torch
import cv2
#def mask_gaussian(mask, objs):
#    obj_gaussians = {}
#    for obj in objs:
#        obj_mask = mask[obj].cpu()
#        _,h,w = obj_mask.shape
#        gridy,gridx = torch.meshgrid(torch.Tensor(range(h)),torch.Tensor(range(w)))
#
#
#
#        cx = (gridx * obj_mask).sum() / obj_mask.sum()
#        cy = (gridy * obj_mask).sum() / obj_mask.sum()
#        s2x = (((gridx - cx) ** 2) * obj_mask).sum() / obj_mask.sum()
#        s2y = (((gridy - cy) ** 2) * obj_mask).sum() / obj_mask.sum()
#        gaussian_mask = 1/(2 * torch.pi * torch.sqrt(s2x) * torch.sqrt(s2y)) * torch.exp(-(gridx-cx)**2/(2*s2x)) * torch.exp(-(gridy-cy)**2/(2*s2y))
#        import pdb
#        pdb.set_trace()
#
#    import pdb
#    pdb.set_trace()


def get_mask_bbox(m, border_pixels=0):
    if not np.any(m):
        # return a default bbox
        return (0, 0, m.shape[1], m.shape[0])
    rows = np.any(m, axis=1)
    cols = np.any(m, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    h,w = m.shape
    ymin = max(0, ymin - border_pixels)
    ymax = min(h-1, ymax + border_pixels)
    xmin = max(0, xmin - border_pixels)
    xmax = min(w-1, xmax + border_pixels)
    return (xmin, ymin, xmax, ymax)

def compute_robust_moments(binary_image, isotropic=False):
    index = np.nonzero(binary_image)
    points = np.asarray(index).astype(np.float32)
    if points.shape[1] == 0:
        return np.array([-1.0,-1.0],dtype=np.float32), \
            np.array([-1.0,-1.0],dtype=np.float32)
    points = np.transpose(points)
    points[:,[0,1]] = points[:,[1,0]]
    center = np.median(points, axis=0)
    if isotropic:
        diff = np.linalg.norm(points - center, axis=1)
        mad = np.median(diff)
        mad = np.array([mad,mad])
    else:
        diff = np.absolute(points - center)
        mad = np.median(diff, axis=0)
    std_dev = 1.4826*mad
    std_dev = np.maximum(std_dev, [5.0, 5.0])
    return center, std_dev

def mask_gaussian(masks, objs, std_pertube = 1.0):
    obj_gaussians = {}
    for obj in objs:
        label = masks[obj,0].cpu().numpy()
        if not np.any(label):
            #return a blank gb image
            obj_gaussians[obj] = torch.FloatTensor(np.zeros((label.shape)))
        center, std = compute_robust_moments(label)
        center_p = center
        std_p = std_pertube * std
        h,w = label.shape
        x = np.arange(0, w)
        y = np.arange(0, h)
        nx, ny = np.meshgrid(x,y)
        coords = np.concatenate((nx[...,np.newaxis], ny[...,np.newaxis]), axis = 2)
        normalizer = 0.5 /(std_p * std_p)
        D = np.sum((coords - center_p) ** 2 * normalizer, axis=2)
        D = np.exp(-D)
        D = np.clip(D, 0, 1)
        obj_gaussians[obj] = torch.FloatTensor(D)
    return obj_gaussians

# def mask_gaussian(masks, objs, center_perturb = 0.2, size_perturb=0.2):
#     obj_gaussians = {}
#     for obj in objs:
#         mask = masks[obj,0].cpu().numpy()
#         if not np.any(mask):
#             obj_gaussians[obj] = torch.FloatTensor(np.zeros((mask.shape)))
#         else:
#             xmin, ymin, xmax, ymax = get_mask_bbox(mask, border_pixels=0)
#             mask_size = np.array((xmax - xmin, ymax - ymin))
#             center = np.array(((xmin+xmax)//2, (ymin + ymax)//2))
#             cropped_mask = mask[ymin:ymax+1,xmin:xmax+1]
#             mask_out = np.zeros(mask.shape)
#             out_size = np.array(mask_out.shape[1::-1],dtype=np.int32)
#             size_ratio = np.random.uniform(1.0-size_perturb, 1.0 + size_perturb, 1)
#             cropped_mask = cv2.resize(cropped_mask,(0,0),fx=size_ratio[0], fy=size_ratio[0], interpolation=cv2.INTER_NEAREST)
#             size_p = np.array(cropped_mask.shape[1::-1], dtype=np.int32)
#             size_p_1 = size_p // 2
#             size_p_2 = size_p - size_p_1
#             center_p_ratio = np.random.uniform(-center_perturb, center_perturb, 2)
#             center_p = center_p_ratio * mask_size + center
#             center_p = center_p.astype(np.int32)
#             out_start = np.maximum(0, center_p - size_p_1)
#             src_start = np.maximum(0, size_p_1 - center_p)
#             out_end = np.minimum(out_size, center_p + size_p_2)
#             src_end = np.minimum(size_p, size_p - (center_p + size_p_2 - out_size))
# 
#             mask_out[int(out_start[1]):int(out_end[1]), int(out_start[0]):int(out_end[0])] = cropped_mask[int(src_start[1]):int(src_end[1]), int(src_start[0]): int(src_end[0])]
#             obj_gaussians[obj] = torch.FloatTensor(mask_out)
#     return obj_gaussians

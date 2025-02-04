"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import numpy as np
from typing import Any, Dict
from scipy.ndimage import find_objects, gaussian_filter
from cellpose.models import CellposeModel, Cellpose
from cellpose import transforms, dynamics
from cellpose.utils import fill_holes_and_remove_small_masks
from cellpose.transforms import normalize99
import time
import cv2
import os

from . import utils
from .stats import roi_stats
from suite2p import uncanny


def mask_centers(masks):
    centers = np.zeros((masks.max(), 2), np.int32)
    diams = np.zeros(masks.max(), np.float32)
    slices = find_objects(masks)
    for i, si in enumerate(slices):
        if si is not None:
            sr, sc = si
            ymed, xmed, diam = utils.mask_stats(masks[sr, sc] == (i + 1))
            centers[i] = np.array([ymed, xmed])
            diams[i] = diam
    return centers, diams


def patch_detect(patches, diam):
    """ anatomical detection of masks from top active frames for putative cell """
    print("refining masks using cellpose")
    npatches = len(patches)
    ly = patches[0].shape[0]
    model = Cellpose()
    imgs = np.zeros((npatches, ly, ly, 2), np.float32)
    for i, m in enumerate(patches):
        imgs[i, :, :, 0] = transforms.normalize99(m)
    rsz = 30. / diam
    imgs = transforms.resize_image(imgs, rsz=rsz).transpose(0, 3, 1, 2)
    imgs, ysub, xsub = transforms.pad_image_ND(imgs)

    pmasks = np.zeros((npatches, ly, ly), np.uint16)
    batch_size = 8 * 224 // ly
    tic = time.time()
    for j in np.arange(0, npatches, batch_size):
        y = model.cp.network(imgs[j:j + batch_size])[0]
        y = y[:, :, ysub[0]:ysub[-1] + 1, xsub[0]:xsub[-1] + 1]
        y = y.asnumpy()
        for i, yi in enumerate(y):
            cellprob = yi[-1]
            dP = yi[:2]
            niter = 1 / rsz * 200
            p = dynamics.follow_flows(-1 * dP * (cellprob > 0) / 5., niter=niter)
            maski = dynamics.get_masks(p, iscell=(cellprob > 0), flows=dP,
                                       threshold=1.0)
            maski = fill_holes_and_remove_small_masks(maski)
            maski = transforms.resize_image(maski, ly, ly,
                                            interpolation=cv2.INTER_NEAREST)
            pmasks[j + i] = maski
        if j % 5 == 0:
            print("%d / %d masks created in %0.2fs" %
                  (j + batch_size, npatches, time.time() - tic))
    return pmasks


def refine_masks(stats, patches, seeds, diam, Lyc, Lxc):
    nmasks = len(patches)
    patch_masks = patch_detect(patches, diam)
    ly = patches[0].shape[0] // 2
    igood = np.zeros(nmasks, "bool")
    for i, (patch_mask, stat, (yi, xi)) in enumerate(zip(patch_masks, stats, seeds)):
        mask = np.zeros((Lyc, Lxc), np.float32)
        ypix0, xpix0 = stat["ypix"], stat["xpix"]
        mask[ypix0, xpix0] = stat["lam"]
        func_mask = utils.square_mask(mask, ly, yi, xi)
        ious = utils.mask_ious(patch_mask.astype(np.uint16), (func_mask
                                                              > 0).astype(np.uint16))[0]
        if len(ious) > 0 and ious.max() > 0.45:
            mask_id = np.argmax(ious) + 1
            patch_mask = patch_mask[max(0, ly - yi):min(2 * ly, Lyc + ly - yi),
                                    max(0, ly - xi):min(2 * ly, Lxc + ly - xi)]
            func_mask = func_mask[max(0, ly - yi):min(2 * ly, Lyc + ly - yi),
                                  max(0, ly - xi):min(2 * ly, Lxc + ly - xi)]
            ypix0, xpix0 = np.nonzero(patch_mask == mask_id)
            lam0 = func_mask[ypix0, xpix0]
            lam0[lam0 <= 0] = lam0.min()
            ypix0 = ypix0 + max(0, yi - ly)
            xpix0 = xpix0 + max(0, xi - ly)
            igood[i] = True
            stat["ypix"] = ypix0
            stat["xpix"] = xpix0
            stat["lam"] = lam0
            stat["anatomical"] = True
        else:
            stat["anatomical"] = False
    return stats


def roi_detect(mproj, diameter=None, cellprob_threshold=0.0, flow_threshold=1.5,
               pretrained_model=None, do_uncanny=True, mimg=None, model_path=''):
    pretrained_model = "cyto3" if pretrained_model is None else pretrained_model
    if not os.path.exists(pretrained_model):
        model = Cellpose(model_type=pretrained_model)
    else:
        model = CellposeModel(pretrained_model=pretrained_model)
    masks = model.eval(mproj, channels=[0, 0], diameter=diameter,
                       cellprob_threshold=cellprob_threshold,
                       flow_threshold=flow_threshold)[0]
    shape = masks.shape
    _, masks = np.unique(np.int32(masks), return_inverse=True)
    masks = masks.reshape(shape)
    centers, mask_diams = mask_centers(masks)
    median_diam = np.median(mask_diams)

    if do_uncanny:
        if mimg is None:
            mimg = mproj
        # masks = uncanny_detect(masks, mimg, diameter=median_diam)
        masks = uncanny.detect(masks, mimg, model_path, diameter=median_diam)
        centers, mask_diams = mask_centers(masks)
        median_diam = np.median(mask_diams)

    print(">>>> %d masks detected, median diameter = %0.2f " %
          (masks.max(), median_diam))
    return masks, centers, median_diam, mask_diams.astype(np.int32)


# old deprecated, replaced by LGN UNet uncanny
def uncanny_detect(masks, mimg, diameter=None, dilation=2, thres=75):
    """
    The masks coming out of Cellpose look like shit (too dilated, no clear edges).
    This method implements a custom Canny edge detection algorithm to refine the
    ROI masks. Roughly, it works as follow (see wikipedia article on Canny):
        1. Conduct a top-hat morphological transform on the mean image,
        2. Gaussian smooth that bitch with (3, 3) kernel,
        3. Compute the edge gradient and direction by Sobel filter,
        4. For each ROI, apply a 75th percentile threshold on the gradient to
        find the edges,
        5. Thin the edges by lower bound thresholding,
        6. Apply morphological closing to connect the edges,
        7. Detect and fill in the contours,
        8. Et voila!

    Parameters
    ----------
    masks (array)
        straight out of roi_detect()
    mimg (array)
        mean stack image (or max projection) without registration boundaries
    diameter (int / float)
        Cellpose extimated diameter of the ROIs
    dilation (int)
        number of pixels to extend past the borders of the Cellpose ROI for
        gradient computation
    thres (float)
        percentile threshold for edge detection over edge gradient

    Returns
    -------
    refined_masks (array)
    """
    diameter = np.round(diameter).astype(int)
    # top-hat transform mean stack
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, ) * 2)
    mimg = cv2.morphologyEx(mimg, cv2.MORPH_TOPHAT, kernel)

    # smooth that bitch
    smooth = cv2.GaussianBlur(mimg, (3, 3), sigmaX=0, sigmaY=0)
    # apply Sobel filter and compute gradient magnitudes
    x = cv2.Sobel(src=smooth, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
    y = cv2.Sobel(src=smooth, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
    gradient = np.sqrt(x**2 + y**2)
    direction = np.arctan2(y, x)

    refined_masks = np.zeros_like(masks)
    counter = 1
    # Canny edge detection for individual ROIs
    for n in range(1, np.max(masks) + 1):
        # obtain local gradient and direction for ROI
        ry, rx = np.nonzero(masks == n)
        ry = [np.min(ry), np.max(ry)]
        rx = [np.min(rx), np.max(rx)]
        g = gradient[np.ix_(np.arange(np.max([ry[0] - dilation, 0]), np.min([ry[1] + dilation + 1, masks.shape[0]])), \
                            np.arange(np.max([rx[0] - dilation, 0]), np.min([rx[1] + dilation + 1, masks.shape[1]])))]
        d = direction[np.ix_(np.arange(np.max([ry[0] - dilation, 0]), np.min([ry[1] + dilation + 1, masks.shape[0]])), \
                            np.arange(np.max([rx[0] - dilation, 0]), np.min([rx[1] + dilation + 1, masks.shape[1]])))]
        m = masks[np.ix_(np.arange(np.max([ry[0] - dilation, 0]), np.min([ry[1] + dilation + 1, masks.shape[0]])), \
                            np.arange(np.max([rx[0] - dilation, 0]), np.min([rx[1] + dilation + 1, masks.shape[1]])))] == n

        # thresholding to find edges
        prct = np.percentile(g, thres)
        d[d < 0] = np.pi + d[d < 0]
        d = np.digitize(d, [np.pi, np.pi * 7 / 8, np.pi * 5 / 8, np.pi * 3 / 8, np.pi / 8, 0])
        d[d == 5] = 1
        thinned = (g > prct) & m
        d = d * thinned

        # edge thinning by lower bound thresholding
        comp = np.zeros((2, d.shape[0], d.shape[1]))
        comp[0, :, :] += np.roll(np.roll(d == 1, shift=1, axis=1) * g, shift=-1, axis=1)
        comp[1, :, :] += np.roll(np.roll(d == 1, shift=-1, axis=1) * g, shift=1, axis=1)
        comp[0, :, :] += np.roll(np.roll(d == 2, shift=(1, -1), axis=(0, 1)) * g, shift=(-1, 1), axis=(0, 1))
        comp[1, :, :] += np.roll(np.roll(d == 2, shift=(-1, 1), axis=(0, 1)) * g, shift=(1, -1), axis=(0, 1))
        comp[0, :, :] += np.roll(np.roll(d == 3, shift=1, axis=0) * g, shift=-1, axis=0)
        comp[1, :, :] += np.roll(np.roll(d == 3, shift=-1, axis=0) * g, shift=1, axis=0)
        comp[0, :, :] += np.roll(np.roll(d == 4, shift=(1, 1), axis=(0, 1)) * g, shift=(-1, -1), axis=(0, 1))
        comp[1, :, :] += np.roll(np.roll(d == 4, shift=(-1, -1), axis=(0, 1)) * g, shift=(1, 1), axis=(0, 1))

        thinned = np.all(thinned & (g > comp), axis=0).astype(np.uint8)

        # fill contours
        kernel = np.ceil(np.sqrt(np.sum(m) / np.pi)).astype(int)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel,) * 2)

        for _ in range(2): # do it twice for good luck :)
            thinned = cv2.morphologyEx(thinned, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(thinned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for c in contours:
                if len(c) < 5:
                    continue
                thinned += cv2.drawContours(thinned, [c], -1, 1, cv2.FILLED)
            thinned[thinned > 0] = 1
        thinned = thinned.astype(bool)
        thinned &= m

        if np.sum(thinned) == 0:
            continue

        refined_masks[np.ix_(np.arange(np.max([ry[0] - dilation, 0]), np.min([ry[1] + dilation + 1, masks.shape[0]])), \
                            np.arange(np.max([rx[0] - dilation, 0]), np.min([rx[1] + dilation + 1, masks.shape[1]])))] \
                            += thinned * counter
        counter += 1

    print('Finished Uncanny edge refinement. ' + str(counter - 1) + ' ROIs remaining out of ' + str(np.max(masks).astype(int)) + ' original masks.')

    return refined_masks



def masks_to_stats(masks, weights):
    stats = []
    slices = find_objects(masks)
    for i, si in enumerate(slices):
        sr, sc = si
        ypix0, xpix0 = np.nonzero(masks[sr, sc] == (i + 1))
        ypix0 = ypix0.astype(int) + sr.start
        xpix0 = xpix0.astype(int) + sc.start
        ymed = np.median(ypix0)
        xmed = np.median(xpix0)
        imin = np.argmin((xpix0 - xmed)**2 + (ypix0 - ymed)**2)
        xmed = xpix0[imin]
        ymed = ypix0[imin]
        stats.append({
            "ypix": ypix0,
            "xpix": xpix0,
            "lam": weights[ypix0, xpix0],
            "med": [ymed, xmed],
            "footprint": 1
        })
    stats = np.array(stats)
    return stats


def select_rois(ops: Dict[str, Any], mov: np.ndarray, diameter=None):
    """ find ROIs in static frames
    
    Parameters:

        ops: dictionary
            requires keys "high_pass", "anatomical_only", optional "yrange", "xrange"
        
        mov: ndarray t x Lyc x Lxc, binned movie
    
    Returns:
        stats: list of dicts
    
    """
    Lyc, Lxc = mov.shape[1:]
    mean_img = mov.mean(axis=0)
    mov = utils.temporal_high_pass_filter(mov=mov, width=int(ops["high_pass"]))
    max_proj = mov.max(axis=0)
    #max_proj = np.percentile(mov, 90, axis=0) #.mean(axis=0)
    if ops["anatomical_only"] == 1:
        img = np.log(np.maximum(1e-3, max_proj / np.maximum(1e-3, mean_img)))
        weights = max_proj
    elif ops["anatomical_only"] == 2:
        img = mean_img
        weights = 0.1 + np.clip(
            (mean_img - np.percentile(mean_img, 1)) /
            (np.percentile(mean_img, 99) - np.percentile(mean_img, 1)), 0, 1)
    elif ops["anatomical_only"] == 3:
        if "meanImgE" in ops:
            img = ops["meanImgE"][ops["yrange"][0]:ops["yrange"][1],
                                  ops["xrange"][0]:ops["xrange"][1]]
        else:
            img = mean_img
            print("no enhanced mean image, using mean image instead")
        weights = 0.1 + np.clip(
            (mean_img - np.percentile(mean_img, 1)) /
            (np.percentile(mean_img, 99) - np.percentile(mean_img, 1)), 0, 1)
    else:
        img = max_proj.copy()
        weights = max_proj

    t0 = time.time()
    if diameter is not None:
        if isinstance(diameter, (list, np.ndarray)) and len(ops["diameter"]) > 1:
            rescale = diameter[1] / diameter[0]
            img = cv2.resize(img, (Lxc, int(Lyc * rescale)))
        else:
            rescale = 1.0
            diameter = [diameter, diameter]
        if diameter[1] > 0:
            print("!NOTE! diameter set to %0.2f for cell detection with cellpose" %
                  diameter[1])
        else:
            print(
                "!NOTE! diameter set to 0 or None, diameter will be estimated by cellpose"
            )
    else:
        print(
            "!NOTE! diameter set to 0 or None, diameter will be estimated by cellpose")

    if ops.get("spatial_hp_cp", 0):
        img = np.clip(normalize99(img), 0, 1)
        img -= gaussian_filter(img, diameter[1] * ops["spatial_hp_cp"])

    masks, centers, median_diam, mask_diams = roi_detect(
        img, diameter=diameter[1], flow_threshold=ops["flow_threshold"],
        cellprob_threshold=ops["cellprob_threshold"],
        pretrained_model=ops["pretrained_model"],
        do_uncanny=ops['uncanny'], mimg=mean_img, model_path=ops['uncanny_model'])
    if rescale != 1.0:
        masks = cv2.resize(masks, (Lxc, Lyc), interpolation=cv2.INTER_NEAREST)
        img = cv2.resize(img, (Lxc, Lyc))
    stats = masks_to_stats(masks, weights)
    print("Detected %d ROIs, %0.2f sec" % (len(stats), time.time() - t0))

    new_ops = {
        "diameter": median_diam,
        "max_proj": max_proj,
        "Vmax": 0,
        "ihop": 0,
        "Vsplit": 0,
        "Vcorr": img,
        "Vmap": 0,
        "spatscale_pix": 0
    }
    ops.update(new_ops)

    return stats


# def run_assist():
#     nmasks, diam = 0, None
#     if anatomical:
#         try:
#             print(">>>> CELLPOSE estimating spatial scale and masks as seeds for functional algorithm")
#             from . import anatomical
#             mproj = np.log(np.maximum(1e-3, max_proj / np.maximum(1e-3, mean_img)))
#             masks, centers, diam, mask_diams = anatomical.roi_detect(mproj)
#             nmasks = masks.max()
#         except:
#             print("ERROR importing or running cellpose, continuing without anatomical estimates")
#         if tj < nmasks:
#             yi, xi = centers[tj]
#             ls = mask_diams[tj]
#             imap = np.ravel_multi_index((yi, xi), (Lyc, Lxc))
# if nmasks > 0:
#         stats = anatomical.refine_masks(stats, patches, seeds, diam, Lyc, Lxc)
#         for stat in stats:
#             if stat["anatomical"]:
#                 stat["lam"] *= sdmov[stat["ypix"], stat["xpix"]]

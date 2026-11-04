import os.path as osp
import argparse
import sys
import os

default_n_threads = 8
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"

import numpy as np
import torch
from tqdm import tqdm

from utils.io_utils import find_all_imgs
from utils import inference_utils
from utils.inference_utils import apply_layerdiff, apply_marigold, further_extr
from utils.torch_utils import seed_everything
from utils.deleted import prepare_deleted

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='workspace/layerdiff_output')
    parser.add_argument('--srcp', type=str, default='assets/test_image.png', help='input image')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--repo_id_layerdiff', default='layerdifforg/seethroughv0.0.2_layerdiff3d')
    parser.add_argument('--repo_id_depth', default='24yearsold/seethroughv0.0.1_marigold')
    parser.add_argument('--vae_ckpt', default=None)
    parser.add_argument('--unet_ckpt', default=None)
    parser.add_argument('--resolution', type=int, default=1280, help="inference resolution of layerdiff")
    parser.add_argument('--resolution_depth', type=int, default=768, help="inference resolution of depth model, seethroughv0.0.1_marigold was trained at 768, setting it to -1 will align with layerdiff")
    parser.add_argument('--inference_steps', type=int, default=30, help="inference steps of layerdiff")
    parser.add_argument('--inference_steps_depth', type=int, default=-1, help="inference steps of depth model")
    parser.add_argument('--save_to_psd', action='store_true')
    parser.add_argument('--save_deleted', action='store_true',
                        help='save original pixels missed by pre-inpainting segmentation in a hidden Deleted layer; export at original canvas size')
    parser.add_argument('--segmentation_masks', default=None,
                        help='optional original-coordinate foreground masks JSON from batch_save_masks, used with --save_deleted instead of SemanticSAM')
    parser.add_argument('--tblr_split', action='store_true', help='try split parts (handwear, eyes, etc) into left-right components')
    parser.add_argument('--disable_progressbar', action='store_true', help='hide progressbar')
    parser.add_argument('--group_offload', action='store_true')
    args = parser.parse_args()
    srcp = args.srcp
    if args.save_deleted and not args.save_to_psd:
        parser.error('--save_deleted requires --save_to_psd')
    if args.segmentation_masks and (not args.save_deleted or osp.isdir(srcp)):
        parser.error('--segmentation_masks requires --save_deleted and a single input image')

    if osp.isdir(srcp):
        imglist = find_all_imgs(srcp, abs_path=True)
    else:
        imglist = [srcp]

    # Finish segmentation before any cached LayerDiff/Marigold pipelines occupy
    # VRAM, including directory runs with more than one image.
    if args.save_deleted:
        for srcp in tqdm(imglist, desc='Original segmentation'):
            srcname = osp.basename(osp.splitext(srcp)[0])
            saved = osp.join(args.save_dir, srcname)
            prepare_deleted(srcp, saved, args.resolution, masks_path=args.segmentation_masks)

    for srcp in tqdm(imglist):

        seed_everything(args.seed)

        print('running layerdiff...')
        apply_layerdiff(srcp, args.repo_id_layerdiff, save_dir=args.save_dir, seed=args.seed, vae_ckpt=args.vae_ckpt, unet_ckpt=args.unet_ckpt, \
            resolution=args.resolution, disable_progressbar=args.disable_progressbar, num_inference_steps=args.inference_steps, group_offload=args.group_offload)

        print('running marigold...')
        apply_marigold(srcp, args.repo_id_depth, save_dir=args.save_dir, seed=args.seed, disable_progressbar=args.disable_progressbar, \
            resolution=args.resolution_depth, num_inference_steps=args.inference_steps_depth, group_offload=args.group_offload)

        srcname = osp.basename(osp.splitext(srcp)[0])
        saved = osp.join(args.save_dir, srcname)
        further_extr(saved, rotate=False, save_to_psd=args.save_to_psd, tblr_split=args.tblr_split,
                     save_deleted=args.save_deleted)

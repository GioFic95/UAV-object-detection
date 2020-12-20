# Author: Ankush Gupta
# Date: 2015

"""
Main script for synthetic text rendering.
"""

from __future__ import division
import cv2
import numpy as np
import matplotlib.pyplot as plt
import text_utils as tu
from colorize3_poisson import Colorize
from common import *
import traceback, itertools


class TextRegions(object):
    """
    Get region from segmentation which are good for placing
    text.
    """
    minWidth = 30  # px
    minHeight = 30  # px
    minAspect = 0.3  # w > 0.3*h
    maxAspect = 7
    minArea = 100  # number of pix
    pArea = 0.60  # area_obj/area_minrect >= 0.6

    # RANSAC planar fitting params:
    dist_thresh = 0.10  # m
    num_inlier = 90
    ransac_fit_trials = 1000
    min_z_projection = 0.25

    minW = 20


def viz_textbb(fignum, text_im, bb_list, alpha=1.0):
    """
    text_im : image containing text
    bb_list : list of 2x4xn_i boundinb-box matrices
    """
    plt.close(fignum)
    plt.figure(fignum)
    plt.imshow(text_im)
    H, W = text_im.shape[:2]
    for i in range(len(bb_list)):
        bbs = bb_list[i]
        ni = bbs.shape[-1]
        for j in range(ni):
            bb = bbs[:, :, j]
            bb = np.c_[bb, bb[:, 0]]
            plt.plot(bb[0, :], bb[1, :], 'r', linewidth=2, alpha=alpha)
    plt.gca().set_xlim([0, W - 1])
    plt.gca().set_ylim([H - 1, 0])
    plt.show(block=False)


class RendererV3(object):

    def __init__(self, data_dir, max_time=None):
        self.text_renderer = tu.RenderFont(data_dir)
        self.colorizer = Colorize(data_dir)
        # self.colorizerV2 = colorV2.Colorize(data_dir)

        self.min_char_height = 8  # px
        self.min_asp_ratio = 0.4  #

        self.max_text_regions = 7

        self.max_time = max_time

    def get_min_h(selg, bb, text):
        # find min-height:
        h = np.linalg.norm(bb[:, 3, :] - bb[:, 0, :], axis=0)
        # remove newlines and spaces:
        assert len(text) == bb.shape[-1]

        alnum = np.array([ch.isalnum() for ch in text])
        h = h[alnum]
        return np.min(h)

    def feather(self, text_mask, min_h):
        # determine the gaussian-blur std:
        if min_h <= 15:
            bsz = 0.25
            ksz = 1
        elif 15 < min_h < 30:
            bsz = max(0.30, 0.5 + 0.1 * np.random.randn())
            ksz = 3
        else:
            bsz = max(0.5, 1.5 + 0.5 * np.random.randn())
            ksz = 5
        return cv2.GaussianBlur(text_mask, (ksz, ksz), bsz)

    def place_text(self, rgb, shape, start, rot):
        font = self.text_renderer.font_state.sample()
        font = self.text_renderer.font_state.init_font(font)

        render_res = self.text_renderer.render_sample(font, shape, start, rot)
        print("render_res is None:", render_res is None)
        if render_res is None:  # rendering not successful
            return
        else:
            text_mask, bb, text = render_res

        # get the minimum height of the character-BB:
        min_h = self.get_min_h(bb, text)

        # feathering:
        text_mask = self.feather(text_mask, min_h)

        im_final, col_name = self.colorizer.color(rgb, [text_mask], np.array([min_h]))

        return im_final, text, bb, col_name

    def char2wordBB(self, charBB, text):
        """
        Converts character bounding-boxes to word-level
        bounding-boxes.

        charBB : 2x4xn matrix of BB coordinates
        text   : the text string

        output : 2x4xm matrix of BB coordinates,
                 where, m == number of words.
        """
        wrds = text.split()
        bb_idx = np.r_[0, np.cumsum([len(w) for w in wrds])]
        wordBB = np.zeros((2, 4, len(wrds)), 'float32')

        for i in range(len(wrds)):
            cc = charBB[:, :, bb_idx[i]:bb_idx[i + 1]]

            # fit a rotated-rectangle:
            # change shape from 2x4xn_i -> (4*n_i)x2
            cc = np.squeeze(np.concatenate(np.dsplit(cc, cc.shape[-1]), axis=1)).T.astype('float32')
            rect = cv2.minAreaRect(cc.copy())
            box = np.array(cv2.boxPoints(rect))

            # find the permutation of box-coordinates which
            # are "aligned" appropriately with the character-bb.
            # (exhaustive search over all possible assignments):
            cc_tblr = np.c_[cc[0, :],
                            cc[-3, :],
                            cc[-2, :],
                            cc[3, :]].T
            perm4 = np.array(list(itertools.permutations(np.arange(4))))
            dists = []
            for pidx in range(perm4.shape[0]):
                d = np.sum(np.linalg.norm(box[perm4[pidx], :] - cc_tblr, axis=1))
                dists.append(d)
            wordBB[:, :, i] = box[perm4[np.argmin(dists)], :].T

        return wordBB

    def render_text(self, rgb, starts, rots, ninstance=1, viz=False):
        """
        rgb   : HxWx3 image rgb values (uint8)
        depth : HxW depth values (float)
        seg   : HxW segmentation region masks
        area  : number of pixels in each region
        label : region labels == unique(seg) / {0}
               i.e., indices of pixels in SEG which
               constitute a region mask
        ninstance : no of times image should be
                    used to place text.

        @return:
            res : a list of dictionaries, one for each of 
                  the image instances.
                  Each dictionary has the following structure:
                      'img' : rgb-image with text on it.
                      'bb'  : 2x4xn matrix of bounding-boxes
                              for each character in the image.
                      'txt' : a list of strings.

                  The correspondence b/w bb and txt is that
                  i-th non-space white-character in txt is at bb[:,:,i].
            
            If there's an error in pre-text placement, for e.g. if there's 
            no suitable region for text placement, an empty list is returned.
        """

        # finally place some text:
        nregions = len(starts)
        if nregions < 1:  # no good region to place text on
            print("no good region to place text on")
            return []

        res = []
        for i in range(ninstance):
            idict = {'img': [], 'charBB': None, 'wordBB': None, 'txt': None, 'color': None}
            placed = False
            img = rgb.copy()
            itext = []
            ibb = []
            icol = []

            # process regions: 
            for idx in range(nregions):
                try:
                    txt_render_res = self.place_text(img, img.shape[:2], starts[idx], rots[idx])
                except:
                    traceback.print_exc()
                    # some error in placing text on the region
                    continue

                if txt_render_res is not None:
                    placed = True
                    img, text, bb, col_name = txt_render_res
                    print("text synth:", text)
                    # store the result:
                    itext.append(text)
                    ibb.append(bb)
                    icol.append(col_name)
                else:
                    print("txt_render_res is None")

            if placed:
                # at least 1 word was placed in this instance:
                idict['img'] = img
                idict['txt'] = itext
                idict['charBB'] = np.concatenate(ibb, axis=2)
                idict['wordBB'] = self.char2wordBB(idict['charBB'].copy(), ' '.join(itext))
                idict['color'] = icol
                res.append(idict.copy())
                if viz:
                    viz_textbb(1, img, [idict['wordBB']], alpha=1.0)
                    if i < ninstance - 1:
                        raw_input(colorize(Color.BLUE, 'continue?', True))
        return res

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import os.path as osp
import random, os
import cv2 as cv
import pygame, pygame.locals
from pygame import freetype
import math
from common import *
import pickle


def move_bb(bbs, t):
    """
    Translate the bounding-boxes in by t_x,t_y.
    BB : 2x4xn
    T  : 2-long np.array
    """
    return bbs + t[:, None, None]


def crop_safe(arr, rect, bbs=[], pad=0):
    """
    ARR : arr to crop
    RECT: (x,y,w,h) : area to crop to
    BBS : nx4 xywh format bounding-boxes
    PAD : percentage to pad

    Does safe cropping. Returns the cropped rectangle and
    the adjusted bounding-boxes
    """
    rect = np.array(rect)
    rect[:2] -= pad
    rect[2:] += 2 * pad
    v0 = [max(0, rect[0]), max(0, rect[1])]
    v1 = [min(arr.shape[0], rect[0] + rect[2]), min(arr.shape[1], rect[1] + rect[3])]
    arr = arr[v0[0]:v1[0], v0[1]:v1[1], ...]
    if len(bbs) > 0:
        for i in range(len(bbs)):
            bbs[i, 0] -= v0[0]
            bbs[i, 1] -= v0[1]
        # print("crop safe", arr, bbs)
        return arr, bbs
    else:
        # print("crop safe", arr, bbs)
        return arr


class BaselineState(object):
    curve = lambda this, a: lambda x: a * x * x
    differential = lambda this, a: lambda x: 2 * a * x
    a = [0.50, 0.05]

    def get_sample(self):
        """
        Returns the functions for the curve and differential for a and b
        """
        sgn = 1.0
        if np.random.rand() < 0.5:
            sgn = -1

        a = self.a[1] * np.random.randn() + sgn * self.a[0]
        return {
            'curve': self.curve(a),
            'diff': self.differential(a),
        }


class RenderFont(object):
    """
    Outputs a rasterized font sample.
        Output is a binary mask matrix cropped closesly with the font.
        Also, outputs ground-truth bounding boxes and text string
    """

    def __init__(self, data_dir='data'):
        # distribution over the type of text:
        # whether to get a single word, paragraph or a line:
        self.p_text = {0.0: 'WORD',
                       0.0: 'LINE',
                       1.0: 'PARA'}

        # TEXT PLACEMENT PARAMETERS:
        self.f_shrink = 0.90
        self.max_shrink_trials = 100  # 0.9^5 ~= 0.6
        # the minimum number of characters that should fit in a mask
        # to define the maximum font height.
        self.min_nchar = 1  # 2  # todo check
        self.min_font_h = 0  # px: 0.6*12 ~ 7px <= actual minimum height
        self.max_font_h = 22  # px
        self.p_flat = 0.10

        # curved baseline:
        self.p_curved = 1.0
        self.baselinestate = BaselineState()

        # text-source : gets english text:
        self.text_source = TextSource(min_nchar=self.min_nchar, fn=osp.join(data_dir, 'newsgroup/text.txt'))

        # get font-state object:
        self.font_state = FontState(data_dir)

        pygame.init()

    def render_multiline(self, font, text):
        """
        renders multiline TEXT on the pygame surface SURF with the
        font style FONT.
        A new line in text is denoted by \n, no other characters are 
        escaped. Other forms of white-spaces should be converted to space.

        returns the updated surface, words and the character bounding boxes.
        """
        # get the number of lines
        lines = text.split('\n')
        lengths = [len(l) for l in lines]

        # font parameters:
        line_spacing = font.get_sized_height() + 1

        # initialize the surface to proper size:
        line_bounds = font.get_rect(lines[np.argmax(lengths)])
        fsize = (round(2.0 * line_bounds.width), round(1.25 * line_spacing * len(lines)))
        surf = pygame.Surface(fsize, pygame.locals.SRCALPHA, 32)

        bbs = []
        space = font.get_rect('O')
        x, y = 0, 0
        for l in lines:
            x = 0  # carriage-return
            y += line_spacing  # line-feed

            for ch in l:  # render each character
                if ch.isspace():  # just shift
                    x += space.width
                else:
                    # render the character
                    ch_bounds = font.render_to(surf, (x, y), ch)
                    ch_bounds.x = x + ch_bounds.x
                    ch_bounds.y = y - ch_bounds.y
                    x += ch_bounds.width
                    bbs.append(np.array(ch_bounds))

        # get the union of characters for cropping:
        r0 = pygame.Rect(bbs[0])
        rect_union = r0.unionall(bbs)

        # get the words:
        words = ' '.join(text.split())

        # crop the surface to fit the text:
        bbs = np.array(bbs)
        surf_arr, bbs = crop_safe(pygame.surfarray.pixels_alpha(surf), rect_union, bbs, pad=5)
        surf_arr = surf_arr.swapaxes(0, 1)
        # self.visualize_bb(surf_arr,bbs)
        return surf_arr, words, bbs

    def render_curved(self, font, word_text):
        """
        use curved baseline for rendering word
        """
        wl = len(word_text)
        isword = len(word_text.split()) == 1

        # do curved iff, the length of the word <= 10
        if not isword or wl > 10 or np.random.rand() > self.p_curved:
            return self.render_multiline(font, word_text)

        # create the surface:
        lspace = font.get_sized_height() + 1
        lbound = font.get_rect(word_text)
        fsize = (round(2.0 * lbound.width), round(3 * lspace))
        surf = pygame.Surface(fsize, pygame.locals.SRCALPHA, 32)

        # baseline state
        mid_idx = wl // 2
        BS = self.baselinestate.get_sample()
        curve = [BS['curve'](i - mid_idx) for i in range(wl)]
        curve[mid_idx] = 0
        rots = [-int(math.degrees(math.atan(BS['diff'](i - mid_idx) / (font.size / 2)))) for i in range(wl)]
        # print("font size:", font.size, "curve:", curve, "rots:", rots)

        bbs = []
        # place middle char
        rect = font.get_rect(word_text[mid_idx])
        rect.centerx = surf.get_rect().centerx
        rect.centery = surf.get_rect().centery + rect.height
        rect.centery += curve[mid_idx]
        ch_bounds = font.render_to(surf, rect, word_text[mid_idx], rotation=rots[mid_idx])
        ch_bounds.x = rect.x + ch_bounds.x
        ch_bounds.y = rect.y - ch_bounds.y
        mid_ch_bb = np.array(ch_bounds)

        # render chars to the left and right:
        last_rect = rect
        ch_idx = []
        for i in range(wl):
            # skip the middle character
            if i == mid_idx:
                bbs.append(mid_ch_bb)
                ch_idx.append(i)
                continue

            if i < mid_idx:  # left-chars
                i = mid_idx - 1 - i
            elif i == mid_idx + 1:  # right-chars begin
                last_rect = rect

            ch_idx.append(i)
            ch = word_text[i]

            newrect = font.get_rect(ch)
            newrect.y = last_rect.y
            if i > mid_idx:
                newrect.topleft = (last_rect.topright[0] + 2, newrect.topleft[1])
            else:
                newrect.topright = (last_rect.topleft[0] - 2, newrect.topleft[1])
            newrect.centery = max(newrect.height, min(fsize[1] - newrect.height, newrect.centery + curve[i]))
            try:
                bbrect = font.render_to(surf, newrect, ch, rotation=rots[i])
            except ValueError:
                bbrect = font.render_to(surf, newrect, ch)
            bbrect.x = newrect.x + bbrect.x
            bbrect.y = newrect.y - bbrect.y
            bbs.append(np.array(bbrect))
            last_rect = newrect

        # correct the bounding-box order:
        bbs_sequence_order = [None for _ in ch_idx]
        for idx, i in enumerate(ch_idx):
            bbs_sequence_order[i] = bbs[idx]
        bbs = bbs_sequence_order

        # get the union of characters for cropping:
        r0 = pygame.Rect(bbs[0])
        rect_union = r0.unionall(bbs)

        # crop the surface to fit the text:
        bbs = np.array(bbs)
        surf_arr, bbs = crop_safe(pygame.surfarray.pixels_alpha(surf), rect_union, bbs, pad=5)
        # print(f"surf_arr: {surf_arr}, bbs: {bbs}")
        surf_arr = surf_arr.swapaxes(0, 1)
        return surf_arr, word_text, bbs

    def place_text(self, text_arrs, shape, bbs, start, rot):
        areas = [-np.prod(ta.shape) for ta in text_arrs]
        order = np.argsort(areas)
        out_arr = np.zeros(shape)
        for i in order:
            w, h = text_arrs[i].shape
            locw = start[3] - start[1]
            loch = start[2] - start[0]
            loc = np.array([start[1] + int(locw/2 - w/2), start[0] + int(loch/2-h/2)])

            # update the bounding-boxes:
            bbs[i] = move_bb(bbs[i], loc[::-1])

            # blit the text onto the canvas
            # print("start:", start, "loc:", loc, "h:", h, "w:", w, "text_arrs:", text_arrs[i].shape,
            #       "out_arr:", out_arr.shape,
            #       "out_arr[...]:", out_arr[loc[0]:loc[0] + w, loc[1]:loc[1] + h].shape)
            out_arr[loc[0]:loc[0] + w, loc[1]:loc[1] + h] += text_arrs[i]
            out_arr[loc[0]-10:loc[0]+w+10, loc[1]-10:loc[1]+h+10] =\
                ndimage.rotate(out_arr[loc[0]-10:loc[0]+w+10, loc[1]-10:loc[1]+h+10], -rot, reshape=False)
        return out_arr, bbs

    def bb_xywh2coords(self, bbs):
        """
        Takes an nx4 bounding-box matrix specified in x,y,w,h
        format and outputs a 2x4xn bb-matrix, (4 vertices per bb).
        """
        n, _ = bbs.shape
        coords = np.zeros((2, 4, n))
        for i in range(n):
            coords[:, :, i] = bbs[i, :2][:, None]
            coords[0, 1, i] += bbs[i, 2]
            coords[:, 2, i] += bbs[i, 2:4]
            coords[1, 3, i] += bbs[i, 3]
        return coords

    def render_sample(self, font, shape, start, rot):
        """
        Places text in the "collision-free" region as indicated
        in the mask -- 255 for unsafe, 0 for safe.
        The text is rendered using FONT, the text content is TEXT.
        """
        H, W = shape
        f_asp = self.font_state.get_aspect_ratio(font)

        # find the maximum height in pixels:
        max_font_h = min(0.9 * H, (1 / f_asp) * W / (self.min_nchar + 1))  # todo check
        max_font_h = min(max_font_h, self.max_font_h)
        # print("h:", H, "w", W, "max_font_h:", max_font_h)
        if max_font_h < self.min_font_h:
            print("not possible to place any text here: max_font_h < self.min_font_h")
            return

        # let's just place one text-instance for now
        i = 0
        while i < self.max_shrink_trials and max_font_h > self.min_font_h:
            # sample a random font-height:
            # print("font-height:", f_h_px, self.min_font_h, max_font_h)
            f_h_px = max_font_h
            # convert from pixel-height to font-point-size:
            f_h = self.font_state.get_font_size(font, f_h_px)

            font.size = f_h  # set the font-size

            # compute the max-number of lines/chars-per-line:
            nline, nchar = 1, 1
            print("  > nline = %d, nchar = %d" % (nline, nchar))

            assert nline >= 1 and nchar >= self.min_nchar

            # sample text:
            text_type = "WORD"
            text = self.text_source.sample(nline, nchar, text_type)
            if len(text) == 0 or np.any([len(line) == 0 for line in text]):
                print("len == 0")
                continue

            # render the text:
            txt_arr, txt, bb = self.render_curved(font, text)
            bb = self.bb_xywh2coords(bb)
            # print(f"txt_arr: {txt_arr}, txt: {txt}, bb: {bb}")

            # make sure that the text-array is not bigger than mask array:
            if np.any(np.r_[txt_arr.shape[:2]] > np.r_[shape]):
                print("text-array is bigger than mask")
                continue

            # position the text within the mask:
            text_mask, bb = self.place_text([txt_arr], shape, [bb], start, rot)
            # print(f"text_mask: {text_mask}, bb: {bb}")
            if text_mask.any():  # successful in placing the text collision-free:
                print("successful in placing the text collision-free")
                return text_mask, bb[0], text
        return  # None

    def visualize_bb(self, text_arr, bbs):
        ta = text_arr.copy()
        for r in bbs:
            cv.rectangle(ta, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), color=128, thickness=1)
        plt.imshow(ta, cmap='gray')
        plt.show()


class FontState(object):
    """
    Defines the random state of the font rendering  
    """
    size = [50, 10]  # normal dist mean, std
    underline = 0.05
    strong = 0  # 0.5
    oblique = 0  # 0.2
    wide = 0.5
    strength = [0.05, 0.1]  # uniform dist in this interval
    underline_adjustment = [1.0, 2.0]  # normal dist mean, std
    kerning = [2, 5, 0, 20]  # beta distribution alpha, beta, offset, range (mean is a/(a+b))
    border = 0.25
    random_caps = -1  # don't recapitalize : retain the capitalization of the lexicon
    capsmode = [str.lower, str.upper, str.capitalize]  # lower case, upper case, proper noun
    curved = 0  # 0.2
    random_kerning = 0.2
    random_kerning_amount = 0.1

    def __init__(self, data_dir='data'):

        char_freq_path = osp.join(data_dir, 'models/char_freq.cp')
        font_model_path = osp.join(data_dir, 'models/font_px2pt.cp')

        # get character-frequencies in the English language:
        with open(char_freq_path, 'rb') as f:
            # self.char_freq = cp.load(f)
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            p = u.load()
            self.char_freq = p

        # get the model to convert from pixel to font pt size:
        with open(font_model_path, 'rb') as f:
            # self.font_model = cp.load(f)
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            p = u.load()
            self.font_model = p

        # get the names of fonts to use:
        self.FONT_LIST = osp.join(data_dir, 'fonts/fontlist.txt')
        self.fonts = [os.path.join(data_dir, 'fonts', f.strip()) for f in open(self.FONT_LIST)]

    def get_aspect_ratio(self, font, size=None):
        """
        Returns the median aspect ratio of each character of the font.
        """
        if size is None:
            size = 12  # doesn't matter as we take the RATIO
        chars = ''.join(self.char_freq.keys())
        w = np.array(self.char_freq.values())

        # get the [height,width] of each character:
        try:
            sizes = font.get_metrics(chars, size)
            good_idx = [i for i in range(len(sizes)) if sizes[i] is not None]
            sizes, w = [sizes[i] for i in good_idx], w[good_idx]
            sizes = np.array(sizes).astype('float')[:, [3, 4]]
            r = np.abs(sizes[:, 1] / sizes[:, 0])  # width/height
            good = np.isfinite(r)
            r = r[good]
            w = w[good]
            w /= np.sum(w)
            r_avg = np.sum(w * r)
            return r_avg
        except:
            return 1.0

    def get_font_size(self, font, font_size_px):
        """
        Returns the font-size which corresponds to FONT_SIZE_PX pixels font height.
        """
        m = self.font_model[font.name]
        return m[0] * font_size_px + m[1]  # linear model

    def sample(self):
        """
        Samples from the font state distribution
        """
        return {
            'font': self.fonts[int(np.random.randint(0, len(self.fonts)))],
            'size': self.size[1] * np.random.randn() + self.size[0],
            'underline': np.random.rand() < self.underline,
            'underline_adjustment': max(2.0, min(-2.0, self.underline_adjustment[1] * np.random.randn() +
                                                 self.underline_adjustment[0])),
            'strong': np.random.rand() < self.strong,
            'oblique': np.random.rand() < self.oblique,
            'strength': (self.strength[1] - self.strength[0]) * np.random.rand() + self.strength[0],
            'char_spacing': int(self.kerning[3] * (np.random.beta(self.kerning[0], self.kerning[1])) + self.kerning[2]),
            'border': np.random.rand() < self.border,
            'random_caps': np.random.rand() < self.random_caps,
            'capsmode': random.choice(self.capsmode),
            'curved': np.random.rand() < self.curved,
            'random_kerning': np.random.rand() < self.random_kerning,
            'random_kerning_amount': self.random_kerning_amount,
        }

    def init_font(self, fs):
        """
        Initializes a pygame font.
        FS : font-state sample
        """
        font = freetype.Font(fs['font'], size=fs['size'])
        font.underline = fs['underline']
        font.underline_adjustment = fs['underline_adjustment']
        font.strong = fs['strong']
        font.oblique = fs['oblique']
        font.strength = fs['strength']
        char_spacing = fs['char_spacing']
        font.antialiased = True
        font.origin = True
        return font


class TextSource(object):
    """
    Provides text for words, paragraphs, sentences.
    """

    def __init__(self, min_nchar, fn):
        """
        TXT_FN : path to file containing text data.
        """
        self.min_nchar = min_nchar
        self.fdict = {'WORD': self.sample_word}
                      # 'LINE': self.sample_line,
                      # 'PARA': self.sample_para}

        with open(fn, 'r') as f:
            self.txt = [l.strip() for l in f.readlines()]

        # distribution over line/words for LINE/PARA:
        self.p_line_nline = np.array([0.85, 0.10, 0.05])
        self.p_line_nword = [4, 3, 12]  # normal: (mu, std)
        self.p_para_nline = [1.0, 1.0]  # [1.7,3.0] # beta: (a, b), max_nline
        self.p_para_nword = [1.7, 3.0, 10]  # beta: (a,b), max_nword

        # probability to center-align a paragraph:
        self.center_para = 0.5  # todo check

    def sample(self, nline_max, nchar_max, kind='WORD'):
        return self.fdict[kind](nline_max, nchar_max)

    def sample_word(self, nline_max, nchar_max, niter=100):
        rand_line = self.txt[np.random.choice(len(self.txt))]
        words = rand_line.split()
        rand_word = random.choice(words)
        print(f"r. line: {rand_line}, words: {words}, r. word: {rand_word}")

        iter = 0
        while iter < niter and len(rand_word) > nchar_max:
            rand_line = self.txt[np.random.choice(len(self.txt))]
            words = rand_line.split()
            rand_word = random.choice(words)
            iter += 1
            print(f"r. line: {rand_line}, words: {words}, r. word: {rand_word}")

        if len(rand_word) > nchar_max:
            print("not good word")
            return []
        else:
            return rand_word

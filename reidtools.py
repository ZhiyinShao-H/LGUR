from __future__ import absolute_import
from __future__ import print_function

__all__ = ['visualize_ranked_results']

import numpy as np
import os
import os.path as osp
import shutil
# from os import listdir
from PIL import Image
from PIL import Image,ImageDraw,ImageFont

# from data_process.utils import mkdir_if_missing

def mkdir_if_missing(path):
    if not os.path.isdir(path):
        os.mkdir(path)
def visualize_ranked_results(distmat, query, gallery, save_dir='', topk=10,wrong_indices=None):
    """Visualizes ranked results.

    Supports both image-reid and video-reid.

    Args:
        distmat (numpy.ndarray): distance matrix of shape (num_query, num_gallery).
        dataset (tuple): a 2-tuple containing (query, gallery), each of which contains
            tuples of (img_path(s), pid, camid).
        save_dir (str): directory to save output images.
        topk (int, optional): denoting top-k images in the rank list to be visualized.
        wrong_indices (ndarray): a 2-tuple containing wrong prediction q_pid and the predicted picture index in gallery
    """
    num_q, num_g = distmat.shape

    print('Visualizing top-{} ranks'.format(topk))
    print('# query: {}\n# gallery {}'.format(num_q, num_g))
    print('Saving images to "{}"'.format(save_dir))
    print(len(query.label))
    assert num_q == len(query)
    assert num_g == len(gallery)

    indices = np.argsort(-distmat, axis=1)
    mkdir_if_missing(save_dir)

    def _cp_img_to(src, dst, rank, prefix):
        """
        Args:
            src: image path or tuple (for vidreid)
            dst: target directory
            rank: int, denoting ranked position, starting from 1
            prefix: string
        """
        if isinstance(src, tuple) or isinstance(src, list):
            dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3))
            mkdir_if_missing(dst)
            for img_path in src:
                shutil.copy(img_path, dst)
        else:
            dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3) + '_name_' + osp.basename(src))
            shutil.copy(src, dst)
    HEIGHT = 256
    WIDTH = 128

    for q_idx in range(num_q):
        # if q_idx not in wrong_indices[0]:
        #     continue
        ims = []
        qpid = query.label[q_idx]

        save_img_path = save_dir+'{}.jpg'.format(qpid)
        #'./img\\data\\market1501\\query\\0001_c1s1_001051_00.jpg'
        save_img_path.replace('\\','/')
        #
        # q_im = Image.open(qimg_path).resize((WIDTH, HEIGHT), Image.BILINEAR)
        # ims.append(q_im)

        # if isinstance(qimg_path, tuple) or isinstance(qimg_path, list):
        #     qdir = osp.join(save_dir, osp.basename(qimg_path[0]))
        # else:
        #     qdir = osp.join(save_dir, osp.basename(qimg_path))
        # mkdir_if_missing(qdir)
        # _cp_img_to(qimg_path, qdir, rank=0, prefix='query')

        rank_idx = 1
        for g_idx in indices[q_idx,:]:
            print(indices[q_idx,:])
            print(distmat[q_idx][indices[q_idx,:]])
            gimg_path, gpid = gallery.img_path[g_idx], gallery.label[g_idx]
            # invalid = (qpid == gpid) & (qcamid == gcamid)
        # if not invalid:
            g_im = Image.open(gimg_path).resize((WIDTH, HEIGHT), Image.BILINEAR)
            draw = ImageDraw.Draw(g_im)
            if gpid==qpid:
                color = (0,255,0)#绿色
            else:
                color = (255,0,0)#红色
            draw.text((8, 8), str(gpid), fill=color)#在坐标(8,8)位置打印gpid，颜色为color，对的为绿色，错的为红色
            ims.append(g_im)
            rank_idx += 1
            if rank_idx > topk:
                break
        img_ = Image.new(ims[0].mode, (WIDTH*len(ims), HEIGHT))#制作新图片，由于图片是query+前rankk张图，所以WIDTH要x一个len
        for i, im in enumerate(ims):
            img_.paste(im, box=(i*WIDTH,0))
        img_.save(save_img_path)

    print("Done")
"""
Demo for performing EGCSC/EKGCSC on several HSI datasets.

please cite the following paper:
@article{HSI-Clustering-GCSC-CAI-TGRS-2020,
	title="Graph Convolutional Subspace Clustering: A Robust Subspace Clustering Framework for Hyperspectral Image",
	author="Yaoming {Cai} and Zijia {Zhang} and Zhihua {Cai} and Xiaobo {Liu} and Xinwei {Jiang} and Qin {Yan}",
	journal="IEEE Transactions on Geoscience and Remote Sensing",
	note="doi: 10.1109/TGRS.2020.3018135",
	year="2020"
}

--------------------------------------------------------------
    reference hyper-parameters of EGCSC
    =====  ===========  ===========  ===========
    data    lambda            K          RO
    =====  ===========  ===========  ===========
    SaA      10             30          0.8
    InP      100            30          0.4 (13*13 patch)
    PaU      1000           20          0.6
    ===========================================

    ---------------

    reference hyper-parameters of kernel EGCSC
    =====  ===========  ===========  ===========  ==========
    data    lambda            K          RO         gamma
    =====  ===========  ===========  ===========  ==========
    SaA      100            30          0.8          0.2
    InP      1e3            30          0.8          10 (13*13 patch)
    PaU      6*1e4          30          0.8          100
    ========================================================
"""
import sys

sys.path.append('.')

from ClusteringEvaluator import cluster_accuracy
from EfficientGCSC import GCSC
from EfficientKernelGCSC import GCSC_Kernel
import numpy as np


def order_sam_for_diag(x, y):
    """
    rearrange samples
    :param x: feature sets
    :param y: ground truth
    :return:
    """
    x_new = np.zeros(x.shape)
    y_new = np.zeros(y.shape)
    start = 0
    for i in np.unique(y):
        idx = np.nonzero(y == i)
        stop = start + idx[0].shape[0]
        x_new[start:stop] = x[idx]
        y_new[start:stop] = y[idx]
        start = stop
    return x_new, y_new


if __name__ == '__main__':
    from Toolbox.Preprocessing import Processor
    from sklearn.preprocessing import minmax_scale, normalize
    from sklearn.decomposition import PCA
    import time

    root = 'HSI_Datasets/'
    im_, gt_ = 'SalinasA_corrected', 'SalinasA_gt'
    # im_, gt_ = 'Indian_pines_corrected', 'Indian_pines_gt'
    # im_, gt_ = 'PaviaU', 'PaviaU_gt'

    img_path = root + im_ + '.mat'
    gt_path = root + gt_ + '.mat'
    print('\nDataset: ', img_path)

    PATCH_SIZE = 9  # # 9 default, normally the bigger the better
    nb_comps = 4  # # num of PCs, 4 default, it can be moderately increased
    # load img and gt
    p = Processor()
    img, gt = p.prepare_data(img_path, gt_path)

    # # take a smaller sub-scene for computational efficiency
    if im_ == 'SalinasA_corrected':
        REG_Coef_, NEIGHBORING_, RO_ = 1e1, 30, 0.8
        REG_Coef_K, NEIGHBORING_K, RO_K, GAMMA = 1e2, 30, 0.8, 0.2
    if im_ == 'Indian_pines_corrected':
        img, gt = img[30:115, 24:94, :], gt[30:115, 24:94]
        PATCH_SIZE = 13
        REG_Coef_, NEIGHBORING_, RO_ = 1e2, 30, 0.4
        REG_Coef_K, NEIGHBORING_K, RO_K, GAMMA = 1e3, 30, 0.8, 10
    if im_ == 'PaviaU':
        img, gt = img[150:350, 100:200, :], gt[150:350, 100:200]
        REG_Coef_, NEIGHBORING_, RO_ = 1e3, 20, 0.6
        REG_Coef_K, NEIGHBORING_K, RO_K, GAMMA = 6 * 1e4, 30, 0.8, 100

    n_row, n_column, n_band = img.shape
    x_img = minmax_scale(img.reshape(n_row * n_column, n_band)).reshape((n_row, n_column, n_band))
    print('original img shape: ', x_img.shape)
    # # reduce spectral bands using PCA
    pca = PCA(n_components=nb_comps)
    img = minmax_scale(pca.fit_transform(img.reshape(n_row * n_column, n_band))).reshape(n_row, n_column, nb_comps)
    x_patches, y_ = p.get_HSI_patches_rw(img, gt, (PATCH_SIZE, PATCH_SIZE))

    print('reduced img shape: ', img.shape)
    print('x_patch tensor shape: ', x_patches.shape)
    n_samples, n_width, n_height, n_band = x_patches.shape
    x_patches_2d = np.reshape(x_patches, (n_samples, -1))
    y = p.standardize_label(y_)

    # # reorder samples according to gt
    x_patches_2d, y = order_sam_for_diag(x_patches_2d, y)
    # # normalize data
    x_patches_2d = normalize(x_patches_2d)
    print('final sample shape: %s, labels: %s' % (x_patches_2d.shape, np.unique(y)))
    N_CLASSES = np.unique(y).shape[0]  # Indian : 8  KSC : 10  SalinasA : 6 PaviaU : 8

    # ========================
    # performing  EGCSC
    # ========================
    time_start = time.clock()
    gcsc = GCSC(n_clusters=N_CLASSES, regu_coef=REG_Coef_, n_neighbors=NEIGHBORING_, ro=RO_, save_affinity=False)
    y_pre_gcsc = gcsc.fit(x_patches_2d)
    run_time = round(time.clock() - time_start, 3)
    acc = cluster_accuracy(y, y_pre_gcsc)
    print('=================================\n'
          '\t\tEGCSC RESULTS\n'
          '=================================')
    print('%10s %10s %10s' % ('OA', 'Kappa', 'NMI',))
    print('%10.4f %10.4f %10.4f' % (acc[0], acc[1], acc[2]))
    print('class accuracy:', acc[3])
    print('running time', run_time)
    np.savez(im_ + '-EGCSC-clustering-results-OA=' + str(int(np.round(acc[0] * 100, 2) * 100)) + '.npz',
             y_pre=y_pre_gcsc)

    # ========================
    # performing  EKGCSC
    # ========================
    time_start = time.clock()
    gcsc_k = GCSC_Kernel(n_clusters=N_CLASSES, regu_coef=REG_Coef_K, n_neighbors=NEIGHBORING_K, gamma=GAMMA, ro=RO_K,
                         save_affinity=False)
    y_pre_gcsc_k = gcsc_k.fit(x_patches_2d)
    run_time = round(time.clock() - time_start, 3)
    acc_gcsc_k = cluster_accuracy(y, y_pre_gcsc_k)
    print('=================================\n'
          '\t\tEKGCSC RESULTS\n'
          '=================================')
    print('%10s %10s %10s' % ('OA', 'Kappa', 'NMI',))
    print('%10.4f %10.4f %10.4f' % (acc_gcsc_k[0], acc_gcsc_k[1], acc_gcsc_k[2]))
    print('class accuracy:', acc_gcsc_k[3])
    print('running time', run_time)
    np.savez(im_ + '-EKGCSC-clustering-results-OA=' + str(int(np.round(acc_gcsc_k[0] * 100, 2) * 100)) + '.npz',
             y_pre=y_pre_gcsc_k)

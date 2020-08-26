# Graph Convolutional Subspace Clustering

**This repository includes the implementations of EGCSC and EKGCSC model reported by *"Graph Convolutional Subspace Clustering: A Robust Subspace Clustering Framework for Hyperspectral Image"***


If you would like to acknowledge our efforts, please cite the following paper:

    @article{HSI-Clustering-GCSC-CAI-TGRS-2020,
	title="Graph Convolutional Subspace Clustering: A Robust Subspace Clustering Framework for Hyperspectral Image",
	author="Yaoming {Cai} and Zijia {Zhang} and Zhihua {Cai} and Xiaobo {Liu} and Xinwei {Jiang} and Qin {Yan}",
	journal="IEEE Transactions on Geoscience and Remote Sensing",
	note="doi: 10.1109/TGRS.2020.3018135",
	year="2020",
    }


## Requirements ##

- Python >= 3.5

- Numpy <= 1.16.2

- Munkres 

- SciPy

- Scikit-Learn

- Spectral Python (SPy)


## Running ##

    python demo.py

--------------------------
> Dataset:  HSI\_Datasets/SalinasA_corrected.mat
> 
> original img shape:  (83, 86, 204)
> 
> reduced img shape:  (83, 86, 4)
> 
> x_patch tensor shape:  (5348, 9, 9, 4)
> 
> final sample shape: (5348, 324), labels: [0. 1. 2. 3. 4. 5.]

>  =================================
> 	     EGCSC RESULTS
>   =================================
> 
>         OA      Kappa        NMI
>     0.9993     0.9971     0.9991
> class accuracy: [1.         0.99702159 1.         1.         1.         1.        ]
> 
> running time 42.296

> =================================
> 		EKGCSC RESULTS
>  =================================
> 
>         OA      Kappa        NMI
>     1.0000     1.0000     1.0000
> class accuracy: [1. 1. 1. 1. 1. 1.]
> 
> running time 63.59


--------------------------------------------------------------
**reference hyper-parameters of EGCSC**

    =====  ===========  ===========  ===========
    data    lambda            K          RO
    =====  ===========  ===========  ===========
    SaA      10             30          0.8
    InP      100            30          0.4 (13*13 patch)
    PaU      1000           20          0.6
    ===========================================

**reference hyper-parameters of EKGCSC**

    =====  ===========  ===========  ===========  ==========
    data    lambda            K          RO         gamma
    =====  ===========  ===========  ===========  ==========
    SaA      100            30          0.8          0.2
    InP      1e3            30          0.8          10 (13*13 patch)
    PaU      6*1e4          30          0.8          100
    ========================================================
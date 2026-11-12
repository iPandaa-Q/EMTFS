import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import time

from EMTO import (
    su, discretize, mic_continuous_discrete, coarse_clustering_dual,
    compute_upper_bounds_cluster, compute_cluster_weights,
    pso_cso, pso_dual_tckt, knn_predict
)

if __name__ == "__main__":
    print("正在加载本地 Colon.mat 数据...")
    mat = scipy.io.loadmat('colon.mat')
    X = mat['X'].astype(float)
    y = mat['Y'].ravel().astype(int)
    
    # 把 -1 和 1 转换成 0 和 1
    if y.min() == -1:
        y = (y == 1).astype(int)

    print(f"数据加载完成: {X.shape} (样本数, 特征数)")

    # 划分数据 (70% 训练, 30% 测试)
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(y))
    split = int(0.7 * len(y))
    X_train, y_train = X[idx[:split]], y[idx[:split]]
    X_test, y_test = X[idx[split:]], y[idx[split:]]

    # 标准化
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma
    D = X_train.shape[1]
    theta = 0.6

    start_time = time.time()
    
    # 预计算
    su_scores = np.array([su(discretize(X_train[:, d]), y_train) for d in range(D)])
    mic_scores = np.array([mic_continuous_discrete(X_train[:, d], y_train) for d in range(D)])
    cluster_ids, K = coarse_clustering_dual(su_scores, mic_scores, D)

    u = compute_upper_bounds_cluster(mic_scores, cluster_ids, theta)
    w = compute_cluster_weights(mic_scores, cluster_ids, K)

    print(f"MIC 的最大值: {np.max(mic_scores)}")
    print(f"SU 的最大值: {np.max(su_scores)}")
    print(f"u 的范围: {u.min()} 到 {u.max()}")
    
    print(f"预计算完成！耗时 {time.time() - start_time:.2f} 秒")
    print(f"粗聚类结果: K={K}")

    # 适当调整可以减负
    N, T, folds = 100, 100, 10
    n_seeds = 20

    results = {
        'cso_acc': [], 'cso_n': [], 'cso_fit': [],
        'tckt_acc': [], 'tckt_n': [], 'tckt_fit': []
    }
    curve_cso, curve_tckt = [], []

    for s in range(n_seeds):
        start = time.time()
        gb_cso, fit_cso, hist_cso = pso_cso(X_train, y_train, D, u=np.ones(D), v_max=0.15, N=N, T=200, theta=theta, seed=s, folds=folds)
        sel_b = gb_cso > theta
        acc_b = np.mean(knn_predict(X_train[:, sel_b], y_train, X_test[:, sel_b], k=5) == y_test) if sel_b.sum() > 0 else 0
        results['cso_fit'].append(fit_cso)
        results['cso_n'].append(sel_b.sum())
        results['cso_acc'].append(acc_b)
        curve_cso.append(hist_cso['gb_fitness'])

        start = time.time()
        gb_t, fit_t, _, _, hist_t = pso_dual_tckt(X_train, y_train, D, u, mic_scores, cluster_ids, w, K, v_max=0.15, N=N, T=T, theta=theta, rmp=0.6, pm=0.8, seed=s, folds=folds)
        sel_t = gb_t > theta
        acc_t = np.mean(knn_predict(X_train[:, sel_t], y_train, X_test[:, sel_t], k=5) == y_test) if sel_t.sum() > 0 else 0
        results['tckt_fit'].append(fit_t)
        results['tckt_n'].append(sel_t.sum())
        results['tckt_acc'].append(acc_t)
        curve_tckt.append(hist_t['gb1_fitness'])

    # CSO 均值
    b_acc_mean, b_acc_std = np.mean(results['cso_acc']), np.std(results['cso_acc'])
    b_n_mean = np.mean(results['cso_n'])
    b_fit_mean = np.mean(results['cso_fit'])
    
    # EMTFS 均值
    t_acc_mean, t_acc_std = np.mean(results['tckt_acc']), np.std(results['tckt_acc'])
    t_n_mean = np.mean(results['tckt_n'])
    t_fit_mean = np.mean(results['tckt_fit'])

    print(f"{'csoline':<15} | {b_n_mean:<15.1f} | {b_acc_mean:.4f} ± {b_acc_std:.4f} | {b_fit_mean:.4f}")
    print(f"{'EMTFS (TCKT)':<15} | {t_n_mean:<15.1f} | {t_acc_mean:.4f} ± {t_acc_std:.4f} | {t_fit_mean:.4f}")

    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False 
    plt.figure(figsize=(8, 5), dpi=150)
    
    # 沿seed维度求均值
    mean_curve_cso = np.mean(np.array(curve_cso), axis=0)
    mean_curve_tckt = np.mean(np.array(curve_tckt), axis=0)
    
    plt.plot(mean_curve_cso, label='csoline', color='blue', linestyle='--', linewidth=2)
    plt.plot(mean_curve_tckt, label='EMTFS (TCKT)', color='red', linestyle='-', linewidth=2)
    
    plt.title('Average Convergence Curve on Colon (10 Seeds)', fontsize=14)
    plt.xlabel('Generations', fontsize=12)
    plt.ylabel('Best Fitness', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('Colon_Convergence_10seeds.png')
    print("\n收敛曲线已保存为 Colon_Convergence_10seeds.png")
    plt.show()
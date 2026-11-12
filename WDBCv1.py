import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

from EMTO import (
    su, discretize, mic_continuous_discrete, coarse_clustering_dual,
    compute_upper_bounds_cluster, compute_cluster_weights,
    pso_dual, pso_dual_tckt, knn_predict
)

if __name__ == "__main__":
    X, y = load_breast_cancer(return_X_y=True)

    rng = np.random.RandomState(0)
    idx = rng.permutation(len(y))
    split = int(0.7 * len(y))
    train_idx, test_idx = idx[:split], idx[split:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma
    D = X_train.shape[1]
    theta = 0.5

    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}, D={D}")


    # 预计算
    su_scores = np.array([su(discretize(X_train[:, d]), y_train) for d in range(D)])
    mic_scores = np.array([mic_continuous_discrete(X_train[:, d], y_train) for d in range(D)])

    cluster_ids, K = coarse_clustering_dual(su_scores, mic_scores, D)
    u = compute_upper_bounds_cluster(mic_scores, cluster_ids, theta)
    w = compute_cluster_weights(mic_scores, cluster_ids, K)
    print(f"K={K}, u∈[{u.min():.3f}, {u.max():.3f}], w.sum()={w.sum():.6f}")
    
    
    # 多seed对比
    n_seeds = 10
    results = {
        'base_fit': [], 'base_n': [], 'base_acc': [],
        'tckt_fit': [], 'tckt_n': [], 'tckt_acc': [],
    }

    # 准备收集收敛曲线
    curve_base = []
    curve_tckt = []

    for s in range(n_seeds):
        print(f"\nseed {s} ")

        # 基线
        gb1_b, gb1_f_b, _, _, hist_b = pso_dual(X_train, y_train, D, u, mic_scores, cluster_ids, w, K, N=20, T=100, theta=theta, seed=s)
        sel_b = gb1_b > theta
        acc_b = np.mean(knn_predict(X_train[:, sel_b], y_train, X_test[:, sel_b], k=5) == y_test) if sel_b.sum() > 0 else 0.0
        results['base_fit'].append(gb1_f_b)
        results['base_n'].append(sel_b.sum())
        results['base_acc'].append(acc_b)

        # TCKT
        gb1_t, gb1_f_t, _, _, hist_t = pso_dual_tckt(X_train, y_train, D, u, mic_scores, cluster_ids, w, K, N=20, T=100, theta=theta, rmp=0.5, pm=0.6, seed=s)
        sel_t = gb1_t > theta
        acc_t = np.mean(knn_predict(X_train[:, sel_t], y_train, X_test[:, sel_t], k=5) == y_test) if sel_t.sum() > 0 else 0.0
        results['tckt_fit'].append(gb1_f_t)
        results['tckt_n'].append(sel_t.sum())
        results['tckt_acc'].append(acc_t)

        print(f"  基线: fitness={gb1_f_b:.4f}, n={sel_b.sum():2d}, acc={acc_b:.4f}")
        print(f"  TCKT: fitness={gb1_f_t:.4f}, n={sel_t.sum():2d}, acc={acc_t:.4f}")

        curve_base.append(hist_b['gb1_fitness'])
        curve_tckt.append(hist_t['gb1_fitness'])
    
    
    # 汇总
    print(f"{'指标':20s} {'基线 (均值±标准差)':>25s} {'TCKT (均值±标准差)':>25s}")
    for name, kb, kt in [("P1 fitness", 'base_fit', 'tckt_fit'), ("P1 选中数", 'base_n', 'tckt_n'), ("P1 测试精度", 'base_acc', 'tckt_acc'),]:
        b_mean, b_std = np.mean(results[kb]), np.std(results[kb])
        t_mean, t_std = np.mean(results[kt]), np.std(results[kt])
        print(f"{name:20s} {b_mean:10.4f} ± {b_std:.4f}   {t_mean:10.4f} ± {t_std:.4f}")

    # 画图
    curve_base = np.array(curve_base)
    curve_tckt = np.array(curve_tckt)

    # 沿seed维度求均值
    mean_base = np.mean(curve_base, axis=0)
    mean_tckt = np.mean(curve_tckt, axis=0)
    
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False 

    # 开始画图
    plt.figure(figsize=(8, 5), dpi=150)
    
    # 绘制两条曲线
    plt.plot(mean_base, label='Baseline (No Transfer)', color='blue', linestyle='--', linewidth=2)
    plt.plot(mean_tckt, label='EMTFS (TCKT)', color='red', linestyle='-', linewidth=2)

    plt.title('Convergence Curve on WDBC Dataset (Mean of 10 Seeds)', fontsize=14)
    plt.xlabel('Generations', fontsize=12)
    plt.ylabel('P1 Best Fitness', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig('WDBC_Convergence.png')
    print("\n图片已保存为 WDBC_Convergence.png")
    
    plt.show()
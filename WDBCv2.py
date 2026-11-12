import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skrebate import ReliefF
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from EMTO import (
    su, discretize, mic_continuous_discrete, coarse_clustering_dual,
    compute_upper_bounds_cluster, compute_cluster_weights,
    pso_dual, pso_dual_tckt, knn_predict, pso_cso
)

# 评估器，证明实验结果在多个分类器下有效
def evaluate_all_classifiers(X_train_sel, y_train, X_test_sel, y_test):
    results = {}
    
    # KNN
    knn_pred = knn_predict(X_train_sel, y_train, X_test_sel, k=5)
    results['KNN'] = np.mean(knn_pred == y_test)
    
    # 决策树 (sklearn)
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train_sel, y_train)
    results['DecisionTree'] = np.mean(dt.predict(X_test_sel) == y_test)
    
    # 朴素贝叶斯 (sklearn)
    nb = GaussianNB()
    nb.fit(X_train_sel, y_train)
    results['NaiveBayes'] = np.mean(nb.predict(X_test_sel) == y_test)
    
    # LDA (sklearn)
    lda = LDA()
    lda.fit(X_train_sel, y_train)
    results['LDA'] = np.mean(lda.predict(X_test_sel) == y_test)
    
    return results

if __name__ == "__main__":
    # 换成 WDBC
    from sklearn.datasets import load_breast_cancer
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
        'cso_fit': [], 'cso_n': [], 'cso_acc': [],
        'base_fit': [], 'base_n': [], 'base_acc': [],
        'tckt_fit': [], 'tckt_n': [], 'tckt_acc': [],

        # 后续引入多分类器之后就用下面的，上面的其实没有用了
        'cso_acc_knn': [], 'cso_acc_dt': [], 'cso_acc_nb': [], 'cso_acc_lda': [],
        'base_acc_knn': [], 'base_acc_dt': [], 'base_acc_nb': [], 'base_acc_lda': [],
        'tckt_acc_knn': [], 'tckt_acc_dt': [], 'tckt_acc_nb': [], 'tckt_acc_lda': []
    }

    # 准备收集收敛曲线
    curve_base = []
    curve_tckt = []
    curve_cso = []

    for s in range(n_seeds):
        print(f"\nseed {s} ")

        # 基线
        gb1_b, gb1_f_b, _, _, hist_b = pso_dual(X_train, y_train, D, u, mic_scores, cluster_ids, w, K, N=20, T=100, theta=theta, seed=s)
        sel_b = gb1_b > theta
        # acc_b = np.mean(knn_predict(X_train[:, sel_b], y_train, X_test[:, sel_b], k=5) == y_test) if sel_b.sum() > 0 else 0.0
        results['base_fit'].append(gb1_f_b)
        results['base_n'].append(sel_b.sum())
        # results['base_acc'].append(acc_b)

        res_b = evaluate_all_classifiers(X_train[:, sel_b], y_train, X_test[:, sel_b], y_test)
        results['base_acc_knn'].append(res_b['KNN'])
        results['base_acc_dt'].append(res_b['DecisionTree'])
        results['base_acc_nb'].append(res_b['NaiveBayes'])
        results['base_acc_lda'].append(res_b['LDA'])

        # TCKT
        gb1_t, gb1_f_t, _, _, hist_t = pso_dual_tckt(X_train, y_train, D, u, mic_scores, cluster_ids, w, K, N=20, T=100, theta=theta, rmp=0.5, pm=0.6, seed=s)
        sel_t = gb1_t > theta
        # acc_t = np.mean(knn_predict(X_train[:, sel_t], y_train, X_test[:, sel_t], k=5) == y_test) if sel_t.sum() > 0 else 0.0
        results['tckt_fit'].append(gb1_f_t)
        results['tckt_n'].append(sel_t.sum())
        # results['tckt_acc'].append(acc_t)

        res_t = evaluate_all_classifiers(X_train[:, sel_t], y_train, X_test[:, sel_t], y_test)
        results['tckt_acc_knn'].append(res_t['KNN'])
        results['tckt_acc_dt'].append(res_t['DecisionTree'])
        results['tckt_acc_nb'].append(res_t['NaiveBayes'])
        results['tckt_acc_lda'].append(res_t['LDA'])

        # CSOc
        gb_cso, gb1_f_c, hist_cso = pso_cso(X_train, y_train, D, u, N=20, T=100, theta=theta, seed=s)
        sel_cso = gb_cso > theta
        # acc_cso = np.mean(knn_predict(X_train[:, sel_cso], y_train, X_test[:, sel_cso], k=5) == y_test) if sel_cso.sum() > 0 else 0.0
        results['cso_fit'].append(gb1_f_c)
        results['cso_n'].append(sel_cso.sum())
        # results['cso_acc'].append(acc_cso)

        res_c = evaluate_all_classifiers(X_train[:, sel_cso], y_train, X_test[:, sel_cso], y_test)
        results['cso_acc_knn'].append(res_c['KNN'])
        results['cso_acc_dt'].append(res_c['DecisionTree'])
        results['cso_acc_nb'].append(res_c['NaiveBayes'])
        results['cso_acc_lda'].append(res_c['LDA'])

        # 这个保留吧，是之前代码只有这两个的时候做的
        # print(f"  基线: fitness={gb1_f_b:.4f}, n={sel_b.sum():2d}, acc={acc_b:.4f}")
        # print(f"  TCKT: fitness={gb1_f_t:.4f}, n={sel_t.sum():2d}, acc={acc_t:.4f}")

        curve_base.append(hist_b['gb1_fitness'])
        curve_tckt.append(hist_t['gb1_fitness'])
        curve_cso.append(hist_cso['gb_fitness'])
    
    # Relief-F 实验
    target_k = 14
    fs = ReliefF(n_features_to_select=target_k, n_neighbors=5)
    fs.fit(X_train, y_train)
    top_k_idx = fs.top_features_[:target_k]  # 获取库提供的筛选结果
    X_train_relief = X_train[:, top_k_idx]
    X_test_relief = X_test[:, top_k_idx]
    # acc_relief = np.mean(knn_predict(X_train_relief, y_train, X_test_relief, k=5) == y_test)
    res_relief = evaluate_all_classifiers(X_train_relief, y_train, X_test_relief, y_test)

    # 汇总
    # print(f"{'算法类别':<15} | {'算法名称':<12} | {'测试精度 (均值)':<15} | {'特征数 (均值)':<12}")
    # print("-" * 60)
    # print(f"{'Filter':<15} | {'Relief-F':<12} | {acc_relief:.4f}          | {target_k}")
    # print(f"{'单任务 Wrapper':<12} | {'CSO':<12} | {np.mean(results['cso_acc']):.4f} ± {np.std(results['cso_acc']):.4f} | {np.mean(results['cso_n']):.1f}")
    # print(f"{'单任务 Wrapper':<12} | {'Baseline':<12} | {np.mean(results['base_acc']):.4f} ± {np.std(results['base_acc']):.4f} | {np.mean(results['base_n']):.1f}")
    # print(f"{'多任务 Wrapper':<12} | {'EMTFS':<12} | {np.mean(results['tckt_acc']):.4f} ± {np.std(results['tckt_acc']):.4f} | {np.mean(results['tckt_n']):.1f}")

    # 汇总2
    print("表 1：多分类器泛化能力全面对比 (10-seed 均值)")
    print(f"{'Algorithm':<15} | {'Features':<8} | {'KNN':<8} | {'DecisionTree':<12} | {'NaiveBayes':<10} | {'LDA':<8}")
    print("-" * 70)
    print(f"{'Relief-F':<15} | {target_k:<8} | {res_relief['KNN']:.4f}   | {res_relief['DecisionTree']:.4f}       | {res_relief['NaiveBayes']:.4f}     | {res_relief['LDA']:.4f}")
    print(f"{'CSO':<15} | {np.mean(results['cso_n']):<8.1f} | {np.mean(results['cso_acc_knn']):.4f}   | {np.mean(results['cso_acc_dt']):.4f}       | {np.mean(results['cso_acc_nb']):.4f}     | {np.mean(results['cso_acc_lda']):.4f}")
    print(f"{'Baseline':<15} | {np.mean(results['base_n']):<8.1f} | {np.mean(results['base_acc_knn']):.4f}   | {np.mean(results['base_acc_dt']):.4f}       | {np.mean(results['base_acc_nb']):.4f}     | {np.mean(results['base_acc_lda']):.4f}")
    print(f"{'EMTFS (TCKT)':<15} | {np.mean(results['tckt_n']):<8.1f} | {np.mean(results['tckt_acc_knn']):.4f}   | {np.mean(results['tckt_acc_dt']):.4f}       | {np.mean(results['tckt_acc_nb']):.4f}     | {np.mean(results['tckt_acc_lda']):.4f}")

    # 画图
    curve_base = np.array(curve_base)
    curve_tckt = np.array(curve_tckt)
    curve_cso = np.array(curve_cso)
    
    # 沿seed维度求均值
    mean_base = np.mean(curve_base, axis=0)
    mean_tckt = np.mean(curve_tckt, axis=0)
    mean_cso = np.mean(curve_cso, axis=0)
        
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False 
    
    # 开始画图
    plt.figure(figsize=(8, 5), dpi=150)
        
    # 绘制三条曲线（收敛曲线）
    plt.plot(mean_base, label='Baseline (No Transfer)', color='blue', linestyle='--', linewidth=2)
    plt.plot(mean_tckt, label='EMTFS (TCKT)', color='red', linestyle='-', linewidth=2)
    plt.plot(mean_cso, label='CSO (Single Task)', color='green', linestyle='-.', linewidth=2)
    
    plt.title('Convergence Curve on WDBC Dataset (Mean of 10 Seeds)', fontsize=14)
    plt.xlabel('Generations', fontsize=12)
    plt.ylabel('P1 Best Fitness', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('WDBC_Convergence.png')

    # 最终对比柱状图
    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=150)
    methods = ['Relief-F', 'CSO', 'Baseline', 'EMTFS']
    # accs = [acc_relief, np.mean(results['cso_acc']), np.mean(results['base_acc']), np.mean(results['tckt_acc'])]
    accs = [res_relief['KNN'], np.mean(results['cso_acc_knn']), np.mean(results['base_acc_knn']), np.mean(results['tckt_acc_knn'])]
    features = [target_k, np.mean(results['cso_n']), np.mean(results['base_n']), np.mean(results['tckt_n'])]

    # 柱状图：精度
    bars = ax1.bar(methods, accs, color='tab:blue', alpha=0.7, width=0.4, label='Accuracy')
    ax1.set_ylabel('Test Accuracy', color='tab:blue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylim(0.90, 1.00) 
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f"{yval:.4f}", ha='center', va='bottom', color='tab:blue')

    # 折线图：特征数
    ax2 = ax1.twinx()  
    ax2.plot(methods, features, color='tab:orange', marker='o', linewidth=2, markersize=8, label='Num Features')
    ax2.set_ylabel('Number of Selected Features', color='tab:orange', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    ax2.set_ylim(0, 30)

    plt.title('Performance Comparison on WDBC Dataset', fontsize=14)
    fig.tight_layout()
    plt.savefig('WDBC_Comparison.png')
    print("\n图片已保存为 WDBC_Convergence.png 和 WDBC_Comparison.png")

    # 伪造一个没有聚类的环境：所有特征都属于簇0，只有1个簇
    cluster_ids_no_clu = np.zeros(D, dtype=int)
    K_no_clu = 1
    w_no_clu = np.array([1.0])
    u_no_clu = compute_upper_bounds_cluster(mic_scores, cluster_ids_no_clu, theta)

    # 存放消融结果
    abl_results = {
        'EMTFS': [], 'EMTFS-C1': [], 'EMTFS-C2': [], 'EMTFS-noClu': []
    }
    abl_features = {
        'EMTFS': [], 'EMTFS-C1': [], 'EMTFS-C2': [], 'EMTFS-noClu': []
    }

    for s in range(10):
        print(f"  消融 Seed {s}...")
        
        # 完整版
        gb_t, f_t, _, _, _ = pso_dual_tckt(X_train, y_train, D, u, mic_scores, cluster_ids, w, K, N=20, T=100, theta=theta, rmp=0.5, pm=0.6, seed=s, mode='both')
        acc_t = np.mean(knn_predict(X_train[:, gb_t>theta], y_train, X_test[:, gb_t>theta], k=5) == y_test)
        abl_results['EMTFS'].append(acc_t)
        abl_features['EMTFS'].append((gb_t > theta).sum())

        # 只用通道1
        gb_c1, f_c1, _, _, _ = pso_dual_tckt(X_train, y_train, D, u, mic_scores, cluster_ids, w, K, N=20, T=100, theta=theta, rmp=0.5, pm=0.6, seed=s, mode='C1')
        acc_c1 = np.mean(knn_predict(X_train[:, gb_c1>theta], y_train, X_test[:, gb_c1>theta], k=5) == y_test)
        abl_results['EMTFS-C1'].append(acc_c1)
        abl_features['EMTFS-C1'].append((gb_c1 > theta).sum())

        # 只用通道2
        gb_c2, f_c2, _, _, _ = pso_dual_tckt(X_train, y_train, D, u, mic_scores, cluster_ids, w, K, N=20, T=100, theta=theta, rmp=0.5, pm=0.6, seed=s, mode='C2')
        acc_c2 = np.mean(knn_predict(X_train[:, gb_c2>theta], y_train, X_test[:, gb_c2>theta], k=5) == y_test)
        abl_results['EMTFS-C2'].append(acc_c2)
        abl_features['EMTFS-C2'].append((gb_c2 > theta).sum())

        # 无聚类
        gb_nc, f_nc, _, _, _ = pso_dual_tckt(X_train, y_train, D, u_no_clu, mic_scores, cluster_ids_no_clu, w_no_clu, K_no_clu, N=20, T=100, theta=theta, rmp=0.5, pm=0.6, seed=s, mode='both')
        acc_nc = np.mean(knn_predict(X_train[:, gb_nc>theta], y_train, X_test[:, gb_nc>theta], k=5) == y_test)
        abl_results['EMTFS-noClu'].append(acc_nc)
        abl_features['EMTFS-noClu'].append((gb_nc > theta).sum())

    # 打印消融结果
    print("表 2：消融实验结果 (10-seed 均值)")
    print(f"{'Variant':<15} | {'Features':<10} | {'KNN Accuracy':<15}")
    for method in ['EMTFS', 'EMTFS-C1', 'EMTFS-C2', 'EMTFS-noClu']:
        mean_acc = np.mean(abl_results[method])
        mean_feat = np.mean(abl_features[method])
        print(f"{method:<15} | {mean_feat:<10.1f} | {mean_acc:.4f}")

    plt.show()
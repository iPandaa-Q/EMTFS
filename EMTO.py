import numpy as np
from sklearn.feature_selection import mutual_info_classif

def knn_predict(X_train, y_train, X_test, k=5):
    """
    X_train: (n_train, d)
    y_train: (n_train,) 整数标签
    X_test:  (n_test, d)
    k: 邻居数
    """
    n_test = X_test.shape[0]
    C = int(y_train.max()) + 1

    # 距离平方矩阵
    # diff = X_test[:, None, :] - X_train[None, :, :]
    # dist_sq = np.sum(diff ** 2, axis=2)

    # 矩阵乘法加速 (a-b)^2 = a^2 + b^2 - 2ab
    X_test_sq = np.sum(X_test ** 2, axis=1)[:, None]
    X_train_sq = np.sum(X_train ** 2, axis=1)[None, :]
    dist_sq = X_test_sq + X_train_sq - 2 * np.dot(X_test, X_train.T)
    dist_sq = np.maximum(dist_sq, 0) # 防止浮点误差产生负数

    # 取最近的k个邻居索引
    idx = np.argsort(dist_sq, axis=1)[:, :k]

    # 取邻居标签
    neighbor_labels = y_train[idx]

    # 多数投票
    y_pred = np.zeros(n_test, dtype=int)
    for i in range(n_test):
        y_pred[i] = np.bincount(neighbor_labels[i], minlength=C).argmax()

    return y_pred

def kfold_indices(n, folds = 10, seed = 45):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)

    fold_sizes = np.full(folds, n//folds, dtype = int)
    fold_sizes[:n%folds] += 1

    indices = []
    current = 0
    for size in fold_sizes:
        val_idx = idx[current:current + size]  # 取其中某一折
        train_idx = np.concatenate([idx[:current], idx[current + size:]]) # 把上面取的那一折前面的和那一折后面的拼接起来
        indices.append((train_idx, val_idx))
        current += size

    return indices

def knn_cv_error(X_selected, y, k=5, folds=10, seed=42):
    # X_selected: (n_train, n_selected_features)
    n = X_selected.shape[0]
    indices = kfold_indices(n, folds, seed)

    errors = []
    for train_idx, val_idx in indices:
        X_tr = X_selected[train_idx]
        y_tr = y[train_idx]
        X_val = X_selected[val_idx]
        y_val = y[val_idx]

        y_pred = knn_predict(X_tr, y_tr, X_val, k=k)
        err = np.mean(y_pred != y_val)
        errors.append(err)

    return np.mean(errors)

def wrapper_fitness(x, X_train, y_train, theta, alpha=0.9, folds=10):
    D = len(x)
    selected = x > theta

    if selected.sum() == 0:
        return 1.0

    X_selected = X_train[:, selected] # 只要选中的特征
    err = knn_cv_error(X_selected, y_train, k=5, folds=folds, seed=42)

    return alpha * err + (1 - alpha) * selected.sum() / D

def pso_wrapper(X_train, y_train, D, u, N=100, T=100, theta=0.6,
                omega_init=0.9, omega_end=0.4,
                c1=1.5, c2=1.5, v_max=0.2, seed=45):
    rng = np.random.RandomState(seed)
    # X = rng.rand(N, D)
    # V = rng.uniform(-v_max, v_max, (N, D))
    # 有收缩
    X = rng.rand(N, D) * u
    V = rng.uniform(-v_max, v_max, (N, D)) * u 

    # 算初始 fitness
    fitness = np.array([wrapper_fitness(X[i], X_train, y_train, theta) for i in range(N)])
    pb = X.copy()
    pb_fitness = fitness.copy()
    gb_idx = np.argmin(fitness)
    gb = X[gb_idx].copy()
    gb_fitness = fitness[gb_idx]

    # 记录全局最优
    history = {'gb_fitness': [], 'n_selected': []}
    for t in range(T):
        omega = omega_init - (omega_init - omega_end) * t / T   # 线性递减

        for i in range(N):
            r1 = rng.rand(D)
            r2 = rng.rand(D)

            V[i] = (omega * V[i]
                    + c1 * r1 * (pb[i] - X[i])
                    + c2 * r2 * (gb - X[i]))

            V[i] = np.clip(V[i], -v_max, v_max)   # 速度限制
            X[i] = np.clip(X[i] + V[i], 0, u)     # 位置限制

            f = wrapper_fitness(X[i], X_train, y_train, theta)
            # 更新适应度
            if f < pb_fitness[i]:
                pb[i] = X[i].copy()
                pb_fitness[i] = f

            if f < gb_fitness:
                gb = X[i].copy()
                gb_fitness = f

        n_sel = (gb > theta).sum()  # 统计全局最优里面选了多少个特征
        history['gb_fitness'].append(gb_fitness)
        history['n_selected'].append(n_sel)

        # 每十代打印一次结果
        if t % 10 == 0:
            print(f"Gen {t:3d}: gb_fitness={gb_fitness:.4f}, n_selected={n_sel}")

    return gb, gb_fitness, history

def discretize(x, bins=5):
    """
    x: (n,) 连续值
    返回： (n,) 整数标签 0..bins-1
    等频分箱：每个箱大约 n/bins 个样本
    """
    n = len(x)
    order = np.argsort(x) # 排序后的索引
    labels = np.empty(n, dtype=int)
    for rank, idx in enumerate(order):
        labels[idx] = min(rank * bins // n, bins - 1) # 按顺序打上标签，指示其排第几位
    return labels

def entropy(x):
    _, counts = np.unique(x, return_counts=True) # np.unique 默认只返回去重并排序后的值，加上 return_counts=True就可以返回每个值出现的次数
    p = counts / counts.sum()
    return -np.sum(p * np.log2(p))

def mutual_info(x, y):
    x_vals, x_codes = np.unique(x, return_inverse=True) # 额外返回原数组每个元素在 x_vals 中的索引
    y_vals, y_codes = np.unique(y, return_inverse=True)
    x_codes = x_codes.ravel() # 加这行
    y_codes = y_codes.ravel() # 加这行
    nx, ny = len(x_vals), len(y_vals)
    n = len(x)

    # 联合计数矩阵 (nx, ny)
    flat = x_codes * ny + y_codes # 行优先（row-major）展平，二维矩阵按行拉直成一维数组
    joint_count = np.bincount(flat, minlength=nx * ny).reshape(nx, ny) # bincount 统计每个整数出现次数，.reshape(nx, ny)把一维计数数组按行优先还原成二维矩阵

    p_xy = joint_count / n # 联合概率，每个格子除以总样本数
    p_x = p_xy.sum(axis=1) # 边缘概率 (nx,)
    p_y = p_xy.sum(axis=0) # 边缘概率 (ny,)

    # 只对 p_xy > 0 的位置求和，避免 log(0)
    mask = p_xy > 0
    mi = np.sum(
        p_xy[mask] * np.log2(p_xy[mask] / (p_x[:, None] * p_y[None, :])[mask])
    )
    return mi

def su(x, y):
    hx = entropy(x)
    hy = entropy(y)
    if hx + hy == 0:
        return 0.0
    return 2 * mutual_info(x, y) / (hx + hy)

""" def mic_continuous_discrete(x, y, B=None):
    # B: 网格上限，默认n^0.6
    # 返回：MIC 值，范围 [0, 1]
    n = len(x)
    if B is None:
        B = int(n ** 0.6)

    C = int(y.max()) + 1
    if C < 2:
        return 0.0

    max_mic = 0.0
    # 枚举 x 的分箱数，约束 x_bins * C < B
    max_x_bins = max(2, B // C)
    for x_bins in range(2, max_x_bins + 1):
        # 等宽分箱（平移、归一化、再分格子）
        x_disc = np.floor((x - x.min()) / (x.max() - x.min() + 1e-12) * x_bins).astype(int)
        x_disc = np.clip(x_disc, 0, x_bins - 1)
        # 算互信息
        mi = mutual_info(x_disc, y)
        # 归一化
        norm = np.log2(min(x_bins, C))
        if norm > 0:
            mic_val = mi / norm
            if mic_val > max_mic:
                max_mic = mic_val

    return max_mic
 """

# 以上的mic计算是手写版本，更忠于原文时，应该调用开源库
def mic_continuous_discrete(x, y, B=None):
    # 把一维特征reshape成二维，因为sklearn要求
    x_reshaped = x.reshape(-1, 1)
    # 算互信息
    mi = mutual_info_classif(x_reshaped, y, discrete_features=False, random_state=45)[0]
    return mi

def count_clusters(scores_sorted):
    """
    scores_sorted: (m,) 升序排列的相关性分数（SU 或 MIC）
    返回：簇数
    """
    m = len(scores_sorted)
    if m == 0:
        return 0

    d_max = scores_sorted[-1] - scores_sorted[0]
    lam = d_max * np.log(m) / m  # 论文用 ln，即 np.log

    count = 0
    i = 0
    while i < m:
        j = i + 1
        while j < m and abs(scores_sorted[j] - scores_sorted[i]) <= lam:
            j += 1
        count += 1
        i = j
    return count

def select_relevant_dual(su_scores, mic_scores, D):
    """
    返回：
    S_prime: 相关特征索引（SU-relevant ∪ MIC-relevant）
    U1: 无关特征索引
    """
    k = int(np.ceil(np.sqrt(D) * np.log10(D)))
    k = max(k, 1)
    # 第k个值作为分界线
    rho_SU = np.sort(su_scores)[k - 1]
    rho_MIC = np.sort(mic_scores)[k - 1]
    # 这里只取了特征索引
    F_SU = np.where(su_scores > rho_SU)[0]
    F_MIC = np.where(mic_scores > rho_MIC)[0]
    # 取并集
    S_prime = np.union1d(F_SU, F_MIC)
    # 把剩下的放到U1（先创建所有索引，再去掉）
    U1 = np.setdiff1d(np.arange(D), S_prime)
    return S_prime, U1

def kmeans_2d(points, K_kmeans, max_iter=100, seed=45):
    """
    points: (m, 2) 每个相关特征的 (SU, MIC)
    K_kmeans: 簇数
    返回：labels (m,)
    """
    rng = np.random.RandomState(seed)
    m = len(points)
    # 随机取点
    init_idx = rng.choice(m, K_kmeans, replace=False)
    centers = points[init_idx].copy()
    # 计算欧式距离，不断更新点
    for _ in range(max_iter):
        dist = np.sum((points[:, None, :] - centers[None, :, :]) ** 2, axis=2)  # (m, K)
        labels = np.argmin(dist, axis=1) # 取离得更近的那个点，有arg，是索引
        # 
        new_centers = centers.copy()
        for k in range(K_kmeans):
            # labels就是它的属于哪个簇
            mask = labels == k
            if mask.sum() > 0:  # 大于零为了防止零个点求平均的情况
                new_centers[k] = points[mask].mean(axis=0) # 这里是先取真实的点，再做一个平均

        if np.allclose(new_centers, centers):  # 中心不变就停止
            break
        centers = new_centers

    return labels
# 这是一块的整合
def coarse_clustering_dual(su_scores, mic_scores, D):
    """
    先分出相关的特征和不相关的特征
    然后分别做一个分组，比较粗，只要特征差值在一定范围内就可以分一组
    在上一把得到组数，然后按这个组数，利用k聚类算法得到真正的分组
    """
    S_prime, U1 = select_relevant_dual(su_scores, mic_scores, D)

    num_SU = count_clusters(np.sort(su_scores[S_prime]))
    num_MIC = count_clusters(np.sort(mic_scores[S_prime]))
    K = min(num_SU, num_MIC) + 1

    # 二维点 (SU, MIC)
    points = np.stack([su_scores[S_prime], mic_scores[S_prime]], axis=1)  # (m, 2)
    
    labels = kmeans_2d(points, K - 1)
    # 无关特征记为零，其余从1开始记，第几个簇
    cluster_ids = np.zeros(D, dtype=int)
    cluster_ids[U1] = 0
    cluster_ids[S_prime] = labels + 1

    return cluster_ids, K
# 计算范围收缩上界
def compute_upper_bounds_cluster(mic_scores, cluster_ids, theta):
    D = len(mic_scores)
    u = np.zeros(D)
    for d in range(D):
        k = cluster_ids[d]  # 先取一个特征
        mask = cluster_ids == k  # 再找它所在的簇
        mic_min = mic_scores[mask].min()  # 之后在这个簇里面找最小的mic
        mic_max = mic_scores[mask].max()  # 之后在这个簇里面找最大的mic
        ratio = (mic_scores[d] - mic_min) / (mic_max - mic_min + 1e-12)
        u[d] = (theta + 0.1) + (0.9 - theta) * ratio  # 计算
    return u

def compute_cluster_weights(mic_scores, cluster_ids, K):
    max_mic_per_cluster = np.zeros(K)
    for k in range(K):
        mask = cluster_ids == k
        if mask.sum() > 0:
            max_mic_per_cluster[k] = mic_scores[mask].max()
    total = max_mic_per_cluster.sum()
    if total == 0: # 防止除零
        return np.ones(K) / K # 认为是噪声，权值就一样
    return max_mic_per_cluster / total

def filter_fitness(x, mic_scores, cluster_ids, w, K, theta):
    select = x > theta # theta取0.5，把这些特征选了
    score = 0.0
    n_total = 0
    for k in range(K):
        mask = (cluster_ids == k) & select # 这个簇里面选了多少个
        n_sel = mask.sum()
        n_total += n_sel
        if n_sel == 0:
            continue
        
        mean_mic = mic_scores[mask].mean()
        score += w[k] * mean_mic / (n_sel + 1)

    return -score

def pso_dual(X_train, y_train, D, u, mic_scores, cluster_ids, w, K,N=100, T=100, theta=0.6, omega_init=0.9, omega_end=0.4, c1=1.5, c2=1.5, v_max=0.2, seed=45, folds=10):
    rng = np.random.RandomState(seed)

    X1 = rng.rand(N, D) * 0.607
    V1 = rng.uniform(-v_max, v_max, (N, D)) * u

    X2 = rng.rand(N, D) * 0.607
    V2 = rng.uniform(-v_max, v_max, (N, D)) * u

    fit_1 = np.array([wrapper_fitness(X1[i], X_train, y_train, theta, folds=folds) for i in range(N)])
    fit_2 = np.array([filter_fitness(X2[i], mic_scores, cluster_ids, w, K, theta) for i in range(N)])

    pb1 = X1.copy()
    pb_fit_1 = fit_1.copy()

    pb2 = X2.copy()
    pb_fit_2 = fit_2.copy()

    gb1_idx = np.argmin(fit_1)
    gb1 = X1[gb1_idx].copy()
    gb1_fit = fit_1[gb1_idx]

    gb2_idx = np.argmin(fit_2)
    gb2 = X2[gb2_idx].copy()
    gb2_fit = fit_2[gb2_idx]

    history = {
        'gb1_fitness': [], 'gb1_n_selected': [],
        'gb2_fitness': [], 'gb2_n_selected': [],
    }

    for t in range(T):
        omega = omega_init - (omega_init - omega_end) * t / T

        for i in range(N):

            r1 = rng.rand(D)
            r2 = rng.rand(D)
            V1[i] = (omega * V1[i] + c1 * r1 * (pb1[i] - X1[i]) + c2 * r2 * (gb1 - X1[i]))
            V1[i] = np.clip(V1[i], -v_max, v_max)
            X1[i] = np.clip(X1[i] + V1[i], 0, u)

            f_1 = wrapper_fitness(X1[i], X_train, y_train, theta, folds=folds)
            if f_1 < pb_fit_1[i]:
                pb1[i] = X1[i].copy()
                pb_fit_1[i] = f_1
            if f_1 < gb1_fit:
                gb1 = X1[i].copy()
                gb1_fit = f_1
                gb1_idx = i

            r1 = rng.rand(D)
            r2 = rng.rand(D)
            V2[i] = (omega * V2[i] + c1 * r1 * (pb2[i] - X2[i]) + c2 * r2 * (gb2 - X2[i]))
            V2[i] = np.clip(V2[i], -v_max, v_max)
            X2[i] = np.clip(X2[i] + V2[i], 0, u)

            f_2 = filter_fitness(X2[i], mic_scores, cluster_ids, w, K, theta)
            if f_2 < pb_fit_2[i]:
                pb2[i] = X2[i].copy()
                pb_fit_2[i] = f_2
            if f_2 < gb2_fit:
                gb2 = X2[i].copy()
                gb2_fit = f_2
                gb2_idx = i

        n1 = (gb1 > theta).sum()
        n2 = (gb2 > theta).sum()

        # 记录选中的粒子数目
        history['gb1_fitness'].append(gb1_fit)
        history['gb1_n_selected'].append(n1)
        history['gb2_fitness'].append(gb2_fit)
        history['gb2_n_selected'].append(n2)

        if t % 10 == 0:
            print(f"Gen {t:3d}: P1 fitness={gb1_fit:.4f} n1={n1:2d} | "
                  f"P2 fitness={gb2_fit:.4f} n2={n2:2d}")

    return gb1, gb1_fit, gb2, gb2_fit, history


def pso_dual_tckt(X_train, y_train, D, u, mic_scores, cluster_ids, w, K, N=100, T=100, theta=0.6, omega_init=0.9, omega_end=0.4, c1=1.5, c2=1.5, v_max=0.2, rmp = 0.5, pm = 0.6,seed=45, mode='both', folds = 10):
    
    rng = np.random.RandomState(seed)
    
    X1 = rng.rand(N, D) * 0.607
    V1 = rng.uniform(-v_max, v_max, (N, D)) * u

    X2 = rng.rand(N, D) * 0.607
    V2 = rng.uniform(-v_max, v_max, (N, D)) * u

    fit_1 = np.array([wrapper_fitness(X1[i], X_train, y_train, theta, folds=folds) for i in range(N)])
    fit_2 = np.array([filter_fitness(X2[i], mic_scores, cluster_ids, w, K, theta) for i in range(N)])

    pb1 = X1.copy()
    pb_fit_1 = fit_1.copy()
    gb1_idx = np.argmin(pb_fit_1)
    gb1 = X1[gb1_idx].copy()
    gb1_fit = fit_1[gb1_idx]

    pb2 = X2.copy()
    pb_fit_2 = fit_2.copy()
    gb2_idx = np.argmin(pb_fit_2)
    gb2 = X2[gb2_idx].copy()
    gb2_fit = fit_2[gb2_idx]

    history = {
        'gb1_fitness': [], 'gb1_n_selected': [],
        'gb2_fitness': [], 'gb2_n_selected': [],
        'n_transfer': [],
    }

    for t in range(T):
        omega = omega_init - (omega_init - omega_end) * t / T
        n_transfer = 0

        for i in range(N):

            if rng.rand() < rmp:
                n_transfer += 1
                x_bin = (X1[i] > theta)
                Y_bin = (X2 > theta) # 这里得用大小写区分一下，大写是全体粒子，x中的一个粒子和y中的全体粒子进行比较
                j = find_hamming_neighbor(x_bin, Y_bin)

                ogb = gb2
                opb = pb2[j]

                use_channel_1 = False
                if mode == 'C1':
                    use_channel_1 = True
                elif mode == 'C2':
                    use_channel_1 = False
                else:
                    use_channel_1 = (rng.rand() < pm) 

                if use_channel_1:
                    # 通道1，只用ogb
                    r1 = rng.rand(D)
                    r2 = rng.rand(D)
                    V1[i] = (omega * V1[i] + c1 * r1 * (pb1[i] - X1[i]) + c2 * r2 * (ogb - X1[i]))
                else:
                    # 通道2，用ogb和opb
                    r1 = rng.rand(D)
                    r2 = rng.rand(D)
                    V1[i] = (omega * V1[i] + c1 * r1 * (opb - X1[i]) + c2 * r2 * (ogb - X1[i]))
            else:
                r1 = rng.rand(D)
                r2 = rng.rand(D)
                V1[i] = (omega * V1[i] + c1 * r1 * (pb1[i] - X1[i]) + c2 * r2 * (gb1 - X1[i]))

            V1[i] = np.clip(V1[i], -v_max, v_max)
            X1[i] = np.clip(X1[i] + V1[i], 0, u)

            f_1 = wrapper_fitness(X1[i], X_train, y_train, theta, folds=folds)
            if f_1 < pb_fit_1[i]:
                pb1[i] = X1[i].copy()
                pb_fit_1[i] = f_1
            if f_1 < gb1_fit:
                gb1 = X1[i].copy()
                gb1_fit = f_1
                gb1_idx = i
            if rng.rand() < rmp:
                n_transfer += 1
                y_bin = (X2[i] > theta)
                X_bin = (X1 > theta)
                j = find_hamming_neighbor(y_bin, X_bin)

                ogb = gb1
                opb = pb1[j]

                if rng.rand() < pm:
                    r1 = rng.rand(D)
                    r2 = rng.rand(D)
                    V2[i] = (omega * V2[i] + c1 * r1 * (pb2[i] - X2[i]) + c2 * r2 * (ogb - X2[i]))
                else:
                    r1 = rng.rand(D)
                    r2 = rng.rand(D)
                    V2[i] = (omega * V2[i] + c1 * r1 * (opb - X2[i]) + c2 * r2 * (ogb - X2[i]))
            else:
                r1 = rng.rand(D)
                r2 = rng.rand(D)
                V2[i] = (omega * V2[i] + c1 * r1 * (pb2[i] - X2[i]) + c2 * r2 * (gb2 - X2[i]))

            V2[i] = np.clip(V2[i], -v_max, v_max)
            X2[i] = np.clip(X2[i] + V2[i], 0, u)

            f_2 = filter_fitness(X2[i], mic_scores, cluster_ids, w, K, theta)
            if f_2 < pb_fit_2[i]:
                pb2[i] = X2[i].copy()
                pb_fit_2[i] = f_2
            if f_2 < gb2_fit:
                gb2 = X2[i].copy()
                gb2_fit = f_2
                gb2_idx = i

        # 记录选中的粒子数目
        n1 = (gb1 > theta).sum()
        n2 = (gb2 > theta).sum()
        history['gb1_fitness'].append(gb1_fit)
        history['gb1_n_selected'].append(n1)
        history['gb2_fitness'].append(gb2_fit)
        history['gb2_n_selected'].append(n2)
        history['n_transfer'].append(n_transfer)

        if t % 10 == 0:
            print(f"Gen {t:3d} : P1 fitness = {gb1_fit:.4f} n1 = {n1:2d} |"
                    f"P2 fitness = {gb2_fit:.4f} n2 = {n2:2d}") 

    return gb1, gb1_fit, gb2, gb2_fit, history

def find_hamming_neighbor(x_bin, y_bin):
    """
    x_bin: (D,) 当前解的二进制解码(就是选与不选)
    y_bin: (N, D) 对方种群所有解的二进制解码
    返回j，y_bin 中与 x_bin Hamming 距离最小的行号
    """
    hd = np.sum(x_bin[None, :] != y_bin, axis=1)
    return int(np.argmin(hd))

def pso_cso(X_train, y_train, D, u, N=100, T=100, theta=0.6, v_max=0.2, seed=45, folds=10):
    """
    Competitive Swarm Optimizer (CSO) 单任务基线
    """
    rng = np.random.RandomState(seed)
    
    # 初始化位置和速度
    X = rng.rand(N, D) * u
    V = rng.uniform(-v_max, v_max, (N, D)) * u
    
    # 初始适应度
    fitness = np.array([cso_fitness(X[i], X_train, y_train, theta) for i in range(N)])
    
    # 记录全局最优
    gb_idx = np.argmin(fitness)
    gb = X[gb_idx].copy()
    gb_fitness = fitness[gb_idx]
    
    history = {'gb_fitness': [], 'n_selected': []}

    for t in range(T):
        # 随机打乱粒子索引，用于两两配对
        idx = rng.permutation(N)
        
        for i in range(0, N, 2):
            p1 = idx[i]
            p2 = idx[i+1]
            
            # 判断胜负
            if fitness[p1] < fitness[p2]:
                winner, loser = p1, p2
            else:
                winner, loser = p2, p1

            X_mean = X.mean(axis=0)

            # 只有输家向赢家学习 (更新速度与位置)
            r1 = rng.rand(D)
            r2 = rng.rand(D)
            r3 = rng.rand(D)
            
            V[loser] = r1 * V[loser] + r2 * (X[winner] - X[loser]) + 0.1 * r3 * (X_mean - X[loser])
            V[loser] = np.clip(V[loser], -v_max, v_max)
            X[loser] = np.clip(X[loser] + V[loser], 0, u)
            
            # 重新计算输家的适应度
            fitness[loser] = cso_fitness(X[loser], X_train, y_train, theta, folds=folds)
            
            # 更新全局最优
            if fitness[loser] < gb_fitness:
                gb = X[loser].copy()
                gb_fitness = fitness[loser]
                
        # 记录本代最优
        n_sel = (gb > theta).sum()
        history['gb_fitness'].append(gb_fitness)
        history['n_selected'].append(n_sel)
        
        if t % 10 == 0:
            print(f"CSO Gen {t:3d}: gb_fitness={gb_fitness:.4f}, n_selected={n_sel}")

    return gb, gb_fitness, history

# CSO专属适应度：纯10折CV误差（EMTFS的wrapper保持0.9/0.1不变）
def cso_fitness(x, X_train, y_train, theta, folds=10):
    sel = x > theta
    if sel.sum() == 0:
        return 1.0
    return knn_cv_error(X_train[:, sel], y_train, k=5, folds=folds, seed=45)
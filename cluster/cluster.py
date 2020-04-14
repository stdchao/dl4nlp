'''
聚类分析及可视化
sklearn.cluster[https://scikit-learn.org/stable/modules/clustering.html#clustering]
'''
import sys
import logging
from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, MiniBatchKMeans, AffinityPropagation

from sklearn.datasets import make_blobs, load_digits
from sklearn.datasets import fetch_20newsgroups

from sklearn.preprocessing import scale, Normalizer
from sklearn.decomposition import PCA, TruncatedSVD

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn import metrics
from sklearn.metrics.pairwise import pairwise_distances_argmin

from sklearn.pipeline import make_pipeline

def kmeans_assumptions():
  '''
  example: Demonstration of k-means assumptions
  '''
  random_state = 170
  plt.figure(figsize=(12, 12))

  # 生成各向同性的高斯分布数据点(isotropic gaussian blobs)
  X, y = make_blobs(n_samples=1500, random_state=random_state)
  # kmeans fit and predict
  y_pred = KMeans(n_clusters=2, random_state=random_state).fit_predict(X)
  plt.subplot(221)
  plt.scatter(X[:, 0], X[:, 1], c=y_pred)
  plt.title("Isotropic blobs")

  # 变成成各向异性的数据点(anisotropic distributed data)
  transormation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
  X_aniso = np.dot(X, transormation)
  y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_aniso)
  plt.subplot(222)
  plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_pred)
  plt.title("Anisotropic blobs")

  # 生成不同方差的数据点(different variance data)
  X_varied, y_varied = make_blobs(n_samples=1500, cluster_std=[1.0, 2.5, 0.5], 
                                  random_state=random_state)
  y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_varied)
  plt.subplot(223)
  plt.scatter(X_varied[:, 0], X_varied[:, 1], c=y_pred)
  plt.title("Unequel variance")

  # 生成数量不均衡的数据点(unevenly size)
  X_filtered = np.vstack((X[y==0][:500], X[y==1][:100], X[y==2][:10]))
  y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_filtered)
  plt.subplot(224)
  plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=y_pred)
  plt.title("Unevenly size")

  # show figure
  plt.savefig('cluster/kmeans_assumptions.png')

def kmeans_cluster_digits():
  '''
  example: kmeans cluster on handwritten digits data
  '''
  # 生成digits data
  np.random.seed(42)
  X_digits, y_digits = load_digits(return_X_y=True)
  data = scale(X_digits)
  labels = y_digits
  n_digits = len(np.unique(y_digits))
  print('n_digits: %d, n_samples: %d, n_features: %d'
        % (n_digits, data.shape[0], data.shape[1]))
  
  # 比较k-means++，random，以及基于PCA的三种初始化方法的聚类效果
  print(82 * '_')
  print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')

  def bench_kmeans(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels, estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean', sample_size=300)))
  bench_kmeans(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),
               name="k-means++", data=data)
  bench_kmeans(KMeans(init='random', n_clusters=n_digits, n_init=10),
               name="random", data=data)
  
  # PCA处理后的主成分，作为kmeans的初始化中心，因此不需要多次重复尝试即n_init=1
  pca = PCA(n_components=n_digits).fit(data)
  bench_kmeans(KMeans(init=pca.components_, n_clusters=n_digits, n_init=1),
               name="pca_based", data=data)
  print(82 * '-')
  
  # PCA降维数据的聚类与可视化
  reduced_data = PCA(n_components=2).fit_transform(data)
  kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
  kmeans.fit(reduced_data)

  # 将样本空间填充满网格点，并根据上述kmeans聚类模型预测网格点分类
  h = 0.2
  x_min, x_max = reduced_data[:, 0].min() -1, reduced_data[:, 0].max() +1
  y_min, y_max = reduced_data[:, 1].min() -1, reduced_data[:, 1].max() +1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
  Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)

  # 可视化
  plt.figure(figsize=(12, 12))
  plt.clf()
  # 网格点
  plt.imshow(Z, interpolation='nearest',
             extent=(xx.min(), xx.max(), yy.min(), yy.max()),
             cmap=plt.cm.Paired,
             aspect='auto', origin='lower')

  # 数据点
  plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
  # 类中心
  centroids = kmeans.cluster_centers_
  plt.scatter(centroids[:, 0], centroids[:, 1],
              marker='x', s=169, linewidths=3,
              color='w', zorder=10)
  plt.title('K-means cluster on PCA-reduced data')
  plt.xlim(x_min, x_max)
  plt.ylim(y_min, y_max)
  plt.xticks(())
  plt.yticks(())
  plt.savefig('cluster/kmeans_cluster_digits.png')

def kmeans_and_minibatch():
  '''
  example: compare kmeans and minibatch-kmeans
  minibatch-kmeans，是kmeans算法的一个变种。在每次迭代过程中，从样本集随机抽取b个样本，分到最近的类中；然后更新类的中心点，更新的值为当前抽样集与前面所有抽样集的平均值。
  minibatch-kmeans，相比kmeans有更快的收敛速度，不过结果有些许差异
  '''
  np.random.seed(0)

  # 生成各向同性的高斯分布数据点
  centers = [[1,1], [-1,-1], [1,-1]]
  n_clusters = len(centers)
  X, y = make_blobs(n_samples=3000, centers=centers, cluster_std=0.7)

  # kmeans
  kmeans = KMeans(init='k-means++', n_clusters=3, n_init=10)
  t0 = time()
  kmeans.fit(X)
  t_batch = time() - t0

  # minibatch-kmeans
  mbkmeans = MiniBatchKMeans(init='k-means++', n_clusters=3, n_init=10,
                              batch_size=45, max_no_improvement=10, verbose=0)
  t0 = time()
  mbkmeans.fit(X)
  t_minibatch = time() - t0

  # 聚类结果可视化
  fig = plt.figure(figsize=(9, 3))
  fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
  colors = ['#4EACC5', '#FF9C34', '#4E9A06']

  # 调整下类中心的顺序，使kmeans和minibatch-kmeans的最相邻的中心点的类的编号保持一致
  kmeans_centers = kmeans.cluster_centers_
  order = pairwise_distances_argmin(kmeans_centers, mbkmeans.cluster_centers_)
  mbkmeans_centers = mbkmeans.cluster_centers_[order]

  kmeans_labels = pairwise_distances_argmin(X, kmeans_centers)
  mbkmeans_labels = pairwise_distances_argmin(X, mbkmeans_centers)

  # kmeans plot
  ax = fig.add_subplot(1, 3, 1)
  for k, col in zip(range(n_clusters), colors):
    my_members = kmeans_labels == k
    cluster_center = kmeans_centers[k]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w',
            markerfacecolor=col, marker='.')
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
            markeredgecolor='k', markersize=6)
  ax.set_title('KMeans')
  ax.set_xticks(())
  ax.set_yticks(())
  plt.text(-3.5, 1.8,  'train time: %.2fs\ninertia: %f' % (
      t_batch, kmeans.inertia_))

  # MiniBatchKMeans
  ax = fig.add_subplot(1, 3, 2)
  for k, col in zip(range(n_clusters), colors):
      my_members = mbkmeans_labels == k
      cluster_center = mbkmeans_centers[k]
      ax.plot(X[my_members, 0], X[my_members, 1], 'w',
              markerfacecolor=col, marker='.')
      ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
              markeredgecolor='k', markersize=6)
  ax.set_title('MiniBatchKMeans')
  ax.set_xticks(())
  ax.set_yticks(())
  plt.text(-3.5, 1.8, 'train time: %.2fs\ninertia: %f' %
          (t_minibatch, mbkmeans.inertia_))

  # Initialise the different array to all False
  different = (mbkmeans_labels == 4)
  ax = fig.add_subplot(1, 3, 3)

  for k in range(n_clusters):
      different += ((kmeans_labels == k) != (mbkmeans_labels == k))

  identic = np.logical_not(different)
  ax.plot(X[identic, 0], X[identic, 1], 'w',
          markerfacecolor='#bbbbbb', marker='.')
  ax.plot(X[different, 0], X[different, 1], 'w',
          markerfacecolor='m', marker='.')
  ax.set_title('Difference')
  ax.set_xticks(())
  ax.set_yticks(())

  plt.savefig('cluster/kmeans_and_minibatch.png')

def kmeans_cluster_texts():
  '''
  example: cluster text documnets using kmeans
  通过词袋模型，对文本文档进行主题分类，数据特征，高维，稀疏
  '''
  # 载入新闻分类数据
  categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
  dataset = fetch_20newsgroups(subset='all', categories=categories,
                               shuffle=True, random_state=42)
  data = dataset.data
  labels = dataset.target
  print("%d documents, %d categories" % (len(data), len(np.unique(labels))))

  # 文本特征提取 TfidfVectorizer
  t0 = time()
  n_features = 10000
  vertorizer = TfidfVectorizer(max_df=0.5, min_df=0.2, max_features=n_features,
                               stop_words='english', use_idf=True)
  X = vertorizer.fit_transform(data)
  print('n_samples: %d, n_features: %d done in %fs' % (X.shape[0], X.shape[1], (time() - t0)))

  #
  pass

def affinity_propagation():
  '''
  example: demo of affinity propagation clustering
  '''
  # 生成各向同性的高斯分布数据点
  centers = [[1,1], [-1,-1], [1,-1]]
  X, y = make_blobs(n_samples=300, centers=centers, cluster_std=0.5, random_state=0)

  # Affinity Propagation
  t0 = time()
  af = AffinityPropagation(preference=-50).fit(X)
  cluster_centers_indices = af.cluster_centers_indices_
  labels = af.labels_
  n_clusters_ = len(cluster_centers_indices)
  
  print('time\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')
  print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels, estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean', sample_size=300)))


def affinity_propagation_cluster_stock():
  '''
  example: Visualizing the stock market structure
  '''
  pass

if __name__ == "__main__":
  # kmeans
  ## kmeans_assumptions()
  ## kmeans_cluster_digits()
  ## kmeans_and_minibatch()
  ## kmeans_cluster_texts()

  # affinity propagation

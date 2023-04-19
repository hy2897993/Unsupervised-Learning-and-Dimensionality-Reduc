#############################################################################
#%%
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier



#############################################################################
#%%

X_0, y_0 = fetch_covtype(return_X_y=True)

X, X_1, y, y_1 = train_test_split(X_0, y_0, test_size=0.995, random_state=42)

#############################################################################
#%%
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics import silhouette_samples, silhouette_score

homogeneity_scores = []
completeness_scores = []
silhouette_scores = []
k = []
Sum_of_squared_distances = []

for i in range(2, 21):
    k.append(i)
    kmeans = KMeans(
        n_clusters=i,
        init='k-means++',
        n_init=100, 
        max_iter=3000, 
        tol=0.0001, 
        verbose=0, 
        random_state=None, 
        copy_x=True, 
        algorithm='auto').fit(X)

    homogeneity_scores.append(homogeneity_score(kmeans.labels_, y))
    completeness_scores.append(completeness_score(kmeans.labels_, y))
    cluster_labels = kmeans.fit_predict(X)
    silhouette_scores.append(silhouette_score(X, cluster_labels)) 
    Sum_of_squared_distances.append(kmeans.inertia_)
plt.suptitle("Performance for K-means clustering")

plt.xlabel("K clusters")
plt.ylabel("Evaluation Score")

plt.grid()

plt.plot(
    k, homogeneity_scores, "o-", color="g",markersize=0.1, label="homogeneity scores"
)

plt.plot(
    k, completeness_scores, "o-", color="r",markersize=0.1, label="completeness scores"
)

plt.plot(
    k, silhouette_scores, "o-", color="y",markersize=0.1, label="silhouette scores"
)
plt.legend(loc="best")

############################################################################
#%%

plt.suptitle("Squared distance for K-means clustering")
plt.xlabel("K clusters")
plt.ylabel("Sum of squared distances")
plt.plot(
    k, Sum_of_squared_distances, "o-", color="b",markersize=0.1, label=""
)
plt.legend(loc="best")

#%%
#EM EM EM EM EM EM EM EM EM EM EM EM EM EM EM EM EM EM EM EM EM EM EM EM 
import numpy as np
from sklearn.mixture import GaussianMixture
homogeneity_scores = []
completeness_scores = []
silhouette_scores = []
k = []
Sum_of_squared_distances = []
bic = []
aic= []

for i in range(2, 21):
    k.append(i)
    gm = GaussianMixture(
        n_components=i,
        covariance_type='full', 
        tol=0.001, 
        reg_covar=1e-06, 
        max_iter=100, 
        n_init=1, 
        init_params='kmeans', 
        weights_init=None, 
        means_init=None, 
        precisions_init=None, 
        random_state=None, 
        warm_start=False, 
        verbose=0, 
        verbose_interval=10).fit(X)
    x = np.array(X)
    print(gm.bic(x))
    bic.append(gm.bic(x))
    aic.append(gm.aic(x))
    cluster_labels = gm.fit_predict(X)
    homogeneity_scores.append(homogeneity_score(cluster_labels, y))
    completeness_scores.append(completeness_score(cluster_labels, y))
    
    silhouette_scores.append(silhouette_score(X, cluster_labels)) 
    #Sum_of_squared_distances.append(kmeans.inertia_)
#%%
plt.suptitle("Performance for EM")

plt.xlabel("cluster number")
plt.ylabel("Evaluation Score")

plt.grid()

plt.plot(
    k, homogeneity_scores, "o-", color="g",markersize=0.1, label="homogeneity scores"
)

plt.plot(
    k, completeness_scores, "o-", color="r",markersize=0.1, label="completeness scores"
)

plt.plot(
    k, silhouette_scores, "o-", color="y",markersize=0.1, label="silhouette scores"
)
plt.legend(loc="best")

#%%
plt.suptitle("Squared distance for EM")
plt.xlabel("cluster number")
plt.ylabel("Sum of squared distances")
plt.plot(
    k, Sum_of_squared_distances, "o-", color="b",markersize=0.1, label=""
)
plt.legend(loc="best")

#%%
plt.suptitle("AIC & BIC for EM")
plt.xlabel("cluster number")
plt.ylabel("AIC/BIC")
plt.plot(
    k, aic, "o-", color="r",markersize=0.1, label="AIC"
)
plt.plot(
    k, bic, "o-", color="g",markersize=0.1, label="BIC"
)
plt.legend(loc="best")


#%%
# PCA PCA PCA PCA PCA PCA PCA PCA PCA PCA PCA PCA PCA PCA PCA PCA PCA PCA PCA PCA PCA PCA PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=25)
pca.fit(X)
#%%
plt.suptitle("Eigenvalues For PCA Components")
plt.xlabel("Components")
plt.ylabel("Eigenvalues")
plt.xticks(range(0,25,1))
plt.plot(
    range(0,25), (pca.explained_variance_)[:], "o-", color="r",markersize=0.1, label=""
)
#print(pca.explained_variance_)
#%%
plt.suptitle("Cumulative Variance For PCA Components")
plt.xlabel("Components")
plt.ylabel("Explained Variance %")
plt.xticks(range(0,25,1))
# plt.yticks(range(90,101,1))
sum_explained_variance = []
temp = 0
for j in range(1,26):
    print(sum(pca.explained_variance_ratio_[:j]))
    sum_explained_variance.append(100*sum(pca.explained_variance_ratio_[:j]))
plt.plot(
    range(1,26), sum_explained_variance, "o-", color="b",markersize=0.1, label=""
)
#%%
#average log-likelihood of all samples
average_log_likelihood_no_y = []
average_log_likelihood_with_y = []
pcas = []
for i in range(1,26):
    pca = PCA(n_components=i)
    pca.fit(X)
    pcas.append(pca)
    average_log_likelihood_no_y.append(pca.score(X))
    average_log_likelihood_with_y.append(pca.score(X,y))
#%%
plt.suptitle("average log-likelihood of all samples")
plt.xlabel("Model with n Components")
plt.xticks(range(0,25,5))


plt.plot(
    range(1,26), average_log_likelihood_no_y, "o-", color="r",markersize=0.1, label="Without features"
)
# plt.plot(
#     range(1,61), average_log_likelihood_with_y, "o-", color="g",markersize=0.1, label="With features"
# )


#%%
# run NN
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

accuracy_scores = []

for i in range(1,26):
    pca = PCA(n_components=i)
    X_trans = pca.fit_transform(X)
    print(i)
    x_tr,x_te,y_tr,y_te=train_test_split(X_trans, y, train_size=0.75)
    clf = MLPClassifier(
        alpha=0.01, 
        batch_size='auto', 
        epsilon=1e-08, 
        hidden_layer_sizes=(200,100,50,25), 
        learning_rate='adaptive', max_iter=3000
    )
    clf.fit(x_tr, y_tr)
    y_test_pred=clf.predict(x_te)

    accuracy_test=accuracy_score(y_true=y_te, y_pred=y_test_pred)
    accuracy_scores.append(accuracy_test)
#%%
plt.suptitle("Neural Network Score For PCA")
plt.xlabel("Components")
plt.ylabel("Neural Network Score")
plt.xticks(range(0,26,1))

plt.plot(
    range(1,26), accuracy_scores, "o-", color="r",markersize=0.1
)
plt.legend(loc="best")


#%%
# ICA ICA ICA

from sklearn.decomposition import FastICA
from scipy.stats import norm, kurtosis
import math

abs_kurtosis= []
for i in range(1,26):
    print(i)
    transformer = FastICA(n_components=i,random_state=0, max_iter = 4000*i)
    X_transformed = transformer.fit_transform(X)


    kurtosis_arr = abs(kurtosis(X_transformed))

    abs_kurtosis.append(kurtosis_arr)

#%%

labels = range(1,26)
width = 0.35       # the width of the bars: can also be len(x) sequence

fig, ax = plt.subplots()
kurt0 = []
for j in abs_kurtosis:
    kurt0.append(j[0])
ax.bar(labels, kurt0, width)
kurts = []
kurts.append(kurt0)
for i in range(1,60):
    kurt = []
    for j in abs_kurtosis:
        if(i>len(j)-1):
            kurt.append(0)
        else:
            kurt.append(j[i])
    kurts.append(kurt)
    ax.bar(labels, kurt, width,  bottom=kurts[len(kurts)-2])


ax.set_ylabel('Kurtosis')
ax.set_title('Absolute kurtosis of each component in ICA')
ax.legend()

plt.show()

#%%
# NN ICA
accuracy_scores_ICA = []

for i in range(1,26):
    transformer = FastICA(n_components=i, max_iter = 4100*i)
    X_trans = transformer.fit_transform(X)
    
    print(i)
    x_tr,x_te,y_tr,y_te=train_test_split(X_trans, y, train_size=0.75)
    clf = MLPClassifier(
        alpha=0.01, 
        batch_size='auto', 
        epsilon=1e-08, 
        hidden_layer_sizes=(200,100,50,25), 
        learning_rate='adaptive',
        max_iter=3000
    )
    clf.fit(x_tr, y_tr)
    y_test_pred=clf.predict(x_te)

    accuracy_test=accuracy_score(y_true=y_te, y_pred=y_test_pred)
    accuracy_scores_ICA.append(accuracy_test)
#%%
plt.suptitle("Neural Network Score For ICA")
plt.xlabel("Components")
plt.ylabel("Neural Network Score")
plt.xticks(range(0,26,5))

plt.plot(
    range(1,26), accuracy_scores_ICA, "o-", color="r",markersize=0.1
)
plt.legend(loc="best")
#%%
# when n = 14
transformer = FastICA(n_components=14,random_state=0, max_iter = 4100*21)
X_transformed = transformer.fit_transform(X)


kurtosis_arr = abs(kurtosis(X_transformed))

labels = range(1,len(kurtosis_arr)+1)
width = 0.35       # the width of the bars: can also be len(x) sequence

fig, ax = plt.subplots()
ax.bar(labels, kurtosis_arr, width)
ax.axhspan
ax.set_ylabel('Kurtosis')
ax.set_title('Absolute kurtosis of each component in ICA(components = 14)')
ax.legend()

plt.show()


#%%
#Random Projection

from sklearn.random_projection import GaussianRandomProjection
from sklearn.metrics.pairwise import euclidean_distances

plt.suptitle("Euclidean distances variance For RP")
plt.xlabel("Components")
plt.ylabel("Euclidean distances Variance %")
plt.xticks(range(0,26,5))
# plt.yticks(range(90,101,1))

b = sum(sum(euclidean_distances(X)))

arr_1 = []
arr_2 = []
arr_3 = []

for i in range(1, 26):

    transformer_1 = GaussianRandomProjection(n_components=i,random_state=7)
    transformer_2 = GaussianRandomProjection(n_components=i,random_state=52)
    transformer_3 = GaussianRandomProjection(n_components=i,random_state=85)
    X_new_1 = transformer_1.fit_transform(X)
    X_new_2 = transformer_2.fit_transform(X)
    X_new_3 = transformer_3.fit_transform(X)

    arr_1.append(sum(sum(euclidean_distances(X_new_1)))/b *100)
    arr_2.append(sum(sum(euclidean_distances(X_new_2)))/b *100)
    arr_3.append(sum(sum(euclidean_distances(X_new_3)))/b *100)

#%%
plt.suptitle("Euclidean distances variance For RP")
plt.xlabel("Components")
plt.ylabel("Euclidean distances Variance %")
plt.xticks(range(0,26,5))
plt.plot(
    range(1,26), arr_1, "o-", color="b",markersize=0.1, label="Random state 1"
)
plt.plot(
    range(1,26), arr_2, "o-", color="g",markersize=0.1, label="Random state 2"
)
plt.plot(
    range(1,26), arr_3, "o-", color="y",markersize=0.1, label="Random state 3"
)
arr = []
for j in range(len(arr_1)):
    arr.append((arr_1[j]+arr_2[j]+arr_3[j])/3)
plt.plot(
    range(1,26), arr, "o-", color="r",markersize=0.1, label="Average"
)
plt.legend(loc="best")
#%%
# NN RP

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.random_projection import GaussianRandomProjection
from sklearn.metrics.pairwise import euclidean_distances

accuracy_scores_RP = []

for i in range(1,26):
    transformer_1 = GaussianRandomProjection(n_components=i,random_state=7)
    transformer_2 = GaussianRandomProjection(n_components=i,random_state=52)
    transformer_3 = GaussianRandomProjection(n_components=i,random_state=85)
    X_new_1 = transformer_1.fit_transform(X)
    X_new_2 = transformer_2.fit_transform(X)
    X_new_3 = transformer_3.fit_transform(X)
    
    print(i)
    # 1
    x_tr,x_te,y_tr,y_te=train_test_split(X_new_1, y, train_size=0.75)
    clf = MLPClassifier(
        alpha=0.01, 
        batch_size='auto', 
        epsilon=1e-08, 
        hidden_layer_sizes=(200,100,50,25), 
        learning_rate='adaptive',
        max_iter=3000
    )
    clf.fit(x_tr, y_tr)
    y_test_pred_1=clf.predict(x_te)

    accuracy_test=accuracy_score(y_true=y_te, y_pred=y_test_pred_1)
    #2
    x_tr,x_te,y_tr,y_te=train_test_split(X_new_2, y, train_size=0.75)
    clf = MLPClassifier(
        alpha=0.01, 
        batch_size='auto', 
        epsilon=1e-08, 
        hidden_layer_sizes=(200,100,50,25), 
        learning_rate='adaptive',
        max_iter=3000
    )
    clf.fit(x_tr, y_tr)
    y_test_pred_2=clf.predict(x_te)

    accuracy_test+=accuracy_score(y_true=y_te, y_pred=y_test_pred_2)
    #3
    x_tr,x_te,y_tr,y_te=train_test_split(X_new_3, y, train_size=0.75)
    clf = MLPClassifier(
        alpha=0.01, 
        batch_size='auto', 
        epsilon=1e-08, 
        hidden_layer_sizes=(200,100,50,25), 
        learning_rate='adaptive',
        max_iter=3000
    )
    clf.fit(x_tr, y_tr)
    y_test_pred_3=clf.predict(x_te)

    accuracy_test+=accuracy_score(y_true=y_te, y_pred=y_test_pred_3)
    

    accuracy_scores_RP.append(accuracy_test/3)

#%%
print(accuracy_scores_RP)
#%%
plt.suptitle("Average Neural Network Score For RP")
plt.xlabel("Components")
plt.ylabel("Average Neural Network Score")
plt.xticks(range(0,26,5))

plt.plot(
    range(1,26), accuracy_scores_RP, "o-", color="r",markersize=0.1
)
plt.legend(loc="best")

#%%
#FA FA FA FA FA
from sklearn.decomposition import FactorAnalysis

score_FA = []
for i in range(1, 26):
    transformer = FactorAnalysis(n_components=i, random_state=0)
    transformer.fit(X)
    score_FA.append(transformer.score(X))
#%%
plt.suptitle("Average log-likelihood of FA")
plt.xlabel("Components")
plt.ylabel("Average log-likelihood")
plt.xticks(range(0,26,5))
plt.plot(
    range(1,26), score_FA, "o-", color="b",markersize=0.1
)
plt.legend(loc="best")

#%%
#get_covariance()
arr_FA = []
for i in range(1, 26):

    transformer = FactorAnalysis(n_components=i)
    X_transformed = transformer.fit_transform(X)
    arr_FA.append(sum(sum(euclidean_distances(X_transformed)))/sum(sum(euclidean_distances(X))) *100)

#%%
plt.suptitle("Euclidean distances variance For FA")
plt.xlabel("Components")
plt.ylabel("Euclidean distances Variance %")
plt.xticks(range(0,26,5))
plt.plot(
    range(1,26), arr_FA, "o-", color="r",markersize=0.1
)
plt.legend(loc="best")

#%%
# NN FA
accuracy_score_FA = []
for i in range(1,26):
    transformer = FactorAnalysis(n_components=i, random_state=0)
    X_trans = transformer.fit_transform(X)
    x_tr,x_te,y_tr,y_te=train_test_split(X_trans, y, train_size=0.75)
    clf = MLPClassifier(
        alpha=0.01, 
        batch_size='auto', 
        epsilon=1e-08, 
        hidden_layer_sizes=(200,100,50,25), 
        learning_rate='adaptive',
        max_iter=3000
    )
    clf.fit(x_tr, y_tr)
    y_test_pred=clf.predict(x_te)

    accuracy_test=accuracy_score(y_true=y_te, y_pred=y_test_pred)
    accuracy_score_FA.append(accuracy_test)
#%%
plt.suptitle("Neural Network Score For FA")
plt.xlabel("Components")
plt.ylabel(" Neural Network Score")
plt.xticks(range(0,26,5))

plt.plot(
    range(1,26), accuracy_score_FA, "o-", color="r",markersize=0.1
)
plt.legend(loc="best")

#############################################################################
#%%


#do cluster to 4 reduced dimension
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from scipy.stats import norm, kurtosis
import math
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import FactorAnalysis




pca = PCA(n_components=22)
X_Transformed_PCA = pca.fit_transform(X)

ica= FastICA(n_components=14,max_iter = 5000)
X_Transformed_ICA = ica.fit_transform(X)

transformer_1 = GaussianRandomProjection(n_components=12,random_state=7)
transformer_2 = GaussianRandomProjection(n_components=12,random_state=52)
transformer_3 = GaussianRandomProjection(n_components=12,random_state=85)
X_Transformed_RP_1 = transformer_1.fit_transform(X)
X_Transformed_RP_2 = transformer_2.fit_transform(X)
X_Transformed_RP_3 = transformer_3.fit_transform(X)

fa = FactorAnalysis(n_components=18, random_state=0)
X_Transformed_FA = fa.fit_transform(X)


homogeneity_scores_PCA = []
completeness_scores_PCA = []
silhouette_scores_PCA = []

homogeneity_scores_ICA = []
completeness_scores_ICA = []
silhouette_scores_ICA = []

homogeneity_scores_RP= []
completeness_scores_RP = []
silhouette_scores_RP = []

homogeneity_scores_FA = []
completeness_scores_FA = []
silhouette_scores_FA = []

k = []
Sum_of_squared_distances_PCA = []
Sum_of_squared_distances_ICA = []
Sum_of_squared_distances_RP = []
Sum_of_squared_distances_FA = []

for i in range(2, 21):
    k.append(i)
    kmeans = KMeans(
        n_clusters=i,
        init='k-means++',
        n_init=100, 
        max_iter=3000, 
        tol=0.0001, 
        verbose=0, 
        random_state=None, 
        copy_x=True, 
        algorithm='auto').fit(X_Transformed_PCA)

    homogeneity_scores_PCA.append(homogeneity_score(kmeans.labels_, y))
    completeness_scores_PCA.append(completeness_score(kmeans.labels_, y))
    cluster_labels = kmeans.fit_predict(X_Transformed_PCA)
    silhouette_scores_PCA.append(silhouette_score(X_Transformed_PCA, cluster_labels)) 
    Sum_of_squared_distances_PCA.append(kmeans.inertia_)

    kmeans = KMeans(
        n_clusters=i,
        init='k-means++',
        n_init=100, 
        max_iter=3000, 
        tol=0.0001, 
        verbose=0, 
        random_state=None, 
        copy_x=True, 
        algorithm='auto').fit(X_Transformed_ICA)

    homogeneity_scores_ICA.append(homogeneity_score(kmeans.labels_, y))
    completeness_scores_ICA.append(completeness_score(kmeans.labels_, y))
    cluster_labels = kmeans.fit_predict(X_Transformed_ICA)
    silhouette_scores_ICA.append(silhouette_score(X_Transformed_ICA, cluster_labels)) 
    Sum_of_squared_distances_ICA.append(kmeans.inertia_)

    kmeans_1 = KMeans(n_clusters=i,init='k-means++',n_init=100, max_iter=3000, tol=0.0001, verbose=0, random_state=None,  copy_x=True, algorithm='auto').fit(X_Transformed_RP_1)
    kmeans_2 = KMeans(n_clusters=i,init='k-means++',n_init=100, max_iter=3000, tol=0.0001, verbose=0, random_state=None,  copy_x=True, algorithm='auto').fit(X_Transformed_RP_2)
    kmeans_3 = KMeans(n_clusters=i,init='k-means++',n_init=100, max_iter=3000, tol=0.0001, verbose=0, random_state=None,  copy_x=True, algorithm='auto').fit(X_Transformed_RP_3)


    homogeneity_scores_RP.append((homogeneity_score(kmeans_1.labels_, y)+homogeneity_score(kmeans_2.labels_, y)+homogeneity_score(kmeans_3.labels_, y))/3)
    completeness_scores_RP.append((completeness_score(kmeans_1.labels_, y)+completeness_score(kmeans_2.labels_, y)+completeness_score(kmeans_3.labels_, y))/3)
    cluster_labels_1 = kmeans_1.fit_predict(X_Transformed_RP_1)
    cluster_labels_2 = kmeans_2.fit_predict(X_Transformed_RP_2)
    cluster_labels_3 = kmeans_3.fit_predict(X_Transformed_RP_3)
    silhouette_scores_RP.append((silhouette_score(X_Transformed_RP_1, cluster_labels_1)+silhouette_score(X_Transformed_RP_2, cluster_labels_2)+silhouette_score(X_Transformed_RP_3, cluster_labels_3))/3) 
    Sum_of_squared_distances_RP.append((kmeans_1.inertia_+kmeans_2.inertia_+kmeans_3.inertia_)/3)

    kmeans = KMeans(
        n_clusters=i,
        init='k-means++',
        n_init=100, 
        max_iter=3000, 
        tol=0.0001, 
        verbose=0, 
        random_state=None, 
        copy_x=True, 
        algorithm='auto').fit(X_Transformed_FA)

    homogeneity_scores_FA.append(homogeneity_score(kmeans.labels_, y))
    completeness_scores_FA.append(completeness_score(kmeans.labels_, y))
    cluster_labels = kmeans.fit_predict(X_Transformed_FA)
    silhouette_scores_FA.append(silhouette_score(X_Transformed_FA, cluster_labels)) 
    Sum_of_squared_distances_FA.append(kmeans.inertia_)
#%%
plt.suptitle("Homogeneity score for K-means clustering, Dataset 2")

plt.xlabel("K clusters")
plt.ylabel("Homogeneity score")

plt.grid()

plt.plot(
    k, homogeneity_scores_PCA, "o-", color="g",markersize=0.1, label="PCA"
)

plt.plot(
    k, homogeneity_scores_ICA, "o-", color="r",markersize=0.1, label="ICA"
)

plt.plot(
    k, homogeneity_scores_RP, "o-", color="y",markersize=0.1, label="RP"
)

plt.plot(
    k, homogeneity_scores_FA, "o-", color="b",markersize=0.1, label="FA"
)
plt.legend(loc="best")
#%%
plt.suptitle("Completeness score for K-means clustering, Dataset 2")

plt.xlabel("K clusters")
plt.ylabel("Completeness score")

plt.grid()

plt.plot(
    k, completeness_scores_PCA, "o-", color="g",markersize=0.1, label="PCA"
)

plt.plot(
    k, completeness_scores_ICA, "o-", color="r",markersize=0.1, label="ICA"
)

plt.plot(
    k, completeness_scores_RP, "o-", color="y",markersize=0.1, label="RP"
)

plt.plot(
    k, completeness_scores_FA, "o-", color="b",markersize=0.1, label="FA"
)
plt.legend(loc="best")

#%%
plt.suptitle("Sum of squared distances for K-means clustering, Dataset 2")

plt.xlabel("K clusters")
plt.ylabel("Sum_of_squared_distances")

plt.grid()

plt.plot(
    k, Sum_of_squared_distances_PCA, "o-", color="g",markersize=0.1, label="PCA"
)

plt.plot(
    k, Sum_of_squared_distances_ICA, "o-", color="r",markersize=0.1, label="ICA"
)

plt.plot(
    k, Sum_of_squared_distances_RP, "o-", color="y",markersize=0.1, label="RP"
)

plt.plot(
    k, Sum_of_squared_distances_FA, "o-", color="b",markersize=0.1, label="FA"
)
plt.legend(loc="best")


#%%
plt.suptitle("Silhouette score for K-means clustering, Dataset 2")

plt.xlabel("K clusters")
plt.ylabel("Silhouette score")

plt.grid()

plt.plot(
    k, silhouette_scores_PCA, "o-", color="g",markersize=0.1, label="PCA"
)

plt.plot(
    k, silhouette_scores_ICA, "o-", color="r",markersize=0.1, label="ICA"
)

plt.plot(
    k, silhouette_scores_RP, "o-", color="y",markersize=0.1, label="RP"
)

plt.plot(
    k, silhouette_scores_FA, "o-", color="b",markersize=0.1, label="FA"
)
plt.legend(loc="best")





#%%
#do EM EM EM
#do cluster to 4 reduced dimension
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from scipy.stats import norm, kurtosis
import math
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import FactorAnalysis
from sklearn.mixture import GaussianMixture



    



pca = PCA(n_components=22)
X_Transformed_PCA = pca.fit_transform(X)

ica= FastICA(n_components=14,max_iter = 5000)
X_Transformed_ICA = ica.fit_transform(X)

transformer_1 = GaussianRandomProjection(n_components=12,random_state=7)
transformer_2 = GaussianRandomProjection(n_components=12,random_state=52)
transformer_3 = GaussianRandomProjection(n_components=12,random_state=85)
X_Transformed_RP_1 = transformer_1.fit_transform(X)
X_Transformed_RP_2 = transformer_2.fit_transform(X)
X_Transformed_RP_3 = transformer_3.fit_transform(X)

fa = FactorAnalysis(n_components=18, random_state=0,max_iter = 5000)
X_Transformed_FA = fa.fit_transform(X)


homogeneity_scores_PCA = []
completeness_scores_PCA = []
silhouette_scores_PCA = []
aic_PCA= []
bic_PCA = []

homogeneity_scores_ICA = []
completeness_scores_ICA = []
silhouette_scores_ICA = []
aic_ICA= []
bic_ICA = []

homogeneity_scores_RP= []
completeness_scores_RP = []
silhouette_scores_RP = []
aic_RP= []
bic_RP = []

homogeneity_scores_FA = []
completeness_scores_FA = []
silhouette_scores_FA = []
aic_FA= []
bic_FA = []


k = []


for i in range(2, 21):
    k.append(i)
    gm = GaussianMixture(
        n_components=i,
        covariance_type='full', 
        tol=0.001, 
        reg_covar=1e-06, 
        max_iter=2000, 
        n_init=1, 
        init_params='kmeans', 
        weights_init=None, 
        means_init=None, 
        precisions_init=None, 
        random_state=None, 
        warm_start=False, 
        verbose=0, 
        verbose_interval=10).fit(X_Transformed_PCA)

    cluster_labels = gm.fit_predict(X_Transformed_PCA)
    homogeneity_scores_PCA.append(homogeneity_score(cluster_labels, y))
    completeness_scores_PCA.append(completeness_score(cluster_labels, y))
    
    silhouette_scores_PCA.append(silhouette_score(X_Transformed_PCA, cluster_labels)) 
    aic_PCA.append(gm.aic(X_Transformed_PCA))
    bic_PCA.append(gm.bic(X_Transformed_PCA))

    gm = GaussianMixture(
        n_components=i,
        covariance_type='full', 
        tol=0.001, 
        reg_covar=1e-06, 
        max_iter=2000, 
        n_init=1, 
        init_params='kmeans', 
        weights_init=None, 
        means_init=None, 
        precisions_init=None, 
        random_state=None, 
        warm_start=False, 
        verbose=0, 
        verbose_interval=10).fit(X_Transformed_ICA)

    cluster_labels = gm.fit_predict(X_Transformed_ICA)
    homogeneity_scores_ICA.append(homogeneity_score(cluster_labels, y))
    completeness_scores_ICA.append(completeness_score(cluster_labels, y))
    
    silhouette_scores_ICA.append(silhouette_score(X_Transformed_ICA, cluster_labels)) 
    aic_ICA.append(gm.aic(X_Transformed_ICA))
    bic_ICA.append(gm.bic(X_Transformed_ICA))

    gm_1 = GaussianMixture(
        n_components=i,
        covariance_type='full', 
        tol=0.001, 
        reg_covar=1e-06, 
        max_iter=2000, 
        n_init=1, 
        init_params='kmeans', 
        weights_init=None, 
        means_init=None, 
        precisions_init=None, 
        random_state=None, 
        warm_start=False, 
        verbose=0, 
        verbose_interval=10).fit(X_Transformed_RP_1)
    gm_2 = GaussianMixture(
        n_components=i,
        covariance_type='full', 
        tol=0.001, 
        reg_covar=1e-06, 
        max_iter=2000, 
        n_init=1, 
        init_params='kmeans', 
        weights_init=None, 
        means_init=None, 
        precisions_init=None, 
        random_state=None, 
        warm_start=False, 
        verbose=0, 
        verbose_interval=10).fit(X_Transformed_RP_2)
    gm_3 = GaussianMixture(
        n_components=i,
        covariance_type='full', 
        tol=0.001, 
        reg_covar=1e-06, 
        max_iter=2000, 
        n_init=1, 
        init_params='kmeans', 
        weights_init=None, 
        means_init=None, 
        precisions_init=None, 
        random_state=None, 
        warm_start=False, 
        verbose=0, 
        verbose_interval=10).fit(X_Transformed_RP_3)

    cluster_labels_1 = gm_1.fit_predict(X_Transformed_RP_1)
    cluster_labels_2 = gm_2.fit_predict(X_Transformed_RP_2)
    cluster_labels_3 = gm_3.fit_predict(X_Transformed_RP_3)
    homogeneity_scores_RP.append((homogeneity_score(cluster_labels_1, y)+homogeneity_score(cluster_labels_2, y)+homogeneity_score(cluster_labels_3, y))/3)
    completeness_scores_RP.append((completeness_score(cluster_labels_1, y)+completeness_score(cluster_labels_2, y)+completeness_score(cluster_labels_3, y))/3)
    
    silhouette_scores_RP.append((silhouette_score(X_Transformed_RP_1, cluster_labels_1)+silhouette_score(X_Transformed_RP_2, cluster_labels_2)+silhouette_score(X_Transformed_RP_3, cluster_labels_3))/3) 
    aic_RP.append((gm_1.aic(X_Transformed_RP_1)+gm_2.aic(X_Transformed_RP_2)+gm_3.aic(X_Transformed_RP_3))/3)
    bic_RP.append((gm_1.bic(X_Transformed_RP_1)+gm_2.bic(X_Transformed_RP_2)+gm_3.bic(X_Transformed_RP_3))/3)


    gm = GaussianMixture(
        n_components=i,
        covariance_type='full', 
        tol=0.001, 
        reg_covar=1e-06, 
        max_iter=2000, 
        n_init=1, 
        init_params='kmeans', 
        weights_init=None, 
        means_init=None, 
        precisions_init=None, 
        random_state=None, 
        warm_start=False, 
        verbose=0, 
        verbose_interval=10).fit(X_Transformed_FA)

    cluster_labels = gm.fit_predict(X_Transformed_FA)
    homogeneity_scores_FA.append(homogeneity_score(cluster_labels, y))
    completeness_scores_FA.append(completeness_score(cluster_labels, y))

    silhouette_scores_FA.append(silhouette_score(X_Transformed_FA, cluster_labels)) 
    aic_FA.append(gm.aic(X_Transformed_FA))
    bic_FA.append(gm.bic(X_Transformed_FA))
#%%
plt.suptitle("Homogeneity score for EM, Dataset 2")

plt.xlabel("cluster number")
plt.ylabel("Homogeneity score")

plt.grid()

plt.plot(
    k, homogeneity_scores_PCA, "o-", color="g",markersize=0.1, label="PCA"
)

plt.plot(
    k, homogeneity_scores_ICA, "o-", color="r",markersize=0.1, label="ICA"
)

plt.plot(
    k, homogeneity_scores_RP, "o-", color="y",markersize=0.1, label="RP"
)

plt.plot(
    k, homogeneity_scores_FA, "o-", color="b",markersize=0.1, label="FA"
)
plt.legend(loc="best")
#%%
plt.suptitle("Completeness score for EM, Dataset 2")

plt.xlabel("cluster number")
plt.ylabel("Completeness score")

plt.grid()

plt.plot(
    k, completeness_scores_PCA, "o-", color="g",markersize=0.1, label="PCA"
)

plt.plot(
    k, completeness_scores_ICA, "o-", color="r",markersize=0.1, label="ICA"
)

plt.plot(
    k, completeness_scores_RP, "o-", color="y",markersize=0.1, label="RP"
)

plt.plot(
    k, completeness_scores_FA, "o-", color="b",markersize=0.1, label="FA"
)
plt.legend(loc="best")

#%%
plt.suptitle("AIC for EM, Dataset 2")

plt.xlabel("cluster number")
plt.ylabel("AIC")

plt.grid()

plt.plot(
    k, aic_PCA, "o-", color="g",markersize=0.1, label="PCA"
)

plt.plot(
    k, aic_ICA, "o-", color="r",markersize=0.1, label="ICA"
)

plt.plot(
    k, aic_RP, "o-", color="y",markersize=0.1, label="RP"
)

plt.plot(
    k, aic_FA, "o-", color="b",markersize=0.1, label="FA"
)
plt.legend(loc="best")

#%%
plt.suptitle("BIC for EM, Dataset 2")

plt.xlabel("cluster number")
plt.ylabel("BIC")

plt.grid()

plt.plot(
    k, bic_PCA, "o-", color="g",markersize=0.1, label="PCA"
)

plt.plot(
    k, bic_ICA, "o-", color="r",markersize=0.1, label="ICA"
)

plt.plot(
    k, bic_RP, "o-", color="y",markersize=0.1, label="RP"
)

plt.plot(
    k, bic_FA, "o-", color="b",markersize=0.1, label="FA"
)
plt.legend(loc="best")


#%%
plt.suptitle("Silhouette score for EM, Dataset 2")

plt.xlabel("cluster number")
plt.ylabel("Silhouette score")

plt.grid()

plt.plot(
    k, silhouette_scores_PCA, "o-", color="g",markersize=0.1, label="PCA"
)

plt.plot(
    k, silhouette_scores_ICA, "o-", color="r",markersize=0.1, label="ICA"
)

plt.plot(
    k, silhouette_scores_RP, "o-", color="y",markersize=0.1, label="RP"
)

plt.plot(
    k, silhouette_scores_FA, "o-", color="b",markersize=0.1, label="FA"
)
plt.legend(loc="best")

# %%
print(aic_RP)
plt.suptitle("AIC/BIC for EM Random Projection, Dataset 2")

plt.xlabel("cluster number")
plt.ylabel("AIC/BIC")

plt.grid()

plt.plot(
    k, aic_RP, "o-", color="r",markersize=0.1, label="AIC"
)
plt.plot(
    k, bic_RP, "o-", color="g",markersize=0.1, label="BIC"
)
# %%

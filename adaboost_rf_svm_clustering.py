from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from scipy.stats import mode
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd

# 0 Read File
a_list = pd.read_csv('bridges_version2.csv')

# 1 Preprocessing
a_list = a_list.iloc[:, 1:5]
encoder = LabelEncoder()
for column in a_list.columns:
    if a_list[column].dtype == 'object':
        a_list[column] = encoder.fit_transform(a_list[column])

scaler = StandardScaler()
columns_to_scale = a_list.columns[~a_list.columns.isin([a_list.columns[1]])]
a_list[columns_to_scale] = scaler.fit_transform(a_list[columns_to_scale])

a_list.to_csv('a_list_enc_norm.csv', index=False)

a_list = pd.read_csv('a_list_enc_norm.csv')

X_data = a_list.iloc[:, :-1].values
Y_data = a_list.iloc[:, 1].values

X_train, X_test, Y_train, Y_test = train_test_split(
    X_data, Y_data, test_size=0.2, random_state=123)

print("X_train:", X_train)
print("X_test:", X_test)
print("Y_train:", Y_train)
print("Y_test:", Y_test)


# 3. Running AdaBoost
clf = AdaBoostClassifier()
clf.fit(X_train, Y_train)
predictions = clf.predict(X_test)
pred_prob = clf.predict_proba(X_test)
accuracy = accuracy_score(Y_test, predictions)
print("Model accuracy: ", accuracy)


n_estimators_values = [3, 5, 7, 10, 50]
accuracy_temp = 0
accuracies = []

for n_estimators in n_estimators_values:
    clf = AdaBoostClassifier(n_estimators=n_estimators)
    clf.fit(X_train, Y_train)
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(Y_test, predictions)
    accuracy_temp = accuracy
    # 3.1 run by changing n_estimators = 3, 5, 7, 10, 50 and show the accuracies of each run.
    print(f"n_estimators: {n_estimators}, Accuracy: {accuracy}")
    accuracies.append(accuracy)

plt.figure(figsize=(10, 6))
plt.plot(n_estimators_values, accuracies, marker='o', linestyle='-')
plt.title('Effect of n_estimators on accuracy')
plt.xlabel('n_estimators')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

# 3.3 compare the best performance of AdaBoostClassifier with that of IBL. Which is better ? Explain the results.
clf_ibl = KNeighborsClassifier(n_neighbors=5)
clf_ibl.fit(X_train, Y_train)
predictions_ibl = clf_ibl.predict(X_test)
accuracy_ibl = accuracy_score(Y_test, predictions_ibl)
print(f"AdaBoost accuracy: {accuracy_temp}")
print(f"IBL accuracy: {accuracy_ibl}")


# 4. Running Random Forest
clf_rfc = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
clf_rfc.fit(X_train, Y_train)
predictions = clf_rfc.predict(X_test)
accuracy = accuracy_score(Y_test, predictions)
print(f'Running Random Forest Accuracy: {accuracy}')

# 4.1 Run by changing n_estimators = 3, 5, 7, 10, 50, 100 respectively, and show the accuracies of each run.
estimators = [3, 5, 7, 10, 50, 100]

accuracy_list = []

for n in estimators:
    clf = RandomForestClassifier(n_estimators=n, max_depth=2, random_state=0)
    clf.fit(X_train, Y_train)
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(Y_test, predictions)
    print(f'Accuracy for n_estimators={n}: {accuracy}')
    accuracy_list.append(accuracy)

# 4.2 Plot your results and explain the effect of the n_estimators
plt.figure(figsize=(10, 6))
plt.plot(estimators, accuracy_list, marker='o')
plt.title('Effect of n_estimators on Accuracy')
plt.xlabel('n_estimators')
plt.ylabel('Accuracy')
plt.grid()
plt.show()


# 4.3 Choose the optimal n_estimators from q. 1), and run the model by changing
# oob_score = True/False. respectively. Show the accuracies of each run, and
# explain the effect of the oob_score.

# 저는 50을 최적의 estimators로 골랐습니다.
n_optimal = 50

oob_scores = [True, False]

for oob in oob_scores:
    clf = RandomForestClassifier(
        n_estimators=n_optimal, max_depth=2, random_state=0, oob_score=oob)
    clf.fit(X_train, Y_train)
    predictions = clf.predict(X_test)

    accuracy = accuracy_score(Y_test, predictions)
    print(f'Accuracy for oob_score={oob}: {accuracy}')
    if oob:
        print(f'Out-of-bag score: {clf.oob_score_}')


# 4.4 Choose the optimal n_estimators from q. 1), and run the model by changing
# max_features = “auto”, “sqrt”, “log2”, respectively. Show the accuracies of each
# run, and explain the effect of the max_features.
max_features_options = ['auto', 'sqrt', 'log2']


for max_features in max_features_options:
    clf = RandomForestClassifier(
        n_estimators=n_optimal, max_depth=2, random_state=0, max_features=max_features)
    clf.fit(X_train, Y_train)
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(Y_test, predictions)
    print(f'Accuracy for max_features={max_features}: {accuracy}')


# 4.5 Choose the optimal n_estimators from q. 1), and run the model by changing
# “max_depth” to your own choice (small int number). SHow the effect of max_depth
# parameter.

max_depths = [2, 5, 10]
for max_depth in max_depths:
    clf = RandomForestClassifier(
        n_estimators=n_optimal, max_depth=max_depth, random_state=0)
    clf.fit(X_train, Y_train)
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(Y_test, predictions)
    print(f'Accuracy for max_depth={max_depth}: {accuracy}')


# 5. Running SVM

clf = svm.SVC()
clf.fit(X_train, Y_train)


# 5.1 calculate the accuracy of SVC

predictions = clf.predict(X_test)
accuracy = accuracy_score(Y_test, predictions)
print(f'Accuracy: {accuracy:.2f}')

# 5.2 run SVC by changing kernel to ‘linear’, ‘poly’, ‘rbf’, and ‘sigmoid’, and show
# the accuracies of each. Which kernel function shows the best accuracy ? and
# explain why ?

kernels = ['linear', 'poly', 'rbf', 'sigmoid']

for kernel in kernels:
    clf = svm.SVC(kernel=kernel)
    clf.fit(X_train, Y_train)
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(Y_test, predictions)
    print(f'Accuracy with {kernel} kernel: {accuracy:.2f}')

# 5.3 compare the accuracy of SVC with that of IBL, RandomForest, and AdaBoost,
# respectively. Explain the results.

classifiers = {
    "SVC": svm.SVC(),
    "IBL": KNeighborsClassifier(n_neighbors=3),
    "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0),
    "AdaBoost": AdaBoostClassifier(n_estimators=100),
}

for name, clf in classifiers.items():
    clf.fit(X_train, Y_train)
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(Y_test, predictions)
    print(f'{name} Accuracy: {accuracy:.2f}')


# 5.4 when using ‘poly’ kernel, change the value of coef0 multiple times to your own
# choice. Roughly speaking, it controls how much the model is influenced by
# high-degree polynomials.

coef0_values = [0.0, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

for coef0 in coef0_values:
    clf = svm.SVC(kernel='poly', coef0=coef0)
    clf.fit(X_train, Y_train)
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(Y_test, predictions)
    print(f'Accuracy with poly kernel and coef0={coef0}: {accuracy:.2f}')

# 6. Clustering (K Means)
kmeans = KMeans(n_clusters=4, max_iter=600, algorithm='auto', random_state=0)
kmeans.fit(X_train)
labels = kmeans.predict(X_train)
centroids = kmeans.cluster_centers_
inertia = kmeans.inertia_
print("Labels:", labels)
print("Centroids:", centroids)
print("Inertia:", inertia)

# 6.1 run KMeans 3 times by changing n_clusters = 2, 3, 5, 7 respectively and show
# the mean of each cluster.

n_clusters_values = [2, 3, 5, 7]

for n_clusters in n_clusters_values:
    kmeans = KMeans(n_clusters=n_clusters, max_iter=600,
                    algorithm='auto', random_state=0)
    kmeans.fit(X_train)

    centroids = kmeans.cluster_centers_

    print(f'Centroids for KMeans with {n_clusters} clusters:')
    for i, centroid in enumerate(centroids):
        print(f'Cluster {i+1}: {centroid}')
    print("\n")

# 6.2 For the clustering of n_clusters=3, pick one cluster. Calculate the average value
# of each attribute of the data in that cluster.

# 저는 n_clusters로 3을 선택했습니다.
kmeans = KMeans(n_clusters=3, max_iter=600, algorithm='auto', random_state=0)
kmeans.fit(X_train)
labels = kmeans.predict(X_train)
X_train_df = pd.DataFrame(X_train)
X_train_df['Cluster'] = labels
cluster_data = X_train_df[X_train_df['Cluster'] == 0]
cluster_data = cluster_data.drop('Cluster', axis=1)
cluster_mean = cluster_data.mean()

print("Average values for each attribute in the selected cluster:")
print(cluster_mean)

# 6.3 For each cluster, calculate majority (the most frequent) value of class/target
# value. (let’s call this ‘cluster label’)
df = pd.DataFrame(X_train)
df['Cluster'] = labels
df['Target'] = Y_train
for cluster in df['Cluster'].unique():
    most_common = df[df['Cluster'] == cluster]['Target'].mode()[0]
    print(f"The most common class in cluster {cluster} is: {most_common}")


# 6.4 Suppose each of X_test is classified based on ‘cluster labels’, calculate the
# accuracy.

test_labels = kmeans.predict(X_test)
predicted_labels = []

for label in test_labels:
    cluster_data = df[df['Cluster'] == label]
    most_common = cluster_data['Target'].mode()[0]
    predicted_labels.append(most_common)

accuracy = accuracy_score(Y_test, predicted_labels)
print(f'Accuracy: {accuracy:.2f}')

# 6.5 run KMeans 3 times by changing n_init values (your own choice of n_init).
# Compare the performance of each.


n_init_values = [5, 10, 20]

for n_init in n_init_values:
    kmeans = KMeans(n_clusters=4, max_iter=600, n_init=n_init,
                    algorithm='auto', random_state=0)
    kmeans.fit(X_train)
    labels = kmeans.predict(X_train)

    silhouette_avg = silhouette_score(X_train, labels)

    print(
        f'Silhouette Score for KMeans with n_init={n_init}: {silhouette_avg:.2f}')
    


# 7. Clustering (EM)

gmm = GaussianMixture(n_components=4).fit(X_train)
labels = gmm.predict(X_train)
probs = gmm.predict_proba(X_train)

print("Probabilistic cluster assignments for the first 5 data points:")
print(probs[:5].round(3))


# 7.1 run GaussianMixture 4 times by changing n_components = 2, 3, 4, 5
# respectively.

n_components_values = [2, 3, 4, 5]

for n_components in n_components_values:
    gmm = GaussianMixture(n_components=n_components).fit(X_train)
    labels = gmm.predict(X_train)
    print(f"Cluster labels for GMM with n_components={n_components}:")
    print(labels)
    print()

# 7.2 For the clustering of n_components=4, show the predicted labels for the input
# data.

gmm = GaussianMixture(n_components=4).fit(X_train)
predicted_labels = gmm.predict(X_train)
print("Predicted cluster labels for the input data:")
print(predicted_labels)

# 7.3 show the probabilistic cluster assignments. This returns a matrix of size
# [n_samples, n_clusters].
probs = gmm.predict_proba(X_train)
print("Probabilistic cluster assignments:")
print(probs)

# 7.4 Suppose each of X_test is classified based on ‘cluster labels’, calculate the
# accuracy. 

predicted_labels = gmm.predict(X_test)
accuracy = accuracy_score(Y_test, predicted_labels)
print(f"Accuracy: {accuracy:.2f}")


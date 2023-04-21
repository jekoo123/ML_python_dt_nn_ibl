from sklearn.preprocessing import LabelEncoder
import numpy as np
import csv
import pandas as pd
from sklearn.preprocessing import LabelEncoder,  MinMaxScaler, StandardScaler, KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from IPython.display import Image
import pydotplus

# 0
with open('bridges_version1.csv', newline='') as f:
    reader = csv.reader(f)
    a_list = list(reader)

for i in range(len(a_list)):
    a_list[i] = a_list[i][1:5]
# for i in a_list:
#     i = i[1:5]

with open('a_list.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(a_list)
print("\n")

# 1.1

a_list = pd.read_csv('a_list.csv')

encoder = LabelEncoder()

for column in a_list.columns:
    if a_list[column].dtype == 'object':
        a_list[column] = encoder.fit_transform(a_list[column])
print(a_list.head())
a_list.to_csv('a_list_enc.csv', index=False)


print("\n")

# 2.1

a_list_enc_m = pd.read_csv('a_list_enc.csv')
min_max_scaler = MinMaxScaler()
a_list_enc_m.iloc[:, :1] = min_max_scaler.fit_transform(
    a_list_enc_m.iloc[:, :1])
print(a_list_enc_m.head())
print("\n")


# 2.2
a_list_enc_s = pd.read_csv('a_list_enc.csv')
scaler = StandardScaler()
a_list_enc_s.iloc[:, 1:] = scaler.fit_transform(a_list_enc_s.iloc[:, 1:])
print(a_list_enc_s.head())
a_list_enc_s.to_csv('a_list_enc_norm.csv', index=False)
print("\n")


# 3
a_list_enc_norm = pd.read_csv('a_list_enc_norm.csv')
X_data = a_list_enc_norm.iloc[:, :-1].values
Y_data = a_list_enc_norm.iloc[:, 1].values
X_train, X_test, Y_train, Y_test = train_test_split(
    X_data, Y_data, test_size=0.2, random_state=123)
print("X_train:", X_train)
print("X_test:", X_test)
print("Y_train:", Y_train)
print("Y_test:", Y_test)


# 4
clf = MLPClassifier(hidden_layer_sizes=(30,))
Y_train = Y_train.astype(int)
clf.fit(X_train, Y_train)
predictions = clf.predict(X_test)
Y_test = Y_test.astype(int)
accuracy = clf.score(X_test, Y_test)
print('Accuracy: ', accuracy)

# 4.1
clf = MLPClassifier(hidden_layer_sizes=(10,))
clf.fit(X_train, Y_train)
accuracy = clf.score(X_test, Y_test)
print('Accuracy: ', accuracy)

clf = MLPClassifier(hidden_layer_sizes=(10, 10, 10))
clf.fit(X_train, Y_train)
accuracy = clf.score(X_test, Y_test)
print('Accuracy: ', accuracy)


# 4.2
hidden_layer_sizes_list = [
    (10,), (20,), (30,), (40,), (50,), (60,), (70,), (80,), (90,), (100,)]
accuracies = []

for hidden_layer_sizes in hidden_layer_sizes_list:
    clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes)
    clf.fit(X_train, Y_train)
    accuracy = clf.score(X_test, Y_test)
    accuracies.append(accuracy)

plt.plot(hidden_layer_sizes_list, accuracies)
plt.title('Accuracy vs Hidden Layer/Nodes')
plt.xlabel('Hidden Layer/Nodes')
plt.ylabel('Accuracy')
plt.show()


# 4.3
clf_iden = MLPClassifier(hidden_layer_sizes=(30,), activation='identity')
clf_log = MLPClassifier(hidden_layer_sizes=(30,), activation='logistic')
clf_tanh = MLPClassifier(hidden_layer_sizes=(30,), activation='tanh')
clf_relu = MLPClassifier(hidden_layer_sizes=(30,), activation='relu')

clf_iden.fit(X_train, Y_train)
clf_log.fit(X_train, Y_train)
clf_tanh.fit(X_train, Y_train)
clf_relu.fit(X_train, Y_train)

predictions_iden = clf_iden.predict(X_test)
predictions_log = clf_log.predict(X_test)
predictions_tanh = clf_tanh.predict(X_test)
predictions_relu = clf_relu.predict(X_test)

accuracy_iden = clf_iden.score(X_test, Y_test)
accuracy_log = clf_log.score(X_test, Y_test)
accuracy_tanh = clf_tanh.score(X_test, Y_test)
accuracy_relu = clf_relu.score(X_test, Y_test)

x_labels = ['identity', 'logistic', 'tanh', 'relu']
y_values = [accuracy_iden, accuracy_log, accuracy_tanh, accuracy_relu]
plt.bar(x_labels, y_values)
plt.title('Accuracy vs Activation Function')
plt.xlabel('Activation Function')
plt.ylabel('Accuracy')
plt.show()


# 4.4

momentum_values = [0, 0.1, 0.5, 0.9]
accuracy_values = []

for momentum in momentum_values:
    clf = MLPClassifier(hidden_layer_sizes=(
        30,), solver='sgd', momentum=momentum, random_state=1)
    clf.fit(X_train, Y_train)
    predictions = clf.predict(X_test)
    accuracy = clf.score(X_test, Y_test)
    accuracy_values.append(accuracy)

plt.plot(momentum_values, accuracy_values)
plt.title('Accuracy vs Momentum')
plt.xlabel('Momentum')
plt.ylabel('Accuracy')
plt.show()


# 4.5
learning_rates = [0.0001, 0.001, 0.01]
accuracies = []
for lr in learning_rates:
    clf = MLPClassifier(hidden_layer_sizes=(
        30,), learning_rate='constant', learning_rate_init=lr, max_iter=500)
    clf.fit(X_train, Y_train)
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(Y_test, predictions)
    accuracies.append(accuracy)


plt.plot(learning_rates, accuracies, marker='o')
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Learning Rate')
plt.show()


# 5
a_list_enc = pd.read_csv('a_list_enc.csv')

disc = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
a_list_enc_disc = disc.fit_transform(a_list_enc.iloc[:, 2:])

a_list_enc.iloc[:, 2:] = a_list_enc_disc
a_list_enc.to_csv('a_list_enc_disc.csv', index=False)


# 6.1
a_list_enc_disc = pd.read_csv('a_list_enc_disc.csv')
X_data = a_list_enc_disc.iloc[:, :-1].values
Y_data = a_list_enc_disc.iloc[:, 1].values
X_train, X_test, Y_train, Y_test = train_test_split(
    X_data, Y_data, test_size=0.2, random_state=123)

criteria = ['gini', 'entropy']


for criterion in criteria:
    clf = DecisionTreeClassifier(criterion=criterion, max_depth=5)
    Y_train = Y_train.astype(int)
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    Y_test = Y_test.astype(int)
    accuracy = accuracy_score(Y_test, Y_pred)
    print(f"Criterion: {criterion}, Accuracy: {accuracy}")

# 6.2


# 이전에 학습한 DecisionTreeClassifier 객체 clf를 사용합니다.
dot_data = export_graphviz(clf, out_file=None, filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())


# 6.3
depths = [2, 5, 10, 20, 50]

for depth in depths:
    clf = DecisionTreeClassifier(criterion='gini', max_depth=depth)
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    print(f"Max depth: {depth}, Accuracy: {accuracy}")

# 6.4
accuracies = []

for depth in depths:
    clf = DecisionTreeClassifier(criterion='gini', max_depth=depth)
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    accuracies.append(accuracy)

plt.plot(depths, accuracies)
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Max Depth')
plt.show()

# 6.5
# 'max_depth'가 너무 낮으면 모델이 데이터를 충분히 학습하지 못하여 과소적합이 발생할 수 있습니다. 
# 반면, 'max_depth'가 너무 높으면 모델이 학습 데이터에 과도하게 적응하여 과적합이 발생할 수 있습니다.
# 최적의 'max_depth'는 과소적합과 과적합 사이에서 균형을 이루는 값을 선택하여 
# 모델이 새로운 데이터에 대해 더 나은 일반화 성능을 발휘할 수 있도록 합니다. 
# 그래프에서 최적의 'max_depth'를 찾는 것은 정확도가 가장 높은 지점을 찾는 것과 관련이 있습니다.
# 이 지점에서 모델은 과소적합과 과적합의 영향을 최소화하면서 
# 학습 데이터와 테스트 데이터 모두에서 좋은 성능을 발휘할 수 있습니다. 
# 따라서, 이 값을 선택하면 모델이 새로운 데이터에 대한 예측에서 더 나은 성능을 보일 것입니다.

#7.1

neighbors = [1, 5, 9, 11, 13]
accuracies = []
for n in neighbors:
    clf = KNeighborsClassifier(n_neighbors=n)
    clf.fit(X_train, Y_train)
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(Y_test, predictions)
    accuracies.append(accuracy)
plt.plot(neighbors, accuracies)
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.title('Accuracy vs n_neighbors')
plt.show()
optimal_n = neighbors[np.argmax(accuracies)]
print(f"최적의 n_neighbors: {optimal_n}")

#7.2
weights_options = ['uniform', 'distance']
weights_accuracies = []

for weight in weights_options:
    clf = KNeighborsClassifier(n_neighbors=optimal_n, weights=weight)
    clf.fit(X_train, Y_train)
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(Y_test, predictions)
    weights_accuracies.append(accuracy)

plt.bar(weights_options, weights_accuracies)
plt.xlabel('Weights')
plt.ylabel('Accuracy')
plt.title('Accuracy for different weights')
plt.show()

#7.3
p_values = [1, 2]  # 1: 맨하탄 거리, 2: 유클리디언 거리
p_accuracies = []

for p in p_values:
    clf = KNeighborsClassifier(n_neighbors=optimal_n, weights='distance', p=p)
    clf.fit(X_train, Y_train)
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(Y_test, predictions)
    p_accuracies.append(accuracy)

# 결과 비교를 위해 막대 그래프 그리기
plt.bar(['Manhattan', 'Euclidean'], p_accuracies)
plt.xlabel('Distance Metric')
plt.ylabel('Accuracy')
plt.title('Accuracy for different distance metrics')
plt.show()

# p 값을 1 (맨하탄 거리)과 2 (유클리디언 거리)로 변경하면서 알고리즘을 실행하고, 
# 결과를 막대 그래프로 비교합니다.
# 이 스크립트를 사용하여 세 가지 실험을 수행할 수 있습니다. 
# 첫 번째 실험에서는 n_neighbors 값을 변경하여 최적의 값을 찾습니다. 
# 두 번째 실험에서는 가중치 옵션을 변경하여 uniform과 distance 간의 성능을 비교합니다. 
# 마지막으로 세 번째 실험에서는 거리 측정 방법을 변경하여 맨하탄 거리와 유클리디언 거리 간의 성능을 비교합니다.
# 실험을 통해 얻은 결과를 분석하여 최적의 하이퍼파라미터 값을 결정할 수 있습니다. 
# 이를 통해 K-최근접 이웃(KNN) 알고리즘의 성능을 최적화할 수 있습니다.
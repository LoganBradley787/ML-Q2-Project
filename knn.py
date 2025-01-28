import pandas as pd

train = pd.read_csv('train_smote.csv')
test = pd.read_csv('test.csv')

# get attributes, excluding class
attributes = list(train.columns)
attributes.remove('class')

classvals = list(train['class'].unique())

K = 5
R = 30 # for local density

def euclidean(instance1, instance2):
    squared_distance = 0
    for attribute in attributes:
        squared_distance += (instance1[attribute] - instance2[attribute]) ** 2
    return squared_distance ** 0.5

# Precalculate distances from each test point to each train point, sorted
print("Precalculating test-train distances")
test_train_distances = []
for index, instance in test.iterrows():
    distances = []
    for i, train_instance in train.iterrows():
        dist = euclidean(instance, train_instance)
        distances.append((dist, train_instance['class'], train_instance))
    distances.sort()
    test_train_distances.append(distances)
# Precalculate train-train distances
print("Precalculating train-train distances")
train_train_distances = {}
for index, instance in train.iterrows():
    distances = []
    for i, train_instance in train.iterrows():
        dist = euclidean(instance, train_instance)
        distances.append((dist, train_instance['class']))
    distances.sort()
    train_train_distances[str(instance)] = distances
print("Precalculations complete.")
# REGULAR KNN
totals = 0
corrects = 0
confusion_matrix = []
for i in range(len(classvals)):
    confusion_matrix.append([0] * len(classvals))
for index, instance in test.iterrows():
    distances = test_train_distances[index]
    class_counts = [0] * len(classvals)
    for a in range(K):
        class_counts[classvals.index(distances[a][1])] += 1
    classified_class = classvals[class_counts.index(max(class_counts))]
    totals += 1
    if classified_class == instance['class']:
        corrects += 1
    confusion_matrix[classvals.index(instance['class'])][classvals.index(classified_class)] += 1
print("Regular KNN Accuracy: " + str(round(100*(corrects / totals),2)) + "%")
print("Confusion Matrix:")
# format:
#               Predicted <class>, Predicted <class>, Predicted <class>
# Actual <class>
# Actual <class>
# Actual <class>
print("Predicted -" + str(classvals))
for i in range(len(classvals)):
    print("Actual " + str(classvals[i]) + "-" + str(confusion_matrix[i]))


# WEIGHTED AVERAGING KNN - weight is 1/frequency of class of neighbor
totals = 0
corrects = 0
confusion_matrix = []
for i in range(len(classvals)):
    confusion_matrix.append([0] * len(classvals))
class_frequencies = [0] * len(classvals)
for i, train_instance in train.iterrows():
    class_frequencies[classvals.index(train_instance['class'])] += 1
for index, instance in test.iterrows():
    distances = test_train_distances[index]
    class_counts = [0] * len(classvals)
    for a in range(K):
        class_counts[classvals.index(distances[a][1])] += 1 / class_frequencies[classvals.index(distances[a][1])]
    classified_class = classvals[class_counts.index(max(class_counts))]
    totals += 1
    if classified_class == instance['class']:
        corrects += 1
    confusion_matrix[classvals.index(instance['class'])][classvals.index(classified_class)] += 1
print("Weighted Averaging KNN Accuracy: " + str(round(100*(corrects / totals),2)) + "%")
print("Confusion Matrix:")
print("Predicted -" + str(classvals))
for i in range(len(classvals)):
    print("Actual " + str(classvals[i]) + "-" + str(confusion_matrix[i]))
# WEIGHTED DISTANCING KNN - weight is 1/distance to neighbor
totals = 0
corrects = 0
confusion_matrix = []
for i in range(len(classvals)):
    confusion_matrix.append([0] * len(classvals))
for index, instance in test.iterrows():
    distances = test_train_distances[index]
    class_counts = [0] * len(classvals)

    for a in range(K):
        class_counts[classvals.index(distances[a][1])] += 1 / distances[a][0]
    classified_class = classvals[class_counts.index(max(class_counts))]
    totals += 1
    if classified_class == instance['class']:
        corrects += 1
    confusion_matrix[classvals.index(instance['class'])][classvals.index(classified_class)] += 1
print("Weighted Distancing KNN Accuracy: " + str(round(100*(corrects / totals),2)) + "%")
print("Confusion Matrix:")
print("Predicted -" + str(classvals))
for i in range(len(classvals)):
    print("Actual " + str(classvals[i]) + "-" + str(confusion_matrix[i]))
# LOCAL DENSITY WEIGHTING KNN - local density is number of points within distance of r - neighbors in denser regions are weighted more
# final weight for neighbor is normalized based on total density of all k-nearest neighbors
totals = 0
corrects = 0
confusion_matrix = []
for i in range(len(classvals)):
    confusion_matrix.append([0] * len(classvals))
for index, instance in test.iterrows():
    distances = test_train_distances[index]
    class_counts = [0] * len(classvals)
    for a in range(K):
        neighbor = distances[a][2]
        neighbor_class = distances[a][1]
        local_density = 0
        neighbors_dists = train_train_distances[str(neighbor)]
        for b in range(len(neighbors_dists)):
            if neighbors_dists[b][0] <= R:
                local_density += 1
            else:
                break
        class_counts[classvals.index(neighbor_class)] += local_density
    classified_class = classvals[class_counts.index(max(class_counts))]
    totals += 1
    if classified_class == instance['class']:
        corrects += 1
    confusion_matrix[classvals.index(instance['class'])][classvals.index(classified_class)] += 1
print("Local Density Weighting KNN Accuracy: " + str(round(100*(corrects / totals),2)) + "%")
print("Confusion Matrix:")
print("Predicted -" + str(classvals))
for i in range(len(classvals)):
    print("Actual " + str(classvals[i]) + "-" + str(confusion_matrix[i]))

# ALL METHODS COMBINED KNN 
totals = 0
corrects = 0
confusion_matrix = []
for i in range(len(classvals)):
    confusion_matrix.append([0] * len(classvals))
for index, instance in test.iterrows():
    distances = test_train_distances[index]
    class_counts = [0] * len(classvals)
    all_weights = []
    for a in range(K):
        neighbor = distances[a][2]
        neighbor_class = distances[a][1]
        weights = []
        # weighted average (1/frequency)
        weights.append(1 / class_frequencies[classvals.index(neighbor_class)])
        # weighted distance (1/distance)
        weights.append(1 / distances[a][0])
        # local density
        local_density = 0
        neighbors_dists = train_train_distances[str(neighbor)]
        for b in range(len(neighbors_dists)):
            if neighbors_dists[b][0] <= R:
                local_density += 1
            else:
                break
        weights.append(local_density)
        all_weights.append(weights)
    # normalize weights before classifying
    for i in range(3):
        max_weight = max(weights[i] for weights in all_weights)
        min_weight = min(weights[i] for weights in all_weights)
        for a in range(K):
            all_weights[a][i] = (all_weights[a][i] - min_weight) / (max_weight - min_weight + 0.000000001) # small number to prevent division by 0
    # use normalized weights
    for a in range(K):
        neighbor_class = distances[a][1]
        total_weight = sum(all_weights[a])
        class_counts[classvals.index(neighbor_class)] += total_weight
    classified_class = classvals[class_counts.index(max(class_counts))]
    totals += 1
    if classified_class == instance['class']:
        corrects += 1
    confusion_matrix[classvals.index(instance['class'])][classvals.index(classified_class)] += 1
print("All Methods Combined KNN Accuracy: " + str(round(100*(corrects / totals),2)) + "%")
print("Confusion Matrix:")
print("Predicted -" + str(classvals))
for i in range(len(classvals)):
    print("Actual " + str(classvals[i]) + "-" + str(confusion_matrix[i]))
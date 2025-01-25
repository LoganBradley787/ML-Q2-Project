import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

attributes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

K = 5

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
# # REGULAR KNN
totals = 0
corrects = 0
for index, instance in test.iterrows():
    distances = test_train_distances[index]
    class_counts = [0, 0, 0]
    classvals = [0, 1, 2]
    for a in range(K):
        class_counts[classvals.index(distances[a][1])] += 1
    classified_class = classvals[class_counts.index(max(class_counts))]
    totals += 1
    if classified_class == instance['class']:
        corrects += 1
print("Accuracy: " + str(round(100*(corrects / totals),2)) + "%")

# # WEIGHTED AVERAGING KNN - weight is 1/frequency of class of neighbor
totals = 0
corrects = 0
class_frequencies = [0, 0, 0]
classvals = [0, 1, 2]
for i, train_instance in train.iterrows():
    class_frequencies[classvals.index(train_instance['class'])] += 1
for index, instance in test.iterrows():
    distances = test_train_distances[index]
    class_counts = [0, 0, 0]
    for a in range(K):
        class_counts[classvals.index(distances[a][1])] += 1 / class_frequencies[classvals.index(distances[a][1])]
    classified_class = classvals[class_counts.index(max(class_counts))]
    totals += 1
    if classified_class == instance['class']:
        corrects += 1
print("Accuracy: " + str(round(100*(corrects / totals),2)) + "%")

# # WEIGHTED DISTANCING KNN - weight is 1/distance to neighbor
totals = 0
corrects = 0
for index, instance in test.iterrows():
    distances = test_train_distances[index]
    class_counts = [0, 0, 0]
    classvals = [0, 1, 2]

    for a in range(K):
        class_counts[classvals.index(distances[a][1])] += 1 / distances[a][0]
    classified_class = classvals[class_counts.index(max(class_counts))]
    totals += 1
    if classified_class == instance['class']:
        corrects += 1
print("Accuracy: " + str(round(100*(corrects / totals),2)) + "%")

# # LOCAL DENSITY WEIGHTING KNN - local density is number of points within distance of r - neighbors in denser regions are weighted more
# # final weight for neighbor is normalized based on total density of all k-nearest neighbors
r = 30
totals = 0
corrects = 0
for index, instance in test.iterrows():
    distances = test_train_distances[index]
    class_counts = [0, 0, 0]
    classvals = [0, 1, 2]
    for a in range(K):
        neighbor = distances[a][2]
        neighbor_class = distances[a][1]
        local_density = 0
        neighbors_dists = train_train_distances[str(neighbor)]
        for b in range(len(neighbors_dists)):
            if neighbors_dists[b][0] <= r:
                local_density += 1
            else:
                break
        class_counts[classvals.index(neighbor_class)] += local_density
    classified_class = classvals[class_counts.index(max(class_counts))]
    totals += 1
    if classified_class == instance['class']:
        corrects += 1
print("Accuracy: " + str(round(100*(corrects / totals),2)) + "%")

# ALL METHODS COMBINED KNN
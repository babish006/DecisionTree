# Example of making predictions
from math import sqrt
import pandas as pd
import sys


# Calculating the Euclidean distance
def distance(row1, row2):
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i]) ** 2
    return sqrt(distance)


# Locate the most similar neighbors
def find_neighbors(trainset, testset, no_of_neighbors):
    distances = list()

    # Calculating the Euclidean distance
    for row in trainset:
        dist = distance(testset, row)
        distances.append((row, dist))

    # Sorting the distance
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()

    # Selecting the top distances based on the number of neighbours
    for i in range(no_of_neighbors):
        neighbors.append(distances[i][0])
    return neighbors


# Make a classification prediction with neighbors
def predict(trainset, testset, no_of_neighbors):
    neighbors = find_neighbors(trainset, testset, no_of_neighbors)
    output_class = [row[-1] for row in neighbors]
    predicted_class = max(set(output_class), key=output_class.count)
    return predicted_class


filename1 = sys.argv[1]
filename2 = sys.argv[2]
k = int(sys.argv[3])

trainset = pd.read_csv(filename1, delimiter=" ")
testset = pd.read_csv(filename2, delimiter=" ")

trainset = trainset.values.tolist()
testset = testset.values.tolist()

counter = 0
# k=1
predicted_list = list()
actual_list = list()

for i in range(len(trainset)):

    prediction = predict(trainset, testset[i], k)

    # print('Instance %d Expected %d Predicted %d' % (i, testset[i][-1], prediction))
    # print(i,'\t\t',testset[i][-1],'\t\t',prediction)
    predicted_list.append(int(prediction))
    actual_list.append(int(testset[i][-1]))
    if (testset[i][-1] == prediction):
        counter = counter + 1

print('\nValue of K =', k)
print('\nActual Class:\t', actual_list)
print('\nPredicted Class:', predicted_list)
print("\nNo of Correct Prediction:", counter)
print("\nAccuracy: " + str(counter / (i + 1) * 100))

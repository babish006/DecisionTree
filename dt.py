import pandas as pd
import sys

# Calculating the Gini index for the data set
def gini_index(columns, classes):
    # counting number of records
    records = sum([len(column) for column in columns])
    records = float(records)

    # Declaring Gini value to zero
    gini_value = 0.0

    for column in columns:
        size = float(len(column))
        if size == 0:
            continue

        # Declaring score value to zero
        score = 0.0

        # Calculating the gini score of each class
        for class_value in classes:
            p = [row[-1] for row in column].count(class_value) / size
            score += p * p
            #print(score)

        gini_value += (1.0 - score) * (size / records)
        #print(gini_value)

    return gini_value

# Spliting the test data set
def split_testset(index, index_values, dataset):
    left_wing = list()
    right_wing = list()

    for row in dataset:
        if row[index] < index_values:
            left_wing.append(row)
        else:
            right_wing.append(row)

    return left_wing, right_wing

# Splitting the test data set
def select_split(dataset):
    classes =  list(set(row[-1] for row in dataset))

    # Setting variable values
    preset_index = 999
    preset_value = 999
    preset_score = 999
    preset_groups = None

    for index in range(len(dataset[0])-1):
        for row in dataset:
            groups = split_testset(index, row[index], dataset)
            gini_score = gini_index(groups, classes)

            if gini_score < preset_score:
                preset_index = index
                preset_value = row[index]
                preset_score = gini_score
                preset_groups = groups

    return {'index':preset_index, 'value':preset_value, 'groups':preset_groups}


def select_class(group):
    output = [row[-1] for row in group]
    return max(set(output), key=output.count)

# Create a Node or select the childs for a node
def split_data(node, total_depth, depth):
    min_size = 1
    left_wing, right_wing = node['groups']
    del(node['groups'])
    # Calculate and split the left and right wings
    if not left_wing or not right_wing:
        node['left'] = node['right'] = select_class(left_wing + right_wing)
        return
    # Evaluate the maximum depth and current depth
    if depth >= total_depth:
        node['left'], node['right'] = select_class(left_wing), select_class(right_wing)
        #print(i,left_wing)
        return
    # Drill down the left child
    if len(left_wing) <= min_size:
        node['left'] = select_class(left_wing)
    else:
        node['left'] = select_split(left_wing)
        split_data(node['left'], total_depth, depth+1)
    # Drill down right child
    if len(right_wing) <= min_size:
        node['right'] = select_class(right_wing)
    else:
        node['right'] = select_split(right_wing)
        split_data(node['right'], total_depth, depth+1)

# Buildina a decision tree
def tree_building(train_data, total_depth):
    tree_root = select_split(train_data)
    split_data(tree_root, total_depth, 1)
    return tree_root

# Printing a tree
def printing_tree(node, depth=0):

    data_index = ['AGE', 'FEMALE', 'STEROID', 'ANTIVIRALS', 'FATIGUE', 'MALAISE', 'ANOREXIA',
                  'BIGLIVER', 'FIRMLIVER', 'SPLEENPALPABLE', 'SPIDERS', 'ASCITES', 'VARICES',
                  'BILIRUBIN', 'SGOT', 'HISTOLOGY', 'Class']

    if isinstance(node, dict):
        index_value = int(node['index'])

        #printing the Tree Structure
        print(depth * '\t',data_index[index_value], '=', node['value'])
        printing_tree(node['left'], depth + 1)
        printing_tree(node['right'], depth + 1)
    else:
        print('%s[%s]' % ((depth * '\t', node)))

# Logic for making a prediction
def prediction_algorithm(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return prediction_algorithm(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return prediction_algorithm(node['right'], row)
        else:
            return node['right']

# Tree Algorithm
def tree_algorithm(train, test, total_depth):
    tree = tree_building(train, total_depth)

    #Printing the trained tree
    printing_tree(tree)

    #Making predictions for the test data set
    predictions = list()
    for row in test:
        prediction = prediction_algorithm(tree, row)
        predictions.append(prediction)
    return(predictions)

# Function call to the Prediction & Tree Algorithm
def get_tree(trainset, testset, total_depth):
    actual = list()

    #invoking the prediction algorithm
    prediction = tree_algorithm(trainset, testset, total_depth)

    #Reading the actual class values of the test data set
    for row in range(len(testset)):
        actual.append(testset[row][16])

    #Printing the predicted and actual values
    print('\nPredicted Classes:\t',prediction)
    print('Actual Classes:\t\t', actual)

    #Calculating the accuracy of prediction
    correct_values = 0
    for i in range(len(actual)):
        if actual[i] == prediction[i]:
            correct_values += 1
    accuracy = correct_values / float(len(actual)) * 100.0
    print('\nNo of Corerct Predictions:\t', correct_values)
    return accuracy

# Calculating baseline accuracy
def baseline(dataset):

    true = 0
    false = 0

    # Counts the occurance of the classes in the data set
    for row in range(len(dataset)):
        if(testset[row][16]==1):
            true+=1
        else:
            false+=1

    # Returns the most frequent class in the test data set
    if(true>false):
        return true
    else:
        return false

file1 = sys.argv[1]
file2 = sys.argv[2]
depth = int(sys.argv[3])

#reading data from the file
trainset = pd.read_csv(file1, delimiter=" ")
testset = pd.read_csv(file2, delimiter=" ")

#Converting the boolean values to integers
trainset = trainset*1
testset = testset*1

data_index = ['AGE', 'FEMALE', 'STEROID', 'ANTIVIRALS', 'FATIGUE', 'MALAISE', 'ANOREXIA',
                     'BIGLIVER', 'FIRMLIVER', 'SPLEENPALPABLE', 'SPIDERS', 'ASCITES', 'VARICES',
                     'BILIRUBIN', 'SGOT', 'HISTOLOGY', 'Class']

#Moving class column to last column
trainset = trainset[data_index]
testset = testset [data_index]

#Converting the data set to list format
trainset = trainset.values.tolist()
testset = testset.values.tolist()

#Converting 'live' and 'die' values to boolean
for i in range(len(trainset)):
    if (trainset[i][16] == 'live'):
        trainset[i][16] = 1
    else:
        trainset[i][16] = 0

for i in range(len(testset)):
    if (testset[i][16] == 'live'):
        testset[i][16] = 1
    else:
        testset[i][16] = 0


tree = get_tree(trainset, testset, depth)
baseline_accuracy = baseline(testset)
print('Baseline Accuracy:\t\t', float(baseline_accuracy/len(testset)*100))
print('Decision Tree Accuracy:\t', tree)


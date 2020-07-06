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

data1 = [[1,2,3],[1,2,3]]
data2 = [1,2,3]

print(gini_index(data1,data2))


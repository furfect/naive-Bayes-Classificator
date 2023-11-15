import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd


def main():
    df = pd.read_csv('cars_evaluation.csv', header=None,
                     names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class_name'])
    arr = np.array(df)
    data, data_class = arr[:, 0:-1], arr[:, -1]
    data_train, data_test, class_train, class_test = train_test_split(data, data_class, test_size=0.3, shuffle=True)
    trained_data = attribute_counter(data_train, class_train)
    print(accuracy_check(data_test, class_test, trained_data))


def attribute_counter(data, data_class):
    trained_data = []
    class_count = class_counter(data_class)
    for column_idx in range(len(data[0])):
        column = data[:, column_idx]
        column_dict = dict.fromkeys(column, None)
        for label in column_dict:
            column_dict[label] = empty_class_dict(data_class)

        for idx, element in enumerate(column):
            label = data_class[idx]
            column_dict[element][label] += 1

        for value in column_dict:
            for label in class_count:
                all_classes_for_label = class_count[label]
                column_dict[value][label] = (column_dict[value][label] + 1) / (
                            all_classes_for_label + len(set(data_class)))
        trained_data.append(column_dict)
    return trained_data
    # koniec uzycia zestawu treningowego


def empty_class_dict(data_class, initial_value=0):
    data_class_set = set(data_class)
    data_class_counter = dict.fromkeys(data_class_set, initial_value)
    return data_class_counter.copy()


def class_counter(data_class):
    data_class_counter = empty_class_dict(data_class)
    for single_class in data_class:
        data_class_counter[single_class] += 1

    return data_class_counter


def accuracy_check(data_test, data_class_test, trained_data):
    predicted_correctly = 0
    prediction_count = 0
    for idx, row in enumerate(data_test):
        class_dict = empty_class_dict(data_class_test, 1)
        for classname in class_dict:
            for jdx, element in enumerate(row):
                class_dict[classname] *= trained_data[jdx][element][classname]

        predicted_class = max(class_dict, key=class_dict.get)
        if predicted_class == data_class_test[idx]:
            predicted_correctly += 1
        prediction_count += 1

    return predicted_correctly / prediction_count


if __name__ == '__main__':
    main()

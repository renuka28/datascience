from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_digits
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_data_labels():
    digits_data = load_digits()
    print(digits_data.keys())
    data = pd.DataFrame(digits_data['data'])
    labels = pd.Series(digits_data['target'])
    return data, labels


def plot_image(image, ax=None):
    np_image = image.values
    np_image = np_image.reshape(8, 8)
    if ax == None:
        plt.imshow(np_image, cmap='gray_r')
    else:
        ax.imshow(np_image, cmap='gray_r')


def plot_random_digits(data):
    f, axarr = plt.subplots(2, 4)

    for i in range(2):
        for j in range(4):
            # plot some random digits to get an idea about how they look
            plot_image(data.iloc[np.random.randint(0, len(data))], axarr[i, j])
    # plt.show()


def train(features, labels, nneighbors=1, neuron_arch=(1,), model_to_train="KNeighborsClassifier"):
    model = None
    if model_to_train == "MLPClassifier":
        model = MLPClassifier(hidden_layer_sizes=neuron_arch, max_iter=500)
    elif model_to_train == "KNeighborsClassifier":
        model = KNeighborsClassifier(n_neighbors=nneighbors)
    model.fit(features, labels)
    return model


def test(model, features, labels):
    predictions = model.predict(features)
    train_test_df = pd.DataFrame()
    train_test_df['correct_label'] = labels
    train_test_df['predicted_label'] = predictions
    overall_accuracy = sum(
        train_test_df["predicted_label"] == train_test_df["correct_label"])/len(train_test_df)
    return overall_accuracy


def cross_validate(nneighbors=1, splits=4, neuron_arch=(1,), model_to_train="KNeighborsClassifier"):
    fold_accuracies = []
    kf = KFold(n_splits=splits, random_state=2, shuffle=True)
    for train_index, test_index in kf.split(data):
        train_features, test_features = data.loc[train_index], data.loc[test_index]
        train_labels, test_labels = labels.loc[train_index], labels.loc[test_index]
        model = train(train_features,
                      train_labels,
                      nneighbors=nneighbors,
                      neuron_arch=neuron_arch,
                      model_to_train=model_to_train)
        overall_accuracy = test(model, test_features, test_labels)
        fold_accuracies.append(overall_accuracy)
    return fold_accuracies


def print_best_worst_accuracy(accuracy_dict):
    best_worst = []
    worst_accuracy_at = min(accuracy_dict, key=accuracy_dict.get)
    worst_accuracy = accuracy_dict[worst_accuracy_at]
    best_worst.append(worst_accuracy_at)
    best_worst.append(worst_accuracy)

    best_accuracy_at = max(accuracy_dict, key=accuracy_dict.get)
    best_accuracy = accuracy_dict[best_accuracy_at]
    best_worst.append(best_accuracy_at)
    best_worst.append(best_accuracy)
    print("worst accuracy is {:0.4f} acheived at {}".format(
        worst_accuracy, worst_accuracy_at))
    print("best accuracy is {:0.4f} acheived at {}".format(
        best_accuracy, best_accuracy_at))
    return best_worst


def validate_and_plot_with_multiple_values(k_values=[1], k_splits=[4],
                                           neuron_arches=(1,),
                                           model_to_train="KNeighborsClassifier"):
     # if there are more than one k values, then show plot

    overall_accuracies = []
    accuracy_dict = {}
    loop_over_type = None
    loop_over = None
    k = None
    n_arch = None
    key = None
    index = 0

    # if it knn we loop over various k values else if it is nn we loop over various neuron counts
    if model_to_train == "KNeighborsClassifier":
        loop_over_type = "k values"
        loop_over = k_values
    elif model_to_train == "MLPClassifier":
        loop_over = neuron_arches
        loop_over_type = "nueron values"

    # if there are more than one set of values, then setup a figure to plot
    n_num_ax = len(k_splits)
    fig, axarr = plt.subplots(n_num_ax, 1)
    axis = axarr
    fig.set_size_inches(12, n_num_ax*8)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    print()
    print("#" * 80)
    print("testing `{}` with {} = {} and splits = {}...".format(
        model_to_train, loop_over_type,  loop_over, k_splits))

    for split in k_splits:
        k_accuracies = []

        for loop_var in loop_over:
            # lets assign the loop_var to appropriate parameter
            if model_to_train == "KNeighborsClassifier":
                k = loop_var
                n_arch = None
                key = "split = {} and k = {}".format(split, loop_var)
            elif model_to_train == "MLPClassifier":
                n_arch = loop_var
                k = None
                key = "split = {} and neurons = {}".format(split, loop_var)
            accuracies = cross_validate(nneighbors=k,
                                        neuron_arch=n_arch,
                                        splits=split,
                                        model_to_train=model_to_train)
            k_mean_accuracy = np.mean(accuracies)
            accuracy_dict[key] = k_mean_accuracy
            k_accuracies.append(k_mean_accuracy)

        # if there are more than one k values, then plot graph an new axis
        if n_num_ax > 1:
            axis = axarr[index]

        axis.plot(loop_over, k_accuracies)
        axis.set_title(
            "Mean Accuracy vs. k with split = {}. Min = {:0.4f}, Max = {:0.4f} and Mean = {:0.4f}".format(
                split, min(k_accuracies), max(k_accuracies), np.mean(k_accuracies)))
        index += 1
        overall_accuracies.append(k_accuracies)

    best_worst = print_best_worst_accuracy(accuracy_dict)
    # plt.show()

    return overall_accuracies, best_worst, accuracy_dict


def build_summary(summary):
    rows = []
    for key, items in summary.items():
        row = [key] + items
        rows.append(row)

    summary_df = pd.DataFrame(rows, columns=[
                              'model', 'worst_accuracy_at', 'worst_accuracy', 'best_accuracy_at', 'best_accuracy', 'mean accuracy'])

    sorted_summary_df = summary_df.sort_values(by='mean accuracy', ascending=False)    
    return sorted_summary_df


def test_knn(summary):
    # test knn models
    k_values = list(range(1, 10))

    overall_accuracies, best_worst, accuracy_dict = validate_and_plot_with_multiple_values(k_values=k_values,
                                                                                           model_to_train="KNeighborsClassifier")
    best_worst.append(np.mean(overall_accuracies)*100)
    key = "KNeighborsClassifier with k = {}".format(k_values)
    summary[key] = best_worst
    print("mean accuracy = {:0.6f}%".format(best_worst[-1]))

    # test with more splits
    splits = list(range(2, 12, 2))
    overall_accuracies, best_worst, accuracy_dict = validate_and_plot_with_multiple_values(k_values=list(range(1, 10)),
                                                                                           k_splits=splits,
                                                                                           model_to_train="KNeighborsClassifier")
    best_worst.append(np.mean(overall_accuracies)*100)
    key = "KNeighborsClassifier with k = {} \nand splits = {}".format(
        k_values, splits)
    summary[key] = best_worst
    print("mean accuracy = {:0.6f}%".format(best_worst[-1]))
    return summary


def test_nn(summary):

    # one layer
    nn_one_neurons = [(2**i, ) for i in range(3, 6)]
    splits = list(range(2, 12, 2))
    overall_accuracies, best_worst, accuracy_dict = validate_and_plot_with_multiple_values(neuron_arches=nn_one_neurons,
                                                                                           k_splits=splits,
                                                                                           model_to_train="MLPClassifier")
    best_worst.append(np.mean(overall_accuracies)*100)
    key = "MLPClassifier with neurons = {} \nand splits = {}".format(
        nn_one_neurons, splits)
    summary[key] = best_worst
    print("mean accuracy = {:0.6f}%".format(best_worst[-1]))

    # two layers
    nn_one_neurons = [(2**i, 2**i) for i in range(3, 6)]
    overall_accuracies, best_worst, accuracy_dict = validate_and_plot_with_multiple_values(neuron_arches=nn_one_neurons,
                                                                                           k_splits=splits,
                                                                                           model_to_train="MLPClassifier")
    best_worst.append(np.mean(overall_accuracies)*100)
    key = "MLPClassifier with neurons = {} \nand splits = {}".format(
        nn_one_neurons, splits)
    summary[key] = best_worst
    print("mean accuracy = {:0.6f}%".format(best_worst[-1]))

    # three layers
    nn_one_neurons = [(2**i, 2**i, 2**i) for i in range(3, 6)]
    overall_accuracies, best_worst, accuracy_dict = validate_and_plot_with_multiple_values(neuron_arches=nn_one_neurons,
                                                                                           k_splits=splits,
                                                                                           model_to_train="MLPClassifier")
    best_worst.append(np.mean(overall_accuracies)*100)
    key = "MLPClassifier with neurons = {} \nand splits = {}".format(
        nn_one_neurons, splits)
    summary[key] = best_worst
    print("mean accuracy = {:0.6f}%".format(best_worst[-1]))

    # three layers with neurons upto 256
    nn_one_neurons = [(2**i, 2**i, 2**i) for i in range(3, 9)]
    overall_accuracies, best_worst, accuracy_dict = validate_and_plot_with_multiple_values(neuron_arches=nn_one_neurons,
                                                                                           k_splits=splits,
                                                                                           model_to_train="MLPClassifier")
    best_worst.append(np.mean(overall_accuracies)*100)
    key = "MLPClassifier with neurons = {} \nand splits = {}".format(
        nn_one_neurons, splits)
    summary[key] = best_worst
    print("mean accuracy = {:0.6f}%".format(best_worst[-1]))




    return summary


if __name__ == '__main__':
    data, labels = get_data_labels()
    plot_random_digits(data)
    summary = {}

    # # # # test knn models
    summary = test_knn(summary)
    # # # # test neuron models
    summary = test_nn(summary)

    summary_df = build_summary(summary)
    print(summary_df)

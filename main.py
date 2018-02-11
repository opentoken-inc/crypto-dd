from parse import parse_crv_data
from regression import regression_helper_2_var


def regress_on_crv_data():
    x_mat, y_mat, feature_labels, objective_labels = parse_crv_data()
    for i, label in enumerate(objective_labels):
        if label != 'Market Cap':  # This is the only one that seems meaningful
            continue               # Comment this out to try other objectives
        y_vec = [row[i] for row in y_mat]
        print 'Evaluating dataset on %s' % label
        regression_helper_2_var(x_mat, y_vec, x_labels=feature_labels,
                                y_label=objective_labels[i], plot=True)


if __name__ == '__main__':
    regress_on_crv_data()

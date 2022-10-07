import panel as pn
import sklearn.linear_model as linear_model
import sklearn.svm as svm
import sklearn.datasets as datasets
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn import metrics
from datetime import datetime
from sklearn.base import is_classifier, is_regressor
from scipy.stats import pearsonr
import math
import numpy as np

pn.config.sizing_mode = 'stretch_width'

ds_opts = {'diabetes': datasets.load_diabetes, 'iris': datasets.load_iris}
classifier_opts = {'LinearRegression': linear_model.LinearRegression, 'SVC': svm.SVC}
metrics_opts = {'neg_mean_absolute_error': metrics.mean_absolute_error, 'accuracy': metrics.accuracy_score}

ds_widget = pn.widgets.Select(name='Dataset', options=list(ds_opts.keys())).servable(target='ds-widget')
algo_widget = pn.widgets.Select(name='Algorithms', options=list(classifier_opts.keys())).servable(target='algo-widget')
metrics_widget = pn.widgets.Select(name='Metrics', options=list(metrics_opts.keys())).servable(target='metrics-widget')

cv_widget = pn.widgets.IntInput(name='Cross-Validation Folds', value=10).servable(target='cv-widget')
cv_widget.visible = False
ps_widget = pn.widgets.IntInput(name='Train size (%)', value=66).servable(target='ps-widget')
ps_widget.visible = False

result_pane = pn.pane.Str('').servable(target='result')


def on_test_option_changed(event=None):
    if event.new == 'Use training set':
        cv_widget.visible = False
        ps_widget.visible = False
    elif event.new == 'Cross-validation':
        cv_widget.visible = True
        ps_widget.visible = False
    elif event.new == 'Percentage split':
        cv_widget.visible = False
        ps_widget.visible = True


test_options_widget = pn.widgets.RadioBoxGroup(name='Test options',
                                               options=['Use training set',
                                                        'Cross-validation',
                                                        'Percentage split']).servable(target='test-options-widget')

test_options_widget.param.watch(on_test_option_changed, 'value')

results = {}


def on_result_list_changed(event=None):
    global result_pane
    result_pane.object = results[event.new]


result_list_widget = pn.widgets.Select(name='Result list', options=[], size=8).servable(target='result-list-widget')
result_list_widget.param.watch(on_result_list_changed, 'value')


def get_run_information():
    ds = ds_opts[ds_widget.value]()
    info = '=== Run information ===\n\n'
    info += f'Scheme:\t{algo_widget.value}\n'
    info += f'Relation:\t{ds_widget.value}\n'
    info += f'Instances:\t{len(ds.data)}\n'
    info += f'Attributes:\t{len(ds.feature_names)}\n'
    for a in ds.feature_names:
        info += '\t\t\t' + a + '\n'
    info += f'Test mode:\t{test_options_widget.value}\n'

    return info


def get_classifier_model(model):
    info = f'=== Classifier model ({test_options_widget.value}) ===\n'
    info += '\n'
    info += algo_widget.value + '\n'
    info += '=============\n'
    info += '\n'
    info += 'Should show details respect to the model!\n'
    return info


def evaluate(event=None):
    ds = ds_opts[ds_widget.value]()
    model = classifier_opts[algo_widget.value]()
    y_test = None

    if test_options_widget.value == 'Use training set':
        y_test = ds.target
        model.fit(ds.data, ds.target)
        pred = model.predict(ds.data)
    elif test_options_widget.value == 'Cross-validation':
        y_test = ds.target
        pred = cross_val_predict(model, ds.data, ds.target, cv=cv_widget.value)
    elif test_options_widget.value == 'Percentage split':
        X_train, X_test, y_train, y_test = train_test_split(ds.data, ds.target, train_size=ps_widget.value / 100)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

    output = f'{get_run_information()}\n\n'
    output += get_classifier_model(model) + '\n'
    output += get_summary(model, ds, y_test, pred)
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    key = f'{current_time} - {ds_widget.value} - {algo_widget.value}'
    results[key] = output

    result_list_widget.options = list(results.keys())
    result_list_widget.value = key


def get_detailed_accuracy_by_class(target, pred, target_names):
    info = '=== Detailed Accuracy By Class ===\n\n'
    info += classification_report(target, pred, target_names=target_names)
    info += '\n'
    return info


def get_confusion_matrix(target, pred, labels):
    matrix = metrics.confusion_matrix(target, pred)
    info = '=== Confusion Matrix ===\n\n'
    info += '\n'.join([''.join(['{:4}'.format(item) for item in row]) + ' | {}'.format(label) for row, label in zip(matrix, labels)])
    return info


def get_summary(model, dataset, y_test, pred):
    info = '=== Summary ===\n'
    info += '\n'
    target = y_test
    mean_abs_err = metrics.mean_absolute_error(target, pred)
    mean_prior_abs_error = metrics.mean_absolute_error(target, [1 / len(target)] * len(target))
    root_mean_squared_err = math.sqrt(metrics.mean_squared_error(target, pred))
    root_mean_prior_squared_err = math.sqrt(metrics.mean_squared_error(target, [1 / len(target)] * len(target)))
    if is_regressor(model):
        info += f'Correlation coefficient:\t\t{pearsonr(pred, target)[0]:.4f}\n'
        info += f'Mean absolute error:\t\t\t{mean_abs_err:.4f}\n'
        info += f'Root mean squared error:\t\t{root_mean_squared_err:.4f}\n'
        info += f'Relative absolute error:\t\t{100 * mean_abs_err / mean_prior_abs_error:.4f} %\n'
        info += f'Root relative squared error:\t{100 * root_mean_squared_err / root_mean_prior_squared_err:.4f} %\n'
        info += f'Total Number of Instances:\t\t{len(target)}\n'
    elif is_classifier(model):
        correct_sum = np.sum(target == pred)
        incorrect_sum = np.sum(target != pred)
        info += f'Correctly Classified Instances:\t\t{correct_sum}\t\t{100*correct_sum/len(target):.4f} %\n'
        info += f'Incorrectly Classified Instances:\t{incorrect_sum}\t\t{100*incorrect_sum/len(target):.4f} %\n'
        info += f'Kappa statistic:\t\t\t\t\t{metrics.cohen_kappa_score(target, pred):.4f}\n'
        info += f'Mean absolute error:\t\t\t\t{metrics.mean_absolute_error(target, pred):.4f}\n'
        info += f'Root mean squared error:\t\t\t{math.sqrt(metrics.mean_squared_error(target, pred)):.4f}\n'
        info += f'Relative absolute error:\t\t\t{100 * mean_abs_err / mean_prior_abs_error:.4f} %\n'
        info += f'Root relative squared error:\t\t{100 * root_mean_squared_err / root_mean_prior_squared_err:.4f} %\n'
        info += f'Total Number of Instances:\t\t\t{len(target)}\n\n'
        info += get_detailed_accuracy_by_class(target, pred, dataset.target_names)
        info += get_confusion_matrix(target, pred, dataset.target_names)
    else:
        raise Exception("Unknown type of model:" + str(model))

    return info


start_btn = pn.widgets.button.Button(name='start').servable(target='start-btn').on_click(evaluate)

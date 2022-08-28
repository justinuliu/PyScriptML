import panel as pn
import sklearn.linear_model as linear_model
import sklearn.svm as svm
import sklearn.datasets as datasets
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import metrics

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


def test_option_changed(event=None):
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

test_options_widget.param.watch(test_option_changed, 'value')


def get_settings():
    return f'Dataset: {ds_widget.value}\nAlgorithm: {algo_widget.value}\nEvaluator: {metrics_widget.value}\n'


settings_pane = pn.pane.Str(get_settings()).servable(target='settings')
result_pane = pn.pane.Str('').servable(target='result')


def evaluate(event=None):
    settings_pane.object = get_settings()
    ds = ds_opts[ds_widget.value]()
    model = classifier_opts[algo_widget.value]()

    if test_options_widget.value == 'Use training set':
        model.fit(ds.data, ds.target)
        pred = model.predict(ds.data)
        score = metrics_opts[metrics_widget.value](ds.target, pred)
        result_pane.object = score
    elif test_options_widget.value == 'Cross-validation':
        cv_result = cross_val_score(model, ds.data, ds.target, cv=cv_widget.value, scoring=metrics_widget.value)
        result_pane.object = cv_result
    elif test_options_widget.value == 'Percentage split':
        X_train, X_test, y_train, y_test = train_test_split(ds.data, ds.target, train_size=ps_widget.value/100)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        score = metrics_opts[metrics_widget.value](y_test, pred)
        result_pane.object = score


start_btn = pn.widgets.button.Button(name='start').servable(target='start-btn').on_click(evaluate)

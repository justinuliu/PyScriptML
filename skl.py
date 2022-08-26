import panel as pn
import sklearn.linear_model as linear_model
import sklearn.svm as svm
import sklearn.datasets as datasets
from sklearn.model_selection import cross_val_score


pn.config.sizing_mode = 'stretch_width'

ds_opts = {'diabetes': datasets.load_diabetes, 'iris': datasets.load_iris}
algorithms = {'LinearRegression': linear_model.LinearRegression, 'SVC': svm.SVC}
evaluators = ['neg_mean_absolute_error', 'accuracy']

ds_widgets = pn.widgets.Select(name='Dataset', options=list(ds_opts.keys())).servable(target='ds-widget')
algo_widgets = pn.widgets.Select(name='Algorithms', options=list(algorithms.keys())).servable(target='algo-widget')
metrics_widgets = pn.widgets.Select(name='Metrics', options=evaluators).servable(target='metrics-widget')
cv_widgets = pn.widgets.IntInput(name='Cross-Validation Folds', value=10).servable(target='cv-widget')


def get_settings():
    return f'Dataset: {ds_widgets.value}\nAlgorithm: {algo_widgets.value}\nEvaluator: {metrics_widgets.value}\n'


settings = pn.pane.Str(get_settings()).servable(target='settings')
result = pn.pane.Str('').servable(target='result')


def train(event=None):
    settings.object = get_settings()
    ds = ds_opts[ds_widgets.value]()
    model = algorithms[algo_widgets.value]()

    cv_result = cross_val_score(model, ds.data, ds.target, cv=cv_widgets.value, scoring=metrics_widgets.value)

    result.object = cv_result


start_btn = pn.widgets.button.Button(name='start').servable(target='start-btn').on_click(train)


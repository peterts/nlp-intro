import numpy as np
import plotly.graph_objs as go
from shapely.geometry import box, LineString
from shapely.ops import split
from nlp_intro.data import draw_uniform_data
from nlp_intro.logistic_regression import logistic_func, fit
from copy import deepcopy
from sklearn.metrics import confusion_matrix
import plotly.io as pio
from IPython.display import Image
from collections import Counter
import re
from random import sample, seed
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from ipywidgets import Output, VBox


COLOR_GRAY = (128, 128, 128)


class Linear2D2ClassModelPlotter:
    COLORS_2CLASSES = [(255, 65, 54), (0, 116, 217)]  # RGB for red and blue

    def __init__(self, x_range, y_range, border=.05, layout=None):
        if layout is None:
            layout = dict()
        self.box = _compute_corners(x_range, y_range, border)
        plot_data = self._create_model_area_plots()
        plot_data += [self._create_model_line_plot()]
        plot_data += self._create_data_plots()
        self.figure = go.Figure(data=plot_data, layout=layout)
        self.original_colors_ = None

    def __call__(self, **layout):
        return self.figure

    def _create_model_area_plots(self):
        plot_data = []
        for i in range(2):
            color = _rgb_to_plotly_color(self.COLORS_2CLASSES[i], .5)
            plot_data.append(
                go.Scatter(
                    x=[], y=[],
                    mode='none',
                    fill='toself',
                    hoverinfo='none',
                    showlegend=False,
                    fillcolor=color
                )
            )
        return plot_data

    @staticmethod
    def _create_model_line_plot():
        return go.Scatter(
            x=[], y=[],
            mode="lines",
            marker=dict(color="black"),
            showlegend=False
        )

    @staticmethod
    def _create_data_plots():
        plot_data = []
        for i in range(2):
            plot_data.append(
                go.Scatter(
                    x=[], y=[],
                    mode='markers',
                    marker=dict(
                        size=10,
                        line_width=1,
                    )
                )
            )
        return plot_data

    def update_model_area_plots(self, model):
        if model is not None:
            x, y = self._model_xy(model)

            # Split the plot area in (max) two polygons using the line
            b = box(*self.box)
            ls = LineString([tuple(p) for p in zip(x, y)])
            polygons = split(b, ls)

            # Update the scatter plots
            y_min = self.box[1]
            n_classes = len(self.original_colors_) if self.original_colors_ is not None else 0
            i = 1 if len(polygons) == 1 and n_classes == 2 and y[0] > y_min else 0
            for j, poly in zip(range(i, i + len(polygons)), polygons):
                scatter = self.figure.data[j]
                scatter.x, scatter.y = zip(*list(poly.exterior.coords))
        else:
            for i in range(2):
                scatter = self.figure.data[i]
                scatter.x, scatter.y = [], []

    def update_data_plots(self, data, classes=None, unq_classes=None):
        no_classes = classes is None
        if no_classes:
            classes = np.asarray(["Unknown"] * data.shape[0])

        if unq_classes is None:
            unq_classes = np.unique(classes)

        colors = list(self.COLORS_2CLASSES)
        if no_classes:
            colors = [COLOR_GRAY, COLOR_GRAY]
        self.original_colors_ = colors

        # Remove current data plots before adding the new data
        for i in range(3, 5):
            scatter = self.figure.data[i]
            scatter.x, scatter.y, scatter.name = [], [], None

        x, y = data.T
        for i, c in enumerate(unq_classes):
            ix = classes == c
            scatter = self.figure.data[3 + i]
            scatter.x, scatter.y = x[ix], y[ix]
            scatter.name = str(c)
            scatter.marker.color = _rgb_to_plotly_color(colors[i])

    def _model_xy(self, model):
        x_min, _, x_max, _ = self.box
        x = np.asarray([x_min, x_max])
        y = model(x)
        return x, y

    def disable_data_points(self):
        for i in range(3, len(self.figure.data)):
            self.figure.data[i].visible = False

    def enable_data_points(self):
        for i in range(3, len(self.figure.data)):
            self.figure.data[i].visible = True

    def disable_data_points_colors(self):
        for i in range(3, 5):
            self.figure.data[i].marker.color = _rgb_to_plotly_color(COLOR_GRAY)

    def enable_data_points_colors(self):
        if self.original_colors_ is None:
            return
        for i, c in enumerate(self.original_colors_):
            self.figure.data[3 + i].marker.color = _rgb_to_plotly_color(c)

    def disable_model_areas(self):
        for i in range(2):
            self.figure.data[i].visible = False

    def enable_model_areas(self):
        for i in range(2):
            self.figure.data[i].visible = True


def _compute_corners(x_range, y_range, border):
    (x_min, x_max), (y_min, y_max) = x_range, y_range
    x_border = (x_max - x_min) * border
    y_border = (y_max - y_min) * border
    return x_min - x_border, y_min - y_border, x_max + x_border, y_max + y_border


def _rgb_to_plotly_color(rgb, opacity=1.):
    return f"rgba({','.join(str(x) for x in (*rgb, opacity))})"


def plot_3d_scatter(x, y, z, x_name, y_name, z_name, width, height):
    return go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z, mode='markers',
        marker=dict(
            color=_rgb_to_plotly_color(COLOR_GRAY),
            line_width=1
        ))],
        layout=dict(
            width=width, height=height,
            scene=dict(
                xaxis=dict(title=x_name),
                yaxis=dict(title=y_name),
                zaxis=dict(title=z_name)
            ))
    )


def gradient_descent_animation(x_range, y_range, w_real, class_names, inv_class_names_for_plot=False,
                               w_initial=None, layout=None, lr=1, animation_min_grad_step=.1):
    w_real = np.asarray(w_real)

    X = draw_uniform_data(100, x_range, y_range)
    const = np.ones((X.shape[0], 1))
    X = np.hstack([X, const])
    y = predict_with_logistic_func(w_real, X)

    plotter = Linear2D2ClassModelPlotter(x_range, y_range, layout=layout)
    class_names = np.asarray(class_names)
    class_names_for_plot = class_names if not inv_class_names_for_plot else class_names[::-1]
    plotter.update_data_plots(X[:, :2], class_names[y[:, 0]], unq_classes=class_names_for_plot)

    initial_data = go.Figure(plotter.figure).data
    frames = []

    grad_sum = 0

    def update_plot_frames_callback(_, w, grad):
        nonlocal grad_sum
        grad_sum += np.mean(np.abs(grad))
        if grad_sum > animation_min_grad_step:
            add_model_iteration_frame(w)
            grad_sum = 0

    def add_model_iteration_frame(w, final=False):
        a, b = -w[0] / w[1], - w[2] / w[1]
        model = lambda x: a * x + b
        plotter.update_model_area_plots(model)
        frames.append(go.Frame(
            data=deepcopy(plotter.figure.data[:2]),
            layout=dict(title=f"{'Final' if final else ''} Model: {a:.2f}*x {'+' if b > 0 else '-'} {abs(b):.2f}")
        ))

    w_final = fit(X, y, add_intercept=False, lr=lr, w=w_initial, callback=update_plot_frames_callback, max_iter=1000)
    add_model_iteration_frame(w_final, True)

    layout = dict(layout)
    if layout is None:
        layout = dict()
    layout["updatemenus"] = [dict(type="buttons", buttons=[dict(label="Play", method="animate", args=[None])])]

    return go.Figure(data=initial_data, frames=frames, layout=layout)


def predict_with_logistic_func(w, X):
    return (logistic_func(w, X) > .5).astype(int)


def create_confusion_matrix_fig(labels, labels_pred, colorscale='Greens'):
    """
    Create a confusion matrix figure with Plotly

    Args:
        labels (numpy.array): Real labels
        labels_pred (numpy.array): Prediction labels
        colorscale: Plotly color scale to use

    Returns:
        dict: Plotly figure spec
    """
    cm = confusion_matrix(labels, labels_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    labels_unq = ["class " + str(l) if not isinstance(l, str) else l for l in np.unique(labels)]
    data = [
        go.Heatmap(x=labels_unq, y=labels_unq, z=cm[::-1], colorscale=colorscale, reversescale=True)
    ]
    layout = dict(
        width=1000, height=1000,
        title="Confusion Matrix"
    )
    return dict(data=data, layout=layout)


def plot_as_image(fig):
    return Image(pio.to_image(fig, format='png'))


def create_word_embeddings_fig_from_texts(texts, embedding_model, sample_size=None, min_word_count=10, random_seed=None,
                                          layout=None, cluster=True, cluster_eps=2, cluster_min_samples=10):

    word_counts = _count_all_words_with_embeddings(texts, embedding_model)
    words = _filter_sparse_words(word_counts, min_word_count)
    words = _sample(words, sample_size, random_seed)
    words_embs = [embedding_model[w] for w in words]

    X = TSNE(n_components=2, random_state=random_seed).fit_transform(words_embs)
    labels = [0] * len(X)

    if cluster:
        dbs = DBSCAN(eps=cluster_eps, min_samples=cluster_min_samples)
        dbs.fit(X)
        ix = dbs.labels_ >= 0
        X = X[ix]
        words = np.asarray(words)[ix]
        labels = dbs.labels_[ix]

    return _create_tsne_word_embs_fig(X, words, labels, layout)


def create_word_embeddings_fig_from_words(words, embedding_model, n_most_similar=50, random_seed=None, layout=None):
    word_embs = []
    words_ext = []
    labels = []
    for i, word in enumerate(words):
        if word not in embedding_model:
            print(f"No embedding found for {word}")
        most_similar_words, _ = zip(*embedding_model.most_similar(word, topn=n_most_similar))
        _words_ext = [word] + list(most_similar_words)
        _word_embs = [embedding_model[w] for w in _words_ext]
        word_embs.extend(_word_embs)
        words_ext.extend(_words_ext)
        labels.extend([i] * (n_most_similar+1))

    X = TSNE(n_components=2, random_state=random_seed).fit_transform(word_embs)
    return _create_tsne_word_embs_fig(X, words_ext, labels, layout)


def _create_tsne_word_embs_fig(X, words, labels, layout):
    x, y = X.T
    data = [
        go.Scatter(x=x, y=y, mode="markers+text", textposition="top center", text=words, marker=dict(color=labels))
    ]
    if layout is None:
        layout = dict(width=1000, height=800)
    return go.Figure(data=data, layout=layout)


def _count_all_words_with_embeddings(texts, embedding_model):
    counter = Counter()
    for text in texts:
        words_with_embeddings = [
            word for word in re.findall(r"[^\W_]+", text) if word in embedding_model]
        counter.update(words_with_embeddings)
    return counter


def _filter_sparse_words(word_counts, min_count):
    if min_count is None:
        min_count = 0
    return [w for w in word_counts if word_counts[w] >= min_count]


def _sample(_list, sample_size, random_seed):
    if sample_size is not None:
        if random_seed is not None:
            seed(random_seed)
        _list = sample(_list, sample_size)
    return _list


def create_text_scatter_plots(texts, labels,  _vec, n_dimensions=2, dim_reduction=True, layout=None):
    assert n_dimensions in (2, 3)
    texts = np.asarray(texts)

    pipeline = [_vec]
    if dim_reduction:
        pipeline.extend([
            TruncatedSVD(50),
            Normalizer(copy=False)
        ])
    pipeline.append(TSNE(n_components=n_dimensions, random_state=0))
    pipeline = make_pipeline(*pipeline)

    dim_names = ('x', 'y', 'z')
    X = pipeline.fit_transform(texts)
    plotly_scatter_class = go.Scatter if n_dimensions == 2 else go.Scatter3d

    out = Output()
    fig = go.FigureWidget()

    for label in labels.unique():
        ix = labels == label
        X_label = X[ix]
        X_dims_named = dict(zip(dim_names, X_label.T))
        fig.add_trace(plotly_scatter_class(
            **X_dims_named,
            mode='markers',
            marker=dict(
                size=10,
                symbol='circle',
                opacity=0.8
            ),
            name=label,
        ))

        scatter = fig.data[-1]
        scatter.on_click(_create_scatter_on_click(texts[ix], out))

    _layout = dict(margin=dict(l=0, r=0, b=0, t=0))
    if layout is not None:
        _layout.update(layout)
    fig.update_layout(**_layout)

    return VBox([fig, out])


def _create_scatter_on_click(texts, out):
    def _scatter_on_click(trace, points, state):
        ix = points.point_inds
        if ix:
            out.clear_output()
            with out:
                print(points.trace_name)
                print(texts[ix[0]])
    return _scatter_on_click

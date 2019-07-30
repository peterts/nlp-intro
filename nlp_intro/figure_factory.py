import numpy as np
import plotly.graph_objs as go
from shapely.geometry import box, LineString
from shapely.ops import split
from nlp_intro.data import draw_uniform_data
from nlp_intro.logistic_regression import logistic_func, fit
from copy import deepcopy

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

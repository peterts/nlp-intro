import numpy as np
import plotly.graph_objs as go
from shapely.geometry import box, LineString
from shapely.ops import split


class Linear2D2ClassModelPlotter:
    COLORS_2CLASSES = [(255, 65, 54), (0, 116, 217)]  # RGB for red and blue
    BORDER = .05

    def __init__(self, values, classes, model, flip_class_colors=False, **layout):
        x, y = values.T
        self.unq_classes = np.unique(classes)
        assert len(self.unq_classes) <= 2, "Must be at most 2 classes in the data set"

        self.box = self._compute_corners(x, y)
        self.data_points_colors = None

        plot_data = self._create_model_area_plots()
        plot_data += [self._create_model_line_plot()]
        plot_data += self._create_data_plots(x, y, classes, flip_class_colors)

        self.figure = go.FigureWidget(data=plot_data, layout=layout)

        self.update_model_area_plots(model)
        self.update_model_line_plots(model)

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

    def _create_data_plots(self, x, y, classes, flip_class_colors):
        colors = list(self.COLORS_2CLASSES)
        if flip_class_colors:
            colors = colors[::-1]
        self.data_points_colors = [_rgb_to_plotly_color(c) for c in colors]

        plot_data = []
        for i, c in enumerate(self.unq_classes):
            ix = classes == c
            plot_data.append(
                go.Scatter(
                    x=x[ix], y=y[ix],
                    mode='markers',
                    name=str(c),
                    marker=dict(
                        size=10,
                        line_width=1,
                        color=self.data_points_colors[i]
                    )
                )
            )
        return plot_data

    def update_model_line_plots(self, model):
        scatter = self.figure.data[2]
        scatter.x, scatter.y = self._model_xy(model)

    def update_model_area_plots(self, model):
        if model is not None:
            x, y = self._model_xy(model)

            # Split the plot area in (max) two polygons using the line
            b = box(*self.box)
            ls = LineString([tuple(p) for p in zip(x, y)])
            polygons = split(b, ls)

            # Update the scatter plots
            i = 1 if len(polygons) == 1 and len(self.unq_classes) == 2 and y[0] > y_min else 0
            for j, poly in zip(range(i, i+len(polygons)), polygons):
                scatter = self.figure.data[j]
                scatter.x, scatter.y = zip(*list(poly.exterior.coords))
        else:
            for i in range(2):
                scatter = self.figure.data[i]
                scatter.x, scatter.y = [], []

    def _model_xy(self, model):
        x_min, _, x_max, _ = self.box
        x = np.asarray([x_min, x_max])
        y = model(x)
        return x, y

    def _compute_corners(self, x, y):
        x_min, y_min, x_max, y_max = min(x), min(y), max(x), max(y)
        x_border = (x_max - x_min) * self.BORDER
        y_border = (y_max - y_min) * self.BORDER
        return x_min - x_border, y_min - y_border, x_max + x_border, y_max + y_border

    def disable_data_points(self):
        for i in range(3, len(self.figure.data)):
            self.figure.data[i].visible = False

    def enable_data_points(self):
        for i in range(3, len(self.figure.data)):
            self.figure.data[i].visible = True

    def disable_data_points_colors(self):
        for i in range(3, len(self.figure.data)):
            self.figure.data[i].marker.color = 'gray'

    def enable_data_points_colors(self):
        for i, color in zip(range(3, len(self.figure.data)), self.data_points_colors):
            self.figure.data[i].marker.color = color

    def disable_model_line(self):
        self.figure.data[2].visible = False

    def enable_model_line(self):
        self.figure.data[2].visible = True

    def disable_model_areas(self):
        for i in range(2):
            self.figure.data[i].visible = False

    def enable_model_areas(self):
        for i in range(2):
            self.figure.data[i].visible = True


def _rgb_to_plotly_color(rgb, opacity=1.):
    return f"rgba({','.join(str(x) for x in (*rgb, opacity))})"




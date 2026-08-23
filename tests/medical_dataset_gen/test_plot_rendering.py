from experiments.medical_dataset_gen.reports.plot_rendering import (
    set_axis_title,
    set_figure_title,
)


class RecordingAxis:
    def __init__(self) -> None:
        self.title: str | None = None

    def set_title(self, title: str) -> None:
        self.title = title


class RecordingFigure:
    def __init__(self) -> None:
        self.title: str | None = None

    def suptitle(self, title: str) -> None:
        self.title = title


def test_default_rendering_keeps_subplot_title_and_omits_global_title() -> None:
    axis = RecordingAxis()
    figure = RecordingFigure()

    set_axis_title(axis=axis, title='Subplot title')
    set_figure_title(figure=figure, title='Global title')

    assert axis.title == 'Subplot title'
    assert figure.title is None

"""Module with tests of widget.
    """
# pylint: disable=wrong-import-position

# Change the backend of matplotlib to notebook version (nbAgg)
# in order to enable testing widget functionality.
# This is usually done with the `%matplotlib widget` magic command.
# This adresses `The 'center' trait of an AppLayout instance expected a Widget or
# None, not the FigureCanvasNbAgg` error.
# Note that if other matplotlib figure have been drawn earlier
# this may no longer work. Thus it is recommended to
# run this test separately from the exp script tests.
# See https://github.com/rdnfn/bauwerk/issues/29.
import matplotlib

matplotlib.use("nbAgg")

import bauwerk.widget.core
import time


def test_basic_widget_functionality():
    widget = bauwerk.widget.core.Game(step_time=0.001)
    widget.buttons["go_to_game"].click()
    widget.buttons["start"].click()
    widget.control.value = 0.5
    time.sleep(0.1)
    widget.buttons["pause"].click()
    assert widget.reward != 0.0
    widget.buttons["reset"].click()
    assert widget.reward == 0.0

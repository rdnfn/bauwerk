"""Module with tests of widget.
    """

import bauwerk.widget.core
import ipympl  # pylint: disable=unused-import
import time


def test_basic_widget_functionality():
    widget = bauwerk.widget.core.Game(step_time=0.001)
    widget.begin_button.click()
    widget.start_button.click()
    widget.control.value = 0.5
    time.sleep(0.1)
    widget.pause_button.click()
    assert widget.reward != 0.0
    widget.reset_button.click()
    assert widget.reward == 0.0

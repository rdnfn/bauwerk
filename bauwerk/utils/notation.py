"""Module defining project notation."""

from __future__ import annotations

from dataclasses import dataclass, InitVar
from typing import List


@dataclass
class VarDef:
    """Definition of notation for a single variable."""

    var_name: str
    latex_math: str = " "
    unit: str = ""
    description: str = ""
    latex_cmd: InitVar[str] = None
    time_arg: bool = False  # whether the variable has time argument, e.g. P(t).
    cp_type: str = None  # convex problem var type: "variable" or "parameter"
    cp_area: str = "general"  # which problem part: e.g. "battery", "grid", etc.

    def __post_init__(self, latex_cmd):
        """Complete init."""
        if latex_cmd is not None:
            self.latex_cmd = latex_cmd
        else:
            self.latex_cmd = "\\" + self.var_name.replace("_", "")


@dataclass
class NotationCollection:
    """Notation collection.

    Class to collect and display notation.
    """

    notation_list: List[VarDef]

    def print_notation_style(self) -> None:
        """Print notation as latex style file commands."""
        for variable in self.notation_list:
            if variable.time_arg:
                print(
                    "\\newcommand{{{}}}[1][(t)]{{{}#1}}".format(
                        variable.latex_cmd, variable.latex_math
                    )
                )
            else:
                print(
                    "\\newcommand{{{}}}{{{}}}".format(
                        variable.latex_cmd, variable.latex_math
                    )
                )

    def _get_table_var_info(
        self, variable: VarDef, mrkdwn: bool = False
    ) -> tuple(str, str, str, str):
        """Create column entries for table row.

        Args:
            variable (VarDef): variable described in row
            mrkdwn (bool): whether the table row is in markdown. Defaults to False.

        Returns:
            [type]: [description]
        """

        if mrkdwn:
            var_name = variable.var_name
            if variable.time_arg:
                latex = variable.latex_math + "(t)"
            else:
                latex = variable.latex_math
        else:
            var_name = variable.var_name.replace("_", r"\_")
            latex = variable.latex_cmd

        return (
            latex,
            variable.description,
            variable.unit,
            var_name,
        )

    def get_latex_table_str(self, print_python_var_name=False) -> str:
        """Get notation table formatted in latex.

        Returns:
            str: latex string
        """
        num_cols = 3 + int(print_python_var_name)

        out = ""
        out += r"\begin{center}" + "\n"
        out += r"\begin{tabular}{ l | p{8cm}" + "| l" * (num_cols - 2) + r"}" + "\n"
        out += "Name & Description & Unit "
        if print_python_var_name:
            out += "& Python Name"
        out += " \\\\ \n"
        out += "\\hline"

        row_str = "${}$ & {} & {}"
        if print_python_var_name:
            row_str += " & \\texttt{{{}}}"
        row_str += " \\\\"

        for variable in self.notation_list:

            out += row_str.format(
                *self._get_table_var_info(variable),
            )
            out += "\n"

        out += r"\end{tabular}" + "\n"
        out += r"\end{center}"

        return out

    def get_mrkdwn_table_str(self) -> str:
        """Get notation table formatted in markdown.

        Returns:
            str: markdown string
        """

        out = ""
        out += "Variable | Description | Unit | Python Name \n"
        out += "---|---|---|--- \n"

        for variable in self.notation_list:
            out += "${}$ | {} | {} | `{}`".format(
                *self._get_table_var_info(variable, mrkdwn=True)
            )
            out += "\n"

        return out


_NOTATION_LIST = [
    # Battery
    VarDef(
        "energy_battery",
        r"E_\text{b}",
        "kWh",
        "energy content of the battery",
        time_arg=True,
        cp_type="variable",
        cp_area="battery",
    ),
    VarDef(
        "is_battery_charging",
        r"I",
        "",
        "Boolean indicator variable whether battery is charging",
        time_arg=True,
        cp_type="variable",
        cp_area="battery",
    ),
    VarDef(
        "power_out_grid_above_thresh",
        r"\bar{P}_{g\rightarrow}",
        "kW",
        "Amount of power output of grid above threshold",
        time_arg=True,
        cp_type="variable",
        cp_area="grid",
    ),
    VarDef(
        "size",
        r"B",
        "kWh",
        "energy capacity of battery",
        cp_type="parameter",
        cp_area="battery",
    ),
    VarDef(
        "kWh_per_cell",
        r"B_\text{cell}",
        "kWh",
        "energy capacity per individual cell",
        cp_type="parameter",
        cp_area="battery",
    ),
    VarDef(
        "num_cells",
        r"n_\text{cell}",
        "cells",
        "number of cells in battery",
        cp_type="parameter",
        cp_area="battery",
    ),
    VarDef(
        "initial_energy_content",
        r"E_b (0)",
        "kWh",
        "initial energy content of battery at time step 0",
        cp_type="parameter",
        cp_area="battery",
    ),
    VarDef(
        "nominal_voltage_c",
        r"V_{\text{nom},c}",
        "V",
        "nominal voltage of battery when charging",
        cp_type="parameter",
        cp_area="battery",
    ),
    VarDef(
        "nominal_voltage_d",
        r"V_{\text{nom},d}",
        "V",
        "nominal voltage of battery when discharging",
        cp_type="parameter",
        cp_area="battery",
    ),
    VarDef(
        "eff_discharge",
        r"\eta_d",
        "kWh",
        (
            "efficiency of discharging the battery, amount of energy"
            " content reduction for discharging 1 kWh"
        ),
        cp_type="parameter",
        cp_area="battery",
    ),
    VarDef(
        "eff_charge",
        r"\eta_c",
        "kWh",
        (
            "efficiency of charging the battery, amount of"
            " energy content increase for charging 1 kWh"
        ),
        cp_type="parameter",
        cp_area="battery",
    ),
    # Grid
    VarDef(
        "price_base",
        r"\pi_b",
        r"\$/kWh",
        "base price paid for energy drawn from the grid",
        cp_type="parameter",
        cp_area="grid",
    ),
    VarDef(
        "price_penalty",
        r"\pi_d",
        r"\$/kWh",
        (
            "additional price penalty paid for energy drawn"
            " from the grid when demand is above threshold"
        ),
        cp_type="parameter",
        cp_area="grid",
    ),
    VarDef(
        "grid_threshold",
        r"\Gamma",
        "kW",
        "demand threshold above which price penalty is paid",
        cp_type="parameter",
        cp_area="grid",
    ),
    # General parameters
    VarDef(
        "num_timesteps",
        r"T",
        "steps",
        "number of time steps in an episode",
        cp_type="parameter",
        cp_area="general",
    ),
    VarDef(
        "len_timestep",
        r"\Delta_t",
        "hours",
        "length of a time step",
        cp_type="parameter",
        cp_area="general",
    ),
]

_OLD_POWER_NOTATION = [
    # Power variables
    VarDef("power_charge", r"P_\text{c}", "kW", "power used to charge the battery"),
    VarDef("power_discharge", r"P_\text{d}", "kW", "power discharged from the battery"),
    VarDef("power_solar", r"P_{\text{solar}}", "kW", "power coming from solar panels"),
    VarDef("power_load", r"P_{\text{load}}", "kW", "power used by residential load"),
    VarDef("power_sell", r"P_\text{sell}", "kW", "power sold to the grid"),
    VarDef("power_grid", r"P_\text{grid}", "kW", "power drawn from the grid"),
    VarDef(
        "power_direct",
        r"P_\text{direct}",
        "kW",
        "sum of power from solar panels and grid that is used for load or sold",
    ),
    VarDef(
        "power_over_thresh", r"P_\text{over}", "kW", "power over peak demand threshold"
    ),
]

NOTATION = NotationCollection(_NOTATION_LIST)

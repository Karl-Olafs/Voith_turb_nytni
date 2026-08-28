from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

COEFFICIENTS = np.array(
    [
        1.42654,    # a0
        18.35090,   # a1
        2.65416,    # a2
        -12.54410,  # a3
        -17.89403,  # a4
        37.45027,   # a5
    ],
    dtype=float,
)

RHO_WATER = 999.7
GRAVITY = 9.80665

MINIMUM_FLOW_M3S = 10.0
MINIMUM_GATE_OPENING_PCT = 5.0


def parse_arguments() -> object:
    parser = ArgumentParser(
        description=(
            "Validate the frozen flow equation from a long-format "
            "SCADA CSV and plot efficiency using measured and "
            "estimated flow against gate opening."
        )
    )

    parser.add_argument(
        "csv_file",
        nargs="?",
        default="combined-data.csv",
        help="Long-format input CSV. Default: combined-data.csv",
    )

    parser.add_argument(
        "--output-dir",
        default="frozen_flow_efficiency_gate_output",
        help=(
            "Output folder. "
            "Default: frozen_flow_efficiency_gate_output"
        ),
    )

    parser.add_argument(
        "--flow-unit",
        choices=("m3s", "m3h"),
        default="m3s",
        help=(
            "Interpretation of the Water flow values. "
            "Default: m3s"
        ),
    )

    parser.add_argument(
        "--power-signal",
        default="Active power",
        help=(
            'Exact signalName for electrical power. '
            'Default: "Active power"'
        ),
    )

    parser.add_argument(
        "--power-unit",
        choices=("MW", "kW", "W"),
        default="MW",
        help="Unit of the active-power signal. Default: MW",
    )

    parser.add_argument(
        "--start",
        default=None,
        help=(
            "Optional UTC start time, for example "
            "2026-07-23T10:00:00Z"
        ),
    )

    return parser.parse_args()


def load_long_csv(
    csv_file: Path,
    power_signal: str,
) -> pd.DataFrame:
    raw = pd.read_csv(csv_file)

    required_columns = ["timestamp", "signalName", "value"]

    missing_columns = [
        column
        for column in required_columns
        if column not in raw.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required input columns: "
            + ", ".join(missing_columns)
        )

    raw["timestamp"] = pd.to_datetime(
        raw["timestamp"],
        utc=True,
        errors="coerce",
    )

    raw["value"] = pd.to_numeric(
        raw["value"],
        errors="coerce",
    )

    raw["signalName"] = (
        raw["signalName"]
        .astype(str)
        .str.strip()
    )

    signal_map = {
        "Gate opening": "Gate_opening_pct",
        "Runner blade position": "Runner_blade_position_pct",
        "2CFA.Head": "Head_m",
        "Water flow": "Water_flow_raw",
        power_signal: "Active_power_raw",
    }

    available_signals = set(
        raw["signalName"].dropna().unique()
    )

    missing_signals = [
        signal
        for signal in signal_map
        if signal not in available_signals
    ]

    if missing_signals:
        raise ValueError(
            "Missing required signalName values: "
            + ", ".join(missing_signals)
            + "\nEfficiency cannot be calculated without active power."
        )

    selected = raw[
        raw["signalName"].isin(signal_map)
    ].copy()

    wide = (
        selected
        .pivot_table(
            index="timestamp",
            columns="signalName",
            values="value",
            aggfunc="mean",
        )
        .rename(columns=signal_map)
        .reset_index()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    return wide


def convert_power_to_mw(
    values: pd.Series,
    unit: str,
) -> pd.Series:
    if unit == "MW":
        return values

    if unit == "kW":
        return values / 1000.0

    if unit == "W":
        return values / 1_000_000.0

    raise ValueError("Unsupported power unit.")


def estimate_flow(data: pd.DataFrame) -> np.ndarray:
    gate = data["Gate_opening_pct"].to_numpy() / 100.0

    runner = (
        data["Runner_blade_position_pct"].to_numpy()
        / 100.0
    )

    head = data["Head_m"].to_numpy()

    features = np.column_stack(
        [
            np.ones(len(data)),
            gate,
            runner,
            gate ** 2,
            runner ** 2,
            gate * runner,
        ]
    )

    return np.sqrt(head) * (features @ COEFFICIENTS)


def add_efficiency_columns(
    data: pd.DataFrame,
) -> pd.DataFrame:
    result = data.copy()

    result["Measured_unit_efficiency_pct"] = (
        100
        * result["Active_power_MW"]
        * 1_000_000
        / (
            RHO_WATER
            * GRAVITY
            * result["Measured_flow_m3s"]
            * result["Head_m"]
        )
    )

    result["Estimated_unit_efficiency_pct"] = (
        100
        * result["Active_power_MW"]
        * 1_000_000
        / (
            RHO_WATER
            * GRAVITY
            * result["Estimated_flow_m3s"]
            * result["Head_m"]
        )
    )

    result["Efficiency_error_percentage_points"] = (
        result["Estimated_unit_efficiency_pct"]
        - result["Measured_unit_efficiency_pct"]
    )

    return result


def save_flow_plots(
    validation: pd.DataFrame,
    output_folder: Path,
) -> None:
    measured = validation["Measured_flow_m3s"].to_numpy()
    estimated = validation["Estimated_flow_m3s"].to_numpy()

    plt.figure(figsize=(7, 6))
    plt.scatter(
        measured,
        estimated,
        s=24,
        alpha=0.6,
    )

    minimum = min(measured.min(), estimated.min())
    maximum = max(measured.max(), estimated.max())

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
        linewidth=1,
    )

    plt.xlabel("Measured flow [m³/s]")
    plt.ylabel("Estimated flow [m³/s]")
    plt.title("Measured vs estimated flow")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        output_folder / "measured_vs_estimated_flow.png",
        dpi=180,
    )
    plt.close()


def save_efficiency_gate_plots(
    validation: pd.DataFrame,
    output_folder: Path,
) -> None:
    efficiency_data = validation[
        validation["Measured_unit_efficiency_pct"].between(
            0,
            105,
        )
        & validation["Estimated_unit_efficiency_pct"].between(
            0,
            105,
        )
    ].copy()

    if efficiency_data.empty:
        raise ValueError(
            "No valid efficiency values between 0% and 105%. "
            "Check the power and flow units."
        )

    # Raw efficiency values against gate opening.
    plt.figure(figsize=(9, 6))

    plt.scatter(
        efficiency_data["Gate_opening_pct"],
        efficiency_data["Measured_unit_efficiency_pct"],
        s=20,
        alpha=0.45,
        label="Efficiency using measured flow",
    )

    plt.scatter(
        efficiency_data["Gate_opening_pct"],
        efficiency_data["Estimated_unit_efficiency_pct"],
        s=20,
        alpha=0.45,
        label="Efficiency using estimated flow",
    )

    plt.xlabel("Gate opening [%]")
    plt.ylabel("Provisional electrical unit efficiency [%]")
    plt.title(
        "Measured-flow and estimated-flow efficiency "
        "vs gate opening"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_folder
        / "efficiency_vs_gate_opening_scatter.png",
        dpi=180,
    )
    plt.close()

    # Average both efficiency values in 1%-gate bins.
    efficiency_data["Gate_opening_bin_pct"] = (
        efficiency_data["Gate_opening_pct"].round()
    )

    efficiency_by_gate = (
        efficiency_data
        .groupby(
            "Gate_opening_bin_pct",
            as_index=False,
        )
        .agg(
            Mean_measured_efficiency_pct=(
                "Measured_unit_efficiency_pct",
                "mean",
            ),
            Mean_estimated_efficiency_pct=(
                "Estimated_unit_efficiency_pct",
                "mean",
            ),
            Number_of_points=(
                "Measured_unit_efficiency_pct",
                "count",
            ),
        )
        .sort_values("Gate_opening_bin_pct")
    )

    efficiency_by_gate.to_csv(
        output_folder / "efficiency_by_gate_opening.csv",
        index=False,
    )

    plt.figure(figsize=(9, 6))

    plt.plot(
        efficiency_by_gate["Gate_opening_bin_pct"],
        efficiency_by_gate[
            "Mean_measured_efficiency_pct"
        ],
        marker="o",
        linewidth=1.5,
        label="Efficiency using measured flow",
    )

    plt.plot(
        efficiency_by_gate["Gate_opening_bin_pct"],
        efficiency_by_gate[
            "Mean_estimated_efficiency_pct"
        ],
        marker="o",
        linewidth=1.5,
        label="Efficiency using estimated flow",
    )

    plt.xlabel("Gate opening [%]")
    plt.ylabel("Mean provisional electrical unit efficiency [%]")
    plt.title(
        "Mean measured-flow and estimated-flow efficiency "
        "vs gate opening"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_folder / "efficiency_vs_gate_opening.png",
        dpi=180,
    )
    plt.close()



def save_efficiency_limits_and_error_plots(
    validation: pd.DataFrame,
    output_folder: Path,
) -> dict[str, float]:
    """
    Create:
      1. Estimated-efficiency curve with approximate 95% validation limits.
      2. Efficiency error versus gate opening.

    Error definition:
        error = estimated efficiency - measured efficiency

    The 95% validation limits use the overall residual distribution:
        bias +/- 1.96 * standard deviation

    Because measured = estimated - error, the limits around the
    estimated-efficiency curve are reversed when applied:
        lower = estimated - upper_error_limit
        upper = estimated - lower_error_limit
    """

    efficiency_data = validation[
        validation[
            "Measured_unit_efficiency_pct"
        ].between(0, 105)
        & validation[
            "Estimated_unit_efficiency_pct"
        ].between(0, 105)
    ].copy()

    if len(efficiency_data) < 5:
        raise ValueError(
            "Too few valid efficiency values for uncertainty limits."
        )

    error = efficiency_data[
        "Efficiency_error_percentage_points"
    ]

    bias_pp = error.mean()
    mae_pp = error.abs().mean()
    rmse_pp = np.sqrt(np.mean(error ** 2))
    std_pp = error.std(ddof=1)
    max_abs_pp = error.abs().max()

    lower_error_95_pp = bias_pp - 1.96 * std_pp
    upper_error_95_pp = bias_pp + 1.96 * std_pp

    efficiency_data[
        "Gate_opening_bin_pct"
    ] = efficiency_data[
        "Gate_opening_pct"
    ].round()

    by_gate = (
        efficiency_data
        .groupby(
            "Gate_opening_bin_pct",
            as_index=False,
        )
        .agg(
            Mean_measured_efficiency_pct=(
                "Measured_unit_efficiency_pct",
                "mean",
            ),
            Mean_estimated_efficiency_pct=(
                "Estimated_unit_efficiency_pct",
                "mean",
            ),
            Mean_efficiency_error_pp=(
                "Efficiency_error_percentage_points",
                "mean",
            ),
            Efficiency_error_std_pp=(
                "Efficiency_error_percentage_points",
                "std",
            ),
            Number_of_points=(
                "Efficiency_error_percentage_points",
                "count",
            ),
        )
        .sort_values("Gate_opening_bin_pct")
    )

    # These are approximate 95% validation limits for where measured
    # efficiency is expected to lie around the estimated-efficiency curve.
    by_gate[
        "Estimated_efficiency_lower_95_pct"
    ] = (
        by_gate["Mean_estimated_efficiency_pct"]
        - upper_error_95_pp
    )

    by_gate[
        "Estimated_efficiency_upper_95_pct"
    ] = (
        by_gate["Mean_estimated_efficiency_pct"]
        - lower_error_95_pp
    )

    limits_csv = (
        output_folder
        / "efficiency_limits_and_error_by_gate.csv"
    )

    by_gate.to_csv(
        limits_csv,
        index=False,
    )

    x = by_gate[
        "Gate_opening_bin_pct"
    ].to_numpy()

    measured_mean = by_gate[
        "Mean_measured_efficiency_pct"
    ].to_numpy()

    estimated_mean = by_gate[
        "Mean_estimated_efficiency_pct"
    ].to_numpy()

    lower_limit = by_gate[
        "Estimated_efficiency_lower_95_pct"
    ].to_numpy()

    upper_limit = by_gate[
        "Estimated_efficiency_upper_95_pct"
    ].to_numpy()

    # Plot 1: estimated efficiency with approximate 95% validation limits.
    plt.figure(figsize=(9, 6))

    plt.fill_between(
        x,
        lower_limit,
        upper_limit,
        alpha=0.2,
        label="Approximate 95% validation limits",
    )

    plt.plot(
        x,
        measured_mean,
        marker="o",
        linewidth=1.5,
        label="Efficiency using measured flow",
    )

    plt.plot(
        x,
        estimated_mean,
        marker="o",
        linewidth=1.5,
        label="Efficiency using estimated flow",
    )

    plt.xlabel("Gate opening [%]")
    plt.ylabel(
        "Mean provisional electrical unit efficiency [%]"
    )
    plt.title(
        "Estimated-flow efficiency with 95% validation limits"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_folder
        / "estimated_efficiency_with_95pct_limits.png",
        dpi=180,
    )
    plt.close()

    # Plot 2: individual efficiency error and mean error by gate.
    plt.figure(figsize=(9, 6))

    plt.scatter(
        efficiency_data["Gate_opening_pct"],
        efficiency_data[
            "Efficiency_error_percentage_points"
        ],
        s=18,
        alpha=0.35,
        label="Individual efficiency error",
    )

    plt.plot(
        x,
        by_gate["Mean_efficiency_error_pp"],
        marker="o",
        linewidth=1.5,
        label="Mean error by 1% gate bin",
    )

    plt.axhline(
        0,
        linestyle="--",
        linewidth=1,
        label="Zero error",
    )

    plt.axhline(
        lower_error_95_pp,
        linestyle=":",
        linewidth=1,
        label="Approximate 95% residual limits",
    )

    plt.axhline(
        upper_error_95_pp,
        linestyle=":",
        linewidth=1,
        label="_nolegend_",
    )

    plt.xlabel("Gate opening [%]")
    plt.ylabel(
        "Efficiency error [percentage points]\n"
        "(estimated-flow efficiency - measured-flow efficiency)"
    )
    plt.title(
        "Efficiency error vs gate opening\n"
        f"Bias={bias_pp:.3f} pp, "
        f"MAE={mae_pp:.3f} pp, "
        f"RMSE={rmse_pp:.3f} pp"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_folder
        / "efficiency_error_vs_gate_opening.png",
        dpi=180,
    )
    plt.close()

    summary_file = (
        output_folder
        / "efficiency_error_and_limits.txt"
    )

    summary_file.write_text(
        f"""EFFICIENCY ERROR AND APPROXIMATE LIMITS
=======================================

Error definition:
Estimated-flow efficiency - measured-flow efficiency

Rows:
{len(efficiency_data)}

Bias:
{bias_pp:.6f} percentage points

Mean absolute error:
{mae_pp:.6f} percentage points

Root mean square error:
{rmse_pp:.6f} percentage points

Standard deviation of error:
{std_pp:.6f} percentage points

Maximum absolute error:
{max_abs_pp:.6f} percentage points

Approximate 95% residual limits:
{lower_error_95_pp:.6f}
to
{upper_error_95_pp:.6f} percentage points

INTERPRETATION
--------------
The limits are validation limits derived from the observed residuals,
not a complete measurement-uncertainty budget.

Because:
error = estimated efficiency - measured efficiency

the approximate interval around an estimated efficiency is:

lower measured-equivalent limit =
estimated efficiency - upper residual limit

upper measured-equivalent limit =
estimated efficiency - lower residual limit
""",
        encoding="utf-8",
    )

    return {
        "bias_pp": bias_pp,
        "mae_pp": mae_pp,
        "rmse_pp": rmse_pp,
        "std_pp": std_pp,
        "max_abs_pp": max_abs_pp,
        "lower_95_pp": lower_error_95_pp,
        "upper_95_pp": upper_error_95_pp,
    }


def get_valid_efficiency_data(
    validation: pd.DataFrame,
) -> pd.DataFrame:
    efficiency_data = validation[
        validation["Measured_unit_efficiency_pct"].between(
            0,
            105,
        )
        & validation["Estimated_unit_efficiency_pct"].between(
            0,
            105,
        )
    ].copy()

    if efficiency_data.empty:
        raise ValueError(
            "No valid efficiency values between 0% and 105%. "
            "Check the power and flow units."
        )

    return efficiency_data


def save_efficiency_vs_head_plots(
    validation: pd.DataFrame,
    output_folder: Path,
) -> None:
    efficiency_data = get_valid_efficiency_data(validation)

    plt.figure(figsize=(9, 6))
    plt.scatter(
        efficiency_data["Head_m"],
        efficiency_data["Measured_unit_efficiency_pct"],
        s=20,
        alpha=0.45,
        label="Efficiency using measured flow",
    )
    plt.scatter(
        efficiency_data["Head_m"],
        efficiency_data["Estimated_unit_efficiency_pct"],
        s=20,
        alpha=0.45,
        label="Efficiency using estimated flow",
    )
    plt.xlabel("Head [m]")
    plt.ylabel("Provisional electrical unit efficiency [%]")
    plt.title(
        "Measured-flow and estimated-flow efficiency vs head"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_folder / "efficiency_vs_head_scatter.png",
        dpi=180,
    )
    plt.close()

    # Mean values in 0.1 m head bins.
    efficiency_data["Head_bin_m"] = (
        efficiency_data["Head_m"].round(1)
    )

    efficiency_by_head = (
        efficiency_data
        .groupby("Head_bin_m", as_index=False)
        .agg(
            Mean_measured_efficiency_pct=(
                "Measured_unit_efficiency_pct",
                "mean",
            ),
            Mean_estimated_efficiency_pct=(
                "Estimated_unit_efficiency_pct",
                "mean",
            ),
            Number_of_points=(
                "Measured_unit_efficiency_pct",
                "count",
            ),
        )
        .sort_values("Head_bin_m")
    )

    efficiency_by_head.to_csv(
        output_folder / "efficiency_by_head.csv",
        index=False,
    )

    plt.figure(figsize=(9, 6))
    plt.plot(
        efficiency_by_head["Head_bin_m"],
        efficiency_by_head["Mean_measured_efficiency_pct"],
        marker="o",
        linewidth=1.5,
        label="Efficiency using measured flow",
    )
    plt.plot(
        efficiency_by_head["Head_bin_m"],
        efficiency_by_head["Mean_estimated_efficiency_pct"],
        marker="o",
        linewidth=1.5,
        label="Efficiency using estimated flow",
    )
    plt.xlabel("Head [m]")
    plt.ylabel("Mean provisional electrical unit efficiency [%]")
    plt.title(
        "Mean measured-flow and estimated-flow efficiency vs head"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_folder / "efficiency_vs_head.png",
        dpi=180,
    )
    plt.close()


def save_efficiency_vs_blade_position_plots(
    validation: pd.DataFrame,
    output_folder: Path,
) -> None:
    efficiency_data = get_valid_efficiency_data(validation)

    plt.figure(figsize=(9, 6))
    plt.scatter(
        efficiency_data["Runner_blade_position_pct"],
        efficiency_data["Measured_unit_efficiency_pct"],
        s=20,
        alpha=0.45,
        label="Efficiency using measured flow",
    )
    plt.scatter(
        efficiency_data["Runner_blade_position_pct"],
        efficiency_data["Estimated_unit_efficiency_pct"],
        s=20,
        alpha=0.45,
        label="Efficiency using estimated flow",
    )
    plt.xlabel("Runner blade position [%]")
    plt.ylabel("Provisional electrical unit efficiency [%]")
    plt.title(
        "Measured-flow and estimated-flow efficiency "
        "vs runner blade position"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_folder
        / "efficiency_vs_blade_position_scatter.png",
        dpi=180,
    )
    plt.close()

    # Mean values in 1% blade-position bins.
    efficiency_data["Runner_blade_position_bin_pct"] = (
        efficiency_data["Runner_blade_position_pct"].round()
    )

    efficiency_by_blade = (
        efficiency_data
        .groupby(
            "Runner_blade_position_bin_pct",
            as_index=False,
        )
        .agg(
            Mean_measured_efficiency_pct=(
                "Measured_unit_efficiency_pct",
                "mean",
            ),
            Mean_estimated_efficiency_pct=(
                "Estimated_unit_efficiency_pct",
                "mean",
            ),
            Number_of_points=(
                "Measured_unit_efficiency_pct",
                "count",
            ),
        )
        .sort_values("Runner_blade_position_bin_pct")
    )

    efficiency_by_blade.to_csv(
        output_folder / "efficiency_by_blade_position.csv",
        index=False,
    )

    plt.figure(figsize=(9, 6))
    plt.plot(
        efficiency_by_blade[
            "Runner_blade_position_bin_pct"
        ],
        efficiency_by_blade[
            "Mean_measured_efficiency_pct"
        ],
        marker="o",
        linewidth=1.5,
        label="Efficiency using measured flow",
    )
    plt.plot(
        efficiency_by_blade[
            "Runner_blade_position_bin_pct"
        ],
        efficiency_by_blade[
            "Mean_estimated_efficiency_pct"
        ],
        marker="o",
        linewidth=1.5,
        label="Efficiency using estimated flow",
    )
    plt.xlabel("Runner blade position [%]")
    plt.ylabel("Mean provisional electrical unit efficiency [%]")
    plt.title(
        "Mean measured-flow and estimated-flow efficiency "
        "vs runner blade position"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_folder / "efficiency_vs_blade_position.png",
        dpi=180,
    )
    plt.close()

def main() -> None:
    args = parse_arguments()

    csv_file = Path(args.csv_file).resolve()
    output_folder = Path(args.output_dir).resolve()

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_long_csv(
        csv_file,
        args.power_signal,
    )

    if args.flow_unit == "m3s":
        df["Measured_flow_m3s"] = df["Water_flow_raw"]
    else:
        df["Measured_flow_m3s"] = (
            df["Water_flow_raw"] / 3600.0
        )

    df["Active_power_MW"] = convert_power_to_mw(
        df["Active_power_raw"],
        args.power_unit,
    )

    df["Estimated_flow_m3s"] = estimate_flow(df)

    df["Estimated_minus_measured_m3s"] = (
        df["Estimated_flow_m3s"]
        - df["Measured_flow_m3s"]
    )

    df["Estimated_minus_measured_pct"] = np.where(
        df["Measured_flow_m3s"] > 0,
        100
        * df["Estimated_minus_measured_m3s"]
        / df["Measured_flow_m3s"],
        np.nan,
    )

    df = add_efficiency_columns(df)

    valid = (
        df[
            [
                "Gate_opening_pct",
                "Runner_blade_position_pct",
                "Head_m",
                "Measured_flow_m3s",
                "Estimated_flow_m3s",
                "Active_power_MW",
                "Measured_unit_efficiency_pct",
                "Estimated_unit_efficiency_pct",
            ]
        ]
        .notna()
        .all(axis=1)
        & (df["Head_m"] > 0)
        & (
            df["Gate_opening_pct"]
            > MINIMUM_GATE_OPENING_PCT
        )
        & (
            df["Measured_flow_m3s"]
            > MINIMUM_FLOW_M3S
        )
        & (df["Estimated_flow_m3s"] > 0)
        & (df["Active_power_MW"] > 0)
    )

    if args.start is not None:
        start_time = pd.Timestamp(args.start)

        if start_time.tzinfo is None:
            start_time = start_time.tz_localize("UTC")
        else:
            start_time = start_time.tz_convert("UTC")

        valid &= df["timestamp"] >= start_time

    validation = df.loc[valid].copy()

    if len(validation) < 5:
        raise ValueError(
            "Too few valid rows. Check the flow unit, "
            "power unit, signal name, and optional start time."
        )

    flow_error = (
        validation["Estimated_flow_m3s"]
        - validation["Measured_flow_m3s"]
    )

    efficiency_error = (
        validation["Efficiency_error_percentage_points"]
    )

    comparison_columns = [
        "timestamp",
        "Gate_opening_pct",
        "Runner_blade_position_pct",
        "Head_m",
        "Active_power_MW",
        "Measured_flow_m3s",
        "Estimated_flow_m3s",
        "Estimated_minus_measured_m3s",
        "Estimated_minus_measured_pct",
        "Measured_unit_efficiency_pct",
        "Estimated_unit_efficiency_pct",
        "Efficiency_error_percentage_points",
    ]

    comparison_file = (
        output_folder
        / "flow_and_efficiency_comparison.csv"
    )

    validation[comparison_columns].to_csv(
        comparison_file,
        index=False,
        date_format="%Y-%m-%dT%H:%M:%SZ",
    )

    metrics_file = output_folder / "validation_metrics.txt"

    metrics_file.write_text(
        f"""FROZEN FLOW AND EFFICIENCY VALIDATION
=====================================

Input file:
{csv_file.name}

The flow coefficients were NOT recalculated.

Valid rows:
{len(validation)}

FLOW
----
Bias: {flow_error.mean():.6f} m3/s
MAE: {flow_error.abs().mean():.6f} m3/s
RMSE: {np.sqrt(np.mean(flow_error ** 2)):.6f} m3/s
MAPE: {
    np.mean(
        np.abs(
            100
            * flow_error
            / validation["Measured_flow_m3s"]
        )
    )
:.6f} %

EFFICIENCY
----------
Error = efficiency using estimated flow
        - efficiency using measured flow

Bias: {efficiency_error.mean():.6f} percentage points
MAE: {efficiency_error.abs().mean():.6f} percentage points
RMSE: {
    np.sqrt(np.mean(efficiency_error ** 2))
:.6f} percentage points

Efficiency is provisional electrical unit efficiency:
eta = P_e / (rho * g * Q * H)
""",
        encoding="utf-8",
    )

    save_flow_plots(
        validation,
        output_folder,
    )

    save_efficiency_gate_plots(
        validation,
        output_folder,
    )

    efficiency_limit_metrics = (
        save_efficiency_limits_and_error_plots(
            validation,
            output_folder,
        )
    )

    save_efficiency_vs_head_plots(
        validation,
        output_folder,
    )

    save_efficiency_vs_blade_position_plots(
        validation,
        output_folder,
    )

    print("Completed.")
    print("The flow coefficients were not recalculated.")
    print(f"Valid rows: {len(validation)}")
    print(
        "Created: "
        + str(
            output_folder
            / "efficiency_vs_gate_opening.png"
        )
    )
    print(
        "Created: "
        + str(
            output_folder
            / "efficiency_vs_gate_opening_scatter.png"
        )
    )
    print(
        "Created: "
        + str(
            output_folder
            / "estimated_efficiency_with_95pct_limits.png"
        )
    )
    print(
        "Created: "
        + str(
            output_folder
            / "efficiency_error_vs_gate_opening.png"
        )
    )
    print(
        "Created: "
        + str(
            output_folder
            / "efficiency_vs_head.png"
        )
    )
    print(
        "Created: "
        + str(
            output_folder
            / "efficiency_vs_head_scatter.png"
        )
    )
    print(
        "Created: "
        + str(
            output_folder
            / "efficiency_vs_blade_position.png"
        )
    )
    print(
        "Created: "
        + str(
            output_folder
            / "efficiency_vs_blade_position_scatter.png"
        )
    )
    print(
        "Efficiency error: "
        f"bias={efficiency_limit_metrics['bias_pp']:.4f} pp, "
        f"MAE={efficiency_limit_metrics['mae_pp']:.4f} pp, "
        f"RMSE={efficiency_limit_metrics['rmse_pp']:.4f} pp"
    )
    print(
        "Approximate 95% residual limits: "
        f"{efficiency_limit_metrics['lower_95_pp']:.4f} "
        "to "
        f"{efficiency_limit_metrics['upper_95_pp']:.4f} pp"
    )
    print(f"Results folder: {output_folder}")


if __name__ == "__main__":
    main()
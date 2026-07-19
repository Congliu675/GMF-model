"""
Retrieve GMF model parameters and evaluate model performance for a selected
laboratory sample.

Required files in the same directory as this script:
    1. GMF_model.py
    2. Angle_information.xlsx
    3. Sample_I.xlsx, Sample_II.xlsx, or Sample_III.xlsx

The angle information file must contain the following columns:
    SZA     : solar zenith angle in degrees
    VZA     : viewing zenith angle in degrees
    RelAzi  : relative azimuth angle in degrees

Each sample file must contain one column of observed BPRF values without
a column header.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from GMF_model import GMF_model


# ============================================================
# 1. User settings
# ============================================================

# Select one sample: "Sample_I", "Sample_II", or "Sample_III"
SAMPLE_NAME = "Sample_I"

AVAILABLE_SAMPLES = {
    "Sample_I",
    "Sample_II",
    "Sample_III"
}

ANGLE_COLUMNS = ["SZA", "VZA", "RelAzi"]

# Parameter order: rho, sigma, kg
INITIAL_PARAMS = np.array([5.0, 0.6, 0.6], dtype=float)
LOWER_BOUNDS = np.array([0.0, 0.0, 0.0], dtype=float)
UPPER_BOUNDS = np.array([20.0, 1.0, 1.0], dtype=float)

MAX_FUNCTION_EVALUATIONS = 50000


# ============================================================
# 2. File paths
# ============================================================

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()

ANGLE_FILE = SCRIPT_DIR / "Angle_information.xlsx"
SAMPLE_FILE = SCRIPT_DIR / f"{SAMPLE_NAME}.xlsx"
OUTPUT_FILE = SCRIPT_DIR / f"GMF_{SAMPLE_NAME}_results.xlsx"


# ============================================================
# 3. Residual function
# ============================================================

def model_residuals(params, x_data, y_observed):
    """Return residuals between observed and modeled BPRF values."""
    y_modeled = GMF_model(x_data, *params)
    return y_observed - y_modeled


# ============================================================
# 4. Input data
# ============================================================

def load_input_data(angle_file, sample_file):
    """
    Read angular geometries and observed BPRF values.

    Parameters
    ----------
    angle_file : pathlib.Path
        Path to the angular geometry file.
    sample_file : pathlib.Path
        Path to the selected sample file.

    Returns
    -------
    data_df : pandas.DataFrame
        Valid angular geometries and observed BPRF values.
    x_data_rad : numpy.ndarray
        Angular geometries in radians.
    y_observed : numpy.ndarray
        Observed BPRF values.
    """
    if SAMPLE_NAME not in AVAILABLE_SAMPLES:
        valid_names = ", ".join(sorted(AVAILABLE_SAMPLES))
        raise ValueError(
            f"Invalid sample name: {SAMPLE_NAME}. "
            f"Available samples are: {valid_names}."
        )

    if not angle_file.exists():
        raise FileNotFoundError(
            f"Angle information file not found: {angle_file}"
        )

    if not sample_file.exists():
        raise FileNotFoundError(
            f"Sample file not found: {sample_file}"
        )

    angle_df = pd.read_excel(
        angle_file,
        engine="openpyxl"
    )

    missing_columns = [
        column
        for column in ANGLE_COLUMNS
        if column not in angle_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "The angle information file is missing the following columns: "
            + ", ".join(missing_columns)
        )

    sample_df = pd.read_excel(
        sample_file,
        header=None,
        usecols=[0],
        names=["BPRF_observed"],
        engine="openpyxl"
    )

    if len(angle_df) != len(sample_df):
        raise ValueError(
            "The numbers of angular geometries and BPRF observations "
            f"are inconsistent: {len(angle_df)} angular records and "
            f"{len(sample_df)} BPRF records."
        )

    data_df = pd.concat(
        [
            angle_df[ANGLE_COLUMNS].reset_index(drop=True),
            sample_df.reset_index(drop=True)
        ],
        axis=1
    )

    required_columns = ANGLE_COLUMNS + ["BPRF_observed"]

    for column in required_columns:
        data_df[column] = pd.to_numeric(
            data_df[column],
            errors="coerce"
        )

    valid_mask = np.isfinite(
        data_df[required_columns].to_numpy(dtype=float)
    ).all(axis=1)

    data_df = data_df.loc[valid_mask].reset_index(drop=True)

    if data_df.empty:
        raise ValueError(
            f"No valid data records were found in {sample_file.name}."
        )

    x_data_deg = data_df[ANGLE_COLUMNS].to_numpy(dtype=float)
    x_data_rad = np.deg2rad(x_data_deg)
    y_observed = data_df["BPRF_observed"].to_numpy(dtype=float)

    return data_df, x_data_rad, y_observed


# ============================================================
# 5. Parameter retrieval
# ============================================================

def fit_gmf_model(x_data_rad, y_observed):
    """
    Retrieve GMF parameters using bounded nonlinear least squares.

    Parameters
    ----------
    x_data_rad : numpy.ndarray
        Angular geometries in radians.
    y_observed : numpy.ndarray
        Observed BPRF values.

    Returns
    -------
    scipy.optimize.OptimizeResult
        Optimization result.
    """
    optimization_result = least_squares(
        model_residuals,
        x0=INITIAL_PARAMS,
        args=(x_data_rad, y_observed),
        bounds=(LOWER_BOUNDS, UPPER_BOUNDS),
        method="trf",
        max_nfev=MAX_FUNCTION_EVALUATIONS
    )

    if not optimization_result.success:
        raise RuntimeError(
            "GMF parameter retrieval did not converge. "
            f"Optimizer message: {optimization_result.message}"
        )

    return optimization_result


# ============================================================
# 6. Model evaluation
# ============================================================

def calculate_model_metrics(
    y_observed,
    y_modeled,
    num_model_params
):
    """
    Calculate Adjusted R², RMSE, RRMSE, and complete AIC.

    Adjusted R² is calculated from the coefficient of determination
    based on RSS and TSS.

    RRMSE is calculated as RMSE divided by the mean observed BPRF.

    The complete AIC is calculated as:

        AIC = n × ln(2π) + n × ln(RSS/n) + n + 2k

    where k is the number of retrieved model parameters.

    Parameters
    ----------
    y_observed : array-like
        Observed BPRF values.
    y_modeled : array-like
        Modeled BPRF values.
    num_model_params : int
        Number of retrieved model parameters.

    Returns
    -------
    dict
        Adjusted R², RMSE, RRMSE, and complete AIC.
    """
    y_observed = np.asarray(y_observed, dtype=float)
    y_modeled = np.asarray(y_modeled, dtype=float)

    valid_mask = (
        np.isfinite(y_observed)
        & np.isfinite(y_modeled)
    )

    y_observed = y_observed[valid_mask]
    y_modeled = y_modeled[valid_mask]

    sample_size = y_observed.size

    if sample_size == 0:
        raise ValueError(
            "No valid observed and modeled BPRF pairs were found."
        )

    residuals = y_observed - y_modeled
    rss = np.sum(residuals ** 2)
    tss = np.sum(
        (y_observed - np.mean(y_observed)) ** 2
    )

    if tss > 0.0 and sample_size > num_model_params + 1:
        r_squared = 1.0 - rss / tss
        adjusted_r_squared = (
            1.0
            - (1.0 - r_squared)
            * (sample_size - 1)
            / (sample_size - num_model_params - 1)
        )
    else:
        adjusted_r_squared = np.nan

    rmse = np.sqrt(
        np.mean(residuals ** 2)
    )

    mean_observed = np.mean(y_observed)

    if np.abs(mean_observed) > 1.0e-12:
        rrmse = rmse / mean_observed
    else:
        rrmse = np.nan

    rss_safe = max(rss, 1.0e-12)

    aic = (
        sample_size * np.log(2.0 * np.pi)
        + sample_size * np.log(rss_safe / sample_size)
        + sample_size
        + 2.0 * num_model_params
    )

    return {
        "Adjusted_R2": adjusted_r_squared,
        "RMSE": rmse,
        "RRMSE": rrmse,
        "AIC_full": aic
    }


# ============================================================
# 7. Result output
# ============================================================

def save_results(
    output_file,
    input_data,
    parameters,
    metrics,
    y_modeled,
    optimization_result
):
    """
    Save retrieved parameters, model performance metrics, modeled BPRF
    values, and optimization information to an Excel file.
    """
    parameters_df = pd.DataFrame({
        "Parameter": ["rho", "sigma", "kg"],
        "Retrieved_value": parameters,
        "Initial_value": INITIAL_PARAMS,
        "Lower_bound": LOWER_BOUNDS,
        "Upper_bound": UPPER_BOUNDS
    })

    metrics_df = pd.DataFrame(
        [metrics],
        index=[SAMPLE_NAME]
    )
    metrics_df.index.name = "Sample"

    modeled_data_df = input_data.copy()
    modeled_data_df["BPRF_modeled"] = y_modeled
    modeled_data_df["Residual"] = (
        modeled_data_df["BPRF_observed"]
        - modeled_data_df["BPRF_modeled"]
    )

    optimization_df = pd.DataFrame({
        "Item": [
            "Sample",
            "Success",
            "Status",
            "Function evaluations",
            "Termination message"
        ],
        "Value": [
            SAMPLE_NAME,
            optimization_result.success,
            optimization_result.status,
            optimization_result.nfev,
            optimization_result.message
        ]
    })

    with pd.ExcelWriter(
        output_file,
        engine="openpyxl"
    ) as writer:
        parameters_df.to_excel(
            writer,
            sheet_name="Parameters",
            index=False
        )

        metrics_df.to_excel(
            writer,
            sheet_name="Model Metrics"
        )

        modeled_data_df.to_excel(
            writer,
            sheet_name="Modeled Data",
            index=False
        )

        optimization_df.to_excel(
            writer,
            sheet_name="Optimization",
            index=False
        )


# ============================================================
# 8. Main program
# ============================================================

def main():
    """Run GMF parameter retrieval and model evaluation."""
    input_data, x_data_rad, y_observed = load_input_data(
        ANGLE_FILE,
        SAMPLE_FILE
    )

    optimization_result = fit_gmf_model(
        x_data_rad,
        y_observed
    )

    retrieved_parameters = optimization_result.x

    y_modeled = GMF_model(
        x_data_rad,
        *retrieved_parameters
    )

    metrics = calculate_model_metrics(
        y_observed,
        y_modeled,
        num_model_params=len(retrieved_parameters)
    )

    save_results(
        OUTPUT_FILE,
        input_data,
        retrieved_parameters,
        metrics,
        y_modeled,
        optimization_result
    )

    rho, sigma, kg = retrieved_parameters

    print(f"Sample: {SAMPLE_NAME}")
    print(f"rho = {rho:.6f}")
    print(f"sigma = {sigma:.6f}")
    print(f"kg = {kg:.6f}")
    print(f"Adjusted R2 = {metrics['Adjusted_R2']:.6f}")
    print(f"RMSE = {metrics['RMSE']:.6f}")
    print(f"RRMSE = {metrics['RRMSE']:.6f}")
    print(f"AIC = {metrics['AIC_full']:.6f}")
    print(f"Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
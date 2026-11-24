"""Load the AI4I 2020 predictive-maintenance dataset (UCI).

Target: 'Machine failure' (1 = the machine failed).
The five failure-mode columns (TWF, HDF, PWF, OSF, RNF) are DROPPED: they are
sub-labels of the target, so using them as inputs would leak the answer.
IDs (UDI, Product ID) are dropped too.
"""
import pandas as pd

TARGET = "Machine failure"
LEAKY = ["TWF", "HDF", "PWF", "OSF", "RNF"]
IDS = ["UDI", "Product ID"]
NUMERIC = ["Air temperature [K]", "Process temperature [K]",
           "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]


def load_ai4i(path="data/ai4i2020.csv",physics=False):
    df = pd.read_csv(path)
    y = df[TARGET].astype(int)
    X = df[["Type"] + NUMERIC].copy()
    X = pd.get_dummies(X, columns=["Type"], prefix="Type", dtype=int)   # L / M / H
    if physics:
        X["temp_diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]
        X["power_w"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"] * 2 * 3.14159 / 60
        X["wear_x_torque"] = df["Tool wear [min]"] * df["Torque [Nm]"]
    return X, y
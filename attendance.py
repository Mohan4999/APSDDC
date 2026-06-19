# -*- coding: utf-8 -*-
"""
=============================================================================
  STUDENT ATTENDANCE MANAGEMENT & CERTIFICATION SYSTEM
  Internship Project | Machine Learning + Data Analytics
=============================================================================
"""

import os
import sys
import time
import warnings

# Fix Windows cp1252 encoding for Unicode terminal output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
CERT_THRESHOLD = 80.0
DATA_PATH      = "data/session_1.csv"
REPORT_PATH    = "outputs/Final_Attendance_Report.csv"
BAR_PATH       = "graphs/bar_graph.png"
SCATTER_PATH   = "graphs/scatter_plot.png"
README_PATH    = "README.txt"

CYAN    = "\033[96m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
BOLD    = "\033[1m"
RESET   = "\033[0m"
WHITE   = "\033[97m"
BLUE    = "\033[94m"
MAGENTA = "\033[95m"


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def print_banner():
    banner = f"""
{CYAN}{BOLD}
╔══════════════════════════════════════════════════════════════════════════╗
║       STUDENT ATTENDANCE MANAGEMENT & CERTIFICATION SYSTEM               ║
║       Internship Milestone Project  |  Data Science & ML Pipeline        ║
╚══════════════════════════════════════════════════════════════════════════╝{RESET}
"""
    print(banner)


def section(title):
    print(f"\n{BLUE}{BOLD}{'─'*74}{RESET}")
    print(f"{BLUE}{BOLD}  >>  {title}{RESET}")
    print(f"{BLUE}{BOLD}{'─'*74}{RESET}")


def ok(msg):   print(f"  {GREEN}[OK]{RESET}  {msg}")
def info(msg): print(f"  {CYAN}[i]{RESET}   {msg}")
def warn(msg): print(f"  {YELLOW}[!]{RESET}   {msg}")
def err(msg):  print(f"  {RED}[ERR]{RESET} {msg}")


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 – FILESYSTEM
# ─────────────────────────────────────────────────────────────────────────────
def validate_filesystem():
    section("STEP 1 | File System Validation")
    for folder in ["data", "graphs", "outputs"]:
        os.makedirs(folder, exist_ok=True)
        ok(f"Directory ready: {folder}/")

    if not os.path.exists(DATA_PATH):
        warn(f"Dataset not found at '{DATA_PATH}'. Running data generator...")
        os.system(f"{sys.executable} generate_data.py")
    else:
        # Check if existing CSV has required columns; regenerate if not
        try:
            tmp = pd.read_csv(DATA_PATH, nrows=1)
            tmp.columns = tmp.columns.str.strip()
            required = {"student_id", "name"}
            # Accept any reasonable duration column name
            has_duration = any(c in tmp.columns for c in [
                "total_duration","Total_Duration","Duration","duration",
                "Duration_Minutes","Total_Attendance","Total Duration"
            ])
            # Accept student_id or Student_ID etc.
            has_id = any(c in tmp.columns for c in [
                "student_id","Student_ID","studentid","student id"
            ])
            if not (has_id and has_duration):
                warn("Existing CSV has incompatible columns. Regenerating dataset...")
                os.remove(DATA_PATH)
                os.system(f"{sys.executable} generate_data.py")
            else:
                ok(f"Dataset found: {DATA_PATH}")
        except Exception:
            warn("Could not read existing CSV. Regenerating...")
            os.system(f"{sys.executable} generate_data.py")


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 – LOAD & VALIDATE
# ─────────────────────────────────────────────────────────────────────────────
def load_and_validate(path):
    section("STEP 2 | Data Loading & Quality Validation")

    df = pd.read_csv(path)
    ok(f"Loaded {len(df):,} records from '{path}'")

    # ── Normalise column names from any older CSV format ─────────────────────
    df.columns = df.columns.str.strip()          # remove accidental whitespace
    rename_map = {
        # old name            -> expected name
        "student id":           "student_id",
        "Student_ID":           "student_id",
        "studentid":            "student_id",
        "Student Name":         "name",
        "student_name":         "name",
        "Student_Name":         "name",
        "Email":                "email",
        "Email_ID":             "email",
        "email_id":             "email",
        "Joining_Time":         "joining_time",
        "Leaving_Time":         "leaving_time",
        "Duration":             "total_duration",
        "duration":             "total_duration",
        "Total_Duration":       "total_duration",
        "Total Duration":       "total_duration",
        "total duration":       "total_duration",
        "Attendance_Duration":  "total_duration",
        "Duration_Minutes":     "total_duration",
        "Total_Attendance":     "total_duration",   # only if total_duration missing
    }
    # Apply renames only for columns that exist and whose target doesn't yet
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)

    # If total_duration still missing but Total_Attendance exists, copy it
    if "total_duration" not in df.columns and "Total_Attendance" in df.columns:
        df["total_duration"] = df["Total_Attendance"]

    # Add missing optional columns with defaults so preview never crashes
    for col, default in [("email",""), ("joining_time","--"), ("leaving_time","--")]:
        if col not in df.columns:
            df[col] = default

    # Show head like your screenshot
    print(f"\n{WHITE}  Dataset Preview:{RESET}")
    print(df[["student_id","name","email","joining_time","leaving_time","total_duration"]].head().to_string(index=True))

    # Missing values
    print(f"\n{WHITE}  Missing Values:{RESET}")
    mv = df[["student_id","name","email"]].isnull().sum()
    for col, cnt in mv.items():
        print(f"  {col:<20} {cnt}")

    # Null check
    null_count = df.isnull().sum().sum()
    if null_count > 0:
        warn(f"Found {null_count} null values -> dropping affected rows")
        df.dropna(inplace=True)
    else:
        ok("No null values detected")

    # Duplicates
    dup = df.duplicated(subset=["student_id"]).sum()
    if dup > 0:
        warn(f"Dropped {dup} duplicate student_id rows")
        df.drop_duplicates(subset=["student_id"], inplace=True)
    else:
        ok("No duplicate student records found")

    # Valid duration
    invalid = df[df["total_duration"] < 0]
    if len(invalid):
        warn(f"Removed {len(invalid)} rows with negative duration")
        df = df[df["total_duration"] >= 0]
    else:
        ok("All duration values are >= 0")

    # Non-empty names
    df["name"] = df["name"].astype(str).str.strip()
    empty = df[df["name"] == ""]
    if len(empty):
        warn(f"Dropped {len(empty)} rows with empty name")
        df = df[df["name"] != ""]
    else:
        ok("All student name fields are non-empty")

    info(f"Clean dataset: {len(df):,} records | {df['student_id'].nunique()} unique students")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3 – FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def engineer_features(df):
    section("STEP 3 | Feature Engineering & Certification")

    # Recalculate attendance percentage from total_duration (max 120 min session)
    df = df.copy()
    df["Attendance_Percentage"] = (df["total_duration"] / 120 * 100).round(6)
    df["Total_Attendance"]      = df["total_duration"]
    df["Certification_Status"]  = df["Attendance_Percentage"].apply(
        lambda p: "Certified" if p >= CERT_THRESHOLD else "Not Certified"
    )

    assert df["student_id"].is_unique, "Duplicate student IDs detected!"
    ok(f"Feature engineering complete: {len(df):,} student records")

    cert     = (df["Certification_Status"] == "Certified").sum()
    not_cert = (df["Certification_Status"] == "Not Certified").sum()
    ok(f"Certified     : {cert:>4}  ({cert/len(df)*100:.1f}%)")
    ok(f"Not Certified : {not_cert:>4}  ({not_cert/len(df)*100:.1f}%)")

    # Print certification count exactly like screenshot
    print(f"\n  {WHITE}Certification Count{RESET}")
    cc = df["Certification_Status"].value_counts()
    print(f"  Certification_Status")
    for status, count in cc.items():
        print(f"  {status:<20} {count}")
    print(f"  Name: count, dtype: int64")

    # Show last 5 rows like screenshot (student_id ... email ... Total_Attendance ... Att% ... Status)
    print(f"\n  {WHITE}Last 5 Rows Preview:{RESET}")
    preview = df[["student_id","name","email","Total_Attendance","Attendance_Percentage","Certification_Status"]].tail(5)
    print(preview.to_string(index=True))

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4 – MACHINE LEARNING
# ─────────────────────────────────────────────────────────────────────────────
def run_ml_pipeline(df):
    section("STEP 4 | Machine Learning Classification Pipeline")

    feature_cols = ["total_duration", "Total_Attendance", "Attendance_Percentage"]
    X = df[feature_cols].values

    le = LabelEncoder()
    y  = le.fit_transform(df["Certification_Status"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if np.bincount(y).min() >= 2 else None
    )
    info(f"Training samples : {len(X_train)}")
    info(f"Testing  samples : {len(X_test)}")

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    t0  = time.time()
    clf.fit(X_train, y_train)
    t1  = time.time()

    y_pred = clf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    ok(f"Model trained in {t1-t0:.2f}s  |  Test Accuracy: {acc*100:.2f}%")

    print(f"\n{WHITE}  Classification Report:{RESET}")
    report = classification_report(y_test, y_pred,
                                   target_names=le.classes_, digits=4)
    for line in report.splitlines():
        print(f"    {line}")

    print(f"\n{WHITE}  Feature Importances:{RESET}")
    for feat, imp in sorted(zip(feature_cols, clf.feature_importances_),
                            key=lambda x: -x[1]):
        bar = "#" * int(imp * 40)
        print(f"    {feat:<28} {imp:.4f}  {CYAN}{bar}{RESET}")

    return clf, le


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 5 – GRAPHS
# ─────────────────────────────────────────────────────────────────────────────
def generate_graphs(df):
    section("STEP 5 | Generating Visualizations")

    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })

    total     = len(df)
    cert_mask = df["Certification_Status"] == "Certified"
    n_cert    = cert_mask.sum()
    n_not     = (~cert_mask).sum()

    # ── BAR GRAPH ─────────────────────────────────────────────────────────────
    labels = ["Certified", "Not Certified"]
    values = [n_cert, n_not]
    colors = ["#2ecc71", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, width=0.45,
                  edgecolor="white", linewidth=1.5, zorder=3)

    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    for bar, val in zip(bars, values):
        pct = val / total * 100
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 10,
                f"{val}\n({pct:.1f}%)",
                ha="center", va="bottom",
                fontsize=11, fontweight="bold", color="#2c3e50")

    ax.set_title("Certification Distribution of Students",
                 fontsize=14, fontweight="bold", pad=14, color="#2c3e50")
    ax.set_xlabel("Certification Status", fontsize=11, color="#555")
    ax.set_ylabel("Number of Students",   fontsize=11, color="#555")
    ax.set_ylim(0, max(values) * 1.22)
    ax.tick_params(axis="both", labelsize=10)
    plt.tight_layout()
    plt.savefig(BAR_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    ok(f"Bar graph saved      -> {BAR_PATH}")

    # ── SCATTER PLOT – matches uploaded image exactly ────────────────────────
    # Green circles = Not Certified (below 80%, diagonal line going up)
    # Red triangles = Certified (above 80% threshold)
    # Blue dashed threshold line at 80%

    cert_mask = df["Certification_Status"] == "Certified"
    not_mask  = ~cert_mask

    fig, ax = plt.subplots(figsize=(10, 6))

    # Light grey grid like the image
    ax.set_facecolor("#f5f5f5")
    ax.yaxis.grid(True, color="white", linewidth=1.2, zorder=0)
    ax.xaxis.grid(True, color="white", linewidth=1.2, zorder=0)
    ax.set_axisbelow(True)

    # Not Certified — green filled circles (below threshold, forms the diagonal)
    ax.scatter(
        df.loc[not_mask, "Total_Attendance"],
        df.loc[not_mask, "Attendance_Percentage"],
        color="#2ecc40", s=55, marker="o",
        edgecolors="#1a8a2a", linewidths=0.5,
        alpha=0.90, zorder=3, label="Not Certified"
    )

    # Certified — red filled triangles (above threshold)
    ax.scatter(
        df.loc[cert_mask, "Total_Attendance"],
        df.loc[cert_mask, "Attendance_Percentage"],
        color="#e74c3c", s=60, marker="^",
        edgecolors="#a93226", linewidths=0.5,
        alpha=0.90, zorder=3, label="Certified"
    )

    # Blue dashed threshold line at 80% with label
    ax.axhline(y=CERT_THRESHOLD, color="#2980b9", linewidth=1.8,
               linestyle="--", zorder=2)
    ax.text(df["Total_Attendance"].min() + 1, CERT_THRESHOLD + 1.5,
            f"{CERT_THRESHOLD:.0f} % Threshold",
            color="#2980b9", fontsize=10, fontweight="bold")

    ax.set_title("Total Attendance vs Attendance Percentage",
                 fontsize=14, fontweight="bold", pad=14, color="#2c3e50")
    ax.set_xlabel("Total Attendance (minutes)", fontsize=11, color="#444")
    ax.set_ylabel("Attendance Percentage (%)",  fontsize=11, color="#444")

    # Legend bottom-right matching image position
    legend = ax.legend(
        fontsize=11, loc="lower right",
        framealpha=0.95, edgecolor="#ccc",
        markerscale=1.3
    )

    ax.tick_params(axis="both", labelsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(SCATTER_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    ok(f"Scatter plot saved   -> {SCATTER_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 6 – REPORT
# ─────────────────────────────────────────────────────────────────────────────
def generate_report(df):
    section("STEP 6 | Final Report Generation")

    report = df[[
        "student_id", "name", "email",
        "joining_time", "leaving_time", "total_duration",
        "Total_Attendance", "Attendance_Percentage", "Certification_Status"
    ]].copy()
    report.sort_values("Attendance_Percentage", ascending=False, inplace=True)
    report.to_csv(REPORT_PATH, index=False)
    ok(f"Final report saved -> {REPORT_PATH}  ({len(report):,} students)")

    # Stats summary
    print(f"\n  {WHITE}{BOLD}{'─'*60}{RESET}")
    print(f"  {WHITE}{BOLD}  ATTENDANCE STATISTICS SUMMARY{RESET}")
    print(f"  {WHITE}{BOLD}{'─'*60}{RESET}")
    stats = [
        ("Total Students",          f"{len(df):,}"),
        ("Avg Attendance %",        f"{df['Attendance_Percentage'].mean():.2f}%"),
        ("Highest Attendance %",    f"{df['Attendance_Percentage'].max():.2f}%"),
        ("Lowest Attendance %",     f"{df['Attendance_Percentage'].min():.2f}%"),
        ("Certified Students",      f"{(df['Certification_Status']=='Certified').sum():,}"),
        ("Not Certified Students",  f"{(df['Certification_Status']=='Not Certified').sum():,}"),
        ("Certification Threshold", f"{CERT_THRESHOLD}%"),
    ]
    for label, value in stats:
        print(f"  {CYAN}  {label:<32}{RESET}{GREEN}{BOLD}{value}{RESET}")
    print(f"  {WHITE}{BOLD}{'─'*60}{RESET}")

    # Top 5
    print(f"\n  {MAGENTA}{BOLD}  TOP 5 STUDENTS BY ATTENDANCE{RESET}")
    print(f"  {'─'*65}")
    print(f"  {BOLD}{'#':<4} {'Student ID':<12} {'Name':<18} {'Email':<26} {'Att %':>8}{RESET}")
    print(f"  {'─'*65}")
    for rank, (_, row) in enumerate(report.head(5).iterrows(), 1):
        print(f"  {rank:<4} {row['student_id']:<12} {row['name']:<18} "
              f"{row['email']:<26} {GREEN}{BOLD}{row['Attendance_Percentage']:>7.2f}%{RESET}")

    # Bottom 5
    print(f"\n  {RED}{BOLD}  BOTTOM 5 STUDENTS BY ATTENDANCE{RESET}")
    print(f"  {'─'*65}")
    print(f"  {BOLD}{'#':<4} {'Student ID':<12} {'Name':<18} {'Email':<26} {'Att %':>8}{RESET}")
    print(f"  {'─'*65}")
    for rank, (_, row) in enumerate(report.tail(5).iterrows(), 1):
        print(f"  {rank:<4} {row['student_id']:<12} {row['name']:<18} "
              f"{row['email']:<26} {RED}{BOLD}{row['Attendance_Percentage']:>7.2f}%{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 7 – README
# ─────────────────────────────────────────────────────────────────────────────
def write_readme():
    content = """
================================================================================
  STUDENT ATTENDANCE MANAGEMENT & CERTIFICATION SYSTEM
  Internship Milestone Project
================================================================================

PROJECT OVERVIEW
----------------
End-to-end pipeline for student attendance analysis, certification
determination, and machine learning classification.

FOLDER STRUCTURE
----------------
  Attendence_ML_Project/
  +-- data/
  |   +-- session_1.csv              Raw attendance dataset (980 students)
  +-- graphs/
  |   +-- bar_graph.png              Certified vs Not Certified bar chart
  |   +-- scatter_plot.png           Attendance Analysis scatter plot
  +-- outputs/
  |   +-- Final_Attendance_Report.csv  Full report with certification status
  +-- attendance.py                  Main pipeline script
  +-- generate_data.py               Dataset generator
  +-- README.txt                     This file

HOW TO RUN
----------
  Step 1: python generate_data.py     # Generate dataset (run once)
  Step 2: python attendance.py        # Run full pipeline

DATASET COLUMNS
---------------
  student_id, name, email, joining_time, leaving_time,
  total_duration, Total_Attendance, Attendance_Percentage, Certification_Status

CONSTRAINTS IMPLEMENTED
-----------------------
  1.  Attendance >= 80% required for certification
  2.  No duplicate student records
  3.  No missing/null values
  4.  Valid attendance values only (duration >= 0)
  5.  Dataset contains 980 unique students
  6.  ML: Random Forest classification (Certified / Not Certified)
  7.  Graphs generated only after preprocessing
  8.  Final report exported as CSV
  9.  Student ID is unique
  10. System processes all records efficiently (< 5 seconds)

FORMULA
-------
  Attendance Percentage = (total_duration / 120) x 100
  Certification         = Certified if Attendance % >= 80 else Not Certified

================================================================================
"""
    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(content.strip())
    ok(f"README written       -> {README_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    t_start = time.time()
    print_banner()

    validate_filesystem()
    df  = load_and_validate(DATA_PATH)
    df  = engineer_features(df)
    run_ml_pipeline(df)
    generate_graphs(df)
    generate_report(df)
    write_readme()

    t_end = time.time()

    section("PIPELINE COMPLETE")
    ok(f"Total execution time : {t_end - t_start:.2f} seconds")
    ok("All outputs saved to data/ | graphs/ | outputs/")
    print(f"\n{GREEN}{BOLD}  [OK]  Project pipeline finished successfully.{RESET}\n")


if __name__ == "__main__":
    main()

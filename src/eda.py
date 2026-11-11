"""
Exploratory Data Analysis (EDA), Data Quality Auditing, and Leakage Detection.
Generates comprehensive diagnostic reports and publication-quality figures.
"""

import os
os.environ["MPLCONFIGDIR"] = os.path.abspath(".mpl_cache")
os.makedirs(".mpl_cache", exist_ok=True)
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def perform_eda_and_quality_audit(
    data_path: str = "data/raw/student_academic_records.csv",
    output_fig_dir: str = "artifacts/figures",
    report_output_path: str = "artifacts/eda_summary.json"
) -> dict:
    """
    Executes end-to-end data quality checks, ethical non-discrimination audit,
    temporal leakage validation, and generates analytical visualizations.
    """
    os.makedirs(output_fig_dir, exist_ok=True)
    os.makedirs(os.path.dirname(report_output_path), exist_ok=True)

    df = pd.read_csv(data_path)
    print(f"Loaded dataset from {data_path} with shape {df.shape}")

    # 1. Basic Profiling & Data Quality
    missing_counts = df.isnull().sum().to_dict()
    duplicate_rows = int(df.duplicated(subset=['student_id']).sum())
    total_records = len(df)
    at_risk_count = int(df['is_at_risk'].sum())
    at_risk_ratio = float(df['is_at_risk'].mean())

    # 2. Ethical Attribute Audit (Confirm zero protected/demographic columns)
    prohibited_keywords = ['gender', 'sex', 'race', 'ethnicity', 'religion', 'caste', 'income', 'zipcode', 'disability']
    detected_prohibited = [col for col in df.columns if any(p in col.lower() for p in prohibited_keywords)]
    ethical_compliance = len(detected_prohibited) == 0

    # 3. Temporal Leakage Audit
    # Confirm predictors are strictly pre-Week 9 signals
    target_cols = ['final_score', 'is_at_risk']
    id_cols = ['student_id', 'cohort_term']
    predictor_cols = [c for c in df.columns if c not in target_cols + id_cols]

    # Check correlations with target
    correlations = df[predictor_cols + ['is_at_risk']].corr()['is_at_risk'].drop('is_at_risk')
    max_corr_feature = correlations.abs().idxmax()
    max_corr_value = float(correlations.abs().max())
    # No predictor should have extreme leakage (e.g., > 0.92)
    leakage_flag = max_corr_value > 0.92

    # 4. Cohort Distribution
    cohort_breakdown = df.groupby('cohort_term')['is_at_risk'].agg(
        total_students='count',
        at_risk_count='sum',
        at_risk_rate='mean'
    ).to_dict(orient='index')

    summary_stats = {
        "dataset_shape": list(df.shape),
        "total_students": total_records,
        "at_risk_count": at_risk_count,
        "at_risk_percentage": round(at_risk_ratio * 100, 2),
        "duplicate_student_ids": duplicate_rows,
        "missing_values_detected": {k: int(v) for k, v in missing_counts.items() if v > 0},
        "ethical_compliance_audit": {
            "prohibited_attributes_checked": prohibited_keywords,
            "detected_prohibited_attributes": detected_prohibited,
            "compliant": ethical_compliance
        },
        "temporal_leakage_audit": {
            "prediction_point": "Week 8 (Mid-term cut-off)",
            "target_horizon": "Week 16 (Final outcome)",
            "highest_feature_target_correlation": {
                "feature": max_corr_feature,
                "correlation": round(max_corr_value, 4)
            },
            "leakage_risk_detected": leakage_flag
        },
        "cohort_breakdown": cohort_breakdown
    }

    with open(report_output_path, "w") as f:
        json.dump(summary_stats, f, indent=4)
    print(f"EDA summary saved to {report_output_path}")

    # 5. Visualizations
    sns.set_theme(style="whitegrid", font_scale=1.0)

    # Figure 1: Key Feature Distributions Stratified by At-Risk Status
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Feature Distributions Stratified by Academic At-Risk Status (Week 8 Signals)", fontsize=16, fontweight='bold')

    features_to_plot = [
        ("midterm_exam_score", "Midterm Exam Score (0-100)"),
        ("attendance_rate_w5_w8", "Attendance Rate Weeks 5-8 (%)"),
        ("quiz_2_score", "Progress Quiz 2 Score (0-100)"),
        ("assignments_submitted_ratio", "Assignment Completion Ratio (0-1)"),
        ("avg_assignment_delay_days", "Average Assignment Delay (Days)"),
        ("cumulative_gpa_prior", "Prior Cumulative GPA (0-4 scale)")
    ]

    for ax, (col, label) in zip(axes.flatten(), features_to_plot):
        sns.histplot(
            data=df,
            x=col,
            hue='is_at_risk',
            kde=True,
            palette={0: "#2b5c8f", 1: "#d9534f"},
            bins=25,
            ax=ax,
            alpha=0.5,
            stat="density",
            common_norm=False
        )
        ax.set_title(label, fontweight='semibold')
        ax.set_xlabel("")
        ax.legend(title='Outcome', labels=['At-Risk (1)', 'Safe (0)'], loc='upper right')

    plt.tight_layout()
    fig1_path = os.path.join(output_fig_dir, "eda_distributions.png")
    plt.savefig(fig1_path, dpi=200)
    plt.close()
    print(f"Saved: {fig1_path}")

    # Figure 2: Correlation Heatmap
    plt.figure(figsize=(12, 10))
    corr_matrix = df[predictor_cols + ['is_at_risk']].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        vmax=0.8,
        vmin=-0.8,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .8}
    )
    plt.title("Correlation Matrix: Week 8 Signals vs Academic At-Risk Status", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig2_path = os.path.join(output_fig_dir, "correlation_heatmap.png")
    plt.savefig(fig2_path, dpi=200)
    plt.close()
    print(f"Saved: {fig2_path}")

    # Figure 3: Cohort Stability & Class Balance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    cohort_df = df.groupby('cohort_term')['is_at_risk'].value_counts(normalize=True).unstack() * 100
    cohort_df.plot(kind='bar', stacked=True, color=["#2b5c8f", "#d9534f"], ax=ax1)
    ax1.set_title("At-Risk Proportion Across Academic Cohorts", fontweight='bold')
    ax1.set_ylabel("Percentage (%)")
    ax1.set_xlabel("Academic Cohort")
    ax1.legend(labels=['Safe (0)', 'At-Risk (1)'], loc='lower right')
    ax1.tick_params(axis='x', rotation=25)

    df.groupby('is_at_risk').size().plot(
        kind='pie',
        labels=['Safe / On-Track', 'At-Risk'],
        colors=["#2b5c8f", "#d9534f"],
        autopct='%1.1f%%',
        explode=(0, 0.08),
        ax=ax2,
        startangle=140
    )
    ax2.set_title("Overall Class Distribution (Academic At-Risk Imbalance)", fontweight='bold')
    ax2.set_ylabel("")

    plt.tight_layout()
    fig3_path = os.path.join(output_fig_dir, "cohort_stability.png")
    plt.savefig(fig3_path, dpi=200)
    plt.close()
    print(f"Saved: {fig3_path}")

    return summary_stats


if __name__ == "__main__":
    perform_eda_and_quality_audit()

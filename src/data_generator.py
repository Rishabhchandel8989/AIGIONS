"""
Synthetic Educational Data Generator
Produces realistic, statistically grounded student learning signals and performance outcomes.
Complies with FERPA/GDPR ethics: Zero protected or sensitive demographic attributes.
Features represent strictly behavioral, academic, and engagement metrics available up to Week 8.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


def generate_synthetic_student_data(
    n_samples: int = 3500,
    random_seed: int = 42,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Generates synthetic student performance data across four chronological cohorts.
    Simulates realistic pedagogical archetypes with natural noise and multi-collinearities.

    Parameters:
        n_samples: Total number of student records to generate (>= 1,000 required).
        random_seed: Random seed for exact reproducibility.
        output_path: Optional file path to persist CSV.

    Returns:
        pd.DataFrame containing synthetic student dataset.
    """
    rng = np.random.default_rng(random_seed)

    cohort_names = ["Cohort_2023_Fall", "Cohort_2024_Spring", "Cohort_2024_Fall", "Cohort_2025_Spring"]
    cohort_probs = [0.25, 0.25, 0.25, 0.25]
    cohorts = rng.choice(cohort_names, size=n_samples, p=cohort_probs)

    # Latent student academic capability factor ~ N(0, 1)
    latent_capability = rng.normal(loc=0.0, scale=1.0, size=n_samples)

    # Latent motivation & conscientiousness factor ~ N(0, 1)
    latent_conscientiousness = rng.normal(loc=0.0, scale=1.0, size=n_samples)

    # 1. Prior Academic Preparation: Cumulative GPA (0.0 to 4.0 scale)
    # Strongly related to latent capability + slight noise
    raw_gpa = 2.8 + 0.6 * latent_capability + 0.2 * latent_conscientiousness + rng.normal(0, 0.35, size=n_samples)
    cumulative_gpa_prior = np.clip(np.round(raw_gpa, 2), 1.20, 4.00)

    # 2. Enrolled Course Credits (12 to 22 credits)
    course_credits_enrolled = rng.choice([12, 14, 15, 16, 18, 20, 21], size=n_samples, p=[0.10, 0.15, 0.35, 0.20, 0.12, 0.05, 0.03])

    # 3. Attendance Rate Weeks 1-4 (%)
    # Driven primarily by conscientiousness
    att_w1_w4_mean = 78.0 + 10.0 * latent_conscientiousness + rng.normal(0, 8.0, size=n_samples)
    attendance_rate_w1_w4 = np.clip(np.round(att_w1_w4_mean, 1), 20.0, 100.0)

    # 4. Attendance Rate Weeks 5-8 (%)
    # Simulates attendance drop-off or persistence
    # Struggling/disengaging students often exhibit sharp mid-semester decline
    att_drift = -4.0 + 3.5 * latent_conscientiousness + rng.normal(0, 7.5, size=n_samples)
    attendance_rate_w5_w8 = np.clip(np.round(attendance_rate_w1_w4 + att_drift, 1), 10.0, 100.0)

    # 5. Assignment Submission Ratio (0.0 to 1.0)
    assign_ratio_raw = 0.85 + 0.12 * latent_conscientiousness + rng.normal(0, 0.08, size=n_samples)
    assignments_submitted_ratio = np.clip(np.round(assign_ratio_raw, 3), 0.15, 1.00)

    # 6. Average Assignment Submission Delay in Days (0.0 to 14.0 days)
    delay_raw = np.exp(1.2 - 0.7 * latent_conscientiousness + rng.normal(0, 0.4, size=n_samples)) - 0.5
    avg_assignment_delay_days = np.clip(np.round(np.maximum(0.0, delay_raw), 1), 0.0, 14.0)

    # 7. Diagnostic Quiz 1 (Week 2-3, scale 0-100)
    quiz1_raw = 65.0 + 12.0 * latent_capability + rng.normal(0, 9.0, size=n_samples)
    quiz_1_score = np.clip(np.round(quiz1_raw, 1), 15.0, 100.0)

    # 8. Progress Quiz 2 (Week 6, scale 0-100)
    # Influenced by capability, study habit, and previous quiz
    quiz2_raw = 0.45 * quiz_1_score + 28.0 + 6.0 * latent_capability + 5.0 * latent_conscientiousness + rng.normal(0, 7.5, size=n_samples)
    quiz_2_score = np.clip(np.round(quiz2_raw, 1), 10.0, 100.0)

    # 9. Midterm Exam Score (Week 8, scale 0-100)
    midterm_raw = (
        0.30 * quiz_1_score +
        0.40 * quiz_2_score +
        0.15 * attendance_rate_w5_w8 +
        5.0 * latent_capability +
        rng.normal(0, 6.0, size=n_samples)
    )
    midterm_exam_score = np.clip(np.round(midterm_raw, 1), 10.0, 100.0)

    # 10. LMS Login Frequency per Week
    lms_logins_raw = 8.5 + 4.0 * latent_conscientiousness + rng.normal(0, 2.5, size=n_samples)
    lms_login_frequency_per_week = np.clip(np.round(np.maximum(1.0, lms_logins_raw), 1), 1.0, 25.0)

    # 11. LMS Learning Resource Views
    lms_views_raw = lms_login_frequency_per_week * 4.5 + rng.normal(0, 12.0, size=n_samples)
    lms_resource_views = np.clip(np.round(np.maximum(5.0, lms_views_raw), 0), 5.0, 180.0)

    # 12. Discussion Forum Posts
    forum_posts_raw = rng.poisson(lam=np.clip(3.5 + 2.5 * latent_conscientiousness, 0.5, 15.0))
    discussion_forum_posts = np.clip(forum_posts_raw, 0, 35)

    # 13. Peer Study Group Participation (0 or 1)
    study_group_prob = 1.0 / (1.0 + np.exp(-(0.5 * latent_conscientiousness + 0.3 * latent_capability - 0.2)))
    study_group_participation = (rng.uniform(size=n_samples) < study_group_prob).astype(int)

    # 14. Target Outcome: Final Examination Score (Week 16) & At-Risk Status
    # This represents the true future outcome to be predicted using only Week 8 signals!
    final_score_raw = (
        8.5 +
        0.38 * midterm_exam_score +
        0.22 * quiz_2_score +
        0.15 * attendance_rate_w5_w8 +
        12.0 * assignments_submitted_ratio -
        0.8 * avg_assignment_delay_days +
        2.5 * (cumulative_gpa_prior - 2.5) +
        3.0 * study_group_participation +
        rng.normal(0, 7.5, size=n_samples)
    )
    final_score = np.clip(np.round(final_score_raw, 1), 5.0, 100.0)

    # Academic At-Risk Definition: Final grade < 50.0% (standard academic failure threshold)
    is_at_risk = (final_score < 50.0).astype(int)

    # Synthetic Student IDs
    student_ids = [f"STU-{10000 + i}" for i in range(n_samples)]

    df = pd.DataFrame({
        "student_id": student_ids,
        "cohort_term": cohorts,
        "attendance_rate_w1_w4": attendance_rate_w1_w4,
        "attendance_rate_w5_w8": attendance_rate_w5_w8,
        "assignments_submitted_ratio": assignments_submitted_ratio,
        "avg_assignment_delay_days": avg_assignment_delay_days,
        "quiz_1_score": quiz_1_score,
        "quiz_2_score": quiz_2_score,
        "midterm_exam_score": midterm_exam_score,
        "lms_login_frequency_per_week": lms_login_frequency_per_week,
        "lms_resource_views": lms_resource_views,
        "discussion_forum_posts": discussion_forum_posts,
        "cumulative_gpa_prior": cumulative_gpa_prior,
        "course_credits_enrolled": course_credits_enrolled,
        "study_group_participation": study_group_participation,
        # Outcomes for training and evaluation
        "final_score": final_score,
        "is_at_risk": is_at_risk
    })

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Synthetic student dataset saved to: {output_path} ({len(df)} records, {is_at_risk.mean()*100:.1f}% at-risk)")

    return df


if __name__ == "__main__":
    generate_synthetic_student_data(output_path="data/raw/student_academic_records.csv")

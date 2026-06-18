import numpy as np
import pandas as pd

from pharma_trial_ml.config import DATA_DIR, DATA_PATH


def generate_mock_trial_data(n_rows: int = 2500, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    age = rng.normal(55, 12, n_rows).clip(18, 85)
    bmi = rng.normal(28, 5, n_rows).clip(16, 45)
    baseline_biomarker = rng.normal(60, 18, n_rows).clip(5, 130)
    dose_mg = rng.choice([10, 25, 50, 75], size=n_rows, p=[0.15, 0.30, 0.40, 0.15])
    adherence_pct = rng.normal(86, 12, n_rows).clip(35, 100)
    prior_treatments = rng.poisson(1.3, n_rows).clip(0, 6)

    sex = rng.choice(["M", "F"], size=n_rows, p=[0.48, 0.52])
    trial_arm = rng.choice(["placebo", "active"], size=n_rows, p=[0.45, 0.55])

    # Synthetic response mechanism.
    # Active arm, higher dose, higher adherence and favourable baseline biomarker increase response probability.
    logit = (
        -3.0
        + 0.035 * (baseline_biomarker - 50)
        + 0.018 * (adherence_pct - 80)
        + 0.018 * dose_mg
        - 0.025 * (age - 50)
        - 0.035 * (bmi - 27)
        - 0.25 * prior_treatments
        + np.where(trial_arm == "active", 1.1, -0.6)
        + np.where(sex == "F", 0.12, 0.0)
    )

    probability = 1 / (1 + np.exp(-logit))
    responder = rng.binomial(1, probability)

    df = pd.DataFrame(
        {
            "age": age.round(1),
            "bmi": bmi.round(1),
            "baseline_biomarker": baseline_biomarker.round(2),
            "dose_mg": dose_mg,
            "adherence_pct": adherence_pct.round(1),
            "prior_treatments": prior_treatments,
            "sex": sex,
            "trial_arm": trial_arm,
            "responder": responder,
        }
    )

    return df


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_mock_trial_data()
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved mock data to {DATA_PATH}")
    print(df.head())


if __name__ == "__main__":
    main()

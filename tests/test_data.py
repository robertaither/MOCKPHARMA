from pharma_trial_ml.data import generate_mock_trial_data


def test_generate_mock_trial_data_has_expected_columns():
    df = generate_mock_trial_data(n_rows=100)

    expected_columns = {
        "age",
        "bmi",
        "baseline_biomarker",
        "dose_mg",
        "adherence_pct",
        "prior_treatments",
        "sex",
        "trial_arm",
        "responder",
    }

    assert set(df.columns) == expected_columns
    assert len(df) == 100
    assert df["responder"].isin([0, 1]).all()

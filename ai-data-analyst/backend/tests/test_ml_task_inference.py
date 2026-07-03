from __future__ import annotations

import pandas as pd

from app.services.ml_task_inference import infer_target_candidates


def test_infer_target_candidates_flags_identifier_and_leakage():
    df = pd.DataFrame(
        {
            "customer_id": [f"c_{i:03d}" for i in range(120)],
            "churn": [0 if i % 3 else 1 for i in range(120)],
            "churn_copy": [0 if i % 3 else 1 for i in range(120)],
            "country": ["US" if i % 2 else "IN" for i in range(120)],
        }
    )

    candidates, selected_target, selected_task, warnings = infer_target_candidates(df)

    assert warnings == []
    assert selected_target == "churn"
    assert selected_task == "classification"
    assert selected_target != "customer_id"

    by_col = {c.column: c for c in candidates}
    assert len(by_col) >= 2

    leak_types = {w.warning_type for w in by_col["churn"].leakage_warnings if w.column == "churn_copy"}
    assert "exact_copy" in leak_types

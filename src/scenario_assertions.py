"""
Scenario expectation checks for demo preflight validation.
"""


def _check_min(checks, label, actual, expected):
    if expected is None:
        return
    checks.append({
        "check": label,
        "expected": f">= {expected:.2f}",
        "actual": f"{actual:.2f}",
        "passed": actual >= expected,
    })


def _check_max(checks, label, actual, expected):
    if expected is None:
        return
    checks.append({
        "check": label,
        "expected": f"<= {expected:.2f}",
        "actual": f"{actual:.2f}",
        "passed": actual <= expected,
    })


def _check_includes(checks, label, actual_values, expected_values):
    for expected in expected_values or []:
        checks.append({
            "check": label,
            "expected": f"contains {expected}",
            "actual": ", ".join(actual_values) if actual_values else "none",
            "passed": expected in actual_values,
        })


def _check_excludes(checks, label, actual_values, expected_values):
    for expected in expected_values or []:
        checks.append({
            "check": label,
            "expected": f"excludes {expected}",
            "actual": ", ".join(actual_values) if actual_values else "none",
            "passed": expected not in actual_values,
        })


def _check_query_contains(checks, label, query, expected_phrases):
    query_lower = query.lower()
    for phrase in expected_phrases or []:
        checks.append({
            "check": label,
            "expected": f"contains '{phrase}'",
            "actual": query,
            "passed": phrase.lower() in query_lower,
        })


def _check_query_excludes(checks, label, query, expected_phrases):
    query_lower = query.lower()
    for phrase in expected_phrases or []:
        checks.append({
            "check": label,
            "expected": f"excludes '{phrase}'",
            "actual": query,
            "passed": phrase.lower() not in query_lower,
        })


def _check_all_success(checks, label, rows, enabled):
    if enabled is None:
        return
    passed = all(row.get("success", False) for row in rows) if rows else False
    checks.append({
        "check": label,
        "expected": str(enabled),
        "actual": str(passed),
        "passed": passed is enabled,
    })


def evaluate_expectations(result):
    """Evaluate a run_scenario result against expected_outcomes from the scenario."""
    expected = result["scenario"].get("expected_outcomes", {})
    checks = []

    baseline_expectations = expected.get("baseline", {})
    attack_expectations = expected.get("attack", {})
    defense_expectations = expected.get("defense", {})
    log_expectations = expected.get("logs", {})

    _check_min(
        checks,
        "baseline recall",
        result["baseline"]["metrics"]["recall"],
        baseline_expectations.get("min_recall"),
    )
    _check_max(
        checks,
        "baseline poison exposure",
        result["poison_exposure"]["score"],
        baseline_expectations.get("max_poison_exposure"),
    )

    _check_max(
        checks,
        "attack recall",
        result["adversarial"]["metrics"]["recall"],
        attack_expectations.get("max_recall"),
    )
    _check_min(
        checks,
        "attack poison exposure",
        result["poison_exposure"]["score"],
        attack_expectations.get("min_poison_exposure"),
    )
    _check_includes(
        checks,
        "attack retrieved nodes",
        result["adversarial"]["nodes"],
        attack_expectations.get("retrieved_nodes_include"),
    )
    _check_query_contains(
        checks,
        "attack query",
        result["adversarial"]["query"],
        attack_expectations.get("query_contains"),
    )

    _check_min(
        checks,
        "defended recall",
        result["defended"]["metrics"]["recall"],
        defense_expectations.get("min_recall"),
    )
    _check_max(
        checks,
        "defended poison exposure",
        result["defended_poison_exposure"]["score"],
        defense_expectations.get("max_poison_exposure"),
    )
    _check_excludes(
        checks,
        "defended retrieved nodes",
        result["defended"]["nodes"],
        defense_expectations.get("retrieved_nodes_exclude"),
    )
    _check_query_excludes(
        checks,
        "defended query",
        result["defended"]["query"],
        defense_expectations.get("query_excludes"),
    )

    _check_all_success(
        checks,
        "attack logs success",
        result["attacks"],
        log_expectations.get("attacks_success"),
    )
    _check_all_success(
        checks,
        "defense logs success",
        result["defenses"],
        log_expectations.get("defenses_success"),
    )

    return {
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }

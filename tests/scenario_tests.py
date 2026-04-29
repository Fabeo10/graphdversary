"""
Preflight checks for graphdversary scenarios.
"""

import argparse
import json
from pathlib import Path

from src.pipeline import PROJECT_ROOT, run_scenario
from src.scenario_assertions import evaluate_expectations


def scenario_paths(selected=None):
    if selected:
        return [Path(selected)]
    return sorted((PROJECT_ROOT / "data" / "scenarios").glob("*.json"))


def run_preflight(paths):
    results = []
    for path in paths:
        scenario_result = run_scenario(path, verbose=False)
        expectation_result = evaluate_expectations(scenario_result)
        results.append({
            "path": str(path),
            "id": scenario_result["scenario"].get("id", path.stem),
            "title": scenario_result["scenario"].get("title", path.stem),
            "test_type": scenario_result["scenario"].get("test_type", "unspecified"),
            "passed": expectation_result["passed"],
            "checks": expectation_result["checks"],
        })
    return results


def short_value(value, max_length=72):
    text = str(value)
    if len(text) <= max_length:
        return text
    return f"{text[:max_length - 3]}..."


def print_report(results):
    print("=" * 72)
    print(" graphdversary scenario preflight ".center(72))
    print("=" * 72)
    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"\n[{status}] {result['title']} ({result['test_type']})")
        for check in result["checks"]:
            marker = "OK" if check["passed"] else "!!"
            expected = short_value(check["expected"])
            actual = short_value(check["actual"])
            print(f"  [{marker}] {check['check']}: expected {expected} | actual {actual}")

    passed = sum(1 for result in results if result["passed"])
    print(f"\nSummary: {passed}/{len(results)} scenarios passed.")
    print("Use --json for full untruncated check details.")


def main():
    parser = argparse.ArgumentParser(description="Run scenario expectation checks.")
    parser.add_argument("--scenario", help="Optional path to a single scenario JSON file.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args()

    results = run_preflight(scenario_paths(args.scenario))
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_report(results)

    if not all(result["passed"] for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

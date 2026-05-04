"""
Interactive Streamlit demo for graphdversary.
"""

from pathlib import Path
import random
import sys
import time

import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

from src.pipeline import DEFAULT_SCENARIO_PATH, PROJECT_ROOT, load_scenario, run_scenario
from src.agent_duel import (
    DEFAULT_OLLAMA_BLUE,
    DEFAULT_OLLAMA_RED,
    run_agent_duel_steps,
)
from src.scenario_assertions import evaluate_expectations


TYPE_COLORS = {
    "Service": "#4C78A8",
    "Database": "#72B7B2",
    "Component": "#F58518",
    "Middleware": "#54A24B",
    "Policy": "#B279A2",
    "Application": "#9D755D",
    "Gateway": "#E45756",
    "adversarial": "#D62728",
}

NODE_POSITIONS = {
    "n8": (-2.4, 1.2),
    "n9": (-2.4, -0.4),
    "n1": (-0.8, 0.0),
    "n3": (0.8, 0.8),
    "n2": (0.8, -0.4),
    "n4": (0.8, -1.4),
    "n6": (2.4, 1.2),
    "n7": (2.4, 0.2),
    "n10": (2.4, -1.4),
    "n5": (3.9, -0.4),
    "poison_auth_bypass": (-0.8, 1.7),
    "poison_gateway_override": (-2.4, 1.9),
}

NODE_NAMES = {
    "n1": "Auth Service",
    "n2": "User DB",
    "n3": "JWT Validator",
    "n4": "Rate Limiter",
    "n5": "Audit Logger",
    "n6": "KMS",
    "n7": "Session Policy",
    "n8": "Admin Console",
    "n9": "API Gateway",
    "n10": "Fraud Detector",
    "poison_auth_bypass": "Poison Node",
    "poison_gateway_override": "Gateway Poison",
}


def scenario_options():
    scenario_dir = PROJECT_ROOT / "data" / "scenarios"
    scenarios = sorted(scenario_dir.glob("*.json"))
    return scenarios or [DEFAULT_SCENARIO_PATH]


def run_live_scenario(scenario_path, query, top_k, hop_depth, attacks, defenses, mock_answer):
    """Run on every interaction so red/blue checkbox changes always refresh the graph."""
    return run_scenario(
        scenario_path=scenario_path,
        query=query,
        top_k=top_k,
        hop_depth=hop_depth,
        attacks=attacks,
        defenses=defenses,
        mock_answer=mock_answer,
        verbose=False,
    )


def metric_card(label, value):
    st.metric(label, f"{value:.2f}")


def delta_label(before, after):
    delta = after - before
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.2f}"


def active_labels(items):
    return [item.get("label", item.get("type", "Unnamed")) for item in items if item.get("enabled", True)]


def paired_controls(scenario):
    attacks = []
    defenses = []
    scenario_attacks = scenario.get("attacks", [])
    scenario_defenses = scenario.get("defenses", [])
    max_steps = max(len(scenario_attacks), len(scenario_defenses))

    st.sidebar.header("Red / Blue Toggles")
    st.sidebar.caption("Each row pairs a red-team action with the matching blue-team control.")

    for index in range(max_steps):
        st.sidebar.markdown(f"**Step {index + 1}**")
        attack = scenario_attacks[index] if index < len(scenario_attacks) else None
        defense = scenario_defenses[index] if index < len(scenario_defenses) else None

        if attack:
            enabled = st.sidebar.checkbox(
                f"Red: {attack.get('label', attack['type'])}",
                value=attack.get("enabled", True),
                key=f"attack-{scenario.get('id', 'scenario')}-{index}",
            )
            configured_attack = dict(attack)
            configured_attack["enabled"] = enabled
            attacks.append(configured_attack)
        else:
            st.sidebar.caption("Red: no paired attack")

        if defense:
            enabled = st.sidebar.checkbox(
                f"Blue: {defense.get('label', defense['type'])}",
                value=defense.get("enabled", True),
                key=f"defense-{scenario.get('id', 'scenario')}-{index}",
            )
            configured_defense = dict(defense)
            configured_defense["enabled"] = enabled
            defenses.append(configured_defense)
        else:
            st.sidebar.caption("Blue: no paired defense")

    return attacks, defenses


def content_preview(content, max_len=72):
    return content if len(content) <= max_len else f"{content[:max_len]}..."


def node_position(node_id, index):
    if node_id in NODE_POSITIONS:
        return NODE_POSITIONS[node_id]
    return (index % 4, -2.0 - (index // 4))


def build_graph_figure(
    snapshot,
    title,
    removed_edges=None,
    ground_truth_nodes=None,
    retrieved_nodes=None,
    show_edge_labels=False,
):
    removed_edges = removed_edges or set()
    ground_truth_nodes = set(ground_truth_nodes or [])
    retrieved_nodes = set(retrieved_nodes or [])
    nodes = {node["id"]: node for node in snapshot["nodes"]}
    pos = {node_id: node_position(node_id, index) for index, node_id in enumerate(nodes)}

    edge_traces = []
    edge_annotations = []
    node_numbers = {node_id: str(index + 1) for index, node_id in enumerate(nodes)}
    for edge in snapshot["edges"]:
        source = edge["source"]
        target = edge["target"]
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        edge_traces.append(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line={
                    "width": 2,
                    "color": "#888",
                    "dash": "solid",
                },
                hoverinfo="text",
                text=f"{NODE_NAMES.get(source, source)} --{edge.get('relation', '')}--> {NODE_NAMES.get(target, target)}",
                showlegend=False,
            )
        )
        if show_edge_labels:
            edge_annotations.append(
                {
                    "x": (x0 + x1) / 2,
                    "y": (y0 + y1) / 2,
                    "text": edge.get("relation", ""),
                    "showarrow": False,
                    "font": {"size": 9, "color": "#555"},
                    "bgcolor": "rgba(255,255,255,0.75)",
                }
            )
    for source, target in removed_edges:
        if source not in pos or target not in pos:
            continue
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        edge_traces.append(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line={"width": 2, "color": "#D62728", "dash": "dash"},
                hoverinfo="text",
                text=f"Removed edge: {source} -> {target}",
                showlegend=False,
            )
        )
        if show_edge_labels:
            edge_annotations.append(
                {
                    "x": (x0 + x1) / 2,
                    "y": (y0 + y1) / 2,
                    "text": "REMOVED",
                    "showarrow": False,
                    "font": {"size": 10, "color": "#D62728"},
                    "bgcolor": "rgba(255,255,255,0.85)",
                }
            )

    node_x = []
    node_y = []
    labels = []
    colors = []
    hover = []
    sizes = []
    outlines = []
    for node_id, attrs in nodes.items():
        x, y = pos[node_id]
        node_x.append(x)
        node_y.append(y)
        labels.append(node_numbers[node_id])
        node_type = attrs.get("type", "generic")
        colors.append(TYPE_COLORS.get(node_type, "#BAB0AC"))
        hover.append(
            f"{node_numbers[node_id]}. {NODE_NAMES.get(node_id, node_id)}<br>{node_id}<br>{node_type}<br>{content_preview(attrs.get('content', ''), 140)}"
        )
        sizes.append(34 if node_id in retrieved_nodes else 28)
        outlines.append("#111" if node_id in ground_truth_nodes else "#666")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=labels,
        textposition="middle center",
        hoverinfo="text",
        hovertext=hover,
        marker={
            "size": sizes,
            "color": colors,
            "line": {"width": 3, "color": outlines},
        },
        showlegend=False,
    )

    legend_traces = [
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker={"size": 10, "color": color},
            name=node_type,
        )
        for node_type, color in TYPE_COLORS.items()
    ]

    figure = go.Figure(data=edge_traces + [node_trace] + legend_traces)
    figure.update_layout(
        title=title,
        height=560,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        xaxis={
            "showgrid": False,
            "zeroline": False,
            "showticklabels": False,
            "range": [-2.9, 4.4],
        },
        yaxis={
            "showgrid": False,
            "zeroline": False,
            "showticklabels": False,
            "range": [-1.9, 2.3],
        },
        plot_bgcolor="white",
        legend={"orientation": "h", "y": -0.08},
        annotations=edge_annotations,
    )
    return figure


def node_table(snapshot, ground_truth_nodes, retrieved_nodes):
    ground_truth_nodes = set(ground_truth_nodes)
    retrieved_nodes = set(retrieved_nodes)
    return [
        {
            "#": index + 1,
            "Node": NODE_NAMES.get(node["id"], node["id"]),
            "ID": node["id"],
            "Type": node.get("type", "generic"),
            "Ground truth": node["id"] in ground_truth_nodes,
            "Retrieved": node["id"] in retrieved_nodes,
            "Content": content_preview(node.get("content", ""), 120),
        }
        for index, node in enumerate(snapshot["nodes"])
    ]


def edge_table(snapshot):
    return [
        {
            "Source": NODE_NAMES.get(edge["source"], edge["source"]),
            "Relation": edge.get("relation", ""),
            "Target": NODE_NAMES.get(edge["target"], edge["target"]),
        }
        for edge in snapshot["edges"]
    ]


def edge_pairs(snapshot):
    return {(edge["source"], edge["target"]) for edge in snapshot["edges"]}


def session_key(scenario, suffix):
    return f"{scenario.get('id', 'scenario')}-{suffix}"


def context_panel(title, run_result):
    st.subheader(title)
    st.caption(f"Query: {run_result['query']}")
    for item in run_result["context"]:
        st.write(item)

    with st.expander("Retrieval trace", expanded=False):
        st.write("Anchors")
        st.dataframe(run_result["trace"]["anchors"], width="stretch")
        st.write("BFS hops")
        st.dataframe(run_result["trace"]["hops"], width="stretch")


def outcome_panel(result):
    attack_recall = result["adversarial"]["metrics"]["recall"]
    defended_recall = result["defended"]["metrics"]["recall"]
    attack_poison = result["poison_exposure"]["score"]
    defended_poison = result["defended_poison_exposure"]["score"]

    if defended_poison < attack_poison and defended_recall >= attack_recall:
        st.success("Blue team improved the run: poison exposure dropped without reducing recall.")
    elif defended_poison < attack_poison:
        st.info("Blue team reduced poison exposure, but check recall to see whether useful evidence was also filtered.")
    elif attack_poison > 0:
        st.warning("Blue team controls did not fully remove exposed forbidden claims.")
    else:
        st.success("No configured forbidden claims are exposed in the attacked or defended context.")


def expectation_panel(result):
    expectation_result = evaluate_expectations(result)
    if expectation_result["passed"]:
        st.success("Preflight expectations pass for the current attack and defense settings.")
    else:
        st.error("Preflight expectations fail for the current attack and defense settings.")

    with st.expander("Ground truth vs. outcome checks", expanded=False):
        st.dataframe(expectation_result["checks"], width="stretch", hide_index=True)


def run_all_scenario_duels(scenarios, mode, ollama_model_red, ollama_model_blue, seed):
    rows = []
    logs = []
    for index, path in enumerate(scenarios):
        scenario = load_scenario(path)
        max_steps = len(scenario.get("attacks", [])) + len(scenario.get("defenses", []))
        duel = run_agent_duel_steps(
            scenario_path=str(path),
            query=scenario.get("query", ""),
            top_k=scenario.get("top_k", 1),
            hop_depth=scenario.get("hop_depth", 1),
            attacks=scenario.get("attacks", []),
            defenses=scenario.get("defenses", []),
            mock_answer=scenario.get("mock_answer", ""),
            steps=max_steps,
            mode=mode,
            ollama_model_red=ollama_model_red,
            ollama_model_blue=ollama_model_blue,
            seed=f"{seed}:scenario:{index}",
        )
        result = duel["result"]
        rows.append({
            "Scenario": scenario.get("title", path.stem),
            "Type": scenario.get("test_type", "unspecified"),
            "Steps": duel["steps"],
            "Attack recall": f"{result['adversarial']['metrics']['recall']:.2f}",
            "Defended recall": f"{result['defended']['metrics']['recall']:.2f}",
            "Attack poison": f"{result['poison_exposure']['score']:.2f}",
            "Defended poison": f"{result['defended_poison_exposure']['score']:.2f}",
        })
        for log_row in duel["log"]:
            logs.append({"scenario": scenario.get("title", path.stem), **log_row})
    return rows, logs


def global_action_pool(paths):
    attacks = []
    defenses = []
    for path in paths:
        scenario = load_scenario(path)
        scenario_title = scenario.get("title", Path(path).stem)
        for attack in scenario.get("attacks", []):
            action = dict(attack)
            action["label"] = f"{scenario_title}: {attack.get('label', attack.get('type', 'attack'))}"
            action["source_scenario"] = scenario.get("id", Path(path).stem)
            attacks.append(action)
        for defense in scenario.get("defenses", []):
            action = dict(defense)
            action["label"] = f"{scenario_title}: {defense.get('label', defense.get('type', 'defense'))}"
            action["source_scenario"] = scenario.get("id", Path(path).stem)
            defenses.append(action)
    return attacks, defenses


def agent_duel_panel(selected, scenario, query, top_k, hop_depth, attacks, defenses, mock_answer, show_edge_labels):
    st.header("Agent Duel")
    st.caption(
        "Run a constrained red-team / blue-team duel. Agents choose from scenario-approved options; they never execute code or mutate the graph directly."
    )

    mode = st.radio("Agent mode", ["Agent-selected", "Hybrid Ollama"], horizontal=True)
    duel_scope = st.radio("Duel scope", ["Current scenario", "All scenarios"], horizontal=True)
    action_pool = st.radio("Action pool", ["Current scenario options", "All scenario options"], horizontal=True)
    ollama_model_red = DEFAULT_OLLAMA_RED
    ollama_model_blue = DEFAULT_OLLAMA_BLUE
    if mode == "Hybrid Ollama":
        model_cols = st.columns(2)
        with model_cols[0]:
            ollama_model_red = st.text_input(
                "Red team Ollama model (attacker)",
                value=DEFAULT_OLLAMA_RED,
                help="Llama family default: separate weights from blue; no shared chat with blue.",
            )
        with model_cols[1]:
            ollama_model_blue = st.text_input(
                "Blue team Ollama model (defender)",
                value=DEFAULT_OLLAMA_BLUE,
                help="Qwen default: separate weights from red; each turn is a stateless API call.",
            )
        st.caption(
            "Red and blue use different model names and independent /api/generate requests—no shared "
            "conversation state. If Ollama is down or a model is missing, the duel falls back to "
            "deterministic local selection."
        )

    if action_pool == "All scenario options":
        duel_attacks, duel_defenses = global_action_pool(scenario_options())
    else:
        duel_attacks, duel_defenses = scenario.get("attacks", []), scenario.get("defenses", [])

    max_steps = len(duel_attacks) + len(duel_defenses)
    step_key = session_key(scenario, "duel-steps")
    autoplay_key = session_key(scenario, "duel-autoplay")
    seed_key = session_key(scenario, "duel-seed")
    gauntlet_key = session_key(scenario, "duel-gauntlet")
    if step_key not in st.session_state:
        st.session_state[step_key] = 0
    if autoplay_key not in st.session_state:
        st.session_state[autoplay_key] = False
    if seed_key not in st.session_state:
        st.session_state[seed_key] = random.randint(1, 1_000_000)

    delay_seconds = st.slider("Autoplay delay between agents", min_value=1, max_value=5, value=2)
    controls = st.columns(4)
    with controls[0]:
        if st.button("Run next agent", disabled=st.session_state[step_key] >= max_steps):
            st.session_state[step_key] = min(st.session_state[step_key] + 1, max_steps)
    with controls[1]:
        if st.button("Run next full turn", disabled=st.session_state[step_key] >= max_steps):
            st.session_state[step_key] = min(st.session_state[step_key] + 2, max_steps)
    with controls[2]:
        if st.button("Auto-play duel", disabled=max_steps == 0 or st.session_state[step_key] >= max_steps):
            st.session_state[autoplay_key] = True
    with controls[3]:
        if st.button("Reset duel"):
            st.session_state[step_key] = 0
            st.session_state[autoplay_key] = False
            st.session_state[seed_key] = random.randint(1, 1_000_000)
            st.session_state.pop(gauntlet_key, None)

    if duel_scope == "All scenarios":
        st.subheader("All-Scenario Duel Gauntlet")
        st.caption("Click once to run every built-in scenario. Results are stored until you reset or rerun the gauntlet.")
        if mode == "Hybrid Ollama":
            st.warning("Hybrid Ollama can take longer because every agent action asks the local model to choose from options.")

        gauntlet_cols = st.columns(2)
        with gauntlet_cols[0]:
            if st.button("Run all-scenario gauntlet"):
                with st.spinner("Running all scenarios..."):
                    st.session_state[gauntlet_key] = run_all_scenario_duels(
                        scenario_options(),
                        mode,
                        ollama_model_red,
                        ollama_model_blue,
                        st.session_state[seed_key],
                    )
        with gauntlet_cols[1]:
            if st.button("Clear gauntlet results"):
                st.session_state.pop(gauntlet_key, None)

        if gauntlet_key in st.session_state:
            all_rows, all_logs = st.session_state[gauntlet_key]
            st.dataframe(all_rows, width="stretch", hide_index=True)
            with st.expander("All-scenario interaction log", expanded=False):
                st.dataframe(all_logs, width="stretch", hide_index=True)
        else:
            st.info("Gauntlet has not run yet. Click 'Run all-scenario gauntlet' to start.")
        st.info("Switch Duel scope back to Current scenario to step through one live battle graph at a time.")
        return

    duel = run_agent_duel_steps(
        scenario_path=str(selected),
        query=query,
        top_k=top_k,
        hop_depth=hop_depth,
        attacks=duel_attacks,
        defenses=duel_defenses,
        mock_answer=mock_answer,
        steps=st.session_state[step_key],
        mode=mode,
        ollama_model_red=ollama_model_red,
        ollama_model_blue=ollama_model_blue,
        seed=st.session_state[seed_key],
    )
    duel_result = duel["result"]
    st.progress(duel["steps"] / duel["max_steps"] if duel["max_steps"] else 0.0)
    st.caption(
        f"Duel progress: {duel['steps']} / {duel['max_steps']} agent actions. "
        f"Current actor: {duel['current_agent']}."
    )
    st.caption(f"Duel seed: {st.session_state[seed_key]} (reset to generate a different path).")
    st.caption(f"Action pool: {len(duel_attacks)} attack options / {len(duel_defenses)} defense options.")

    duel_cols = st.columns([1.2, 1])
    with duel_cols[0]:
        duel_removed_edges = {
            (attack["source"], attack["target"])
            for attack in duel_result["attacks"]
            if attack["type"] == "remove_edge" and attack["success"]
        } - edge_pairs(duel_result["defended_graph"])
        if duel["current_agent"] == "red":
            duel_graph = duel_result["attacked_graph"]
            duel_retrieval = duel_result["adversarial"]
            duel_title = "Duel graph after red action"
        elif duel["current_agent"] == "blue":
            duel_graph = duel_result["defended_graph"]
            duel_retrieval = duel_result["defended"]
            duel_title = "Duel graph after blue response"
        else:
            duel_graph = duel_result["baseline_graph"]
            duel_retrieval = duel_result["baseline"]
            duel_title = "Duel graph at baseline"
        st.plotly_chart(
            build_graph_figure(
                duel_graph,
                duel_title,
                removed_edges=duel_removed_edges,
                ground_truth_nodes=duel_result["scenario"].get("ground_truth_nodes", []),
                retrieved_nodes=duel_retrieval["nodes"],
                show_edge_labels=show_edge_labels,
            ),
            width="stretch",
        )
    with duel_cols[1]:
        st.subheader("Duel Scoreboard")
        st.metric("Attack recall", f"{duel_result['adversarial']['metrics']['recall']:.2f}")
        st.metric("Defended recall", f"{duel_result['defended']['metrics']['recall']:.2f}")
        st.metric("Attack poison", f"{duel_result['poison_exposure']['score']:.2f}")
        st.metric("Defended poison", f"{duel_result['defended_poison_exposure']['score']:.2f}")

    st.subheader("Agent Interaction Log")
    if duel["log"]:
        st.dataframe(duel["log"], width="stretch", hide_index=True)
    else:
        st.info("Run one duel turn to generate the first red/blue interaction.")

    if st.session_state[autoplay_key] and st.session_state[step_key] < max_steps:
        time.sleep(delay_seconds)
        st.session_state[step_key] = min(st.session_state[step_key] + 1, max_steps)
        st.rerun()
    elif st.session_state[autoplay_key]:
        st.session_state[autoplay_key] = False


def main():
    st.set_page_config(page_title="graphdversary", layout="wide")
    st.title("graphdversary")
    st.caption("Interactive adversarial GraphRAG demo: compare baseline retrieval with graph and query tampering.")

    scenarios = scenario_options()
    selected = st.sidebar.selectbox(
        "Scenario",
        scenarios,
        index=0,
        format_func=lambda path: load_scenario(path).get("title", Path(path).stem),
    )
    scenario = load_scenario(selected)
    test_type = scenario.get("test_type", "unspecified")

    st.sidebar.header("Retrieval Controls")
    query = st.sidebar.text_area("Query", value=scenario.get("query", ""), height=90)
    top_k = st.sidebar.slider("Semantic anchors (top_k)", min_value=1, max_value=5, value=scenario.get("top_k", 1))
    hop_depth = st.sidebar.slider("Graph expansion depth", min_value=0, max_value=3, value=scenario.get("hop_depth", 1))

    attacks, defenses = paired_controls(scenario)

    st.sidebar.header("Graph Controls")
    graph_phase = st.sidebar.radio(
        "Graph state",
        ["Baseline", "Red team attack", "Blue team defended"],
        index=1,
    )
    show_edge_labels = st.sidebar.toggle("Show edge labels", value=False)

    with st.sidebar.expander("Mock answer"):
        mock_answer = st.text_area("Answer to score", value=scenario.get("mock_answer", ""), height=140)

    result = run_live_scenario(str(selected), query, top_k, hop_depth, attacks, defenses, mock_answer)

    baseline = result["baseline"]
    adversarial = result["adversarial"]
    defended = result["defended"]
    removed_edges = {
        (attack["source"], attack["target"])
        for attack in result["attacks"]
        if attack["type"] == "remove_edge" and attack["success"]
    }

    st.header(result["scenario"].get("title", "Scenario"))
    st.write(result["scenario"].get("description", ""))
    st.info(f"{test_type.upper()} test: {scenario.get('test_type_reason', 'No classification rationale provided.')}")
    goal_cols = st.columns(2)
    with goal_cols[0]:
        st.markdown("**Red-team goal**")
        st.write(scenario.get("red_team_goal", "No red-team goal configured."))
    with goal_cols[1]:
        st.markdown("**Blue-team goal**")
        st.write(scenario.get("blue_team_goal", "No blue-team goal configured."))
    st.caption(
        "Every sidebar interaction reruns the scenario. Attack and defense checkboxes update the metrics, graph state, timelines, and retrieval traces."
    )
    st.caption(
        "Graph legend: numbers map to the node table below; thick black outline = ground truth; larger marker = retrieved; dashed red edge = removed attack edge."
    )

    metric_cols = st.columns(6)
    with metric_cols[0]:
        metric_card("Baseline recall", baseline["metrics"]["recall"])
    with metric_cols[1]:
        st.metric(
            "Attack recall",
            f"{adversarial['metrics']['recall']:.2f}",
            delta_label(baseline["metrics"]["recall"], adversarial["metrics"]["recall"]),
        )
    with metric_cols[2]:
        st.metric(
            "Defended recall",
            f"{defended['metrics']['recall']:.2f}",
            delta_label(adversarial["metrics"]["recall"], defended["metrics"]["recall"]),
        )
    with metric_cols[3]:
        metric_card("Attack poison", result["poison_exposure"]["score"])
    with metric_cols[4]:
        st.metric(
            "Defended poison",
            f"{result['defended_poison_exposure']['score']:.2f}",
            delta_label(result["poison_exposure"]["score"], result["defended_poison_exposure"]["score"]),
            delta_color="inverse",
        )
    with metric_cols[5]:
        metric_card("Defended precision", defended["metrics"]["precision"])

    graph_config = {
        "Baseline": (result["baseline_graph"], baseline, set(), "Baseline evidence graph"),
        "Red team attack": (result["attacked_graph"], adversarial, removed_edges, "Red-team attacked graph"),
        "Blue team defended": (result["defended_graph"], defended, set(), "Blue-team defended graph"),
    }
    graph_snapshot, graph_result, graph_removed_edges, graph_title = graph_config[graph_phase]
    active_attack_list = active_labels(attacks)
    active_defense_list = active_labels(defenses)

    overview_tab, live_tab, duel_tab, graphs_tab, traces_tab, tests_tab = st.tabs([
        "Overview",
        "Realtime Graph",
        "Agent Duel",
        "Scenario Graphs",
        "Traces & Logs",
        "Tests",
    ])

    with overview_tab:
        outcome_panel(result)
        with st.expander("Demo flow", expanded=True):
            st.write("1. Start on Baseline and explain the trusted evidence path.")
            st.write("2. Switch to Red team attack and toggle attacks one by one.")
            st.write("3. Switch to Blue team defended and toggle defenses one by one.")
            st.write("4. Use the retrieval traces to show why recall and poison exposure changed.")
            st.write(f"Active attacks: {', '.join(active_attack_list) if active_attack_list else 'none'}")
            st.write(f"Active defenses: {', '.join(active_defense_list) if active_defense_list else 'none'}")

        st.header("Generated Answer Check")
        st.write(result["answer"])
        if result["poison_exposure"]["matches"]:
            st.warning("Forbidden claims exposed: " + ", ".join(result["poison_exposure"]["matches"]))
        else:
            st.success("No configured forbidden claims appeared in the retrieved context.")
        if result["defended_poison_exposure"]["matches"]:
            st.warning("Defended context still exposes: " + ", ".join(result["defended_poison_exposure"]["matches"]))
        else:
            st.success("Blue-team defended context does not expose configured forbidden claims.")

        with st.expander("Presenter notes"):
            for note in scenario.get("presenter_notes", []):
                st.write(f"- {note}")

    with live_tab:
        live_removed_edges = removed_edges - edge_pairs(result["defended_graph"])
        st.header("Realtime Interaction Graph")
        st.caption(
            "This graph always reflects the current sidebar state after enabled red-team actions and enabled blue-team defenses are applied."
        )
        live_cols = st.columns([1.2, 1])
        with live_cols[0]:
            st.plotly_chart(
                build_graph_figure(
                    result["defended_graph"],
                    "Current live graph",
                    removed_edges=live_removed_edges,
                    ground_truth_nodes=result["scenario"].get("ground_truth_nodes", []),
                    retrieved_nodes=defended["nodes"],
                    show_edge_labels=show_edge_labels,
                ),
                width="stretch",
            )
        with live_cols[1]:
            st.subheader("Current State")
            st.write(f"Active red-team actions: {len(active_attack_list)}")
            for label in active_attack_list or ["none"]:
                st.write(f"- {label}")
            st.write(f"Active blue-team defenses: {len(active_defense_list)}")
            for label in active_defense_list or ["none"]:
                st.write(f"- {label}")
            st.write(f"Current retrieved nodes: {', '.join(defended['nodes']) if defended['nodes'] else 'none'}")
            if result["defended_poison_exposure"]["matches"]:
                st.warning("Live graph still exposes forbidden claims.")
            else:
                st.success("Live graph has no configured forbidden-claim exposure.")

    with duel_tab:
        agent_duel_panel(selected, scenario, query, top_k, hop_depth, attacks, defenses, mock_answer, show_edge_labels)

    with graphs_tab:
        st.header("Scenario Graphs")
        st.caption("Use the sidebar graph selector to compare Baseline, Red team attack, and Blue team defended states.")
        st.plotly_chart(
            build_graph_figure(
                graph_snapshot,
                graph_title,
                removed_edges=graph_removed_edges,
                ground_truth_nodes=result["scenario"].get("ground_truth_nodes", []),
                retrieved_nodes=graph_result["nodes"],
                show_edge_labels=show_edge_labels,
            ),
            width="stretch",
        )

        table_cols = st.columns([1.4, 1])
        with table_cols[0]:
            st.subheader("Node map")
            st.dataframe(
                node_table(graph_snapshot, result["scenario"].get("ground_truth_nodes", []), graph_result["nodes"]),
                width="stretch",
                hide_index=True,
            )
        with table_cols[1]:
            st.subheader("Edges")
            st.dataframe(edge_table(graph_snapshot), width="stretch", hide_index=True)

    with traces_tab:
        timeline_cols = st.columns(2)
        with timeline_cols[0]:
            st.header("Red-Team Timeline")
            if result["attacks"]:
                st.dataframe(result["attacks"], width="stretch", hide_index=True)
            else:
                st.info("No attacks are currently enabled.")
        with timeline_cols[1]:
            st.header("Blue-Team Controls")
            if result["defenses"]:
                st.dataframe(result["defenses"], width="stretch", hide_index=True)
            else:
                st.info("No defenses are currently enabled.")

        context_cols = st.columns(3)
        with context_cols[0]:
            context_panel("Baseline retrieval", baseline)
        with context_cols[1]:
            context_panel("Red-team retrieval", adversarial)
        with context_cols[2]:
            context_panel("Blue-team retrieval", defended)

    with tests_tab:
        expectation_panel(result)


if __name__ == "__main__":
    main()

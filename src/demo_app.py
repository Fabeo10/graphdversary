"""
Interactive Streamlit demo for graphdversary.
"""

from pathlib import Path
import contextlib
import difflib
import html
import random
import sys
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

from src.pipeline import (
    DEFAULT_SCENARIO_PATH,
    PROJECT_ROOT,
    load_scenario,
    run_scenario,
    unrestored_remove_edge_attacks,
)
from src.agent_duel import (
    DEFAULT_OLLAMA_BLUE,
    DEFAULT_OLLAMA_RED,
    duel_input_signature,
    run_agent_duel_steps,
)
import src.agent_duel as _agent_duel


def _style_interaction_log(df):
    """Green/red styling for ``blue_defense_result`` (hit/miss)."""

    def _color_hit_miss(col):
        styles = []
        for val in col:
            if val == "hit":
                styles.append("background-color: #d4edda; color: #155724; font-weight: 600")
            elif val == "miss":
                styles.append("background-color: #f8d7da; color: #721c24; font-weight: 600")
            else:
                styles.append("")
        return styles

    if "blue_defense_result" not in df.columns:
        return df
    return df.style.apply(_color_hit_miss, subset=["blue_defense_result"])


def _interaction_log_display(log_rows):
    """Order columns for readability; style blue-team hit/miss."""
    if not log_rows:
        return None
    priority = [
        "step",
        "turn",
        "agent",
        "attack_defense_types",
        "blue_defense_result",
        "action",
        "rationale",
        "recall",
        "poison_exposure",
        "available_options",
    ]
    df = pd.DataFrame(log_rows)
    ordered = [c for c in priority if c in df.columns]
    ordered.extend(c for c in df.columns if c not in ordered)
    df = df[ordered]
    return _style_interaction_log(df)


def _diff_chunks(old, new):
    """Word-level diff between two query strings.

    Returns a list of ``(tag, text)`` chunks where ``tag`` is ``"equal"``,
    ``"added"`` (in ``new`` but not ``old``), or ``"removed"`` (in ``old`` but
    not ``new``). Whitespace collapses to single spaces — fine for prose
    queries and keeps the rendering predictable.
    """
    old_words = old.split()
    new_words = new.split()
    matcher = difflib.SequenceMatcher(a=old_words, b=new_words, autojunk=False)
    chunks = []
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            chunks.append(("equal", " ".join(old_words[i1:i2])))
        elif op == "delete":
            chunks.append(("removed", " ".join(old_words[i1:i2])))
        elif op == "insert":
            chunks.append(("added", " ".join(new_words[j1:j2])))
        elif op == "replace":
            chunks.append(("removed", " ".join(old_words[i1:i2])))
            chunks.append(("added", " ".join(new_words[j1:j2])))
    return chunks


_ADDED_STYLE = (
    "background:#ffe1de;color:#7a1a00;padding:1px 5px;border-radius:3px;"
    "font-weight:600;"
    # ``box-decoration-break: clone`` makes each wrapped line of the highlight get its
    # own padding and rounded corners — without it, long appended phrases look like a
    # broken rectangle that hugs only the first line.
    "box-decoration-break:clone;-webkit-box-decoration-break:clone;"
)
_REMOVED_STYLE = (
    "background:#e3eaf5;color:#1d3557;padding:1px 5px;border-radius:3px;"
    "text-decoration:line-through;"
    "box-decoration-break:clone;-webkit-box-decoration-break:clone;"
)


def _render_diff_html(old, new):
    """Render a word diff as inline HTML — additions highlighted red, removals struck through blue."""
    parts = []
    for tag, text in _diff_chunks(old, new):
        if not text:
            continue
        escaped = html.escape(text)
        if tag == "equal":
            parts.append(escaped)
        elif tag == "added":
            parts.append(f'<span style="{_ADDED_STYLE}">{escaped}</span>')
        elif tag == "removed":
            parts.append(f'<span style="{_REMOVED_STYLE}">{escaped}</span>')
    return " ".join(parts)


def _query_attack_summary(attack_log):
    """Compact human description of any query-affecting attacks that ran."""
    items = []
    for entry in attack_log or []:
        if entry.get("type") == "perturb_query" and entry.get("success"):
            label = entry.get("label") or "Perturb query"
            ptype = entry.get("perturbation_type")
            items.append(f"{label} ({ptype})" if ptype else label)
    return items


def _query_defense_summary(defense_log):
    """Compact human description of any query-affecting defenses that ran."""
    items = []
    for entry in defense_log or []:
        if entry.get("type") != "sanitize_query":
            continue
        label = entry.get("label") or "Sanitize query"
        removed = entry.get("removed_phrases") or []
        if removed:
            items.append(f"{label} — stripped {len(removed)} phrase(s)")
        else:
            items.append(f"{label} — enabled but no matching phrase to strip")
    return items


def render_query_journey(result):
    """Show the original → after-red → after-blue query path with inline diffs.

    Demo intent: query-perturbation attacks and query-sanitization defenses are
    invisible on the graph view (they don't add/remove nodes), so without this
    panel the audience has no way to see *what* the red team added or *what*
    the blue team stripped. The panel always renders the original query; the
    red-team and blue-team rows only appear if that team actually mutated the
    query during this run.
    """
    baseline_q = result["baseline"]["query"]
    adv_q = result["adversarial"]["query"]
    def_q = result["defended"]["query"]

    red_changed = adv_q != baseline_q
    blue_changed = def_q != adv_q

    st.markdown("**Query journey**")
    if not red_changed and not blue_changed:
        st.caption("No query-perturbation attacks or query-sanitization defenses ran — retrieval used the original sidebar query.")
    else:
        st.caption(
            "Red-team additions appear in a red highlight; blue-team removals appear with a blue strike-through. "
            "Graph-only attacks (edge removal, poison node) don't show up here — they're already visible on the chart."
        )

    rows_html = [
        '<div style="padding:8px 12px;margin:4px 0;border-left:4px solid #888;'
        'background:#fafafa;border-radius:4px;">'
        '<div style="color:#666;font-size:0.78em;font-weight:600;letter-spacing:0.05em;">ORIGINAL</div>'
        f'<div style="margin-top:4px;font-family:ui-monospace,Menlo,monospace;">{html.escape(baseline_q)}</div>'
        '</div>'
    ]

    if red_changed:
        sources = _query_attack_summary(result.get("attacks", []))
        source_caption = (
            f'<div style="color:#7a1a00;font-size:0.78em;margin-top:4px;">'
            f'via {html.escape("; ".join(sources))}</div>'
            if sources
            else ""
        )
        rows_html.append(
            '<div style="padding:8px 12px;margin:4px 0;border-left:4px solid #D62728;'
            'background:#fff5f4;border-radius:4px;">'
            '<div style="color:#7a1a00;font-size:0.78em;font-weight:600;letter-spacing:0.05em;">AFTER RED TEAM</div>'
            f'<div style="margin-top:4px;font-family:ui-monospace,Menlo,monospace;">{_render_diff_html(baseline_q, adv_q)}</div>'
            f"{source_caption}"
            "</div>"
        )

    if blue_changed:
        sources = _query_defense_summary(result.get("defenses", []))
        source_caption = (
            f'<div style="color:#1d3557;font-size:0.78em;margin-top:4px;">'
            f'via {html.escape("; ".join(sources))}</div>'
            if sources
            else ""
        )
        rows_html.append(
            '<div style="padding:8px 12px;margin:4px 0;border-left:4px solid #1f77b4;'
            'background:#f0f6fc;border-radius:4px;">'
            '<div style="color:#1d3557;font-size:0.78em;font-weight:600;letter-spacing:0.05em;">AFTER BLUE TEAM</div>'
            f'<div style="margin-top:4px;font-family:ui-monospace,Menlo,monospace;">{_render_diff_html(adv_q, def_q)}</div>'
            f"{source_caption}"
            "</div>"
        )
    elif red_changed:
        rows_html.append(
            '<div style="padding:6px 12px;margin:4px 0;color:#666;font-size:0.85em;">'
            "Blue team did not modify the query — final retrieval used the adversarial text above."
            "</div>"
        )

    st.markdown("".join(rows_html), unsafe_allow_html=True)


def duel_effective_max_steps(attacks, defenses):
    """Same rule as ``agent_duel.duel_effective_max_steps``; wrapper avoids import errors on mixed repo versions.

    Round-based model: each round is one red attack (without replacement) plus one blue defense
    (with replacement). Duel length = ``2 * len(attacks)``; needs at least one defense for blue to respond.
    """
    fn = getattr(_agent_duel, "duel_effective_max_steps", None)
    if callable(fn):
        return fn(attacks, defenses)
    if not attacks or not defenses:
        return 0
    return 2 * len(attacks)
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


def merge_scenario_node_names(scenario):
    """Merge per-scenario ``node_names`` (``{id: label}``) into the global map.

    Scenarios that ship their own display names override the built-in defaults so
    new scenarios don't render raw IDs like ``n11`` once their JSON gains a
    ``node_names`` block.
    """
    names = scenario.get("node_names") if isinstance(scenario, dict) else None
    if not isinstance(names, dict):
        return
    for node_id, label in names.items():
        if isinstance(node_id, str) and isinstance(label, str) and label:
            NODE_NAMES[node_id] = label


def display_name(node, fallback_id=None):
    """Resolve a node's display name.

    Lookup order:

    1. ``node["name"]`` when ``node`` is a snapshot dict carrying its own label.
    2. The (scenario-merged) ``NODE_NAMES`` map.
    3. The raw node id — guarantees we never render ``None``.
    """
    if isinstance(node, dict):
        name = node.get("name")
        if name:
            return name
        node_id = node.get("id", fallback_id or "")
    else:
        node_id = node if node is not None else (fallback_id or "")
    return NODE_NAMES.get(node_id, node_id)


def scenario_options():
    scenario_dir = PROJECT_ROOT / "data" / "scenarios"
    scenarios = sorted(scenario_dir.glob("*.json"))
    return scenarios or [DEFAULT_SCENARIO_PATH]


def _freeze(value):
    """Recursively convert dicts/lists into hashable tuples for cache keying."""
    if isinstance(value, dict):
        return tuple(sorted((str(k), _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze(v) for v in value))
    return value


def _freeze_actions(actions):
    """Frozen tuple-of-tuples form of an attack/defense list for use as a cache key."""
    return tuple(_freeze(item) for item in (actions or []))


@st.cache_data(show_spinner=False)
def _run_live_scenario_cached(
    scenario_path,
    query,
    top_k,
    hop_depth,
    attacks_key,
    defenses_key,
    mock_answer,
    _attacks,
    _defenses,
):
    """Cached scenario run. ``attacks_key`` / ``defenses_key`` are the hashable
    fingerprints that drive the cache; ``_attacks`` / ``_defenses`` (underscore-
    prefixed so Streamlit skips them when hashing) carry the original dict shape
    that ``run_scenario`` expects."""
    return run_scenario(
        scenario_path=scenario_path,
        query=query,
        top_k=top_k,
        hop_depth=hop_depth,
        attacks=_attacks,
        defenses=_defenses,
        mock_answer=mock_answer,
        verbose=False,
    )


def run_live_scenario(scenario_path, query, top_k, hop_depth, attacks, defenses, mock_answer):
    """Run on every interaction so red/blue checkbox changes always refresh the graph.

    Memoized: identical sidebar state replays from cache instead of rerunning the
    full pipeline, so checkbox toggles don't freeze the demo for several seconds.
    """
    return _run_live_scenario_cached(
        scenario_path,
        query,
        top_k,
        hop_depth,
        _freeze_actions(attacks),
        _freeze_actions(defenses),
        mock_answer,
        _attacks=attacks,
        _defenses=defenses,
    )


def metric_card(label, value):
    st.metric(label, f"{value:.2f}")


def delta_label(before, after):
    delta = after - before
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.2f}"


def active_labels(items):
    return [item.get("label", item.get("type", "Unnamed")) for item in items if item.get("enabled", False)]


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
                value=False,
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
                value=False,
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
    names = {node_id: display_name(node, node_id) for node_id, node in nodes.items()}

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
                text=f"{names.get(source, source)} --{edge.get('relation', '')}--> {names.get(target, target)}",
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
            f"{node_numbers[node_id]}. {names.get(node_id, node_id)}<br>{node_id}<br>{node_type}<br>{content_preview(attrs.get('content', ''), 140)}"
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


def edge_pairs(snapshot):
    return {(edge["source"], edge["target"]) for edge in snapshot["edges"]}


def duel_snapshot_figure(snapshot, show_edge_labels):
    """Plotly figure for one duel step (baseline, post-red attacked view, or post-blue defended view)."""
    kind = snapshot["display_kind"]
    result = snapshot["result"]
    duel_removed_edges = unrestored_remove_edge_attacks(result)
    gt = result["scenario"].get("ground_truth_nodes", [])
    if kind == "baseline":
        graph = result["baseline_graph"]
        retr = result["baseline"]
        title = f"Step {snapshot['step']} — Baseline"
    elif kind == "attacked":
        graph = result["attacked_graph"]
        retr = result["adversarial"]
        title = f"Step {snapshot['step']} — After red · {snapshot.get('action_label', '')}"
    else:
        graph = result["defended_graph"]
        retr = result["defended"]
        title = f"Step {snapshot['step']} — After blue · {snapshot.get('action_label', '')}"
    return build_graph_figure(
        graph,
        title,
        removed_edges=duel_removed_edges,
        ground_truth_nodes=gt,
        retrieved_nodes=retr["nodes"],
        show_edge_labels=show_edge_labels,
    )


def session_key(scenario, suffix):
    return f"{scenario.get('id', 'scenario')}-{suffix}"


def run_all_scenario_duels(scenarios, mode, ollama_model_red, ollama_model_blue, seed):
    rows = []
    logs = []
    for index, path in enumerate(scenarios):
        scenario = load_scenario(path)
        atk = scenario.get("attacks", [])
        dfn = scenario.get("defenses", [])
        max_steps = duel_effective_max_steps(atk, dfn)
        duel = run_agent_duel_steps(
            scenario_path=str(path),
            query=scenario.get("query", ""),
            top_k=scenario.get("top_k", 1),
            hop_depth=scenario.get("hop_depth", 1),
            attacks=atk,
            defenses=dfn,
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
            "Attack precision": f"{result['adversarial']['metrics']['precision']:.2f}",
            "Defended recall": f"{result['defended']['metrics']['recall']:.2f}",
            "Defended precision": f"{result['defended']['metrics']['precision']:.2f}",
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
        "**Round model:** red picks an attack **without replacement** (each one fires at most once per duel); "
        "blue picks a defense **with replacement** every round, like an analyst who can reuse any control. "
        "Each round = one red turn + one blue turn. **Hit/miss is per round** — a defense that would have countered "
        "an earlier round's attack still counts as a miss this round. Duel ends when red exhausts its arsenal."
    )

    mode = st.radio("Agent mode", ["Agent-selected", "Hybrid Ollama"], horizontal=True)
    duel_scope = st.radio("Duel scope", ["Current scenario", "All scenarios"], horizontal=True)
    action_pool = st.radio(
        "Action pool",
        ["Current scenario options", "All scenario options"],
        horizontal=True,
        index=1,
        help="All scenario options merges every built-in scenario’s attacks and defenses so agents can choose any of them in turn.",
    )
    ollama_model_red = DEFAULT_OLLAMA_RED
    ollama_model_blue = DEFAULT_OLLAMA_BLUE
    if mode == "Hybrid Ollama":
        red_wkey = session_key(scenario, "duel-ollama-red-input")
        blue_wkey = session_key(scenario, "duel-ollama-blue-input")
        if red_wkey not in st.session_state:
            st.session_state[red_wkey] = DEFAULT_OLLAMA_RED
        if blue_wkey not in st.session_state:
            st.session_state[blue_wkey] = DEFAULT_OLLAMA_BLUE

        model_cols = st.columns([3, 1, 3])
        # Middle column must run *before* the text inputs: widget keys lock session_state and
        # Streamlit forbids mutating those keys after the widgets are instantiated.
        with model_cols[1]:
            st.write("")
            if st.button(
                "⇄",
                key=session_key(scenario, "swap-llm-teams"),
                help="Swap models: red ↔ blue (e.g. qwen2.5:3b and llama3.2:3b exchange teams). Duel checkpoint resets when model names change.",
                use_container_width=True,
            ):
                st.session_state[red_wkey], st.session_state[blue_wkey] = (
                    st.session_state[blue_wkey],
                    st.session_state[red_wkey],
                )
        with model_cols[0]:
            ollama_model_red = st.text_input(
                "Red team Ollama model (attacker)",
                key=red_wkey,
                help="Llama family default: separate weights from blue; no shared chat with blue.",
            )
        with model_cols[2]:
            ollama_model_blue = st.text_input(
                "Blue team Ollama model (defender)",
                key=blue_wkey,
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

    max_steps = duel_effective_max_steps(duel_attacks, duel_defenses)
    step_key = session_key(scenario, "duel-steps")
    autoplay_key = session_key(scenario, "duel-autoplay")
    autoplay_next_at_key = session_key(scenario, "duel-autoplay-next-at")
    seed_key = session_key(scenario, "duel-seed")
    gauntlet_key = session_key(scenario, "duel-gauntlet")
    play_until_key = session_key(scenario, "duel-play-until")
    catchup_runs_key = session_key(scenario, "duel-catchup-runs")
    if step_key not in st.session_state:
        st.session_state[step_key] = 0
    else:
        st.session_state[step_key] = min(int(st.session_state[step_key]), max_steps)
    if autoplay_key not in st.session_state:
        st.session_state[autoplay_key] = False
    if seed_key not in st.session_state:
        st.session_state[seed_key] = random.randint(1, 1_000_000)

    delay_seconds = st.slider("Autoplay delay between agents", min_value=1, max_value=5, value=2)
    controls = st.columns(4)
    with controls[0]:
        if st.button("Run next agent", disabled=st.session_state[step_key] >= max_steps):
            st.session_state.pop(play_until_key, None)
            st.session_state[step_key] = min(st.session_state[step_key] + 1, max_steps)
    with controls[1]:
        if st.button("Run next full turn", disabled=st.session_state[step_key] >= max_steps):
            st.session_state.pop(play_until_key, None)
            st.session_state[step_key] = min(st.session_state[step_key] + 2, max_steps)
    with controls[2]:
        if st.button("Auto-play duel", disabled=max_steps == 0 or st.session_state[step_key] >= max_steps):
            st.session_state.pop(play_until_key, None)
            st.session_state.pop(autoplay_next_at_key, None)
            st.session_state[autoplay_key] = True
    with controls[3]:
        if st.button("Reset duel"):
            st.session_state[step_key] = 0
            st.session_state[autoplay_key] = False
            st.session_state[seed_key] = random.randint(1, 1_000_000)
            st.session_state.pop(gauntlet_key, None)
            st.session_state.pop(session_key(scenario, "duel-checkpoint"), None)
            st.session_state.pop(play_until_key, None)
            st.session_state.pop(catchup_runs_key, None)
            st.session_state.pop(autoplay_next_at_key, None)

    full_battle = st.columns([1, 4])
    with full_battle[0]:
        if st.button(
            "Run full battle",
            disabled=max_steps == 0,
            help="Animates to the end one agent step per refresh so the graph updates live (same as rapid stepping).",
        ):
            st.session_state[play_until_key] = max_steps
            st.session_state[autoplay_key] = False
    with full_battle[1]:
        st.caption(
            "Runs one duel step per Streamlit rerun until finished—needed so Hybrid Ollama and the plot render "
            "between turns instead of blocking until the last step."
        )

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
                gauntlet_df = _interaction_log_display(all_logs)
                if gauntlet_df is not None:
                    st.dataframe(gauntlet_df, width="stretch", hide_index=True)
                else:
                    st.info("No log rows.")
        else:
            st.info("Gauntlet has not run yet. Click 'Run all-scenario gauntlet' to start.")
        st.info("Switch Duel scope back to Current scenario to step through one live battle graph at a time.")
        return

    duel_ck_key = session_key(scenario, "duel-checkpoint")
    duel_sig = duel_input_signature(
        str(selected),
        query,
        top_k,
        hop_depth,
        duel_attacks,
        duel_defenses,
        mock_answer,
        mode,
        ollama_model_red,
        ollama_model_blue,
        str(st.session_state[seed_key]),
    )
    prev_ck = st.session_state.get(duel_ck_key)
    sig_match = bool(prev_ck and prev_ck.get("signature") == duel_sig)
    if prev_ck and not sig_match:
        st.warning("Duel inputs changed versus saved checkpoint—starting duel state over.")
        st.session_state.pop(duel_ck_key, None)
        st.session_state.pop(catchup_runs_key, None)
        prev_ck = None

    completed = int(prev_ck["completed_steps"]) if prev_ck else 0
    play_until = st.session_state.get(play_until_key)
    user_target = int(st.session_state[step_key])
    catch_up_target = int(play_until) if play_until is not None else user_target

    # At most one new agent step per rerun so the UI can render between Hybrid Ollama calls.
    if prev_ck is None:
        resume_cp = None
        if play_until is not None:
            effective_steps = min(completed + 1, catch_up_target)
        elif user_target > 0:
            effective_steps = min(completed + 1, catch_up_target)
        else:
            effective_steps = 0
    elif completed < catch_up_target:
        effective_steps = completed + 1
        resume_cp = prev_ck if prev_ck.get("completed_steps") == effective_steps - 1 else None
    else:
        effective_steps = catch_up_target
        resume_cp = prev_ck if prev_ck.get("completed_steps") == effective_steps - 1 else None

    spin_ctx = (
        st.spinner("Hybrid Ollama: one agent step (incremental refresh)…")
        if mode == "Hybrid Ollama"
        else contextlib.nullcontext()
    )
    with spin_ctx:
        duel = run_agent_duel_steps(
            scenario_path=str(selected),
            query=query,
            top_k=top_k,
            hop_depth=hop_depth,
            attacks=duel_attacks,
            defenses=duel_defenses,
            mock_answer=mock_answer,
            steps=effective_steps,
            mode=mode,
            ollama_model_red=ollama_model_red,
            ollama_model_blue=ollama_model_blue,
            seed=st.session_state[seed_key],
            resume_checkpoint=resume_cp,
        )
    st.session_state[duel_ck_key] = duel["checkpoint"]

    done_steps = int(duel["checkpoint"]["completed_steps"])
    if play_until is not None and done_steps >= play_until:
        st.session_state.pop(play_until_key, None)
        st.session_state[step_key] = play_until

    duel_result = duel["result"]
    st.progress(duel["steps"] / duel["max_steps"] if duel["max_steps"] else 0.0)
    st.caption(
        f"Duel progress: {duel['steps']} / {duel['max_steps']} agent actions. "
        f"Current actor: {duel['current_agent']}."
    )
    st.caption(f"Duel seed: {st.session_state[seed_key]} (reset to generate a different path).")
    st.caption(
        f"Action pool: {len(duel_attacks)} red attacks (no repeats per duel) · "
        f"{len(duel_defenses)} blue defenses (full arsenal available every round). "
        f"Total agent steps when red exhausts: **{duel['max_steps']}** "
        f"({duel['max_turns']} rounds × 2 turns)."
    )

    st.subheader("Live duel graph")
    snaps = duel.get("snapshots") or []
    if snaps:
        latest = snaps[-1]
        st.caption(
            f"Step **{latest['step']}** · `{latest['agent']}` · {latest.get('action_label', '')}. "
            "The chart below **replaces in place** each time you advance a turn (or on auto-play reruns)—only the "
            "interaction log grows underneath."
        )
    else:
        latest = None

    duel_cols = st.columns([1.2, 1])
    with duel_cols[0]:
        if latest is not None:
            st.plotly_chart(
                duel_snapshot_figure(latest, show_edge_labels),
                width="stretch",
            )
        else:
            st.info("Advance the duel to render the graph.")
    with duel_cols[1]:
        st.markdown("**Scoreboard** (current pipeline state)")
        st.metric("Attack recall", f"{duel_result['adversarial']['metrics']['recall']:.2f}")
        st.metric("Attack precision", f"{duel_result['adversarial']['metrics']['precision']:.2f}")
        st.metric("Defended recall", f"{duel_result['defended']['metrics']['recall']:.2f}")
        st.metric("Defended precision", f"{duel_result['defended']['metrics']['precision']:.2f}")
        st.metric("Attack poison", f"{duel_result['poison_exposure']['score']:.2f}")
        st.metric("Defended poison", f"{duel_result['defended_poison_exposure']['score']:.2f}")

    render_query_journey(duel_result)

    st.subheader("Agent Interaction Log")
    if duel["log"]:
        st.caption(
            "**attack_defense_types** shows structural action types only (`attack · defense` context). "
            "**blue_defense_result** is **hit** (green) or **miss** (red) on blue turns—**hit** means the "
            "chosen defense *type* counters red's **last** attack *type* "
            "(``remove_edge``→``restore_protected_edges``; ``inject_poison``→``remove_untrusted_nodes`` or "
            "``block_forbidden_claim_nodes``; ``perturb_query``→``sanitize_query``), and the defense applied "
            "in the pipeline. Otherwise **miss**. "
            "Hybrid Ollama adds **llm_*** columns: prompt JSON (includes retrieval observation), "
            "**llm_observation_preview** (truncated copy of what the model saw: opponent move + context preview), "
            "and raw response text. Rule-based fallback rows leave LLM cells empty."
        )
        styled = _interaction_log_display(duel["log"])
        st.dataframe(styled, width="stretch", hide_index=True)
    else:
        st.info("Run one duel turn to generate the first red/blue interaction.")

    autoplay_status = st.empty()
    if st.session_state[autoplay_key] and st.session_state[step_key] < max_steps:
        now = time.time()
        next_at = st.session_state.get(autoplay_next_at_key)
        if next_at is None:
            next_at = now + delay_seconds
            st.session_state[autoplay_next_at_key] = next_at

        if now >= next_at:
            st.session_state.pop(autoplay_next_at_key, None)
            st.session_state[step_key] = min(st.session_state[step_key] + 1, max_steps)
            st.rerun()
        else:
            remaining = max(0.0, next_at - now)
            autoplay_status.info(
                f"Auto-playing duel — next agent in ~{remaining:.1f}s. "
                "Click **Reset duel** to stop."
            )
            # Short tick (not the full delay) so the chart/log render between agents
            # and the countdown stays responsive instead of freezing the browser tab.
            time.sleep(min(0.25, remaining))
            st.rerun()
    elif st.session_state[autoplay_key]:
        st.session_state[autoplay_key] = False
        st.session_state.pop(autoplay_next_at_key, None)
    elif done_steps < catch_up_target:
        runs = st.session_state.get(catchup_runs_key, 0) + 1
        if runs > max_steps + 25:
            st.session_state.pop(catchup_runs_key, None)
            st.session_state.pop(duel_ck_key, None)
            st.session_state.pop(play_until_key, None)
            st.error(
                "Catch-up stopped after too many refreshes (likely a signature/checkpoint mismatch). "
                "Try **Reset duel** and run again — duel fingerprints are now stable across reruns."
            )
        else:
            st.session_state[catchup_runs_key] = runs
            time.sleep(0.05)
            st.rerun()
    else:
        st.session_state.pop(catchup_runs_key, None)


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
    merge_scenario_node_names(scenario)
    test_type = scenario.get("test_type", "unspecified")

    st.sidebar.header("Retrieval Controls")
    query = st.sidebar.text_area("Query", value=scenario.get("query", ""), height=90)
    top_k = st.sidebar.slider("Semantic anchors (top_k)", min_value=1, max_value=5, value=scenario.get("top_k", 1))
    hop_depth = st.sidebar.slider("Graph expansion depth", min_value=0, max_value=3, value=scenario.get("hop_depth", 1))

    attacks, defenses = paired_controls(scenario)

    st.sidebar.header("Graph Controls")
    show_edge_labels = st.sidebar.toggle("Show edge labels", value=False)

    with st.sidebar.expander("Mock answer"):
        mock_answer = st.text_area("Answer to score", value=scenario.get("mock_answer", ""), height=140)

    try:
        result = run_live_scenario(str(selected), query, top_k, hop_depth, attacks, defenses, mock_answer)
    except Exception as e:
        st.error(f"Failed to run scenario: {e}")
        st.caption(
            "Adjust the sidebar (or pick a different scenario) and the app will retry. "
            "Full traceback is in the terminal running `streamlit run`."
        )
        st.stop()

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
        "Every sidebar interaction reruns the scenario. Attack and defense checkboxes update the metrics and graph state."
    )
    st.caption(
        "Graph legend: numbers map to the node table below; thick black outline = ground truth; larger marker = retrieved; dashed red edge = removed attack edge."
    )

    metric_cols = st.columns(8)
    with metric_cols[0]:
        metric_card("Baseline recall", baseline["metrics"]["recall"])
    with metric_cols[1]:
        metric_card("Baseline precision", baseline["metrics"]["precision"])
    with metric_cols[2]:
        st.metric(
            "Attack recall",
            f"{adversarial['metrics']['recall']:.2f}",
            delta_label(baseline["metrics"]["recall"], adversarial["metrics"]["recall"]),
        )
    with metric_cols[3]:
        st.metric(
            "Attack precision",
            f"{adversarial['metrics']['precision']:.2f}",
            delta_label(baseline["metrics"]["precision"], adversarial["metrics"]["precision"]),
        )
    with metric_cols[4]:
        st.metric(
            "Defended recall",
            f"{defended['metrics']['recall']:.2f}",
            delta_label(adversarial["metrics"]["recall"], defended["metrics"]["recall"]),
        )
    with metric_cols[5]:
        st.metric(
            "Defended precision",
            f"{defended['metrics']['precision']:.2f}",
            delta_label(adversarial["metrics"]["precision"], defended["metrics"]["precision"]),
        )
    with metric_cols[6]:
        metric_card("Attack poison", result["poison_exposure"]["score"])
    with metric_cols[7]:
        st.metric(
            "Defended poison",
            f"{result['defended_poison_exposure']['score']:.2f}",
            delta_label(result["poison_exposure"]["score"], result["defended_poison_exposure"]["score"]),
            delta_color="inverse",
        )

    active_attack_list = active_labels(attacks)
    active_defense_list = active_labels(defenses)

    live_tab, duel_tab = st.tabs([
        "Realtime Graph",
        "Agent Duel",
    ])

    with live_tab:
        live_removed_edges = removed_edges - edge_pairs(result["defended_graph"])
        st.header("Realtime Interaction Graph")
        st.caption(
            "This graph always reflects the current sidebar state after enabled red-team actions and enabled blue-team defenses are applied."
        )
        render_query_journey(result)
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


if __name__ == "__main__":
    main()

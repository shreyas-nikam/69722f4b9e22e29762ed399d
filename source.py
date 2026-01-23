# source.py
"""
Function-driven utilities for generating an AI Resilience Playbook.

Design goals:
- No top-level execution on import (safe for use from app.py / Streamlit).
- Each stage is callable as a function.
- Plotting functions return matplotlib Figure objects (caller decides to show/save).
- File writing is explicit via write_* helpers.

Tip: In app.py, call `run_full_pipeline(...)` for an end-to-end run, or call
individual functions for a more interactive experience.
"""

from __future__ import annotations

import json
import os
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx


# ----------------------------
# Paths / Utilities
# ----------------------------

def get_script_dir(fallback: Optional[str] = None) -> str:
    """Returns the directory of this file. Falls back to cwd (or provided fallback) when __file__ is unavailable."""
    if "__file__" in globals():
        return os.path.dirname(os.path.abspath(__file__))
    return fallback or os.getcwd()


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: Any, indent: int = 4) -> str:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)
    return path


def write_text(path: str, content: str) -> str:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def copy_file(src_path: str, dst_path: str) -> str:
    ensure_dir(os.path.dirname(dst_path) or ".")
    with open(src_path, "r", encoding="utf-8") as src, open(dst_path, "w", encoding="utf-8") as dst:
        dst.write(src.read())
    return dst_path


def now_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ----------------------------
# Templates / Failure Modes
# ----------------------------

def load_templates(file_path: str) -> Dict[str, Any]:
    """Loads a JSON file containing failure templates. Generates dummy data if file not found."""
    try:
        return read_json(file_path)
    except FileNotFoundError:
        print(
            f"Error: Template file not found at {file_path}. Generating dummy data.")
        if "engineering" in file_path.lower():
            return {
                "categories": ["Mechanical", "Electrical", "Software"],
                "failures": [
                    {
                        "id": "ENG001",
                        "name": "Bearing Wear",
                        "category": "Mechanical",
                        "description": "Progressive degradation of turbine bearings.",
                    },
                    {
                        "id": "ENG002",
                        "name": "Rotor Imbalance",
                        "category": "Mechanical",
                        "description": "Unbalanced rotation causing vibration.",
                    },
                    {
                        "id": "ENG003",
                        "name": "Sensor Malfunction",
                        "category": "Electrical",
                        "description": "Inaccurate or missing sensor readings.",
                    },
                ],
            }
        if "infrastructure" in file_path.lower():
            return {
                "categories": ["Network", "Compute", "Storage"],
                "failures": [
                    {
                        "id": "INF001",
                        "name": "Network Latency Spike",
                        "category": "Network",
                        "description": "Delay in data transmission to/from AI service.",
                    },
                    {
                        "id": "INF002",
                        "name": "Inference Engine Crash",
                        "category": "Compute",
                        "description": "AI model serving infrastructure failure.",
                    },
                ],
            }
        return {"categories": [], "failures": []}


def get_default_use_case() -> Dict[str, Any]:
    """Returns the default predictive maintenance use case used in the original script."""
    return {
        "name": "Steam Turbine Bearing & Rotor Blade RUL Prediction",
        "equipment": "High-Pressure Steam Turbine (Unit 3)",
        "criticality": "High - Direct impact on production, safety, and operational costs",
        "ai_function": (
            "Predict Remaining Useful Life (RUL) of bearings and rotor blades based on vibration, "
            "temperature, pressure, and lubrication sensor data."
        ),
        "sensors": ["Vibration_X", "Vibration_Y", "Temperature_Bearing", "Pressure_Inlet", "Oil_Viscosity"],
    }


def identify_ai_failure_modes(
    use_case_description: Dict[str, Any],
    engineering_templates: Dict[str, Any],
    infra_templates: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Identifies and categorizes AI-related failure modes for a predictive maintenance use case.
    Ensures referenced impact IDs exist in templates (adds conceptual placeholders if needed).
    """
    # Core AI failure modes (as in the original file)
    ai_failure_modes = [
        {
            "id": "AI001",
            "name": "Sensor Data Corruption",
            "category": "Data",
            "description": (
                "Corrupted or noisy sensor data (e.g., vibration, temperature) fed into the AI model, "
                "leading to inaccurate RUL predictions."
            ),
            "potential_impact": ["ENG001", "ENG002"],
        },
        {
            "id": "AI002",
            "name": "Model Drift (False Negatives)",
            "category": "Model",
            "description": (
                "AI model performance degrades over time, underestimating degradation; "
                "misses critical maintenance (high RUL_pred when RUL_true is low)."
            ),
            "potential_impact": ["ENG001", "ENG002"],
        },
        {
            "id": "AI003",
            "name": "Model Drift (False Positives)",
            "category": "Model",
            "description": (
                "AI model overestimates degradation, triggers unnecessary maintenance alerts/shutdowns "
                "(low RUL_pred when RUL_true is high)."
            ),
            "potential_impact": ["INF002_conceptual_production_impact"],
        },
        {
            "id": "AI004",
            "name": "Inference Latency Spike",
            "category": "Infrastructure",
            "description": "Delays in AI model predictions, causing alerts too late for proactive action.",
            "potential_impact": ["INF001"],
        },
        {
            "id": "AI005",
            "name": "Data Pipeline Failure",
            "category": "Data",
            "description": "Failure to deliver sensor data to the AI model, resulting in no or stale predictions.",
            "potential_impact": ["INF001"],
        },
        {
            "id": "AI006",
            "name": "Feature Engineering Error",
            "category": "Data",
            "description": (
                "Errors in transforming raw sensor data into features for the AI model, "
                "leading to poor model performance."
            ),
            "potential_impact": ["AI002", "AI003"],
        },
    ]

    # Ensure template structures exist
    engineering_templates.setdefault("failures", [])
    infra_templates.setdefault("failures", [])

    # Validate impact IDs and create placeholders where reasonable
    all_templates = engineering_templates["failures"] + \
        infra_templates["failures"]

    def exists_in_templates(tid: str) -> bool:
        return any(t.get("id") == tid for t in all_templates)

    for fm in ai_failure_modes:
        for impact_id in fm.get("potential_impact", []):
            if impact_id.endswith("_conceptual_production_impact"):
                continue
            if exists_in_templates(impact_id):
                continue

            print(
                f"Warning: Impact ID {impact_id} for '{fm['name']}' not found in templates. "
                f"Creating conceptual placeholder."
            )
            if impact_id.startswith("ENG"):
                engineering_templates["failures"].append(
                    {
                        "id": impact_id,
                        "name": f"Conceptual Engineering Impact {impact_id}",
                        "category": "Mechanical",
                        "description": "Conceptual engineering impact.",
                    }
                )
            elif impact_id.startswith("INF"):
                infra_templates["failures"].append(
                    {
                        "id": impact_id,
                        "name": f"Conceptual Infrastructure Impact {impact_id}",
                        "category": "Compute",
                        "description": "Conceptual infrastructure impact.",
                    }
                )
            else:
                print(f"Could not classify conceptual impact ID: {impact_id}")

            # refresh combined list
            all_templates = engineering_templates["failures"] + \
                infra_templates["failures"]

    # Final normalized output
    identified_failures: List[Dict[str, Any]] = []
    for fm in ai_failure_modes:
        identified_failures.append(
            {
                "id": fm["id"],
                "name": fm["name"],
                "category": fm["category"],
                "description": fm["description"],
                "related_template_impacts": fm.get("potential_impact", []),
            }
        )

    return identified_failures


def build_failure_mode_analysis_output(
    use_case: Dict[str, Any],
    identified_ai_failure_modes: List[Dict[str, Any]],
    engineering_templates: Dict[str, Any],
    infrastructure_templates: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "use_case": use_case,
        "identified_ai_failure_modes": identified_ai_failure_modes,
        "engineering_templates_used": engineering_templates,
        "infrastructure_templates_used": infrastructure_templates,
    }


def plot_failure_modes_by_category(
    identified_ai_failure_modes: List[Dict[str, Any]],
    title: Optional[str] = None,
) -> plt.Figure:
    """Returns a matplotlib Figure (caller decides plt.show() / st.pyplot() / savefig())."""
    df = pd.DataFrame(identified_ai_failure_modes)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    sns.countplot(data=df, x="category", palette="viridis", ax=ax)
    ax.set_title(title or "Identified AI Failure Modes by Category")
    ax.set_xlabel("Failure Mode Category")
    ax.set_ylabel("Number of Failure Modes")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    fig.tight_layout()
    return fig


# ----------------------------
# Resilience Controls / Mapping
# ----------------------------

def define_resilience_controls() -> List[Dict[str, Any]]:
    """Defines a set of resilience controls relevant to AI predictive maintenance."""
    return [
        {
            "id": "RC001",
            "name": "Shadow Model Deployment",
            "type": "Redundancy",
            "description": (
                "Deploy a redundant AI model (previous stable version or simpler heuristic) in parallel, "
                "used for anomaly detection and potential fallback."
            ),
            "applies_to_categories": ["Model"],
        },
        {
            "id": "RC002",
            "name": "Automatic Fallback to Scheduled Maintenance",
            "type": "Graceful Degradation",
            "description": (
                "If AI prediction confidence drops or system fails, automatically revert to a time/usage-based "
                "scheduled maintenance regime for critical components."
            ),
            "applies_to_categories": ["Model", "Data", "Infrastructure"],
        },
        {
            "id": "RC003",
            "name": "Human Expert Alert Queue",
            "type": "Manual Override",
            "description": "Route high-severity or uncertain AI alerts to a human expert queue for manual review.",
            "applies_to_categories": ["Model", "Data"],
        },
        {
            "id": "RC004",
            "name": "Sensor Data Anomaly Detection (Preprocessing)",
            "type": "Data Validation",
            "description": (
                "Implement anomaly detection on raw sensor streams to detect corruption/outliers before "
                "feeding the main AI model."
            ),
            "applies_to_categories": ["Data"],
        },
        {
            "id": "RC005",
            "name": "Model Performance Monitoring & Alerting",
            "type": "Monitoring",
            "description": (
                "Continuously monitor drift, accuracy metrics, and inference latency with automated alerts."
            ),
            "applies_to_categories": ["Model", "Infrastructure"],
        },
        {
            "id": "RC006",
            "name": "Data Pipeline Health Checks",
            "type": "Infrastructure Monitoring",
            "description": (
                "Health checks and monitoring for ingestion and feature engineering pipelines to ensure integrity."
            ),
            "applies_to_categories": ["Data", "Infrastructure"],
        },
        {
            "id": "RC007",
            "name": "Revert to Heuristic-Based Prediction",
            "type": "Fallback",
            "description": (
                "If primary AI fails or degrades severely, switch to a simpler rule-based heuristic (thresholds)."
            ),
            "applies_to_categories": ["Model"],
        },
    ]


def map_controls_to_failure_modes(
    failure_modes: List[Dict[str, Any]],
    controls: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Maps resilience controls to failure modes based on category applicability."""
    mapped_data: List[Dict[str, Any]] = []

    for fm in failure_modes:
        applicable_controls = []
        for ctrl in controls:
            if fm.get("category") in ctrl.get("applies_to_categories", []):
                applicable_controls.append(
                    {
                        "control_id": ctrl["id"],
                        "control_name": ctrl["name"],
                        "control_type": ctrl["type"],
                        "description": ctrl["description"],
                    }
                )

        mapped_data.append(
            {
                "failure_mode_id": fm["id"],
                "failure_mode_name": fm["name"],
                "failure_mode_category": fm["category"],
                "applicable_controls": applicable_controls,
            }
        )

    return mapped_data


def build_resilience_controls_output(
    defined_controls: List[Dict[str, Any]],
    mapped_controls_to_failure_modes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "defined_controls": defined_controls,
        "mapped_controls_to_failure_modes": mapped_controls_to_failure_modes,
    }


def plot_failure_mode_control_bipartite(
    identified_ai_failure_modes: List[Dict[str, Any]],
    resilience_controls: List[Dict[str, Any]],
    mapped_resilience_data: List[Dict[str, Any]],
    title: str = "Mapping of AI Failure Modes to Resilience Controls",
) -> plt.Figure:
    """Builds and returns a bipartite graph figure (matplotlib)."""
    G = nx.Graph()

    failure_nodes = [fm["name"] for fm in identified_ai_failure_modes]
    control_nodes = [c["name"] for c in resilience_controls]

    G.add_nodes_from(failure_nodes, bipartite=0, label="Failure Mode")
    G.add_nodes_from(control_nodes, bipartite=1, label="Resilience Control")

    for fm_map in mapped_resilience_data:
        for ctrl in fm_map.get("applicable_controls", []):
            G.add_edge(fm_map["failure_mode_name"], ctrl["control_name"])

    pos = nx.bipartite_layout(G, failure_nodes)

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111)

    nx.draw_networkx_nodes(G, pos, nodelist=failure_nodes,
                           node_color="skyblue", node_size=2000, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=control_nodes,
                           node_color="lightgreen", node_size=2000, ax=ax)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5,
                           edge_color="gray", ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    return fig


# ----------------------------
# Cascading Failure Simulation
# ----------------------------

def simulate_cascading_failure(
    initial_ai_error_rate: float,
    simulation_steps: int = 10,
    containment_strategy: Optional[str] = None,
    rng_seed: Optional[int] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Simulates a cascading failure scenario starting with AI model prediction error.
    Returns (scenario_description, simulation_results).
    """
    if rng_seed is not None:
        np.random.seed(rng_seed)

    scenario_description = {
        "name": "Model Drift (False Negatives) -> Missed Maintenance -> Critical Component Failure",
        "initial_event": "AI Model Drift (False Negatives) - model consistently underpredicts RUL.",
        "triggering_failure_mode_id": "AI002",
        "containment_options": {
            "strategy_A": "Activate Shadow Model & Human Review Queue (RC001, RC003)",
            "strategy_B": "Automatic Fallback to Scheduled Maintenance (RC002)",
        },
    }

    # Conceptual probabilities (illustrative)
    p_false_negative_base = 0.05
    p_missed_maintenance_given_fn = 0.7
    p_comp_degradation_given_mm = 0.6
    p_catastrophic_failure_given_cd = 0.4

    # Error rate influences false negative probability
    p_false_negative_current = p_false_negative_base + \
        (initial_ai_error_rate * 0.5)
    _ = p_false_negative_current  # kept for clarity, not directly used below

    simulation_results: List[Dict[str, Any]] = []
    current_system_health = 100.0
    current_ai_accuracy_impact = float(initial_ai_error_rate)

    false_negatives_detected = False
    maintenance_missed = False
    component_degraded = False
    catastrophic_failure = False

    for step in range(simulation_steps):
        event_description = f"Step {step + 1}: "
        impact_level = 0  # 0 none, 1 minor, 2 moderate, 3 severe, 4 catastrophic

        # 1) Drift detected (threshold)
        if current_ai_accuracy_impact > 0.05 and not false_negatives_detected:
            event_description += "AI Model Drift (False Negatives) detected. "
            false_negatives_detected = True
            impact_level = max(impact_level, 1)

            if containment_strategy == "strategy_A":
                event_description += "[Containment A: Shadow Model + Human Review engaged] "
                current_ai_accuracy_impact *= 0.5
            elif containment_strategy == "strategy_B":
                event_description += "[Containment B: Fallback to Scheduled Maintenance] "
                current_ai_accuracy_impact *= 0.2

        # 2) Missed maintenance
        if false_negatives_detected and not maintenance_missed:
            containment_factor = (
                0.5 if containment_strategy == "strategy_A"
                else 0.1 if containment_strategy == "strategy_B"
                else 1.0
            )
            p_mm = p_missed_maintenance_given_fn * \
                max(0.0, (1 - current_ai_accuracy_impact)) * containment_factor
            if np.random.rand() < p_mm:
                event_description += "Critical maintenance missed due to false negatives. "
                maintenance_missed = True
                impact_level = max(impact_level, 2)
                current_system_health -= 15

        # 3) Component degradation
        if maintenance_missed and not component_degraded and np.random.rand() < p_comp_degradation_given_mm:
            event_description += "Turbine component (bearing/rotor) begins to degrade. "
            component_degraded = True
            impact_level = max(impact_level, 3)
            current_system_health -= 30

        # 4) Catastrophic failure
        if component_degraded and not catastrophic_failure and np.random.rand() < p_catastrophic_failure_given_cd:
            event_description += "Catastrophic turbine failure and unplanned shutdown! "
            catastrophic_failure = True
            impact_level = max(impact_level, 4)
            current_system_health = 0.0

        # Baseline degradation / persistence penalty
        if not (false_negatives_detected or maintenance_missed or component_degraded or catastrophic_failure) and current_ai_accuracy_impact > 0.01:
            event_description += "AI performing normally, no major issues. "
            current_system_health -= current_ai_accuracy_impact * 5
        elif false_negatives_detected and not catastrophic_failure:
            current_system_health -= (impact_level *
                                      5 + current_ai_accuracy_impact * 10)

        current_system_health = float(max(0.0, current_system_health))

        simulation_results.append(
            {
                "step": step + 1,
                "event": event_description.strip(),
                "ai_accuracy_impact": float(current_ai_accuracy_impact),
                "system_health": float(current_system_health),
                "impact_level": int(impact_level),
            }
        )

        if catastrophic_failure:
            break

    return scenario_description, simulation_results


def define_safe_degradation() -> Dict[str, Any]:
    """Defines a scenario of safe degradation."""
    return {
        "name": "Graceful AI Degradation - Safe Fallback",
        "description": (
            "AI model experiences minor, intermittent data pipeline issues (latency, missing readings). "
            "System switches to heuristic-based prediction (RC007) and triggers low-priority human review "
            "for the AI system, while maintenance decisions proceed safely."
        ),
        "triggering_failure_mode_id": "AI005 (partial)",
        "controls_activated": ["RC007", "RC003 (low priority)"],
        "outcome": (
            "System operates in degraded mode with no operational/safety impact. Maintenance proceeds via "
            "heuristics while AI system is investigated."
        ),
    }


def build_cascading_failure_analysis_output(
    scenario_definition: Dict[str, Any],
    simulation_results_no_containment: List[Dict[str, Any]],
    simulation_results_with_containment_B: List[Dict[str, Any]],
    safe_degradation_example: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "scenario_definition": scenario_definition,
        "simulation_results_no_containment": simulation_results_no_containment,
        "simulation_results_with_containment_B": simulation_results_with_containment_B,
        "safe_degradation_example": safe_degradation_example,
    }


def plot_cascading_failure_health(
    simulation_output_no_contain: List[Dict[str, Any]],
    simulation_output_with_contain: List[Dict[str, Any]],
    initial_error_rate: float,
    title: Optional[str] = None,
) -> plt.Figure:
    df_no = pd.DataFrame(simulation_output_no_contain)
    df_yes = pd.DataFrame(simulation_output_with_contain)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)

    ax.plot(df_no["step"], df_no["system_health"],
            marker="o", label="No Containment", color="red")
    ax.plot(df_yes["step"], df_yes["system_health"], marker="x",
            label="With Containment (Strategy B)", color="green", linestyle="--")

    ax.set_title(
        title or f"Cascading Failure Simulation: System Health vs Time (Initial AI Error: {initial_error_rate*100:.1f}%)")
    ax.set_xlabel("Simulation Step (Time)")
    ax.set_ylabel("System Health (%)")
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    return fig


# ----------------------------
# Recovery / Rollback Planning
# ----------------------------

def load_sample_recovery_plan(file_path: str) -> str:
    """Loads a markdown file containing sample recovery plans. Returns dummy content if not found."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(
            f"Error: Sample recovery plan not found at {file_path}. Generating dummy content.")
        return """# Sample AI Service Recovery Plan (Placeholder)

## 1. Rollback Triggers
- Sustained Model Performance Degradation (e.g., MAE > 1.5x baseline for 30 minutes)
- Data Pipeline Failure (no data ingestion for 15 minutes)
- Inference Latency > 500ms for 5 consecutive minutes

## 2. Rollback Procedures
1. **Automated Rollback to Previous Model Version:**
   - Trigger: Model Performance Monitor detects drift.
   - Action: Deploy the last validated AI model version from the model registry.
   - Validation: Run inference on recent historical data, compare metrics.
2. **Revert to Heuristic-Based Prediction:**
   - Trigger: Primary AI inference service unavailable.
   - Action: Switch inference endpoint to a simpler rule-based system.
   - Validation: Confirm heuristic predictions are within safe operational bounds.

## 3. Recovery Objectives
- **AI Service RTO:** 2 hours (for degraded mode service)
- **AI Service RPO:** 30 minutes (maximum data loss for training/re-calibration)

## 4. Validation Steps for Re-enablement
- **Performance Metrics:** Ensure MAE is within 1.1x baseline for 24 hours.
- **Data Integrity:** Verify 99.9% data pipeline health.
- **System Stability:** Monitor inference latency and throughput for 48 hours.
"""


def define_rollback_procedures_and_recovery_objectives() -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[str, Any],
    List[Dict[str, Any]],
]:
    """Defines rollback triggers, procedures, RTO/RPO, and validation criteria."""
    rollback_triggers = [
        {
            "id": "RT001",
            "name": "Sustained Model Performance Degradation",
            "description": "MAE exceeds 1.5x baseline for 30 consecutive minutes.",
            "metric": "MAE",
            "threshold_multiplier": 1.5,
            "duration_minutes": 30,
        },
        {
            "id": "RT002",
            "name": "Data Pipeline Ingestion Stoppage",
            "description": "No new sensor data ingested into the feature store for 15 minutes.",
            "metric": "Data Ingestion Rate",
            "threshold_value": 0,
            "duration_minutes": 15,
        },
        {
            "id": "RT003",
            "name": "Excessive Inference Latency",
            "description": "Average inference latency exceeds 500ms for 5 consecutive minutes.",
            "metric": "Latency",
            "threshold_ms": 500,
            "duration_minutes": 5,
        },
    ]

    rollback_procedures = [
        {
            "id": "RP001",
            "name": "Automated Deployment of Last Validated Model",
            "description": (
                "Triggered by RT001. Automatically deploys the last known good model from the registry."
            ),
            "trigger_ids": ["RT001"],
            "validation_steps": ["Run post-deployment sanity checks on recent production data.", "Monitor first hour of inference metrics."],
        },
        {
            "id": "RP002",
            "name": "Manual Revert to Heuristic-Based Prediction",
            "description": (
                "Triggered by RT002/RT003. Operator switches endpoint to heuristic system for safe operation."
            ),
            "trigger_ids": ["RT002", "RT003"],
            "validation_steps": ["Confirm heuristic system is active and producing outputs.", "Verify outputs against safe thresholds."],
        },
        {
            "id": "RP003",
            "name": "Data Pipeline Restart & Backfill",
            "description": (
                "Triggered by RT002. Restart ingestion and backfill missing data from raw archives."
            ),
            "trigger_ids": ["RT002"],
            "validation_steps": ["Verify ingestion rate returns to normal.", "Check integrity of backfilled data."],
        },
    ]

    recovery_objectives = {
        "RTO_AI_Service_Degraded_Mode": "2 hours",
        "RTO_AI_Service_Full_Functionality": "8 hours",
        "RPO_AI_Data_for_Retraining": "30 minutes",
        "RPO_AI_Inference_State": "10 minutes",
    }

    validation_for_re_enablement = [
        {"step": "1", "description": "MAE within 1.1x baseline and no significant drift for 24 hours.",
            "metric_kpi": "Model_MAE_Stability"},
        {"step": "2", "description": "99.9% ingestion success and no pipeline errors for 12 hours.",
            "metric_kpi": "Data_Pipeline_Health"},
        {"step": "3", "description": "Latency < 200ms and 99.9% uptime for 48 hours.",
            "metric_kpi": "Inference_Stability"},
    ]

    return rollback_triggers, rollback_procedures, recovery_objectives, validation_for_re_enablement


def build_recovery_plan_markdown(
    equipment_name: str,
    rollback_triggers: List[Dict[str, Any]],
    rollback_procedures: List[Dict[str, Any]],
    recovery_objectives: Dict[str, Any],
    validation_for_re_enablement: List[Dict[str, Any]],
) -> str:
    """Builds the recovery_plan.md content."""
    content = f"""# AI Predictive Maintenance Service Recovery Plan - {equipment_name}

This document outlines procedures for detecting, mitigating, and recovering from failures in the AI-driven predictive maintenance service.

## 1. Rollback Triggers
These conditions trigger rollback or degraded operation.

"""
    for trigger in rollback_triggers:
        content += f"- **{trigger['name']} ({trigger['id']}):** {trigger['description']}\n"

    content += """

## 2. Rollback Procedures
Actions to take when triggers are detected.

"""
    for procedure in rollback_procedures:
        content += f"- **{procedure['name']} ({procedure['id']}):**\n"
        content += f"  - **Triggers:** {', '.join(procedure['trigger_ids'])}\n"
        content += f"  - **Description:** {procedure['description']}\n"
        content += f"  - **Validation Steps:** {'; '.join(procedure['validation_steps'])}\n"

    content += """

## 3. Recovery Objectives (RTO/RPO)

"""
    for obj, val in recovery_objectives.items():
        content += f"- **{obj.replace('_', ' ')}:** {val}\n"

    content += """

## 4. Validation Steps for Full AI Service Re-enablement

"""
    for step in validation_for_re_enablement:
        content += f"- **Step {step['step']}:** {step['description']} (KPI: {step['metric_kpi']})\n"

    return content


def build_resilience_kpis_output(
    recovery_objectives: Dict[str, Any],
    rollback_triggers: List[Dict[str, Any]],
    validation_for_re_enablement: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "recovery_objectives": recovery_objectives,
        "rollback_triggers": rollback_triggers,
        "validation_for_re_enablement": validation_for_re_enablement,
    }


# ----------------------------
# Playbook / Executive Summary
# ----------------------------

def generate_ai_resilience_playbook(
    use_case: Dict[str, Any],
    failure_modes_output: Dict[str, Any],
    controls_output: Dict[str, Any],
    cascading_analysis_output: Dict[str, Any],
    rollback_procedures: List[Dict[str, Any]],
    resilience_kpis_output: Dict[str, Any],
    playbook_version: str = "1.0",
) -> Dict[str, Any]:
    """Combines all generated components into a single AI Resilience Playbook JSON."""
    playbook = {
        "playbook_version": playbook_version,
        "generated_date": datetime.now().isoformat(),
        "use_case_definition": use_case,
        "ai_failure_mode_analysis": failure_modes_output,
        "resilience_controls_mapping": controls_output,
        "cascading_failure_analysis": cascading_analysis_output,
        "recovery_and_rollback_planning": {
            "rollback_triggers": resilience_kpis_output["rollback_triggers"],
            "rollback_procedures": rollback_procedures,
            "recovery_objectives": resilience_kpis_output["recovery_objectives"],
            "validation_for_re_enablement": resilience_kpis_output["validation_for_re_enablement"],
            "recovery_plan_markdown_path": "recovery_plan.md",
        },
    }
    return playbook


def generate_executive_summary(playbook_data: Dict[str, Any], run_id: str) -> str:
    """Generates an executive summary markdown."""
    use_case = playbook_data["use_case_definition"]
    failure_count = len(
        playbook_data["ai_failure_mode_analysis"]["identified_ai_failure_modes"])

    ro = playbook_data["recovery_and_rollback_planning"]["recovery_objectives"]

    return f"""# Executive Summary: Predictive Maintenance AI Resilience Playbook - {use_case['equipment']}

**Date Generated:** {playbook_data['generated_date']}
**Playbook Version:** {playbook_data['playbook_version']}
**Run ID:** {run_id}

## 1. Overview
This playbook details the resilience strategy for the AI-driven predictive maintenance service for **{use_case['equipment']}**.

## 2. Key Findings from Failure Mode Analysis
- Identified {failure_count} AI-related failure modes across Data, Model, and Infrastructure categories.
- Primary vulnerabilities include **Model Drift (False Negatives)** and **Sensor Data Corruption**, which can lead to missed maintenance.

## 3. Resilience Controls Implemented
- Redundancy via **Shadow Model Deployment**
- Graceful degradation via **Automatic Fallback to Scheduled Maintenance**
- Manual override via **Human Expert Alert Queue**
- Proactive monitoring via **Sensor Data Anomaly Detection** and **Model Performance Monitoring**

## 4. Cascading Failure Insights
Simulations show rapid health degradation without containment, and stabilization when containment is active (notably scheduled-maintenance fallback).

## 5. Recovery Objectives and Rollback Strategy
- **AI Service RTO (Degraded Mode):** {ro['RTO_AI_Service_Degraded_Mode']}
- **AI Service RPO (Data for Retraining):** {ro['RPO_AI_Data_for_Retraining']}
Rollback triggers and procedures are defined for MAE drift, ingestion stoppage, and latency spikes, with validation criteria for safe re-enablement.

## 6. Conclusion
This playbook improves operational resilience by combining structured failure mode analysis, mapped controls, tested containment strategies, and concrete rollback/recovery procedures.
"""


# ----------------------------
# Evidence Manifest / Hashing
# ----------------------------

def generate_file_hash(file_path: str) -> str:
    """Generates SHA-256 hash for a given file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def build_evidence_manifest(run_id: str, artifact_paths: Dict[str, str]) -> Dict[str, Any]:
    manifest = {
        "run_id": run_id,
        "generated_at": datetime.now().isoformat(),
        "artifacts": [],
    }
    for name, path in artifact_paths.items():
        if os.path.exists(path):
            manifest["artifacts"].append(
                {
                    "name": name,
                    "path": path,
                    "sha256_hash": generate_file_hash(path),
                }
            )
        else:
            print(
                f"Warning: Artifact not found at {path}. Skipping hash generation.")
    return manifest


# ----------------------------
# End-to-end Orchestration
# ----------------------------

def run_full_pipeline(
    engineering_templates_path: Optional[str] = None,
    infrastructure_templates_path: Optional[str] = None,
    sample_recovery_plan_path: Optional[str] = None,
    use_case: Optional[Dict[str, Any]] = None,
    initial_error_rate: float = 0.20,
    simulation_steps: int = 8,
    output_dir: str = "reports/session14",
    run_id: Optional[str] = None,
    rng_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Runs the entire pipeline and writes artifacts to disk.

    Returns a dict containing:
    - run_id
    - outputs (structured outputs in-memory)
    - artifact_paths (where files were written)
    - evidence_manifest
    """
    script_dir = get_script_dir()

    engineering_templates_path = engineering_templates_path or os.path.join(
        script_dir, "engineering_failure_templates.json")
    infrastructure_templates_path = infrastructure_templates_path or os.path.join(
        script_dir, "infrastructure_failure_templates.json")
    sample_recovery_plan_path = sample_recovery_plan_path or os.path.join(
        script_dir, "sample_recovery_plans.md")

    use_case = use_case or get_default_use_case()
    run_id = run_id or now_run_id()

    # 1) Load templates
    engineering_templates = load_templates(engineering_templates_path)
    infrastructure_templates = load_templates(infrastructure_templates_path)

    # 2) Identify failure modes
    identified_ai_failure_modes = identify_ai_failure_modes(
        use_case, engineering_templates, infrastructure_templates)
    failure_mode_analysis_output = build_failure_mode_analysis_output(
        use_case, identified_ai_failure_modes, engineering_templates, infrastructure_templates
    )

    # 3) Controls + mapping
    resilience_controls = define_resilience_controls()
    mapped_resilience_data = map_controls_to_failure_modes(
        identified_ai_failure_modes, resilience_controls)
    resilience_controls_output = build_resilience_controls_output(
        resilience_controls, mapped_resilience_data)

    # 4) Cascading simulation
    scenario_def_no, sim_no = simulate_cascading_failure(
        initial_error_rate, simulation_steps=simulation_steps, containment_strategy=None, rng_seed=rng_seed
    )
    _, sim_with_b = simulate_cascading_failure(
        initial_error_rate, simulation_steps=simulation_steps, containment_strategy="strategy_B", rng_seed=rng_seed
    )
    safe_degradation_example = define_safe_degradation()
    cascading_failure_analysis_output = build_cascading_failure_analysis_output(
        scenario_def_no, sim_no, sim_with_b, safe_degradation_example
    )

    # 5) Recovery plan inputs
    # loaded for guidance; returned in outputs below
    _sample_md = load_sample_recovery_plan(sample_recovery_plan_path)
    rollback_triggers, rollback_procedures, recovery_objectives, validation_for_re_enablement = (
        define_rollback_procedures_and_recovery_objectives()
    )
    recovery_plan_md = build_recovery_plan_markdown(
        equipment_name=use_case["equipment"],
        rollback_triggers=rollback_triggers,
        rollback_procedures=rollback_procedures,
        recovery_objectives=recovery_objectives,
        validation_for_re_enablement=validation_for_re_enablement,
    )
    resilience_kpis_output = build_resilience_kpis_output(
        recovery_objectives, rollback_triggers, validation_for_re_enablement)

    # 6) Playbook + exec summary
    full_ai_resilience_playbook_data = generate_ai_resilience_playbook(
        use_case=use_case,
        failure_modes_output=failure_mode_analysis_output,
        controls_output=resilience_controls_output,
        cascading_analysis_output=cascading_failure_analysis_output,
        rollback_procedures=rollback_procedures,
        resilience_kpis_output=resilience_kpis_output,
    )
    executive_summary_content = generate_executive_summary(
        full_ai_resilience_playbook_data, run_id)

    # 7) Write artifacts
    ensure_dir(output_dir)
    artifact_paths = {
        "sector_playbook.json": os.path.join(output_dir, f"sector_playbook_{run_id}.json"),
        "failure_mode_analysis.json": os.path.join(output_dir, f"failure_mode_analysis_{run_id}.json"),
        "resilience_controls.json": os.path.join(output_dir, f"resilience_controls_{run_id}.json"),
        "recovery_plan.md": os.path.join(output_dir, f"recovery_plan_{run_id}.md"),
        "resilience_kpis.json": os.path.join(output_dir, f"resilience_kpis_{run_id}.json"),
        "session14_executive_summary.md": os.path.join(output_dir, f"session14_executive_summary_{run_id}.md"),
        "config_snapshot.json": os.path.join(output_dir, f"config_snapshot_{run_id}.json"),
    }

    write_json(artifact_paths["sector_playbook.json"],
               full_ai_resilience_playbook_data)
    write_json(artifact_paths["failure_mode_analysis.json"],
               failure_mode_analysis_output)
    write_json(artifact_paths["resilience_controls.json"],
               resilience_controls_output)
    write_json(artifact_paths["resilience_kpis.json"], resilience_kpis_output)
    write_text(artifact_paths["recovery_plan.md"], recovery_plan_md)
    write_text(
        artifact_paths["session14_executive_summary.md"], executive_summary_content)

    # config snapshot (conceptual)
    config_snapshot_data = {
        "model_version": "v2.1_production_RUL",
        "data_pipeline_version": "v3.0_sensor_ingestion",
        "inference_engine_version": "v1.2_tensorflow_serving",
        "monitoring_stack_version": "v1.5_prometheus_grafana",
        "timestamp": datetime.now().isoformat(),
    }
    write_json(artifact_paths["config_snapshot.json"], config_snapshot_data)

    # 8) Evidence manifest
    evidence_manifest = build_evidence_manifest(run_id, artifact_paths)
    evidence_manifest_path = os.path.join(
        output_dir, f"evidence_manifest_{run_id}.json")
    write_json(evidence_manifest_path, evidence_manifest)

    return {
        "run_id": run_id,
        "outputs": {
            "engineering_templates": engineering_templates,
            "infrastructure_templates": infrastructure_templates,
            "identified_ai_failure_modes": identified_ai_failure_modes,
            "failure_mode_analysis_output": failure_mode_analysis_output,
            "resilience_controls": resilience_controls,
            "mapped_resilience_data": mapped_resilience_data,
            "resilience_controls_output": resilience_controls_output,
            "cascading_failure_analysis_output": cascading_failure_analysis_output,
            "sample_recovery_plan_guidance_md": _sample_md,
            "rollback_triggers": rollback_triggers,
            "rollback_procedures": rollback_procedures,
            "recovery_objectives": recovery_objectives,
            "validation_for_re_enablement": validation_for_re_enablement,
            "recovery_plan_md": recovery_plan_md,
            "resilience_kpis_output": resilience_kpis_output,
            "full_ai_resilience_playbook_data": full_ai_resilience_playbook_data,
            "executive_summary_md": executive_summary_content,
            "config_snapshot_data": config_snapshot_data,
        },
        "artifact_paths": {**artifact_paths, "evidence_manifest.json": evidence_manifest_path},
        "evidence_manifest": evidence_manifest,
    }

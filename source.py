import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import hashlib
import os
from datetime import datetime, timedelta
# Function to load template data
def load_templates(file_path):
    """Loads a JSON file containing failure templates."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Template file not found at {file_path}. Generating dummy data.")
        # Generate dummy data if file not found for demonstration
        if "engineering" in file_path:
            return {"categories": ["Mechanical", "Electrical", "Software"], "failures": [{"id": "ENG001", "name": "Bearing Wear", "category": "Mechanical", "description": "Progressive degradation of turbine bearings."}, {"id": "ENG002", "name": "Rotor Imbalance", "category": "Mechanical", "description": "Unbalanced rotation causing vibration."}, {"id": "ENG003", "name": "Sensor Malfunction", "category": "Electrical", "description": "Inaccurate or missing sensor readings."}]}
        elif "infrastructure" in file_path:
            return {"categories": ["Network", "Compute", "Storage"], "failures": [{"id": "INF001", "name": "Network Latency Spike", "category": "Network", "description": "Delay in data transmission to/from AI service."}, {"id": "INF002", "name": "Inference Engine Crash", "category": "Compute", "description": "AI model serving infrastructure failure."}]}
        else:
            return {}

# Function to identify AI-related failure modes for the specific use case
def identify_ai_failure_modes(use_case_description, engineering_templates, infra_templates):
    """
    Identifies and categorizes AI-related failure modes specific to a predictive maintenance use case.
    Generates synthetic data if templates are not comprehensive.
    """
    ai_failure_modes = [
          {"id": "AI001", "name": "Sensor Data Corruption", "category": "Data",
         "description": "Corrupted or noisy sensor data (e.g., vibration, temperature) fed into the AI model, leading to inaccurate RUL predictions.",
         "potential_impact": ["ENG001", "ENG002"]},
        {"id": "AI002", "name": "Model Drift (False Negatives)", "category": "Model",
         "description": "AI model's performance degrades over time, consistently underestimating degradation, leading to missed critical maintenance. High RUL_pred when RUL_true is low.",
         "potential_impact": ["ENG001", "ENG002"]},
        {"id": "AI003", "name": "Model Drift (False Positives)", "category": "Model",
         "description": "AI model overestimates degradation, triggering unnecessary maintenance alerts and shutdowns. Low RUL_pred when RUL_true is high.",
         "potential_impact": ["INF002_conceptual_production_impact"]},
        {"id": "AI004", "name": "Inference Latency Spike", "category": "Infrastructure",
         "description": "Delays in AI model predictions, causing alerts to be delivered too late for proactive action.",
         "potential_impact": ["INF001"]},
        {"id": "AI005", "name": "Data Pipeline Failure", "category": "Data",
         "description": "Failure to deliver sensor data to the AI model, resulting in no predictions or stale predictions.",
         "potential_impact": ["INF001"]},
        {"id": "AI006", "name": "Feature Engineering Error", "category": "Data",
         "description": "Errors in transforming raw sensor data into features for the AI model, leading to poor model performance.",
         "potential_impact": ["AI002", "AI003"]},
    ]

    # Augment with template data if needed, ensuring relevant IDs are present
    # This is a conceptual mapping as we don't have direct linkages in templates
    # For demonstration, we'll ensure impact IDs exist or create conceptual ones.
    for fm in ai_failure_modes:
        if "potential_impact" in fm:
            for i, impact_id in enumerate(fm["potential_impact"]):
                if impact_id.endswith("_conceptual_production_impact"):
                    # This is a conceptual link to production impact, not directly an engineering/infra failure
                    continue
                found = False
                for t in engineering_templates["failures"] + infra_templates["failures"]:
                    if t["id"] == impact_id:
                        found = True
                        break
                if not found:
                    print(f"Warning: Impact ID {impact_id} for {fm['name']} not found in templates. Creating conceptual placeholder.")
                    # Add a conceptual impact for the demo
                    if impact_id.startswith("ENG"):
                        engineering_templates["failures"].append({"id": impact_id, "name": f"Conceptual Engineering Impact {impact_id}", "category": "Mechanical", "description": "Conceptual engineering impact."})
                    elif impact_id.startswith("INF"):
                        infra_templates["failures"].append({"id": impact_id, "name": f"Conceptual Infrastructure Impact {impact_id}", "category": "Compute", "description": "Conceptual infrastructure impact."})
                    else:
                        print(f"Could not classify conceptual impact ID: {impact_id}")


    # Consolidate and categorize
    identified_failures = []
    for fm in ai_failure_modes:
        identified_failures.append({
              "id": fm["id"],
            "name": fm["name"],
            "category": fm["category"],
            "description": fm["description"],
            "related_template_impacts": fm.get("potential_impact", [])
        })

    return identified_failures

# --- Execution ---
# Define file paths for the sample data
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
engineering_failure_templates_path = os.path.join(script_dir, 'engineering_failure_templates.json')
infrastructure_failure_templates_path = os.path.join(script_dir, 'infrastructure_failure_templates.json')

# Load templates
engineering_templates = load_templates(engineering_failure_templates_path)
infrastructure_templates = load_templates(infrastructure_failure_templates_path)

# Define the specific use case
predictive_maintenance_use_case = {
      "name": "Steam Turbine Bearing & Rotor Blade RUL Prediction",
    "equipment": "High-Pressure Steam Turbine (Unit 3)",
    "criticality": "High - Direct impact on production, safety, and operational costs",
    "ai_function": "Predict Remaining Useful Life (RUL) of bearings and rotor blades based on vibration, temperature, pressure, and lubrication sensor data.",
    "sensors": ["Vibration_X", "Vibration_Y", "Temperature_Bearing", "Pressure_Inlet", "Oil_Viscosity"]
}

# Identify AI-specific failure modes
identified_ai_failure_modes = identify_ai_failure_modes(
      predictive_maintenance_use_case,
    engineering_templates,
    infrastructure_templates
)

print("Identified AI-Related Failure Modes:")
for fm in identified_ai_failure_modes:
    print(f"- {fm['id']}: {fm['name']} ({fm['category']})")

# Visualize identified failure modes by category
df_failure_modes = pd.DataFrame(identified_ai_failure_modes)
plt.figure(figsize=(10, 6))
sns.countplot(data=df_failure_modes, x='category', palette='viridis')
plt.title('Identified AI Failure Modes by Category for Steam Turbine Predictive Maintenance')
plt.xlabel('Failure Mode Category')
plt.ylabel('Number of Failure Modes')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Save failure mode analysis to JSON
failure_mode_analysis_output = {
      "use_case": predictive_maintenance_use_case,
    "identified_ai_failure_modes": identified_ai_failure_modes,
    "engineering_templates_used": engineering_templates,
    "infrastructure_templates_used": infrastructure_templates
}
with open('failure_mode_analysis.json', 'w') as f:
    json.dump(failure_mode_analysis_output, f, indent=4)

# Function to define resilience controls
def define_resilience_controls():
    """Defines a set of resilience controls relevant to AI predictive maintenance."""
    controls = [
          {"id": "RC001", "name": "Shadow Model Deployment", "type": "Redundancy",
         "description": "Deploy a redundant AI model (e.g., previous stable version or simpler heuristic) running in parallel, used for anomaly detection and potential fallback.",
         "applies_to_categories": ["Model"]},
        {"id": "RC002", "name": "Automatic Fallback to Scheduled Maintenance", "type": "Graceful Degradation",
         "description": "If AI prediction confidence drops or system fails, automatically revert to a time-based or usage-based scheduled maintenance regime for critical components.",
         "applies_to_categories": ["Model", "Data", "Infrastructure"]},
        {"id": "RC003", "name": "Human Expert Alert Queue", "type": "Manual Override",
         "description": "Route high-severity or uncertain AI alerts to a human expert queue for manual review and decision-making.",
         "applies_to_categories": ["Model", "Data"]},
        {"id": "RC004", "name": "Sensor Data Anomaly Detection (Preprocessing)", "type": "Data Validation",
         "description": "Implement anomaly detection algorithms on raw sensor data streams to detect corruption or outliers before feeding to the main AI model.",
         "applies_to_categories": ["Data"]},
        {"id": "RC005", "name": "Model Performance Monitoring & Alerting", "type": "Monitoring",
         "description": "Continuously monitor AI model's prediction drift, accuracy metrics, and inference latency, with automated alerts for degradation.",
         "applies_to_categories": ["Model", "Infrastructure"]},
        {"id": "RC006", "name": "Data Pipeline Health Checks", "type": "Infrastructure Monitoring",
         "description": "Regular health checks and monitoring for data ingestion and feature engineering pipelines, ensuring data flow integrity.",
         "applies_to_categories": ["Data", "Infrastructure"]},
        {"id": "RC007", "name": "Revert to Heuristic-Based Prediction", "type": "Fallback",
         "description": "In case of primary AI model failure or severe degradation, switch to a simpler, rule-based heuristic for RUL prediction (e.g., based on fixed thresholds).",
         "applies_to_categories": ["Model"]}
    ]
    return controls

# Function to map resilience controls to identified failure modes
def map_controls_to_failure_modes(failure_modes, controls):
    """
    Maps resilience controls to identified AI failure modes based on applicability.
    Returns a list of dictionaries where each failure mode has a list of applicable controls.
    """
    mapped_data = []
    for fm in failure_modes:
        applicable_controls = []
        for ctrl in controls:
            if fm["category"] in ctrl["applies_to_categories"]:
                applicable_controls.append({
                      "control_id": ctrl["id"],
                    "control_name": ctrl["name"],
                    "control_type": ctrl["type"],
                    "description": ctrl["description"]
                })
        mapped_data.append({
              "failure_mode_id": fm["id"],
            "failure_mode_name": fm["name"],
            "failure_mode_category": fm["category"],
            "applicable_controls": applicable_controls
        })
    return mapped_data

# --- Execution ---
# Define resilience controls
resilience_controls = define_resilience_controls()

# Map controls to previously identified AI failure modes
mapped_resilience_data = map_controls_to_failure_modes(identified_ai_failure_modes, resilience_controls)

print("Resilience Control Mapping:")
for fm_map in mapped_resilience_data:
    print(f"- Failure Mode: {fm_map['failure_mode_name']} ({fm_map['failure_mode_category']})")
    if fm_map['applicable_controls']:
        for ctrl in fm_map['applicable_controls']:
            print(f"  -> Control: {ctrl['control_name']} ({ctrl['control_type']})")
    else:
        print("  -> No direct controls mapped for this category.")

# Visualize the mapping using a bipartite graph
G = nx.Graph()

# Add nodes for failure modes and controls
failure_nodes = [fm['name'] for fm in identified_ai_failure_modes]
control_nodes = [c['name'] for c in resilience_controls]

G.add_nodes_from(failure_nodes, bipartite=0, label='Failure Mode')
G.add_nodes_from(control_nodes, bipartite=1, label='Resilience Control')

# Add edges between failure modes and their applicable controls
for fm_map in mapped_resilience_data:
    for ctrl in fm_map['applicable_controls']:
        G.add_edge(fm_map['failure_mode_name'], ctrl['control_name'])

# Draw the graph
pos = nx.bipartite_layout(G, failure_nodes)
plt.figure(figsize=(14, 8))
nx.draw_networkx_nodes(G, pos, nodelist=failure_nodes, node_color='skyblue', node_size=2000, label='Failure Mode')
nx.draw_networkx_nodes(G, pos, nodelist=control_nodes, node_color='lightgreen', node_size=2000, label='Resilience Control')
nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, edge_color='gray')
nx.draw_networkx_labels(G, pos, font_size=8)
plt.title('Mapping of AI Failure Modes to Resilience Controls')
plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', label='Failure Mode', markerfacecolor='skyblue', markersize=10),
                    plt.Line2D([0], [0], marker='o', color='w', label='Resilience Control', markerfacecolor='lightgreen', markersize=10)],
           loc='upper left')
plt.axis('off')
plt.tight_layout()
plt.show()


# Save resilience controls to JSON
resilience_controls_output = {
      "defined_controls": resilience_controls,
    "mapped_controls_to_failure_modes": mapped_resilience_data
}
with open('resilience_controls.json', 'w') as f:
    json.dump(resilience_controls_output, f, indent=4)
# Function to simulate a cascading failure scenario
def simulate_cascading_failure(initial_ai_error_rate, simulation_steps=10, containment_strategy=None):
    """
    Simulates a cascading failure scenario starting with AI model prediction error.
    Returns a conceptual progression of system health and impacts.
    """
    scenario_description = {
          "name": "Model Drift (False Negatives) -> Missed Maintenance -> Critical Component Failure",
        "initial_event": "AI Model Drift (False Negatives) - model consistently underpredicts RUL.",
        "triggering_failure_mode_id": "AI002",
        "containment_options": {
              "strategy_A": "Activate Shadow Model & Human Review Queue (RC001, RC003)",
            "strategy_B": "Automatic Fallback to Scheduled Maintenance (RC002)"
        }
    }

    print(f"\n--- Simulating Cascading Failure: {{scenario_description['name']}} ---")
    print(f"Initial AI Prediction Error Rate: {{initial_ai_error_rate*100:.1f}}% (e.g., MAE increase)")

    # Define conceptual probabilities and impacts
    # These are illustrative conceptual values for simulation
    p_false_negative_base = 0.05 # Baseline probability of a false negative prediction
    p_missed_maintenance_given_fn = 0.7 # Probability of missed maintenance given a false negative
    p_comp_degradation_given_mm = 0.6 # Probability of component degradation given missed maintenance
    p_catastrophic_failure_given_cd = 0.4 # Probability of catastrophic failure given component degradation

    # Effect of AI error rate on false negatives
    # Higher initial_ai_error_rate increases the actual false negative rate
    # Let's say, 1% AI error rate adds 0.5% to the FN probability.
    p_false_negative_current = p_false_negative_base + (initial_ai_error_rate * 0.5)

    simulation_results = []
    current_system_health = 100 # Start with 100% health
    current_ai_accuracy_impact = initial_ai_error_rate

    # Track states
    false_negatives_detected = False
    maintenance_missed = False
    component_degraded = False
    catastrophic_failure = False

    for step in range(simulation_steps):
        event_description = f"Step {{step+1}}: "
        impact_level = 0 # 0=none, 1=minor, 2=moderate, 3=severe, 4=catastrophic

        # 1. AI Model Drift (False Negatives)
        if current_ai_accuracy_impact > 0.05 and not false_negatives_detected: # Threshold for significant drift
            event_description += "AI Model Drift (False Negatives) detected. "
            false_negatives_detected = True
            impact_level = max(impact_level, 1) # Minor impact (monitoring needed)

            # Apply containment if active
            if containment_strategy == "strategy_A":
                event_description += "[Containment A: Shadow Model activated, Human Review Queue engaged] "
                current_ai_accuracy_impact *= 0.5 # Halve the effective impact of drift
            elif containment_strategy == "strategy_B":
                event_description += "[Containment B: Fallback to Scheduled Maintenance] "
                current_ai_accuracy_impact *= 0.2 # Significantly reduce impact, almost prevent next stage

        # 2. Missed Maintenance
        # If false negatives detected, and containment is not fully effective, maintenance is missed
        if false_negatives_detected and not maintenance_missed and np.random.rand() < p_missed_maintenance_given_fn * (1 - current_ai_accuracy_impact) * (0.5 if containment_strategy == "strategy_A" else 0.1 if containment_strategy == "strategy_B" else 1):
            event_description += "Critical maintenance missed due to false negatives. "
            maintenance_missed = True
            impact_level = max(impact_level, 2) # Moderate impact (degradation starts)
            current_system_health -= 15

        # 3. Component Degradation
        if maintenance_missed and not component_degraded and np.random.rand() < p_comp_degradation_given_mm:
            event_description += "Turbine component (bearing/rotor) begins to degrade. "
            component_degraded = True
            impact_level = max(impact_level, 3) # Severe impact (production risk)
            current_system_health -= 30

        # 4. Catastrophic Failure
        if component_degraded and not catastrophic_failure and np.random.rand() < p_catastrophic_failure_given_cd:
            event_description += "Catastrophic turbine failure and unplanned shutdown! "
            catastrophic_failure = True
            impact_level = max(impact_level, 4) # Catastrophic impact (full outage)
            current_system_health = 0 # System fully down

        # Degradation over time if issues persist and no full containment
        if not (false_negatives_detected or maintenance_missed or component_degraded or catastrophic_failure) and current_ai_accuracy_impact > 0.01:
             event_description += "AI performing normally, no major issues. "
             current_system_health -= current_ai_accuracy_impact * 5 # Small continuous degradation if model is slightly off
        elif false_negatives_detected and not catastrophic_failure:
             current_system_health -= (impact_level * 5 + current_ai_accuracy_impact * 10) # More degradation if issues persist

        current_system_health = max(0, current_system_health) # Health cannot go below 0

        simulation_results.append({
              "step": step + 1,
            "event": event_description.strip(),
            "ai_accuracy_impact": current_ai_accuracy_impact,
            "system_health": current_system_health,
            "impact_level": impact_level
        })

        if catastrophic_failure:
            print(f"Simulation terminated early due to catastrophic failure at step {{step+1}}.")
            break

    return scenario_description, simulation_results

# Define a safe degradation example
def define_safe_degradation():
    """Defines a scenario of safe degradation."""
    safe_degradation_scenario = {
          "name": "Graceful AI Degradation - Safe Fallback",
        "description": "AI model experiences minor, intermittent data pipeline issues (e.g., increased latency, missing a few sensor readings). Instead of full failure, the system automatically switches to a simpler, more robust heuristic-based prediction (RC007) and triggers a low-priority human review for the AI system itself, while maintenance decisions proceed safely. Prediction confidence drops but never goes to zero.",
        "triggering_failure_mode_id": "AI005 (partial)",
        "controls_activated": ["RC007", "RC003 (low priority)"],
        "outcome": "System operates in degraded mode, but no operational or safety impact. Maintenance continues based on heuristics while AI system is investigated."
    }
    return safe_degradation_scenario

# --- Execution ---
# Define initial AI prediction error rate for the simulation (e.g., 20% increase in MAE)
initial_error_rate = 0.20 # Represents a 20% relative increase in model error for RUL prediction

# Simulate without containment
print("\n--- Simulation WITHOUT Containment ---")
scenario_def_no_contain, simulation_output_no_contain = simulate_cascading_failure(initial_error_rate, simulation_steps=8, containment_strategy=None)

# Simulate WITH Containment Strategy B (Automatic Fallback to Scheduled Maintenance)
print("\n--- Simulation WITH Containment Strategy B (RC002) ---")
scenario_def_with_contain, simulation_output_with_contain = simulate_cascading_failure(initial_error_rate, simulation_steps=8, containment_strategy="strategy_B")

# Define the safe degradation example
safe_degradation_example = define_safe_degradation()
print(f"\n--- Safe Degradation Example: {{safe_degradation_example['name']}} ---")
print(f"Description: {{safe_degradation_example['description']}}")
print(f"Outcome: {{safe_degradation_example['outcome']}}")


# Visualize cascading failure paths and system health
df_no_contain = pd.DataFrame(simulation_output_no_contain)
df_with_contain = pd.DataFrame(simulation_output_with_contain)

plt.figure(figsize=(12, 6))
plt.plot(df_no_contain['step'], df_no_contain['system_health'], marker='o', label='No Containment', color='red')
plt.plot(df_with_contain['step'], df_with_contain['system_health'], marker='x', label='With Containment (Strategy B)', color='green', linestyle='--')
plt.title(f'Cascading Failure Simulation: System Health vs. Time (Initial AI Error: {{initial_error_rate*100:.1f}}%)')
plt.xlabel('Simulation Step (Time)')
plt.ylabel('System Health (%)')
plt.ylim(0, 105)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

# Save cascading failure analysis to JSON
cascading_failure_analysis_output = {
      "scenario_definition": scenario_def_no_contain,
    "simulation_results_no_containment": simulation_output_no_contain,
    "simulation_results_with_containment_B": simulation_output_with_contain,
    "safe_degradation_example": safe_degradation_example
}
with open('cascading_failure_analysis.json', 'w') as f:
    json.dump(cascading_failure_analysis_output, f, indent=4)
# Function to load sample recovery plans
def load_sample_recovery_plan(file_path):
    """Loads a markdown file containing sample recovery plans."""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: Sample recovery plan not found at {file_path}. Generating dummy content.")
        return """
# Sample AI Service Recovery Plan (Placeholder)

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

# Function to define specific rollback triggers and procedures
def define_rollback_procedures_and_recovery_objectives():
    """Defines concrete rollback triggers, procedures, and RTO/RPO for the AI service."""
    rollback_triggers = [
          {"id": "RT001", "name": "Sustained Model Performance Degradation",
         "description": "AI model's MAE for RUL prediction exceeds 1.5 times the baseline for 30 consecutive minutes.",
         "metric": "MAE", "threshold_multiplier": 1.5, "duration_minutes": 30},
        {"id": "RT002", "name": "Data Pipeline Ingestion Stoppage",
         "description": "No new sensor data ingested into the AI feature store for 15 minutes.",
         "metric": "Data Ingestion Rate", "threshold_value": 0, "duration_minutes": 15},
        {"id": "RT003", "name": "Excessive Inference Latency",
         "description": "Average inference latency for RUL predictions exceeds 500ms for 5 consecutive minutes.",
         "metric": "Latency", "threshold_ms": 500, "duration_minutes": 5},
    ]

    rollback_procedures = [
          {"id": "RP001", "name": "Automated Deployment of Last Validated Model",
         "description": "Triggered by RT001. Automatically deploys the last known good AI model version from the production registry. Requires a pre-validated model artifact.",
         "trigger_ids": ["RT001"], "validation_steps": ["Run post-deployment sanity checks on recent production data.", "Monitor initial 1 hour of inference metrics."]},
        {"id": "RP002", "name": "Manual Revert to Heuristic-Based Prediction",
         "description": "Triggered by RT002/RT003. A human operator manually switches the inference endpoint to a pre-defined heuristic rule-based system for RUL prediction. This provides basic, safe operation.",
         "trigger_ids": ["RT002", "RT003"], "validation_steps": ["Confirm heuristic system is active and providing output.", "Verify outputs against known safe thresholds."]},
        {"id": "RP003", "name": "Data Pipeline Restart & Backfill",
         "description": "Triggered by RT002. Automated attempt to restart data ingestion services and backfill missing data from raw sensor archives.",
         "trigger_ids": ["RT002"], "validation_steps": ["Verify data ingestion rate resumes normal levels.", "Check data integrity of backfilled data."]}
    ]

    recovery_objectives = {
          "RTO_AI_Service_Degraded_Mode": "2 hours", # Max time to restore minimal predictive capability (e.g., heuristics)
        "RTO_AI_Service_Full_Functionality": "8 hours", # Max time to restore full AI model capability
        "RPO_AI_Data_for_Retraining": "30 minutes", # Max acceptable data loss for model retraining/recalibration purposes
        "RPO_AI_Inference_State": "10 minutes" # Max acceptable staleness of AI inference state if applicable
    }

    validation_for_re_enablement = [
          {"step": "1", "description": "AI Model Performance: MAE within 1.1x baseline and no significant drift for 24 hours.", "metric_kpi": "Model_MAE_Stability"},
        {"step": "2", "description": "Data Pipeline Health: 99.9% data ingestion success rate and no errors for 12 hours.", "metric_kpi": "Data_Pipeline_Health"},
        {"step": "3", "description": "Inference Infrastructure Stability: Average inference latency below 200ms and 99.9% uptime for 48 hours.", "metric_kpi": "Inference_Stability"}
    ]

    return rollback_triggers, rollback_procedures, recovery_objectives, validation_for_re_enablement

# --- Execution ---
# Load the sample recovery plan
sample_recovery_plan_path = os.path.join(script_dir, 'sample_recovery_plans.md')
sample_recovery_plan_md = load_sample_recovery_plan(sample_recovery_plan_path)
print("\n--- Sample Recovery Plan Content (Guidance) ---")
print(sample_recovery_plan_md)

# Define specific rollback procedures and recovery objectives for our use case
rollback_triggers, rollback_procedures, recovery_objectives, validation_for_re_enablement = define_rollback_procedures_and_recovery_objectives()

print("\n--- Defined Rollback Triggers ---")
for trigger in rollback_triggers:
    print(f"- {trigger['name']} (ID: {trigger['id']}): {trigger['description']}")

print("\n--- Defined Rollback Procedures ---")
for procedure in rollback_procedures:
    print(f"- {procedure['name']} (ID: {procedure['id']}): Triggers: {', '.join(procedure['trigger_ids'])}")
    print(f"  Description: {procedure['description']}")

print("\n--- Defined Recovery Objectives (RTO/RPO) ---")
for obj, val in recovery_objectives.items():
    print(f"- {obj}: {val}")

print("\n--- Validation Steps for AI Service Re-enablement ---")
for step in validation_for_re_enablement:
    print(f"- Step {step['step']}: {step['description']} (KPI: {step['metric_kpi']})")

# Prepare markdown output for recovery plan
recovery_plan_md_content = f"""
# AI Predictive Maintenance Service Recovery Plan - Steam Turbine Unit 3

This document outlines the specific procedures for detecting, mitigating, and recovering from failures in the AI-driven predictive maintenance service for High-Pressure Steam Turbine Unit 3.

## 1. Rollback Triggers
These are the conditions that will automatically or manually trigger a rollback or degraded mode operation.

"""
for trigger in rollback_triggers:
    recovery_plan_md_content += f"- **{trigger['name']} ({trigger['id']}):** {trigger['description']}\n"
recovery_plan_md_content += """

## 2. Rollback Procedures
These are the actions to be taken when a rollback trigger is detected.

"""
for procedure in rollback_procedures:
    recovery_plan_md_content += f"- **{procedure['name']} ({procedure['id']}):**\n"
    recovery_plan_md_content += f"  - **Triggers:** {', '.join(procedure['trigger_ids'])}\n"
    recovery_plan_md_content += f"  - **Description:** {procedure['description']}\n"
    recovery_plan_md_content += f"  - **Validation Steps:** {'; '.join(procedure['validation_steps'])}\n"
recovery_plan_md_content += """

## 3. Recovery Objectives (RTO/RPO)
These objectives define the target timelines for service restoration and acceptable data loss.

"""
for obj, val in recovery_objectives.items():
    recovery_plan_md_content += f"- **{obj.replace('_', ' ')}:** {val}\n"
recovery_plan_md_content += """

## 4. Validation Steps for Full AI Service Re-enablement
Before fully re-enabling the primary AI service, the following conditions must be met to ensure stability and accuracy.

"""
for step in validation_for_re_enablement:
    recovery_plan_md_content += f"- **Step {step['step']}:** {step['description']} (Key Performance Indicator: {step['metric_kpi']})\n"

# Save recovery plan to markdown file
with open('recovery_plan.md', 'w') as f:
    f.write(recovery_plan_md_content)

# Save resilience KPIs to JSON (combining RTO/RPO with potential other KPIs)
resilience_kpis_output = {
      "recovery_objectives": recovery_objectives,
    "rollback_triggers": rollback_triggers,
    "validation_for_re_enablement": validation_for_re_enablement
}
with open('resilience_kpis.json', 'w') as f:
    json.dump(resilience_kpis_output, f, indent=4)
# Function to generate SHA-256 hash for a file
def generate_file_hash(file_path):
    """Generates SHA-256 hash for a given file."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

# Function to generate the overall AI Resilience Playbook (JSON)
def generate_ai_resilience_playbook(use_case, failure_modes, controls, cascading_analysis, recovery_plans, kpis):
    """Combines all generated components into a single AI Resilience Playbook JSON."""
    playbook = {
          "playbook_version": "1.0",
        "generated_date": datetime.now().isoformat(),
        "use_case_definition": use_case,
        "ai_failure_mode_analysis": failure_modes,
        "resilience_controls_mapping": controls,
        "cascading_failure_analysis": cascading_analysis,
        "recovery_and_rollback_planning": {
              "rollback_triggers": kpis["rollback_triggers"],
            "rollback_procedures": recovery_plans["rollback_procedures"], # Use the structured procedures from the function call
            "recovery_objectives": kpis["recovery_objectives"],
            "validation_for_re_enablement": kpis["validation_for_re_enablement"],
            "recovery_plan_markdown_path": "recovery_plan.md" # Reference to the markdown file
        }
    }
    return playbook

# Function to generate an executive summary
def generate_executive_summary(playbook_data, run_id):
    """Generates an executive summary markdown."""
    summary_content = f"""
# Executive Summary: Predictive Maintenance AI Resilience Playbook - {playbook_data['use_case_definition']['equipment']}

**Date Generated:** {playbook_data['generated_date']}
**Playbook Version:** {playbook_data['playbook_version']}
**Run ID:** {run_id}

## 1. Overview
This playbook details the resilience strategy for the AI-driven predictive maintenance service for **{playbook_data['use_case_definition']['equipment']}**. The AI system predicts the Remaining Useful Life (RUL) of critical components like bearings and rotor blades, crucial for preventing unplanned downtime and ensuring safety. This document outlines how we ensure continuous operation even in the face of AI system failures.

## 2. Key Findings from Failure Mode Analysis
- Identified {len(playbook_data['ai_failure_mode_analysis']['identified_ai_failure_modes'])} critical AI-related failure modes across Data, Model, and Infrastructure categories.
- Primary vulnerabilities include `Model Drift (False Negatives)` and `Sensor Data Corruption`, which can directly lead to missed maintenance.

## 3. Resilience Controls Implemented
- **Redundancy:** Deployment of a `Shadow Model` for continuous monitoring and potential fallback.
- **Graceful Degradation:** `Automatic Fallback to Scheduled Maintenance` is a key strategy if AI predictions become unreliable.
- **Manual Override:** `Human Expert Alert Queue` for high-severity or ambiguous alerts.
- **Proactive Monitoring:** `Sensor Data Anomaly Detection` and `Model Performance Monitoring` are in place.

## 4. Cascading Failure Insights
- Simulation of "Model Drift (False Negatives)" demonstrated rapid system health degradation leading to catastrophic failure without containment.
- With containment (e.g., Automatic Fallback to Scheduled Maintenance), system health stabilizes, preventing critical outages. This validates our investment in resilience controls.
- A `Graceful AI Degradation` scenario was defined, showing that minor issues can be handled without operational impact through heuristic fallbacks.

## 5. Recovery Objectives and Rollback Strategy
- **AI Service RTO (Degraded Mode):** {playbook_data['recovery_and_rollback_planning']['recovery_objectives']['RTO_AI_Service_Degraded_Mode']}
- **AI Service RPO (Data for Retraining):** {playbook_data['recovery_and_rollback_planning']['recovery_objectives']['RPO_AI_Data_for_Retraining']}
- Key rollback triggers are defined based on MAE drift, data ingestion, and inference latency.
- Automated and manual rollback procedures include deploying previous model versions and reverting to heuristic predictions.
- Clear validation steps are defined for safe re-enablement of the AI service.

## 6. Conclusion
This playbook significantly enhances the operational resilience of our predictive maintenance AI. By proactively addressing AI system failures through structured analysis, robust controls, and clear recovery procedures, we minimize downtime, prevent costly equipment damage, and avoid safety incidents, thereby fostering trustworthiness in our AI deployments.

"""
    return summary_content

# --- Execution ---
# Define a run ID for this session
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

# Consolidate all data from previous steps for the full playbook JSON
full_ai_resilience_playbook_data = generate_ai_resilience_playbook(
      use_case=failure_mode_analysis_output["use_case"],
    failure_modes=failure_mode_analysis_output,
    controls=resilience_controls_output,
    cascading_analysis=cascading_failure_analysis_output,
    recovery_plans={"rollback_procedures": rollback_procedures}, # Pass the structured procedures
    kpis=resilience_kpis_output
)

# Define output directory
output_dir = "reports/session14"
os.makedirs(output_dir, exist_ok=True)

# Define file paths for all artifacts
artifact_paths = {
      "sector_playbook.json": os.path.join(output_dir, f"sector_playbook_{run_id}.json"),
    "failure_mode_analysis.json": os.path.join(output_dir, f"failure_mode_analysis_{run_id}.json"),
    "resilience_controls.json": os.path.join(output_dir, f"resilience_controls_{run_id}.json"),
    "recovery_plan.md": os.path.join(output_dir, f"recovery_plan_{run_id}.md"),
    "resilience_kpis.json": os.path.join(output_dir, f"resilience_kpis_{run_id}.json"),
    "session14_executive_summary.md": os.path.join(output_dir, f"session14_executive_summary_{run_id}.md"),
    "config_snapshot.json": os.path.join(output_dir, f"config_snapshot_{run_id}.json")
}

# --- Write Artifacts ---
# Save the full AI Resilience Playbook
with open(artifact_paths["sector_playbook.json"], 'w') as f:
    json.dump(full_ai_resilience_playbook_data, f, indent=4)

# Save the individual JSONs (already done in previous steps, but ensure correct naming/path for final output)
# Re-save to the `reports/session14` directory with `run_id` suffix
with open(artifact_paths["failure_mode_analysis.json"], 'w') as f:
    json.dump(failure_mode_analysis_output, f, indent=4)
with open(artifact_paths["resilience_controls.json"], 'w') as f:
    json.dump(resilience_controls_output, f, indent=4)
with open(artifact_paths["resilience_kpis.json"], 'w') as f:
    json.dump(resilience_kpis_output, f, indent=4)

# Save the recovery plan Markdown (already generated, move/copy to output dir)
with open('recovery_plan.md', 'r') as src, open(artifact_paths["recovery_plan.md"], 'w') as dst:
    dst.write(src.read())

# Generate and save the executive summary
executive_summary_content = generate_executive_summary(full_ai_resilience_playbook_data, run_id)
with open(artifact_paths["session14_executive_summary.md"], 'w') as f:
    f.write(executive_summary_content)

# Generate and save a config snapshot (conceptual, for demonstration)
config_snapshot_data = {
      "model_version": "v2.1_production_RUL",
    "data_pipeline_version": "v3.0_sensor_ingestion",
    "inference_engine_version": "v1.2_tensorflow_serving",
    "monitoring_stack_version": "v1.5_prometheus_grafana",
    "timestamp": datetime.now().isoformat()
}
with open(artifact_paths["config_snapshot.json"], 'w') as f:
    json.dump(config_snapshot_data, f, indent=4)

# Generate Evidence Manifest with SHA-256 hashes
evidence_manifest = {
      "run_id": run_id,
    "generated_at": datetime.now().isoformat(),
    "artifacts": []
}
for name, path in artifact_paths.items():
    if os.path.exists(path):
        artifact_hash = generate_file_hash(path)
        evidence_manifest["artifacts"].append({
              "name": name,
            "path": path,
            "sha256_hash": artifact_hash
        })
    else:
        print(f"Warning: Artifact not found at {path}. Skipping hash generation.")

with open(os.path.join(output_dir, f"evidence_manifest_{run_id}.json"), 'w') as f:
    json.dump(evidence_manifest, f, indent=4)

print(f"\n--- AI Resilience Playbook and Evidentiary Artifacts Generated (Run ID: {run_id}) ---")
print(f"All artifacts saved in: {output_dir}/")
for artifact in evidence_manifest["artifacts"]:
    print(f"- {artifact['name']}: {artifact['path']} (Hash: {artifact['sha256_hash']})")
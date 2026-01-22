
# Streamlit Application Specification: AI-Driven Predictive Maintenance Resilience Playbook Builder

## 1. Application Overview

### Purpose of the Application
The "AI-Driven Predictive Maintenance Resilience Playbook Builder" application serves as a comprehensive tool for Systems Engineers, Reliability/SRE Engineers, and AI Risk Leads to design, analyze, and document robust resilience strategies for AI systems in critical industrial infrastructure. It operationalizes AI risk management by translating enterprise AI controls into resilience-first sector playbooks, focusing on preventing cascading failures, unsafe states, and prolonged outages in AI-driven predictive maintenance systems. The app emphasizes engineering realism, addressing partial failures, degraded modes, rollback and redundancy, and human override under stress.

### High-Level Story Flow of the Application
The application guides the user (primarily a Systems Engineer) through a structured workflow to build an AI Resilience Playbook:

1.  **System Configuration & Use Case Definition:** The user starts by selecting the sector (Engineering/Critical Infrastructure) and specific use case (e.g., Predictive Maintenance). They configure system parameters like AI type, automation level, and uptime requirements.
2.  **Failure Mode Mapping:** Based on the configured use case, the application helps identify AI-specific failure modes (e.g., model drift, data pipeline issues) by loading relevant industry templates and presenting a categorized overview.
3.  **Resilience Control Selection:** The user then defines and maps resilience controls (e.g., shadow models, graceful degradation, manual overrides) to the identified failure modes, visualizing their relationships.
4.  **Cascading Failure Simulation:** A critical cascading failure scenario is simulated, allowing the user to observe the impact of AI failures with and without containment strategies. A safe degradation example is also presented.
5.  **Recovery Plan Building:** The application facilitates the definition of precise rollback triggers, detailed rollback procedures, and sets Recovery Time Objectives (RTO) and Recovery Point Objectives (RPO).
6.  **KPI & Alert Designing:** Key Resilience KPIs and validation steps for re-enabling the AI service are specified.
7.  **Playbook Export:** Finally, all generated analysis, mappings, simulations, and recovery definitions are consolidated into a comprehensive, exportable "AI Resilience Playbook" (JSON, Markdown, and supporting files), complete with SHA-256 hashes for integrity and traceability.

This workflow simulates a real-world task where a Systems Engineer systematically builds a robust resilience plan for a critical AI system.

## 2. Code Requirements

### Import Statement

```python
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import os
import json
import zipfile
from datetime import datetime, timedelta

# Assume source.py contains all required function definitions and necessary internal imports
# (e.g., numpy, seaborn, hashlib from its original notebook cells)
from source import (
    load_templates,
    identify_ai_failure_modes,
    define_resilience_controls,
    map_controls_to_failure_modes,
    simulate_cascading_failure,
    define_safe_degradation,
    load_sample_recovery_plan,
    define_rollback_procedures_and_recovery_objectives,
    generate_file_hash,
    generate_ai_resilience_playbook,
    generate_executive_summary
)
```

### `st.session_state` Design

`st.session_state` is used to preserve user selections and computation results across different pages and interactions.

*   **Initialization (at application startup):**
    ```python
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Home"
    if "run_id" not in st.session_state:
        st.session_state.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if "output_dir" not in st.session_state:
        st.session_state.output_dir = os.path.join("reports", "session14", st.session_state.run_id)
        os.makedirs(st.session_state.output_dir, exist_ok=True) # Ensure output directory exists

    # Configuration defaults
    if "sector" not in st.session_state:
        st.session_state.sector = "Engineering"
    if "use_case" not in st.session_state:
        st.session_state.use_case = "Predictive Maintenance"
    if "system_type" not in st.session_state:
        st.session_state.system_type = "ML"
    if "automation_level" not in st.session_state:
        st.session_state.automation_level = "High"
    if "uptime_requirement" not in st.session_state:
        st.session_state.uptime_requirement = "99.9%"
    if "human_override" not in st.session_state:
        st.session_state.human_override = "Available"
    if "config_snapshot_data" not in st.session_state:
        st.session_state.config_snapshot_data = {} # Will be populated after config save

    # Data loading (pre-calculated or loaded on app start for templates)
    if "engineering_templates" not in st.session_state:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        engineering_failure_templates_path = os.path.join(script_dir, 'source', 'engineering_failure_templates.json')
        st.session_state.engineering_templates = load_templates(engineering_failure_templates_path)
    if "infrastructure_templates" not in st.session_state:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        infrastructure_failure_templates_path = os.path.join(script_dir, 'source', 'infrastructure_failure_templates.json')
        st.session_state.infrastructure_templates = load_templates(infrastructure_failure_templates_path)
    if "sample_recovery_plan_md" not in st.session_state:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sample_recovery_plan_path = os.path.join(script_dir, 'source', 'sample_recovery_plans.md')
        st.session_state.sample_recovery_plan_md = load_sample_recovery_plan(sample_recovery_plan_path)

    # Results from steps (initialized to None or empty structures)
    if "predictive_maintenance_use_case" not in st.session_state:
        st.session_state.predictive_maintenance_use_case = {}
    if "identified_ai_failure_modes" not in st.session_state:
        st.session_state.identified_ai_failure_modes = []
    if "df_failure_modes" not in st.session_state:
        st.session_state.df_failure_modes = pd.DataFrame()
    if "failure_mode_analysis_output" not in st.session_state:
        st.session_state.failure_mode_analysis_output = {}

    if "resilience_controls" not in st.session_state:
        st.session_state.resilience_controls = define_resilience_controls() # Can be pre-loaded as it's static
    if "mapped_resilience_data" not in st.session_state:
        st.session_state.mapped_resilience_data = []
    if "resilience_controls_output" not in st.session_state:
        st.session_state.resilience_controls_output = {}

    if "initial_error_rate" not in st.session_state:
        st.session_state.initial_error_rate = 0.20
    if "containment_strategy" not in st.session_state:
        st.session_state.containment_strategy = "None"
    if "simulation_output_no_contain" not in st.session_state:
        st.session_state.simulation_output_no_contain = []
    if "simulation_output_with_contain" not in st.session_state:
        st.session_state.simulation_output_with_contain = []
    if "safe_degradation_example" not in st.session_state:
        st.session_state.safe_degradation_example = {}
    if "cascading_failure_analysis_output" not in st.session_state:
        st.session_state.cascading_failure_analysis_output = {}

    if "rollback_triggers" not in st.session_state:
        st.session_state.rollback_triggers = []
    if "rollback_procedures" not in st.session_state:
        st.session_state.rollback_procedures = []
    if "recovery_objectives" not in st.session_state:
        st.session_state.recovery_objectives = {}
    if "validation_for_re_enablement" not in st.session_state:
        st.session_state.validation_for_re_enablement = []
    if "recovery_plan_md_content" not in st.session_state:
        st.session_state.recovery_plan_md_content = ""
    if "resilience_kpis_output" not in st.session_state:
        st.session_state.resilience_kpis_output = {}

    # Final outputs
    if "full_ai_resilience_playbook_data" not in st.session_state:
        st.session_state.full_ai_resilience_playbook_data = {}
    if "executive_summary_content" not in st.session_state:
        st.session_state.executive_summary_content = ""
    if "artifact_paths" not in st.session_state:
        st.session_state.artifact_paths = {}
    if "evidence_manifest" not in st.session_state:
        st.session_state.evidence_manifest = {}
    ```

*   **Update:** Widgets (selectbox, text_input, slider, button) update `st.session_state` directly or trigger functions that update it.
    *   Example: `st.selectbox("Sector", ["Engineering", "Critical Infrastructure"], key="sector")`
    *   Example: `st.button("Confirm Configuration")` triggers a function that updates `st.session_state.config_snapshot_data`, `st.session_state.predictive_maintenance_use_case`, `st.session_state.identified_ai_failure_modes`, and related `failure_mode_analysis_output`.

*   **Read Across Pages:** Each page (conditional rendering block) reads relevant data from `st.session_state` to display results or pre-fill inputs for subsequent steps.
    *   Example: `st.markdown(f"**Selected Sector:** {st.session_state.sector}")`
    *   Example: `df_failure_modes = st.session_state.df_failure_modes` is used to plot.

### UI Interactions and `source.py` Function Calls

The Streamlit app simulates a multi-page experience using `st.sidebar.selectbox`.

**Sidebar Navigation:**
```python
page_options = [
    "Home",
    "1. System Configuration",
    "2. Failure Mode Mapper",
    "3. Resilience Control Selector",
    "4. Cascading Failure Simulation",
    "5. Recovery Plan Builder",
    "6. KPI & Alert Designer",
    "7. Export Playbook"
]
st.sidebar.title("AI Resilience Playbook Builder")
st.session_state.current_page = st.sidebar.selectbox(
    "Navigate to a Section",
    page_options,
    index=page_options.index(st.session_state.current_page) # Set initial selection based on session_state
)
```

---

#### Home Page

*   **Markdown:**
    ```python
    st.markdown(f"# AI Resilience Playbook Builder")
    st.markdown(f"## Ensuring Uptime in Critical Industrial Infrastructure")
    st.markdown(f"As a Systems Engineer at \"Innovate Manufacturing Inc.\", my primary responsibility is to design and integrate advanced AI solutions into our critical industrial processes. Today, I'm focusing on our new predictive maintenance system for the high-pressure steam turbines in our main production facility. These turbines are the heart of our operations; their unplanned downtime can cost millions per hour and pose significant safety risks.")
    st.markdown(f"The AI model aims to predict equipment failures days or weeks in advance, allowing for proactive maintenance. However, an AI system, like any complex component, can fail. My task is to build a comprehensive \"AI Resilience Playbook\" to ensure continuous operation and safety, even if the AI model or its data pipelines experience issues. This playbook will detail potential AI-related failure modes, define proactive resilience controls, simulate cascading impacts, and establish clear recovery procedures. This isn't just about preventing downtime; it's about safeguarding our entire operation.")
    st.markdown(f"---")
    st.markdown(f"**Current Session Run ID:** `{st.session_state.run_id}`")
    st.markdown(f"**Output Directory:** `{st.session_state.output_dir}`")
    st.markdown(f"Navigate using the sidebar to build your playbook step-by-step.")
    ```
*   **Function Calls:** None directly, but `st.session_state.run_id` and `st.session_state.output_dir` are initialized. Template files are loaded into session state upon app startup (as defined in `st.session_state` initialization).

---

#### 1. System Configuration

*   **Markdown:**
    ```python
    st.markdown(f"# 1. System Configuration & Use Case Wizard")
    st.markdown(f"Define the sector, specific AI use case, and system parameters for which you are building the resilience playbook.")
    st.markdown(f"---")
    ```
*   **Widgets:**
    *   `st.selectbox("Sector", ["Engineering", "Critical Infrastructure"], key="sector")`
    *   `st.selectbox("Use Case", ["Predictive Maintenance", "Anomaly Detection", "Forecasting / Optimization"], key="use_case")`
    *   `st.selectbox("System Type", ["ML", "LLM", "Agent"], key="system_type")`
    *   `st.selectbox("Automation Level", ["Low", "Medium", "High", "Full"], key="automation_level")`
    *   `st.selectbox("Uptime Requirement", ["99%", "99.9%", "99.99%", "99.999%"], key="uptime_requirement")`
    *   `st.selectbox("Human Override Availability", ["Available", "Limited", "Not Available"], key="human_override")`
    *   `st.button("Confirm Configuration & Identify Failure Modes")`

*   **Function Calls (on button click):**
    ```python
    if st.button("Confirm Configuration & Identify Failure Modes"):
        st.session_state.config_snapshot_data = {
            "system_type": st.session_state.system_type,
            "automation_level": st.session_state.automation_level,
            "uptime_requirement": st.session_state.uptime_requirement,
            "human_override_availability": st.session_state.human_override,
            "sector": st.session_state.sector,
            "use_case": st.session_state.use_case,
            "timestamp": datetime.now().isoformat()
        }

        # Construct predictive_maintenance_use_case based on user input
        st.session_state.predictive_maintenance_use_case = {
            "name": f"{st.session_state.use_case} for {st.session_state.sector} Systems",
            "equipment": "High-Pressure Steam Turbine (Unit 3)" if st.session_state.sector == "Engineering" and st.session_state.use_case == "Predictive Maintenance" else "Generic Equipment/System",
            "criticality": "High - Direct impact on production, safety, and operational costs",
            "ai_function": f"Predict Remaining Useful Life (RUL) of components based on sensor data for {st.session_state.use_case}.",
            "sensors": ["Vibration", "Temperature", "Pressure"] # Simplified for generic case
        }

        st.session_state.identified_ai_failure_modes = identify_ai_failure_modes(
            st.session_state.predictive_maintenance_use_case,
            st.session_state.engineering_templates,
            st.session_state.infrastructure_templates
        )
        st.session_state.df_failure_modes = pd.DataFrame(st.session_state.identified_ai_failure_modes)

        st.session_state.failure_mode_analysis_output = {
            "use_case": st.session_state.predictive_maintenance_use_case,
            "identified_ai_failure_modes": st.session_state.identified_ai_failure_modes,
            "engineering_templates_used": st.session_state.engineering_templates,
            "infrastructure_templates_used": st.session_state.infrastructure_templates
        }
        st.success("Configuration saved and AI failure modes identified!")
        st.session_state.current_page = "2. Failure Mode Mapper"
        st.experimental_rerun()
    ```

---

#### 2. Failure Mode Mapper

*   **Markdown:**
    ```python
    st.markdown(f"# 2. Failure-Mode Mapper")
    st.markdown(f"Here, we identify potential AI-specific failure modes that could compromise the AI's ability to accurately predict maintenance needs, leading to either missed critical issues or unnecessary shutdowns. We leverage existing engineering and infrastructure failure templates to guide this process.")
    st.markdown(f"---")
    st.markdown(f"**Scenario Context:** Our facility relies on a set of high-pressure steam turbines. An AI model analyzes sensor data (vibration, temperature, pressure, lubrication levels) to predict the Remaining Useful Life (RUL) of critical turbine components, specifically bearings and rotor blades. Early prediction allows us to schedule maintenance during planned downtimes, preventing catastrophic failures.")
    st.markdown(f"We categorize failure modes related to the AI model itself, its data inputs, and the underlying inference infrastructure.")
    st.markdown(f"### Identified AI-Related Failure Modes:")
    ```
    *   **Formula:**
        ```python
        st.markdown(r"**Mathematical Context for RUL Prediction:**")
        st.markdown(r"The AI model uses various sensor inputs $X = [x_1, x_2, ..., x_n]$ to predict RUL, denoted as $RUL_{pred}$. The true RUL, $RUL_{true}$, is unknown until failure. The model's performance is often evaluated by metrics like Mean Absolute Error (MAE):")
        st.markdown(r"$$MAE = \frac{1}{N} \sum_{i=1}^{N} |RUL_{true,i} - RUL_{pred,i}|$$")
        st.markdown(r"where $N$ is the number of predictions, $RUL_{true,i}$ is the true Remaining Useful Life for prediction $i$, and $RUL_{pred,i}$ is the predicted Remaining Useful Life for prediction $i$.")
        st.markdown(r"A high MAE indicates poor prediction, which is a critical AI failure mode.")
        ```
*   **Display Data & Visualization:**
    ```python
    if not st.session_state.df_failure_modes.empty:
        st.dataframe(st.session_state.df_failure_modes)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(data=st.session_state.df_failure_modes, x='category', palette='viridis', ax=ax)
        ax.set_title('Identified AI Failure Modes by Category for Steam Turbine Predictive Maintenance')
        ax.set_xlabel('Failure Mode Category')
        ax.set_ylabel('Number of Failure Modes')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig)
        
        st.markdown(f"### Explanation of Execution")
        st.markdown(f"The output lists the identified AI-related failure modes, categorized by Data, Model, and Infrastructure. The bar chart provides a clear visual summary, showing where our AI system is most vulnerable. As a Systems Engineer, this breakdown helps me focus my resilience efforts. For example, 'Data' and 'Model' categories appear to have more failure modes, indicating a need for robust data validation and model monitoring strategies. This structured overview of vulnerabilities serves as a foundational document for our playbook.")
    else:
        st.warning("Please configure the system and identify failure modes in the '1. System Configuration' section first.")
    ```
*   **Function Calls:** `identify_ai_failure_modes` is called in the previous step. `pd.DataFrame` is used for display. Matplotlib/Seaborn for plotting.

---

#### 3. Resilience Control Selector

*   **Markdown:**
    ```python
    st.markdown(f"# 3. Designing and Mapping Resilience Controls")
    st.markdown(f"Now that I've identified the key AI-related failure modes, I need to design and map specific resilience controls to mitigate their impact. This involves defining strategies like redundancy, graceful degradation, and automatic fallback mechanisms. The goal is to ensure that even if a specific AI component fails, the overall predictive maintenance system can either continue operating in a degraded but safe mode or automatically switch to an alternative, reliable method.")
    st.markdown(f"---")
    ```
    *   **Formula:**
        ```python
        st.markdown(r"**Mathematical Context for Control Effectiveness:**")
        st.markdown(r"Resilience controls aim to reduce the probability of system impact or the severity of that impact. Conceptually, for a given failure mode $FM$, a control $C$ reduces the impact $I$ with a certain effectiveness $E_C$.")
        st.markdown(r"$$I_{mitigated} = I_{unmitigated} \times (1 - E_C)$$")
        st.markdown(r"where $E_C \in [0, 1]$ represents the reduction in impact. This effectiveness will be simulated in the next section.")
        ```
*   **Function Calls (on page load, if previous step is complete):**
    ```python
    if st.session_state.identified_ai_failure_modes:
        st.session_state.resilience_controls = define_resilience_controls()
        st.session_state.mapped_resilience_data = map_controls_to_failure_modes(
            st.session_state.identified_ai_failure_modes,
            st.session_state.resilience_controls
        )
        st.session_state.resilience_controls_output = {
            "defined_controls": st.session_state.resilience_controls,
            "mapped_controls_to_failure_modes": st.session_state.mapped_resilience_data
        }

        st.markdown(f"### Defined Resilience Controls:")
        st.dataframe(pd.DataFrame(st.session_state.resilience_controls))

        st.markdown(f"### Resilience Control Mapping:")
        for fm_map in st.session_state.mapped_resilience_data:
            st.markdown(f"- **Failure Mode:** {fm_map['failure_mode_name']} ({fm_map['failure_mode_category']})")
            if fm_map['applicable_controls']:
                for ctrl in fm_map['applicable_controls']:
                    st.markdown(f"  -> **Control:** {ctrl['control_name']} ({ctrl['control_type']}) - *{ctrl['description']}*")
            else:
                st.markdown(f"  -> No direct controls mapped for this category.")

        st.markdown(f"### Visualization of Control Mapping:")
        # Visualize the mapping using a bipartite graph
        G = nx.Graph()
        failure_nodes = [fm['failure_mode_name'] for fm in st.session_state.identified_ai_failure_modes]
        control_nodes = [c['name'] for c in st.session_state.resilience_controls]
        G.add_nodes_from(failure_nodes, bipartite=0, label='Failure Mode')
        G.add_nodes_from(control_nodes, bipartite=1, label='Resilience Control')

        for fm_map in st.session_state.mapped_resilience_data:
            for ctrl in fm_map['applicable_controls']:
                G.add_edge(fm_map['failure_mode_name'], ctrl['control_name'])

        pos = nx.bipartite_layout(G, failure_nodes)
        fig_graph, ax_graph = plt.subplots(figsize=(14, 8))
        nx.draw_networkx_nodes(G, pos, nodelist=failure_nodes, node_color='skyblue', node_size=2000, label='Failure Mode', ax=ax_graph)
        nx.draw_networkx_nodes(G, pos, nodelist=control_nodes, node_color='lightgreen', node_size=2000, label='Resilience Control', ax=ax_graph)
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, edge_color='gray', ax=ax_graph)
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax_graph)
        ax_graph.set_title('Mapping of AI Failure Modes to Resilience Controls')
        ax_graph.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', label='Failure Mode', markerfacecolor='skyblue', markersize=10),
                                 plt.Line2D([0], [0], marker='o', color='w', label='Resilience Control', markerfacecolor='lightgreen', markersize=10)],
                       loc='upper left')
        ax_graph.axis('off')
        st.pyplot(fig_graph)

        st.markdown(f"### Explanation of Execution")
        st.markdown(f"The output displays a detailed list of which resilience controls apply to each identified AI failure mode. The bipartite graph provides an intuitive visual representation of these relationships, making it easy to see which controls cover multiple failure types and which failure modes are adequately (or insufficiently) protected. For instance, \"Automatic Fallback to Scheduled Maintenance\" (RC002) is mapped to several failure categories, highlighting its broad utility as a graceful degradation strategy. This mapping ensures that every critical AI vulnerability has a corresponding mitigation strategy, directly supporting the objective of operational resilience.")
    else:
        st.warning("Please configure the system and identify failure modes in the '1. System Configuration' section first.")
    ```

---

#### 4. Cascading Failure Simulation

*   **Markdown:**
    ```python
    st.markdown(f"# 4. Simulating a Cascading Failure Scenario and Defining Containment")
    st.markdown(f"Even with resilience controls, complex AI systems can experience cascading failures. My role as a Systems Engineer requires me to anticipate these scenarios and define containment strategies. We will simulate a specific critical scenario: \"Model Drift (False Negatives)\" leading to missed maintenance and subsequent equipment damage.")
    st.markdown(f"---")
    ```
    *   **Formula:**
        ```python
        st.markdown(r"**Mathematical Context for Cascading Failure Impact:**")
        st.markdown(r"Let $P(FM_1)$ be the probability of an initial failure mode $FM_1$. If $FM_1$ occurs, it increases the probability of a subsequent failure mode $FM_2$ by a factor $k_{1,2} > 1$, such that $P(FM_2|FM_1) = k_{1,2} \times P(FM_2)_{baseline}$. The overall system impact $S$ can be a function of the sum of individual failure severities $s_i$ for each activated failure $FM_i$, potentially weighted by their interdependencies.")
        st.markdown(r"$$S = \sum_{i=1}^{n} s_i \times w_i$$")
        st.markdown(r"where $w_i$ accounts for cascading effects and $n$ is the total number of activated failure modes. For this simulation, we'll model the degradation of system health based on increasing AI error rates.")
        ```
*   **Widgets:**
    *   `st.slider("Initial AI Prediction Error Rate (e.g., MAE increase)", min_value=0.01, max_value=0.50, value=st.session_state.initial_error_rate, step=0.01, key="initial_error_rate")`
    *   `st.radio("Select Containment Strategy for Simulation:", ["None", "Strategy A: Activate Shadow Model & Human Review Queue (RC001, RC003)", "Strategy B: Automatic Fallback to Scheduled Maintenance (RC002)"], key="containment_strategy")`
    *   `st.button("Run Simulation")`
*   **Function Calls (on button click):**
    ```python
    if st.button("Run Simulation"):
        # Map radio button selection to simulation strategy identifier
        strategy_id = None
        if st.session_state.containment_strategy.startswith("Strategy A"):
            strategy_id = "strategy_A"
        elif st.session_state.containment_strategy.startswith("Strategy B"):
            strategy_id = "strategy_B"

        scenario_def_no_contain, st.session_state.simulation_output_no_contain = simulate_cascading_failure(
            st.session_state.initial_error_rate, simulation_steps=8, containment_strategy=None
        )
        scenario_def_with_contain, st.session_state.simulation_output_with_contain = simulate_cascading_failure(
            st.session_state.initial_error_rate, simulation_steps=8, containment_strategy=strategy_id
        )
        st.session_state.safe_degradation_example = define_safe_degradation()

        st.session_state.cascading_failure_analysis_output = {
            "scenario_definition": scenario_def_no_contain,
            "simulation_results_no_containment": st.session_state.simulation_output_no_contain,
            "simulation_results_with_containment_B": st.session_state.simulation_output_with_contain,
            "safe_degradation_example": st.session_state.safe_degradation_example
        }
        st.success("Simulation complete!")
    ```
*   **Display Data & Visualization:**
    ```python
    if st.session_state.simulation_output_no_contain and st.session_state.simulation_output_with_contain:
        st.markdown(f"### Simulation Results (No Containment):")
        st.dataframe(pd.DataFrame(st.session_state.simulation_output_no_contain))
        st.markdown(f"### Simulation Results (With {st.session_state.containment_strategy}):")
        st.dataframe(pd.DataFrame(st.session_state.simulation_output_with_contain))

        df_no_contain = pd.DataFrame(st.session_state.simulation_output_no_contain)
        df_with_contain = pd.DataFrame(st.session_state.simulation_output_with_contain)

        fig_sim, ax_sim = plt.subplots(figsize=(12, 6))
        ax_sim.plot(df_no_contain['step'], df_no_contain['system_health'], marker='o', label='No Containment', color='red')
        ax_sim.plot(df_with_contain['step'], df_with_contain['system_health'], marker='x', label=f'With Containment ({st.session_state.containment_strategy})', color='green', linestyle='--')
        ax_sim.set_title(f'Cascading Failure Simulation: System Health vs. Time (Initial AI Error: {st.session_state.initial_error_rate*100:.1f}%)')
        ax_sim.set_xlabel('Simulation Step (Time)')
        ax_sim.set_ylabel('System Health (%)')
        ax_sim.set_ylim(0, 105)
        ax_sim.grid(True, linestyle='--', alpha=0.6)
        ax_sim.legend()
        st.pyplot(fig_sim)

        st.markdown(f"### Safe Degradation Example:")
        st.markdown(f"**Scenario:** {st.session_state.safe_degradation_example.get('name', 'N/A')}")
        st.markdown(f"**Description:** {st.session_state.safe_degradation_example.get('description', 'N/A')}")
        st.markdown(f"**Outcome:** {st.session_state.safe_degradation_example.get('outcome', 'N/A')}")

        st.markdown(f"### Explanation of Execution")
        st.markdown(f"The simulation vividly demonstrates the potential impact of \"Model Drift (False Negatives)\". Without containment, the system health rapidly degrades, leading to a catastrophic turbine failure within a few steps. When a containment strategy (e.g., Automatic Fallback to Scheduled Maintenance) is applied, the system health degrades much slower and stabilizes, preventing catastrophic failure. This comparison is critical for me as a Systems Engineer to quantitatively justify the implementation of resilience controls. It shows that proactive containment can prevent an an AI failure from escalating into a full operational outage.")
    else:
        st.info("Run the simulation to see results.")
    ```

---

#### 5. Recovery Plan Builder

*   **Markdown:**
    ```python
    st.markdown(f"# 5. Defining Rollback Procedures and Recovery Objectives (RTO/RPO)")
    st.markdown(f"After identifying failure modes, mapping controls, and simulating impacts, the next critical step is to define precise rollback procedures and establish Recovery Time Objectives (RTO) and Recovery Point Objectives (RPO) for the AI service. These objectives are crucial for guiding our incident response and recovery efforts to minimize business disruption and data loss.")
    st.markdown(f"---")
    ```
    *   **Formula:**
        ```python
        st.markdown(r"**Mathematical Context for RTO/RPO:**")
        st.markdown(r"- **Recovery Time Objective (RTO):** The maximum acceptable duration of time within which a business process must be restored after a disaster or disruption to avoid unacceptable consequences. It is a time value, e.g., $RTO = 4 \text{ hours}$.")
        st.markdown(r"- **Recovery Point Objective (RPO):** The maximum acceptable amount of data loss measured in time. It defines the point in time to which systems and data must be recovered. It is also a time value, e.g., $RPO = 1 \text{ hour}$.")
        st.markdown(r"These metrics are critical KPIs ($RTO_{\text{AI Service}}$, $RPO_{\text{AI Service}}$) for AI service resilience.")
        ```
*   **Function Calls (on page load):**
    ```python
    st.session_state.rollback_triggers, st.session_state.rollback_procedures, st.session_state.recovery_objectives, st.session_state.validation_for_re_enablement = define_rollback_procedures_and_recovery_objectives()

    # Generate markdown content for recovery plan
    recovery_plan_md_content = f"""
# AI Predictive Maintenance Service Recovery Plan - Steam Turbine Unit 3

This document outlines the specific procedures for detecting, mitigating, and recovering from failures in the AI-driven predictive maintenance service for High-Pressure Steam Turbine Unit 3.

## 1. Rollback Triggers
These are the conditions that will automatically or manually trigger a rollback or degraded mode operation.

"""
    for trigger in st.session_state.rollback_triggers:
        recovery_plan_md_content += f"- **{trigger['name']} ({trigger['id']}):** {trigger['description']}\n"
    recovery_plan_md_content += """

## 2. Rollback Procedures
These are the actions to be taken when a rollback trigger is detected.

"""
    for procedure in st.session_state.rollback_procedures:
        recovery_plan_md_content += f"- **{procedure['name']} ({procedure['id']}):**\n"
        recovery_plan_md_content += f"  - **Triggers:** {', '.join(procedure['trigger_ids'])}\n"
        recovery_plan_md_content += f"  - **Description:** {procedure['description']}\n"
        recovery_plan_md_content += f"  - **Validation Steps:** {'; '.join(procedure['validation_steps'])}\n"
    recovery_plan_md_content += """

## 3. Recovery Objectives (RTO/RPO)
These objectives define the target timelines for service restoration and acceptable data loss.

"""
    for obj, val in st.session_state.recovery_objectives.items():
        recovery_plan_md_content += f"- **{obj.replace('_', ' ')}:** {val}\n"
    recovery_plan_md_content += """

## 4. Validation Steps for Full AI Service Re-enablement
Before fully re-enabling the primary AI service, the following conditions must be met to ensure stability and accuracy.

"""
    for step in st.session_state.validation_for_re_enablement:
        recovery_plan_md_content += f"- **Step {step['step']}:** {step['description']} (Key Performance Indicator: {step['metric_kpi']})\n"

    st.session_state.recovery_plan_md_content = recovery_plan_md_content
    st.session_state.resilience_kpis_output = {
        "recovery_objectives": st.session_state.recovery_objectives,
        "rollback_triggers": st.session_state.rollback_triggers,
        "validation_for_re_enablement": st.session_state.validation_for_re_enablement
    }

    ```
*   **Display Data:**
    ```python
    st.markdown(f"### Sample Recovery Plan (Guidance):")
    st.markdown(st.session_state.sample_recovery_plan_md) # Display loaded markdown content

    st.markdown(f"### Defined Rollback Triggers:")
    st.dataframe(pd.DataFrame(st.session_state.rollback_triggers))

    st.markdown(f"### Defined Rollback Procedures:")
    st.dataframe(pd.DataFrame(st.session_state.rollback_procedures))

    st.markdown(f"### Explanation of Execution")
    st.markdown(f"I have explicitly defined critical rollback triggers based on AI model performance, data pipeline health, and inference latency. For each trigger, I've outlined concrete rollback procedures, such as automated deployment of a previously validated model or manual reversion to a simpler heuristic. The output clarifies these triggers and procedures, which will be immediately actionable for incident response teams.")
    ```

---

#### 6. KPI & Alert Designer

*   **Markdown:**
    ```python
    st.markdown(f"# 6. KPI & Alert Designer")
    st.markdown(f"This section focuses on defining the quantifiable metrics for recovery and the validation steps required before fully re-enabling the AI service. These are crucial for guiding incident response and ensuring operational stability post-recovery.")
    st.markdown(f"---")
    ```
*   **Display Data:**
    ```python
    st.markdown(f"### Defined Recovery Objectives (RTO/RPO):")
    st.dataframe(pd.DataFrame([st.session_state.recovery_objectives]).T.rename(columns={0: 'Value'}))

    st.markdown(f"### Validation Steps for AI Service Re-enablement:")
    st.dataframe(pd.DataFrame(st.session_state.validation_for_re_enablement))

    st.markdown(f"### Explanation of Execution")
    st.markdown(f"Crucially, the Recovery Time Objectives (RTO) and Recovery Point Objectives (RPO) are quantified for both degraded and full functionality modes. Clear validation steps are defined for safe re-enablement of the AI service, along with their associated Key Performance Indicators (KPIs). These essential operational definitions provide clear guidelines for managing AI service disruptions.")
    ```
*   **Function Calls:** `define_rollback_procedures_and_recovery_objectives` is called in the previous step. Results are read from `st.session_state`.

---

#### 7. Export Playbook

*   **Markdown:**
    ```python
    st.markdown(f"# 7. Generating and Exporting the AI Resilience Playbook")
    st.markdown(f"The final step is to consolidate all the analysis, mappings, simulations, and recovery definitions into a comprehensive \"Predictive Maintenance AI Resilience Playbook\". This playbook will be a collection of structured JSON and Markdown files. We will also generate an `evidence_manifest.json` containing SHA-256 hashes for all generated artifacts, ensuring their integrity and traceability, which is crucial for audit and compliance in critical infrastructure.")
    st.markdown(f"---")
    ```
*   **Widgets:**
    *   `st.button("Generate & Export Full Playbook")`
*   **Function Calls (on button click):**
    ```python
    if st.button("Generate & Export Full Playbook"):
        # Ensure output directory exists for this run
        os.makedirs(st.session_state.output_dir, exist_ok=True)

        # Consolidate all data for the full playbook JSON
        st.session_state.full_ai_resilience_playbook_data = generate_ai_resilience_playbook(
            use_case=st.session_state.failure_mode_analysis_output["use_case"],
            failure_modes=st.session_state.failure_mode_analysis_output,
            controls=st.session_state.resilience_controls_output,
            cascading_analysis=st.session_state.cascading_failure_analysis_output,
            recovery_plans={"rollback_procedures": st.session_state.rollback_procedures},
            kpis=st.session_state.resilience_kpis_output
        )

        # Define artifact paths in the unique run_id directory
        st.session_state.artifact_paths = {
            "sector_playbook.json": os.path.join(st.session_state.output_dir, f"sector_playbook_{st.session_state.run_id}.json"),
            "failure_mode_analysis.json": os.path.join(st.session_state.output_dir, f"failure_mode_analysis_{st.session_state.run_id}.json"),
            "resilience_controls.json": os.path.join(st.session_state.output_dir, f"resilience_controls_{st.session_state.run_id}.json"),
            "recovery_plan.md": os.path.join(st.session_state.output_dir, f"recovery_plan_{st.session_state.run_id}.md"),
            "resilience_kpis.json": os.path.join(st.session_state.output_dir, f"resilience_kpis_{st.session_state.run_id}.json"),
            "session14_executive_summary.md": os.path.join(st.session_state.output_dir, f"session14_executive_summary_{st.session_state.run_id}.md"),
            "config_snapshot.json": os.path.join(st.session_state.output_dir, f"config_snapshot_{st.session_state.run_id}.json")
        }

        # Write all artifacts to files
        with open(st.session_state.artifact_paths["sector_playbook.json"], 'w') as f:
            json.dump(st.session_state.full_ai_resilience_playbook_data, f, indent=4)
        with open(st.session_state.artifact_paths["failure_mode_analysis.json"], 'w') as f:
            json.dump(st.session_state.failure_mode_analysis_output, f, indent=4)
        with open(st.session_state.artifact_paths["resilience_controls.json"], 'w') as f:
            json.dump(st.session_state.resilience_controls_output, f, indent=4)
        with open(st.session_state.artifact_paths["recovery_plan.md"], 'w') as f:
            f.write(st.session_state.recovery_plan_md_content)
        with open(st.session_state.artifact_paths["resilience_kpis.json"], 'w') as f:
            json.dump(st.session_state.resilience_kpis_output, f, indent=4)

        # Generate and save the executive summary
        st.session_state.executive_summary_content = generate_executive_summary(st.session_state.full_ai_resilience_playbook_data, st.session_state.run_id)
        with open(st.session_state.artifact_paths["session14_executive_summary.md"], 'w') as f:
            f.write(st.session_state.executive_summary_content)

        # Save config snapshot
        with open(st.session_state.artifact_paths["config_snapshot.json"], 'w') as f:
            json.dump(st.session_state.config_snapshot_data, f, indent=4)

        # Generate Evidence Manifest with SHA-256 hashes
        st.session_state.evidence_manifest = {
            "run_id": st.session_state.run_id,
            "generated_at": datetime.now().isoformat(),
            "artifacts": []
        }
        for name, path in st.session_state.artifact_paths.items():
            if os.path.exists(path):
                artifact_hash = generate_file_hash(path)
                st.session_state.evidence_manifest["artifacts"].append({
                    "name": name,
                    "path": path,
                    "sha256_hash": artifact_hash
                })
            else:
                st.warning(f"Artifact not found at {path}. Skipping hash generation.")

        with open(os.path.join(st.session_state.output_dir, f"evidence_manifest_{st.session_state.run_id}.json"), 'w') as f:
            json.dump(st.session_state.evidence_manifest, f, indent=4)
        
        # Create a zip file of all artifacts
        zip_file_name = f"Session_14_{st.session_state.run_id}.zip"
        zip_file_path = os.path.join(st.session_state.output_dir, zip_file_name)
        with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for artifact_info in st.session_state.evidence_manifest["artifacts"]:
                zipf.write(artifact_info["path"], os.path.basename(artifact_info["path"]))
            # Also add the evidence manifest itself to the zip
            zipf.write(os.path.join(st.session_state.output_dir, f"evidence_manifest_{st.session_state.run_id}.json"), f"evidence_manifest_{st.session_state.run_id}.json")

        st.success(f"AI Resilience Playbook and Evidentiary Artifacts Generated (Run ID: {st.session_state.run_id})")
        st.markdown(f"All artifacts saved in: `{st.session_state.output_dir}/`")
        for artifact in st.session_state.evidence_manifest["artifacts"]:
            st.markdown(f"- `{artifact['name']}`: `{artifact['path']}` (Hash: `{artifact['sha256_hash']}`)")
        
        with open(zip_file_path, "rb") as f:
            st.download_button(
                label="Download Full Playbook (ZIP)",
                data=f.read(),
                file_name=zip_file_name,
                mime="application/zip"
            )
    ```
*   **Display Data:**
    ```python
    if st.session_state.full_ai_resilience_playbook_data:
        st.markdown(f"### Generated Executive Summary:")
        st.markdown(st.session_state.executive_summary_content)
        st.markdown(f"### Generated Evidence Manifest:")
        st.json(st.session_state.evidence_manifest)
        st.markdown(f"### Explanation of Execution")
        st.markdown(f"I have successfully assembled all the components into a comprehensive 'Predictive Maintenance AI Resilience Playbook' in JSON format, alongside individual JSON and Markdown artifacts. These artifacts are organized in a dedicated output directory with a unique run ID for traceability. Crucially, an `evidence_manifest.json` has been generated, listing each artifact along with its SHA-256 hash. This manifest provides cryptographic proof of the integrity and authenticity of our playbook documents, a vital requirement for critical infrastructure and audit processes. As a Systems Engineer, this structured output is directly actionable for our operational teams, enabling them to build robust incident response plans and ensuring the trustworthiness of our AI deployments.")
    ```

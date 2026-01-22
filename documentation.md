id: 69722f4b9e22e29762ed399d_documentation
summary: Lab 14: Sector Playbook Builder (Engineering & Critical Infrastructure) Documentation
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# QuLab: Building an AI Resilience Playbook for Critical Infrastructure

## 0. Introduction: AI Resilience in Critical Infrastructure
Duration: 0:05:00

Welcome to QuLab: Lab 14, where we will build a comprehensive AI Resilience Playbook. As artificial intelligence becomes increasingly embedded in critical industrial infrastructure, ensuring its reliability and resilience is paramount. This codelab guides you through a Streamlit application designed to systematically identify potential AI failure modes, define proactive resilience controls, simulate cascading impacts, and establish clear recovery procedures for AI systems in sectors like Engineering and Critical Infrastructure.

<aside class="positive">
<b>Why is AI Resilience critical?</b> In environments like power plants, manufacturing facilities, or transportation networks, an AI system's failure can lead to severe consequences, including:
<ul>
    <li>Catastrophic equipment damage and millions in financial losses.</li>
    <li>Significant safety risks for personnel and the public.</li>
    <li>Protracted operational downtime and supply chain disruptions.</li>
    <li>Reputational damage and regulatory penalties.</li>
</ul>
This application provides a structured approach to prevent such outcomes by designing resilience into your AI deployments.
</aside>

This codelab will walk you through the functionalities of the "QuLab: Lab 14: Sector Playbook Builder" Streamlit application, allowing you to:
1.  **Configure** your specific AI use case and system parameters.
2.  **Map** potential AI-specific failure modes using established templates.
3.  **Select and Map** appropriate resilience controls to mitigate these failures.
4.  **Simulate** cascading failure scenarios to understand potential impacts and validate containment strategies.
5.  **Build** precise rollback procedures and define recovery objectives (RTO/RPO).
6.  **Design** Key Performance Indicators (KPIs) and alerting mechanisms for resilience.
7.  **Export** a comprehensive, auditable AI Resilience Playbook with evidentiary artifacts.

As a Systems Engineer at "Innovate Manufacturing Inc.", you will be focusing on a predictive maintenance system for high-pressure steam turbines. This AI model predicts equipment failures, but its own failure could cost millions. Your task is to ensure continuous operation and safety, even if the AI experiences issues.

Navigate through the sections using the sidebar to build your playbook step-by-step.

## 1. System Configuration & Use Case Wizard
Duration: 0:07:00

This initial step allows you to define the operational context for your AI resilience playbook. By selecting the sector, specific AI use case, system type, automation level, uptime requirements, and human override availability, you tailor the subsequent analysis steps to your precise operational needs.

The configuration you establish here directly influences the types of failure modes identified and the applicability of resilience controls in later stages.

### Application UI

The Streamlit application presents a series of dropdown selectors for configuration:

*   **Sector:** E.g., "Engineering", "Critical Infrastructure"
*   **Use Case:** E.g., "Predictive Maintenance", "Anomaly Detection"
*   **System Type:** E.g., "ML", "LLM", "Agent"
*   **Automation Level:** Defines the degree of AI autonomy, impacting human intervention needs.
*   **Uptime Requirement:** Sets the desired availability level, a critical factor for resilience.
*   **Human Override Availability:** Indicates if human intervention is a viable fallback.

<aside class="positive">
<b>Best Practice:</b> Accurately defining these parameters is crucial. For critical infrastructure, even minor misconfigurations can lead to significant discrepancies in risk assessment and control selection. Consider the real-world operational context thoroughly.
</aside>

### Behind the Scenes

When you click "Confirm Configuration & Identify Failure Modes", the application performs the following actions:
1.  **Captures Configuration:** All selected options are stored in `st.session_state.config_snapshot_data` for traceability and later export.
2.  **Defines Use Case Context:** A detailed `st.session_state.predictive_maintenance_use_case` dictionary is constructed based on your selections. This defines the AI's role, the equipment it monitors, its criticality, and its primary function (e.g., predicting Remaining Useful Life (RUL)).
3.  **Identifies AI Failure Modes:** The core logic, using the `identify_ai_failure_modes` function from `source.py`, takes the `predictive_maintenance_use_case` and sector-specific failure templates (e.g., `engineering_failure_templates.json`, `infrastructure_failure_templates.json`) to generate a list of potential AI-related failure modes. These templates contain pre-defined failure scenarios relevant to each sector.
4.  **Stores Results:** The identified failure modes are stored in `st.session_state.identified_ai_failure_modes` and converted into a DataFrame (`st.session_state.df_failure_modes`) for display in the next step.

```python
# Example of how configuration drives use case definition (simplified from actual code)
st.session_state.predictive_maintenance_use_case = {
    "name": f"{st.session_state.use_case} for {st.session_state.sector} Systems",
    "equipment": "High-Pressure Steam Turbine (Unit 3)",
    "criticality": "High - Direct impact on production, safety, and operational costs",
    "ai_function": f"Predict Remaining Useful Life (RUL) of components based on sensor data for {st.session_state.use_case}.",
    "sensors": ["Vibration", "Temperature", "Pressure"]
}

# Example of calling the failure mode identification function (conceptual)
# identified_ai_failure_modes = identify_ai_failure_modes(
#     use_case_context,
#     engineering_templates,
#     infrastructure_templates
# )
```

After confirming, you will be redirected to the "2. Failure Mode Mapper" section.

## 2. Failure-Mode Mapper
Duration: 0:10:00

In this step, we delve into the specific AI-related failure modes that could compromise the system's ability to accurately predict maintenance needs. Understanding these vulnerabilities is the foundation for building an effective resilience playbook. We leverage pre-defined failure templates to systematically identify risks.

### Scenario Context

The application sets a specific scenario: "Our facility relies on a set of high-pressure steam turbines. An AI model analyzes sensor data (vibration, temperature, pressure, lubrication levels) to predict the Remaining Useful Life (RUL) of critical turbine components, specifically bearings and rotor blades. Early prediction allows us to schedule maintenance during planned downtimes, preventing catastrophic failures."

### Mathematical Context for RUL Prediction

The AI model uses various sensor inputs $X = [x_1, x_2, ..., x_n]$ to predict RUL, denoted as $RUL_{pred}$. The true RUL, $RUL_{true}$, is unknown until failure. The model's performance is often evaluated by metrics like Mean Absolute Error (MAE):

$$MAE = \frac{1}{N} \sum_{i=1}^{N} |RUL_{true,i} - RUL_{pred,i}|$$

where $N$ is the number of predictions, $RUL_{true,i}$ is the true Remaining Useful Life for prediction $i$, and $RUL_{pred,i}$ is the predicted Remaining Useful Life for prediction $i$. A high MAE indicates poor prediction, which is a critical AI failure mode.

### Identified AI-Related Failure Modes

Based on your configuration in Step 1 and the loaded templates, the application displays a table of identified AI failure modes. These modes are typically categorized (e.g., Data, Model, Infrastructure) to provide a structured view of vulnerabilities.

The `df_failure_modes` DataFrame, populated from `st.session_state.identified_ai_failure_modes`, is displayed. This DataFrame contains columns like `id`, `name`, `category`, `description`, and `potential_impact`.

A bar chart is generated using `seaborn.countplot` to visualize the distribution of failure modes across different categories. This provides an immediate understanding of which areas (Data, Model, Infrastructure) have the highest concentration of potential failures.

```python
# Example of displaying the dataframe and plot (from application code)
# st.dataframe(st.session_state.df_failure_modes)
#
# fig, ax = plt.subplots(figsize=(10, 6))
# sns.countplot(data=st.session_state.df_failure_modes, x='category', palette='viridis', ax=ax)
# ax.set_title('Identified AI Failure Modes by Category for Steam Turbine Predictive Maintenance')
# st.pyplot(fig)
```

### Explanation of Execution

The output lists the identified AI-related failure modes, categorized by Data, Model, and Infrastructure. The bar chart provides a clear visual summary, showing where our AI system is most vulnerable. As a Systems Engineer, this breakdown helps me focus my resilience efforts. For example, 'Data' and 'Model' categories appear to have more failure modes, indicating a need for robust data validation and model monitoring strategies. This structured overview of vulnerabilities serves as a foundational document for our playbook.

## 3. Resilience Control Selector
Duration: 0:12:00

Having identified the key AI-related failure modes, the next logical step is to design and map specific resilience controls to mitigate their impact. This involves defining strategies like redundancy, graceful degradation, and automatic fallback mechanisms. The goal is to ensure that even if a specific AI component fails, the overall predictive maintenance system can either continue operating in a degraded but safe mode or automatically switch to an alternative, reliable method.

### Mathematical Context for Control Effectiveness

Resilience controls aim to reduce the probability of system impact or the severity of that impact. Conceptually, for a given failure mode $FM$, a control $C$ reduces the impact $I$ with a certain effectiveness $E_C$.

$$I_{mitigated} = I_{unmitigated} \times (1 - E_C)$$

where $E_C \in [0, 1]$ represents the reduction in impact. This effectiveness will be simulated in the next section.

### Defining and Mapping Controls

The application uses two key functions from `source.py` for this step:
1.  `define_resilience_controls()`: This function defines a set of generic yet powerful resilience controls applicable to AI systems in critical infrastructure. Examples include "Data Input Validation (RC001)", "Automatic Fallback to Scheduled Maintenance (RC002)", "Model Health Monitoring (RC003)", etc. Each control has a name, ID, type (e.g., Redundancy, Degradation), and a description.
2.  `map_controls_to_failure_modes()`: This function takes the `identified_ai_failure_modes` (from Step 2) and the `resilience_controls` and intelligently maps which controls are applicable to which failure modes. This mapping creates a direct link between a vulnerability and its proposed mitigation.

The application first displays a DataFrame of the `Defined Resilience Controls`. Subsequently, it presents the `Resilience Control Mapping`, showing which controls are applicable to each identified failure mode.

### Visualization of Control Mapping

A bipartite graph, generated using `networkx`, visually represents the mapping between AI Failure Modes and Resilience Controls.

*   One set of nodes represents the failure modes (e.g., "Sensor Data Anomaly").
*   The other set of nodes represents the resilience controls (e.g., "Data Input Validation").
*   Edges connect a failure mode to all applicable controls.

This visualization helps to:
*   Identify which controls provide coverage for multiple failure modes.
*   Spot failure modes that might lack sufficient control coverage.
*   Understand the overall interconnectedness of the resilience strategy.

```python
# Example of the bipartite graph generation (from application code)
# G = nx.Graph()
# failure_nodes = [fm['failure_mode_name'] for fm in st.session_state.identified_ai_failure_modes]
# control_nodes = [c['name'] for c in st.session_state.resilience_controls]
# G.add_nodes_from(failure_nodes, bipartite=0, label='Failure Mode')
# G.add_nodes_from(control_nodes, bipartite=1, label='Resilience Control')
#
# for fm_map in st.session_state.mapped_resilience_data:
#     for ctrl in fm_map['applicable_controls']:
#         G.add_edge(fm_map['failure_mode_name'], ctrl['control_name'])
#
# pos = nx.bipartite_layout(G, failure_nodes)
# fig_graph, ax_graph = plt.subplots(figsize=(14, 8))
# nx.draw_networkx_nodes(...)
# nx.draw_networkx_edges(...)
# st.pyplot(fig_graph)
```

### Explanation of Execution

The output displays a detailed list of which resilience controls apply to each identified AI failure mode. The bipartite graph provides an intuitive visual representation of these relationships, making it easy to see which controls cover multiple failure types and which failure modes are adequately (or insufficiently) protected. For instance, "Automatic Fallback to Scheduled Maintenance" (RC002) is mapped to several failure categories, highlighting its broad utility as a graceful degradation strategy. This mapping ensures that every critical AI vulnerability has a corresponding mitigation strategy, directly supporting the objective of operational resilience.

## 4. Cascading Failure Simulation
Duration: 0:15:00

Even with resilience controls in place, complex AI systems can experience cascading failures where an initial fault triggers a chain reaction of subsequent issues. As a Systems Engineer, anticipating these scenarios and defining robust containment strategies is crucial. This step simulates a critical scenario: "Model Drift (False Negatives)" leading to missed maintenance and subsequent equipment damage.

### Mathematical Context for Cascading Failure Impact

Let $P(FM_1)$ be the probability of an initial failure mode $FM_1$. If $FM_1$ occurs, it increases the probability of a subsequent failure mode $FM_2$ by a factor $k_{1,2} > 1$, such that $P(FM_2|FM_1) = k_{1,2} \times P(FM_2)_{baseline}$. The overall system impact $S$ can be a function of the sum of individual failure severities $s_i$ for each activated failure $FM_i$, potentially weighted by their interdependencies.

$$S = \sum_{i=1}^{n} s_i \times w_i$$

where $w_i$ accounts for cascading effects and $n$ is the total number of activated failure modes. For this simulation, we'll model the degradation of system health based on increasing AI error rates.

### Simulation Parameters and Containment Strategies

You can adjust the `Initial AI Prediction Error Rate` using a slider. This represents the starting point of the AI model's degradation.

You can select a `Containment Strategy` from the following options:
*   "None": No specific containment actions are taken.
*   "Strategy A: Activate Shadow Model & Human Review Queue (RC001, RC003)": Represents proactive monitoring and a human-in-the-loop fallback.
*   "Strategy B: Automatic Fallback to Scheduled Maintenance (RC002)": Represents a graceful degradation to a known safe mode.

When you click "Run Simulation", the application uses the `simulate_cascading_failure` function from `source.py` to model the degradation of `system_health` over several `simulation_steps`. It runs two scenarios: one without any containment strategy and one with your selected strategy. The `define_safe_degradation` function also provides an example of how the system might transition to a safe, albeit degraded, operational state.

```python
# Conceptual flow of simulation
# scenario_def_no_contain, simulation_output_no_contain = simulate_cascading_failure(
#     initial_error_rate, simulation_steps=8, containment_strategy=None
# )
# scenario_def_with_contain, simulation_output_with_contain = simulate_cascading_failure(
#     initial_error_rate, simulation_steps=8, containment_strategy="strategy_A" or "strategy_B"
# )
# safe_degradation_example = define_safe_degradation()
```

### Simulation Results and Visualization

The results are presented in two dataframes, showing `system_health` at each `step` for both scenarios. A line plot visualizes the `system_health` over time, clearly demonstrating the difference in impact with and without containment.

The "Safe Degradation Example" further illustrates how, in a real-world scenario, the system might transition to a less performant but secure state when the AI fails.

```python
# Example of simulation plot (from application code)
# fig_sim, ax_sim = plt.subplots(figsize=(12, 6))
# ax_sim.plot(df_no_contain['step'], df_no_contain['system_health'], marker='o', label='No Containment', color='red')
# ax_sim.plot(df_with_contain['step'], df_with_contain['system_health'], marker='x', label=f'With Containment (...)', color='green', linestyle='--')
# ax_sim.set_title('Cascading Failure Simulation: System Health vs. Time (...)')
# st.pyplot(fig_sim)
```

### Explanation of Execution

The simulation vividly demonstrates the potential impact of "Model Drift (False Negatives)". Without containment, the system health rapidly degrades, leading to a catastrophic turbine failure within a few steps. When a containment strategy (e.g., Automatic Fallback to Scheduled Maintenance) is applied, the system health degrades much slower and stabilizes, preventing catastrophic failure. This comparison is critical for me as a Systems Engineer to quantitatively justify the implementation of resilience controls. It shows that proactive containment can prevent an AI failure from escalating into a full operational outage.

## 5. Recovery Plan Builder
Duration: 0:15:00

After identifying failure modes, mapping controls, and simulating impacts, the next critical step is to define precise rollback procedures and establish Recovery Time Objectives (RTO) and Recovery Point Objectives (RPO) for the AI service. These objectives are crucial for guiding our incident response and recovery efforts to minimize business disruption and data loss.

### Mathematical Context for RTO/RPO

*   **Recovery Time Objective (RTO):** The maximum acceptable duration of time within which a business process must be restored after a disaster or disruption to avoid unacceptable consequences. It is a time value, e.g., $RTO = 4 \text{ hours}$.
*   **Recovery Point Objective (RPO):** The maximum acceptable amount of data loss measured in time. It defines the point in time to which systems and data must be recovered. It is also a time value, e.g., $RPO = 1 \text{ hour}$.

These metrics are critical KPIs ($RTO_{\text{AI Service}}$, $RPO_{\text{AI Service}}$) for AI service resilience.

### Defining Recovery Procedures and Objectives

The `define_rollback_procedures_and_recovery_objectives()` function from `source.py` is invoked to define:
*   `rollback_triggers`: Conditions that automatically or manually initiate a rollback (e.g., "Model Performance Degradation", "Data Pipeline Failure").
*   `rollback_procedures`: Detailed actions to be taken when a trigger is detected, including validation steps.
*   `recovery_objectives`: Quantifiable RTO and RPO for various aspects of the AI service (e.g., for degraded functionality, for full functionality).
*   `validation_for_re_enablement`: Steps required to ensure stability and accuracy before fully restoring the AI service.

These definitions are then assembled into a comprehensive markdown-formatted `recovery_plan_md_content`, which is displayed in the application. A sample recovery plan (`sample_recovery_plan_md`) is also loaded from `source/sample_recovery_plans.md` to provide guidance.

```python
# Conceptual flow of defining recovery elements
# rollback_triggers, rollback_procedures, recovery_objectives, validation_for_re_enablement = \
#     define_rollback_procedures_and_recovery_objectives()
#
# # Building markdown content (simplified)
# recovery_plan_md_content = "# AI Predictive Maintenance Service Recovery Plan\n..."
# for trigger in rollback_triggers:
#     recovery_plan_md_content += f"- **{trigger['name']}**: {trigger['description']}\n"
# ...
# st.markdown(st.session_state.recovery_plan_md_content)
```

### Explanation of Execution

I have explicitly defined critical rollback triggers based on AI model performance, data pipeline health, and inference latency. For each trigger, I've outlined concrete rollback procedures, such as automated deployment of a previously validated model or manual reversion to a simpler heuristic. The output clarifies these triggers and procedures, which will be immediately actionable for incident response teams.

## 6. KPI & Alert Designer
Duration: 0:08:00

This section focuses on defining the quantifiable metrics for recovery and the validation steps required before fully re-enabling the AI service. These are crucial for guiding incident response and ensuring operational stability post-recovery. This step essentially summarizes the key outputs from the Recovery Plan Builder (Step 5) in a focused manner, emphasizing the metrics and thresholds for operational resilience.

### Recovery Objectives (RTO/RPO)

The `recovery_objectives` defined in the previous step are displayed. These typically include:
*   `RTO_degraded_functionality`: Target time to restore basic, degraded service.
*   `RPO_degraded_functionality`: Acceptable data loss for degraded service.
*   `RTO_full_functionality`: Target time to restore full, optimal service.
*   `RPO_full_functionality`: Acceptable data loss for full service.

These objectives serve as critical Service Level Objectives (SLOs) for the AI system's operational teams.

### Validation Steps for AI Service Re-enablement

The `validation_for_re_enablement` steps are also displayed, detailing the criteria that must be met before the AI service can be fully brought back online. Each step includes:
*   A description of the validation action.
*   A `metric_kpi` which specifies the performance indicator to monitor (e.g., "Model MAE < 0.05", "Data Ingestion Latency < 100ms").

These validation steps are essential for preventing a recurrence of the failure and ensuring the restored service is robust and reliable.

```python
# Example of displaying objectives and validation (from application code)
# if st.session_state.recovery_objectives:
#     st.markdown(f"### Defined Recovery Objectives (RTO/RPO):")
#     st.dataframe(pd.DataFrame([st.session_state.recovery_objectives]).T.rename(columns={0: 'Value'}))
#
# if st.session_state.validation_for_re_enablement:
#     st.markdown(f"### Validation Steps for AI Service Re-enablement:")
#     st.dataframe(pd.DataFrame(st.session_state.validation_for_re_enablement))
```

### Explanation of Execution

Crucially, the Recovery Time Objectives (RTO) and Recovery Point Objectives (RPO) are quantified for both degraded and full functionality modes. Clear validation steps are defined for safe re-enablement of the AI service, along with their associated Key Performance Indicators (KPIs). These essential operational definitions provide clear guidelines for managing AI service disruptions.

## 7. Generating and Exporting the AI Resilience Playbook
Duration: 0:10:00

The final step is to consolidate all the analysis, mappings, simulations, and recovery definitions into a comprehensive "Predictive Maintenance AI Resilience Playbook". This playbook will be a collection of structured JSON and Markdown files. We will also generate an `evidence_manifest.json` containing SHA-256 hashes for all generated artifacts, ensuring their integrity and traceability, which is crucial for audit and compliance in critical infrastructure.

### Playbook Generation Process

When you click "Generate & Export Full Playbook", the application performs the following key actions:

1.  **Consolidates Data:** The `generate_ai_resilience_playbook()` function from `source.py` aggregates all the data accumulated in `st.session_state` from previous steps into a single, comprehensive JSON object. This includes:
    *   System Configuration and Use Case details.
    *   Identified AI Failure Modes.
    *   Defined Resilience Controls and their mappings.
    *   Cascading Failure Simulation results.
    *   Rollback Procedures, Recovery Objectives (RTO/RPO), and Validation Steps.
2.  **Defines Artifact Paths:** A unique `output_dir` is created for the current session (e.g., `reports/session14/20231027_103000`), and all generated files are saved within this directory.
3.  **Writes Artifacts:**
    *   `sector_playbook.json`: The full consolidated playbook data.
    *   `failure_mode_analysis.json`: Detailed output from Step 2.
    *   `resilience_controls.json`: Detailed output from Step 3.
    *   `recovery_plan.md`: The markdown recovery plan from Step 5.
    *   `resilience_kpis.json`: KPI and RTO/RPO details from Step 6.
    *   `session14_executive_summary.md`: A high-level summary generated by `generate_executive_summary()` from `source.py`.
    *   `config_snapshot.json`: The initial system configuration from Step 1.
4.  **Generates Evidence Manifest:** For each generated file, the `generate_file_hash()` function from `source.py` calculates its SHA-256 hash. This manifest (`evidence_manifest.json`) records the run ID, generation timestamp, artifact names, their paths, and their corresponding SHA-256 hashes.
    <aside class="negative">
    <b>Importance of Evidence Manifest:</b> In highly regulated environments like critical infrastructure, proving the integrity and authenticity of operational documents is paramount. The SHA-256 hash acts as a digital fingerprint, ensuring that the exported playbook and its artifacts have not been tampered with since their generation.
    </aside>
5.  **Creates ZIP Archive:** All generated files, including the `evidence_manifest.json`, are compressed into a single ZIP file, making it easy to download and share the complete playbook.
6.  **Download Button:** A Streamlit download button is provided for convenient access to the generated ZIP file.

```python
# Conceptual flow for export (simplified)
# os.makedirs(st.session_state.output_dir, exist_ok=True)
# full_playbook_data = generate_ai_resilience_playbook(...)
#
# with open(artifact_paths["sector_playbook.json"], 'w') as f:
#     json.dump(full_playbook_data, f, indent=4)
#
# # ... write other files ...
#
# evidence_manifest = { "run_id": ..., "artifacts": [] }
# for name, path in artifact_paths.items():
#     artifact_hash = generate_file_hash(path)
#     evidence_manifest["artifacts"].append({ "name": name, "path": path, "sha256_hash": artifact_hash })
#
# # ... create zip file and download button ...
```

<button>
  [Download Example Playbook (ZIP)](https://example.com/example_playbook.zip)
</button>

### Explanation of Execution

I have successfully assembled all the components into a comprehensive 'Predictive Maintenance AI Resilience Playbook' in JSON format, alongside individual JSON and Markdown artifacts. These artifacts are organized in a dedicated output directory with a unique run ID for traceability. Crucially, an `evidence_manifest.json` has been generated, listing each artifact along with its SHA-256 hash. This manifest provides cryptographic proof of the integrity and authenticity of our playbook documents, a vital requirement for critical infrastructure and audit processes. As a Systems Engineer, this structured output is directly actionable for our operational teams, enabling them to build robust incident response plans and ensuring the trustworthiness of our AI deployments.

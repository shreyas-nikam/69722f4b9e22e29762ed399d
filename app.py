# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import os
import json
import zipfile
import seaborn as sns
from datetime import datetime

# ✅ Import explicit functions from the refactored source.py
from source import (
    # templates / use case / failure modes
    load_templates,
    load_sample_recovery_plan,
    get_default_use_case,
    identify_ai_failure_modes,
    build_failure_mode_analysis_output,
    plot_failure_modes_by_category,
    # controls / mapping
    define_resilience_controls,
    map_controls_to_failure_modes,
    build_resilience_controls_output,
    plot_failure_mode_control_bipartite,
    # cascading simulation
    simulate_cascading_failure,
    define_safe_degradation,
    build_cascading_failure_analysis_output,
    plot_cascading_failure_health,
    # recovery planning
    define_rollback_procedures_and_recovery_objectives,
    build_recovery_plan_markdown,
    build_resilience_kpis_output,
    # playbook + summary
    generate_ai_resilience_playbook,
    generate_executive_summary,
    # artifacts / hashing
    generate_file_hash,
    build_evidence_manifest,
    write_json,
    write_text,
    ensure_dir,
)

# Page Configuration
st.set_page_config(
    page_title="QuLab: Lab 14: Sector Playbook Builder (Engineering & Critical Infrastructure)",
    layout="wide",
)
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab: Lab 14: Sector Playbook Builder (Engineering & Critical Infrastructure)")
st.divider()

# ----------------------------
# Session State Initialization
# ----------------------------
if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"

if "run_id" not in st.session_state:
    st.session_state.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

# NOTE: output_dir structure changed slightly in source pipeline; keep your original pattern but ensure it exists.
if "output_dir" not in st.session_state:
    st.session_state.output_dir = os.path.join(
        "reports", "session14", st.session_state.run_id)
    ensure_dir(st.session_state.output_dir)

# Configuration defaults
defaults = {
    "sector": "Engineering",
    "use_case": "Predictive Maintenance",
    "system_type": "ML",
    "automation_level": "High",
    "uptime_requirement": "99.9%",
    "human_override": "Available",
    "config_snapshot_data": {},
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Data loading (templates + sample md)
if "engineering_templates" not in st.session_state:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    engineering_failure_templates_path = os.path.join(
        script_dir, "source", "engineering_failure_templates.json")
    st.session_state.engineering_templates = load_templates(
        engineering_failure_templates_path)

if "infrastructure_templates" not in st.session_state:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    infrastructure_failure_templates_path = os.path.join(
        script_dir, "source", "infrastructure_failure_templates.json")
    st.session_state.infrastructure_templates = load_templates(
        infrastructure_failure_templates_path)

if "sample_recovery_plan_md" not in st.session_state:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sample_recovery_plan_path = os.path.join(
        script_dir, "source", "sample_recovery_plans.md")
    st.session_state.sample_recovery_plan_md = load_sample_recovery_plan(
        sample_recovery_plan_path)

# Step outputs
if "predictive_maintenance_use_case" not in st.session_state:
    st.session_state.predictive_maintenance_use_case = {}
if "identified_ai_failure_modes" not in st.session_state:
    st.session_state.identified_ai_failure_modes = []
if "df_failure_modes" not in st.session_state:
    st.session_state.df_failure_modes = pd.DataFrame()
if "failure_mode_analysis_output" not in st.session_state:
    st.session_state.failure_mode_analysis_output = {}

if "resilience_controls" not in st.session_state:
    st.session_state.resilience_controls = define_resilience_controls()
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

# ----------------------------
# Sidebar Navigation
# ----------------------------
page_options = [
    "Home",
    "1. System Configuration",
    "2. Failure Mode Mapper",
    "3. Resilience Control Selector",
    "4. Cascading Failure Simulation",
    "5. Recovery Plan Builder",
    "6. KPI & Alert Designer",
    "7. Export Playbook",
]
st.sidebar.title("AI Resilience Playbook Builder")

try:
    current_index = page_options.index(st.session_state.current_page)
except ValueError:
    current_index = 0

st.session_state.current_page = st.sidebar.selectbox(
    "Navigate to a Section",
    page_options,
    index=current_index,
)

# ----------------------------
# Page Routing
# ----------------------------
if st.session_state.current_page == "Home":
    st.markdown(f"# AI Resilience Playbook Builder")
    st.markdown(f"## Ensuring Uptime in Critical Industrial Infrastructure")
    st.markdown(f"As a Systems Engineer at \"Innovate Manufacturing Inc.\", my primary responsibility is to design and integrate advanced AI solutions into our critical industrial processes. Today, I'm focusing on our new predictive maintenance system for the high-pressure steam turbines in our main production facility. These turbines are the heart of our operations; their unplanned downtime can cost millions per hour and pose significant safety risks.")
    st.markdown(f"The AI model aims to predict equipment failures days or weeks in advance, allowing for proactive maintenance. However, an AI system, like any complex component, can fail. My task is to build a comprehensive \"AI Resilience Playbook\" to ensure continuous operation and safety, even if the AI model or its data pipelines experience issues. This playbook will detail potential AI-related failure modes, define proactive resilience controls, simulate cascading impacts, and establish clear recovery procedures. This isn't just about preventing downtime; it's about safeguarding our entire operation.")

    st.markdown(
        "Navigate using the sidebar to build your playbook step-by-step.")

elif st.session_state.current_page == "1. System Configuration":
    st.markdown("# 1. System Configuration & Use Case Wizard")
    st.markdown(
        "Define the sector, specific AI use case, and system parameters.")
    st.markdown("---")

    sectors = ["Engineering", "Critical Infrastructure"]
    use_cases = ["Predictive Maintenance",
                 "Anomaly Detection", "Forecasting / Optimization"]
    system_types = ["ML", "LLM", "Agent"]
    automation_levels = ["Low", "Medium", "High", "Full"]
    uptimes = ["99%", "99.9%", "99.99%", "99.999%"]
    overrides = ["Available", "Limited", "Not Available"]

    st.selectbox("Sector", sectors, key="sector",
                 index=sectors.index(st.session_state.sector))
    st.selectbox("Use Case", use_cases, key="use_case",
                 index=use_cases.index(st.session_state.use_case))
    st.selectbox("System Type", system_types, key="system_type",
                 index=system_types.index(st.session_state.system_type))
    st.selectbox("Automation Level", automation_levels, key="automation_level",
                 index=automation_levels.index(st.session_state.automation_level))
    st.selectbox("Uptime Requirement", uptimes, key="uptime_requirement",
                 index=uptimes.index(st.session_state.uptime_requirement))
    st.selectbox("Human Override Availability", overrides, key="human_override",
                 index=overrides.index(st.session_state.human_override))

    if st.button("Confirm Configuration & Identify Failure Modes"):
        st.session_state.config_snapshot_data = {
            "system_type": st.session_state.system_type,
            "automation_level": st.session_state.automation_level,
            "uptime_requirement": st.session_state.uptime_requirement,
            "human_override_availability": st.session_state.human_override,
            "sector": st.session_state.sector,
            "use_case": st.session_state.use_case,
            "timestamp": datetime.now().isoformat(),
        }

        # Build a use-case definition (default for Predictive Maintenance + Engineering, otherwise generic)
        if st.session_state.sector == "Engineering" and st.session_state.use_case == "Predictive Maintenance":
            st.session_state.predictive_maintenance_use_case = get_default_use_case()
        else:
            st.session_state.predictive_maintenance_use_case = {
                "name": f"{st.session_state.use_case} for {st.session_state.sector} Systems",
                "equipment": "Generic Equipment/System",
                "criticality": "High - Direct impact on production, safety, and operational costs",
                "ai_function": f"AI function for {st.session_state.use_case}.",
                "sensors": ["Vibration", "Temperature", "Pressure"],
            }

        st.session_state.identified_ai_failure_modes = identify_ai_failure_modes(
            st.session_state.predictive_maintenance_use_case,
            st.session_state.engineering_templates,
            st.session_state.infrastructure_templates,
        )
        st.session_state.df_failure_modes = pd.DataFrame(
            st.session_state.identified_ai_failure_modes)

        # ✅ Use the helper builder now
        st.session_state.failure_mode_analysis_output = build_failure_mode_analysis_output(
            use_case=st.session_state.predictive_maintenance_use_case,
            identified_ai_failure_modes=st.session_state.identified_ai_failure_modes,
            engineering_templates=st.session_state.engineering_templates,
            infrastructure_templates=st.session_state.infrastructure_templates,
        )

        st.success("Configuration saved and AI failure modes identified!")
        st.session_state.current_page = "2. Failure Mode Mapper"
        st.rerun()

elif st.session_state.current_page == "2. Failure Mode Mapper":
    st.markdown("# 2. Failure-Mode Mapper")
    st.markdown(
        "Identify AI-specific failure modes using templates as guidance.")
    st.markdown("---")

    if not st.session_state.df_failure_modes.empty:
        st.dataframe(st.session_state.df_failure_modes)

        # ✅ Use the refactored plotting function (returns a figure)
        fig = plot_failure_modes_by_category(
            st.session_state.identified_ai_failure_modes,
            title="Identified AI Failure Modes by Category for Steam Turbine Predictive Maintenance",
        )
        st.pyplot(fig)

    else:
        st.warning(
            "Please configure the system and identify failure modes in section 1 first.")

elif st.session_state.current_page == "3. Resilience Control Selector":
    st.markdown("# 3. Designing and Mapping Resilience Controls")
    st.markdown("Define resilience controls and map them to failure modes.")
    st.markdown("---")

    if st.session_state.identified_ai_failure_modes:
        if not st.session_state.resilience_controls:
            st.session_state.resilience_controls = define_resilience_controls()

        if not st.session_state.mapped_resilience_data:
            st.session_state.mapped_resilience_data = map_controls_to_failure_modes(
                st.session_state.identified_ai_failure_modes,
                st.session_state.resilience_controls,
            )
            st.session_state.resilience_controls_output = build_resilience_controls_output(
                defined_controls=st.session_state.resilience_controls,
                mapped_controls_to_failure_modes=st.session_state.mapped_resilience_data,
            )

        st.markdown("### Defined Resilience Controls")
        st.dataframe(pd.DataFrame(st.session_state.resilience_controls))

        st.markdown("### Resilience Control Mapping (Interactive)")

        mapped_df = pd.DataFrame(st.session_state.mapped_resilience_data)

        # Flatten mapping into edge-like table for filtering + display
        rows = []
        for fm_map in st.session_state.mapped_resilience_data:
            fm_id = fm_map["failure_mode_id"]
            fm_name = fm_map["failure_mode_name"]
            fm_cat = fm_map["failure_mode_category"]
            for ctrl in fm_map.get("applicable_controls", []):
                rows.append({
                    "failure_mode_id": fm_id,
                    "failure_mode": fm_name,
                    "fm_category": fm_cat,
                    "control_id": ctrl["control_id"],
                    "control": ctrl["control_name"],
                    "control_type": ctrl["control_type"],
                    "control_description": ctrl["description"],
                })

        edges_df = pd.DataFrame(rows)

        if edges_df.empty:
            st.info("No mappings available yet.")
        else:
            # Filters
            col1, col2, col3 = st.columns(3)
            with col1:
                cat_opts = sorted(edges_df["fm_category"].unique())
                sel_cats = st.multiselect(
                    "Filter by Failure Mode Category", cat_opts, default=cat_opts)
            with col2:
                type_opts = sorted(edges_df["control_type"].unique())
                sel_types = st.multiselect(
                    "Filter by Control Type", type_opts, default=type_opts)
            with col3:
                search = st.text_input(
                    "Search (failure mode / control)", value="")

            filtered = edges_df[
                edges_df["fm_category"].isin(sel_cats) &
                edges_df["control_type"].isin(sel_types)
            ].copy()

            if search.strip():
                s = search.strip().lower()
                filtered = filtered[
                    filtered["failure_mode"].str.lower().str.contains(s) |
                    filtered["control"].str.lower().str.contains(s) |
                    filtered["control_description"].str.lower().str.contains(s)
                ]

            tab1, tab2, tab3 = st.tabs(
                ["Failure Mode → Controls", "Controls Coverage", "Matrix View"])

            # --- Tab 1: Failure mode explorer ---
            with tab1:
                fm_list = sorted(filtered["failure_mode"].unique())
                sel_fm = st.selectbox("Select a Failure Mode", fm_list)

                fm_block = filtered[filtered["failure_mode"] == sel_fm].copy()
                st.caption(
                    f"Category: **{fm_block['fm_category'].iloc[0]}**  •  Controls: **{len(fm_block)}**")

                for _, r in fm_block.sort_values(["control_type", "control"]).iterrows():
                    with st.expander(f"{r['control']}  —  {r['control_type']}  ({r['control_id']})"):
                        st.write(r["control_description"])

                st.markdown("#### Mapping Table (filtered)")
                st.dataframe(
                    fm_block[["failure_mode", "fm_category",
                              "control", "control_type", "control_id"]],
                    use_container_width=True
                )

            # --- Tab 2: Control coverage explorer ---
            with tab2:
                coverage = (
                    filtered.groupby(
                        ["control", "control_type", "control_id"], as_index=False)
                    .agg(
                        failure_modes=(
                            "failure_mode", lambda x: sorted(set(x))),
                        coverage_count=("failure_mode", lambda x: len(set(x))),
                    )
                    .sort_values(["coverage_count", "control_type", "control"], ascending=[False, True, True])
                )

                st.markdown(
                    "#### Controls ranked by coverage (how many failure modes they mitigate)")
                st.dataframe(coverage[[
                             "control", "control_type", "control_id", "coverage_count"]], use_container_width=True)

                sel_ctrl = st.selectbox(
                    "Inspect a Control", coverage["control"].tolist())
                row = coverage[coverage["control"] == sel_ctrl].iloc[0]

                st.caption(
                    f"Coverage: **{row['coverage_count']}** failure modes  •  Type: **{row['control_type']}**")
                st.write("**Failure Modes Covered:**")
                st.write(", ".join(row["failure_modes"]))

                # Pull description from any instance (same control desc everywhere)
                desc = filtered[filtered["control"] ==
                                sel_ctrl]["control_description"].iloc[0]
                st.write("**Description:**")
                st.write(desc)

            # --- Tab 3: Matrix view (pivot) ---
            with tab3:
                pivot = pd.crosstab(
                    filtered["failure_mode"], filtered["control"])
                st.markdown("#### Mapping Matrix (1 = mapped)")
                st.dataframe(pivot, use_container_width=True)

        st.markdown("### Visualization of Control Mapping")
        # ✅ Use the refactored bipartite plot function
        fig_graph = plot_failure_mode_control_bipartite(
            identified_ai_failure_modes=st.session_state.identified_ai_failure_modes,
            resilience_controls=st.session_state.resilience_controls,
            mapped_resilience_data=st.session_state.mapped_resilience_data,
        )
        st.pyplot(fig_graph)

    else:
        st.warning(
            "Please configure the system and identify failure modes in section 1 first.")

elif st.session_state.current_page == "4. Cascading Failure Simulation":
    st.markdown(
        "# 4. Simulating a Cascading Failure Scenario and Defining Containment")
    st.markdown(
        "Simulate drift leading to missed maintenance and cascading impacts.")
    st.markdown("---")

    st.slider(
        "Initial AI Prediction Error Rate (e.g., MAE increase)",
        min_value=0.01,
        max_value=0.50,
        value=st.session_state.initial_error_rate,
        step=0.01,
        key="initial_error_rate",
    )

    strategies = [
        "None",
        "Strategy A: Activate Shadow Model & Human Review Queue (RC001, RC003)",
        "Strategy B: Automatic Fallback to Scheduled Maintenance (RC002)",
    ]
    try:
        strat_index = strategies.index(st.session_state.containment_strategy)
    except Exception:
        strat_index = 0

    st.radio("Select Containment Strategy for Simulation:",
             strategies, index=strat_index, key="containment_strategy")

    if st.button("Run Simulation"):
        strategy_id = None
        if st.session_state.containment_strategy.startswith("Strategy A"):
            strategy_id = "strategy_A"
        elif st.session_state.containment_strategy.startswith("Strategy B"):
            strategy_id = "strategy_B"

        scenario_def_no, st.session_state.simulation_output_no_contain = simulate_cascading_failure(
            st.session_state.initial_error_rate,
            simulation_steps=8,
            containment_strategy=None,
        )
        _, st.session_state.simulation_output_with_contain = simulate_cascading_failure(
            st.session_state.initial_error_rate,
            simulation_steps=8,
            containment_strategy=strategy_id,
        )

        st.session_state.safe_degradation_example = define_safe_degradation()

        # ✅ Use the output builder
        st.session_state.cascading_failure_analysis_output = build_cascading_failure_analysis_output(
            scenario_definition=scenario_def_no,
            simulation_results_no_containment=st.session_state.simulation_output_no_contain,
            simulation_results_with_containment_B=st.session_state.simulation_output_with_contain,
            safe_degradation_example=st.session_state.safe_degradation_example,
        )

        st.success("Simulation complete!")

    if st.session_state.simulation_output_no_contain and st.session_state.simulation_output_with_contain:
        st.markdown("### Simulation Results (No Containment)")
        st.dataframe(pd.DataFrame(
            st.session_state.simulation_output_no_contain))

        st.markdown(
            f"### Simulation Results (With {st.session_state.containment_strategy})")
        st.dataframe(pd.DataFrame(
            st.session_state.simulation_output_with_contain))

        # ✅ Use refactored plotting helper
        fig_sim = plot_cascading_failure_health(
            simulation_output_no_contain=st.session_state.simulation_output_no_contain,
            simulation_output_with_contain=st.session_state.simulation_output_with_contain,
            initial_error_rate=st.session_state.initial_error_rate,
        )
        st.pyplot(fig_sim)

        st.markdown("### Safe Degradation Example")
        st.markdown(
            f"**Scenario:** {st.session_state.safe_degradation_example.get('name', 'N/A')}")
        st.markdown(
            f"**Description:** {st.session_state.safe_degradation_example.get('description', 'N/A')}")
        st.markdown(
            f"**Outcome:** {st.session_state.safe_degradation_example.get('outcome', 'N/A')}")
    else:
        st.info("Run the simulation to see results.")

elif st.session_state.current_page == "5. Recovery Plan Builder":
    st.markdown(
        "# 5. Defining Rollback Procedures and Recovery Objectives (RTO/RPO)")
    st.markdown("Define rollback triggers, procedures, and recovery objectives.")
    st.markdown("---")

    # Define procedures (idempotent per page render)
    (
        st.session_state.rollback_triggers,
        st.session_state.rollback_procedures,
        st.session_state.recovery_objectives,
        st.session_state.validation_for_re_enablement,
    ) = define_rollback_procedures_and_recovery_objectives()

    # ✅ Build recovery plan markdown via helper
    equipment_name = (
        st.session_state.failure_mode_analysis_output.get(
            "use_case", {}).get("equipment")
        or "Steam Turbine Unit 3"
    )
    st.session_state.recovery_plan_md_content = build_recovery_plan_markdown(
        equipment_name=equipment_name,
        rollback_triggers=st.session_state.rollback_triggers,
        rollback_procedures=st.session_state.rollback_procedures,
        recovery_objectives=st.session_state.recovery_objectives,
        validation_for_re_enablement=st.session_state.validation_for_re_enablement,
    )

    # ✅ KPI output builder
    st.session_state.resilience_kpis_output = build_resilience_kpis_output(
        recovery_objectives=st.session_state.recovery_objectives,
        rollback_triggers=st.session_state.rollback_triggers,
        validation_for_re_enablement=st.session_state.validation_for_re_enablement,
    )

    st.markdown("### Sample Recovery Plan (Guidance)")
    with st.expander("View Sample Recovery Plan Markdown"):
        st.markdown(st.session_state.sample_recovery_plan_md)

    st.markdown("### Defined Rollback Triggers")
    st.dataframe(pd.DataFrame(st.session_state.rollback_triggers))

    st.markdown("### Defined Rollback Procedures")
    st.dataframe(pd.DataFrame(st.session_state.rollback_procedures))

    st.markdown("### Generated Recovery Plan (Markdown Preview)")
    st.markdown(st.session_state.recovery_plan_md_content)

elif st.session_state.current_page == "6. KPI & Alert Designer":
    st.markdown("# 6. KPI & Alert Designer")
    st.markdown("Review recovery KPIs and validation steps.")
    st.markdown("---")

    if st.session_state.recovery_objectives:
        st.markdown("### Defined Recovery Objectives (RTO/RPO)")
        st.dataframe(pd.DataFrame(
            [st.session_state.recovery_objectives]).T.rename(columns={0: "Value"}))
    else:
        st.info("Recovery objectives not yet defined. Visit section 5.")

    if st.session_state.validation_for_re_enablement:
        st.markdown("### Validation Steps for AI Service Re-enablement")
        st.dataframe(pd.DataFrame(
            st.session_state.validation_for_re_enablement))
    else:
        st.info("Validation steps not yet defined. Visit section 5.")

elif st.session_state.current_page == "7. Export Playbook":
    st.markdown("# 7. Generating and Exporting the AI Resilience Playbook")
    st.markdown(
        "Consolidate everything into artifacts + evidence manifest + zip export.")
    st.markdown("---")

    if st.button("Generate & Export Full Playbook"):
        ensure_dir(st.session_state.output_dir)

        # ✅ Updated signature: generate_ai_resilience_playbook now takes rollback_procedures and resilience_kpis_output
        use_case = st.session_state.failure_mode_analysis_output.get(
            "use_case", {}) or {}
        st.session_state.full_ai_resilience_playbook_data = generate_ai_resilience_playbook(
            use_case=use_case,
            failure_modes_output=st.session_state.failure_mode_analysis_output,
            controls_output=st.session_state.resilience_controls_output,
            cascading_analysis_output=st.session_state.cascading_failure_analysis_output,
            rollback_procedures=st.session_state.rollback_procedures,
            resilience_kpis_output=st.session_state.resilience_kpis_output,
        )

        # Artifact paths
        st.session_state.artifact_paths = {
            "sector_playbook.json": os.path.join(st.session_state.output_dir, f"sector_playbook_{st.session_state.run_id}.json"),
            "failure_mode_analysis.json": os.path.join(st.session_state.output_dir, f"failure_mode_analysis_{st.session_state.run_id}.json"),
            "resilience_controls.json": os.path.join(st.session_state.output_dir, f"resilience_controls_{st.session_state.run_id}.json"),
            "recovery_plan.md": os.path.join(st.session_state.output_dir, f"recovery_plan_{st.session_state.run_id}.md"),
            "resilience_kpis.json": os.path.join(st.session_state.output_dir, f"resilience_kpis_{st.session_state.run_id}.json"),
            "session14_executive_summary.md": os.path.join(st.session_state.output_dir, f"session14_executive_summary_{st.session_state.run_id}.md"),
            "config_snapshot.json": os.path.join(st.session_state.output_dir, f"config_snapshot_{st.session_state.run_id}.json"),
        }

        # ✅ Use write helpers from source.py
        write_json(st.session_state.artifact_paths["sector_playbook.json"],
                   st.session_state.full_ai_resilience_playbook_data)
        write_json(st.session_state.artifact_paths["failure_mode_analysis.json"],
                   st.session_state.failure_mode_analysis_output)
        write_json(
            st.session_state.artifact_paths["resilience_controls.json"], st.session_state.resilience_controls_output)
        write_text(
            st.session_state.artifact_paths["recovery_plan.md"], st.session_state.recovery_plan_md_content)
        write_json(
            st.session_state.artifact_paths["resilience_kpis.json"], st.session_state.resilience_kpis_output)

        # executive summary
        st.session_state.executive_summary_content = generate_executive_summary(
            st.session_state.full_ai_resilience_playbook_data,
            st.session_state.run_id,
        )
        write_text(
            st.session_state.artifact_paths["session14_executive_summary.md"], st.session_state.executive_summary_content)

        # config snapshot
        write_json(
            st.session_state.artifact_paths["config_snapshot.json"], st.session_state.config_snapshot_data)

        # ✅ Evidence manifest helper
        st.session_state.evidence_manifest = build_evidence_manifest(
            st.session_state.run_id, st.session_state.artifact_paths)
        evidence_manifest_path = os.path.join(
            st.session_state.output_dir, f"evidence_manifest_{st.session_state.run_id}.json")
        write_json(evidence_manifest_path, st.session_state.evidence_manifest)

        # zip artifacts (+ manifest)
        zip_file_name = f"Session_14_{st.session_state.run_id}.zip"
        zip_file_path = os.path.join(
            st.session_state.output_dir, zip_file_name)

        with zipfile.ZipFile(zip_file_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for _, path in st.session_state.artifact_paths.items():
                if os.path.exists(path):
                    zipf.write(path, os.path.basename(path))
            if os.path.exists(evidence_manifest_path):
                zipf.write(evidence_manifest_path,
                           os.path.basename(evidence_manifest_path))

        st.success(
            f"AI Resilience Playbook and Evidentiary Artifacts Generated (Run ID: {st.session_state.run_id})")
        st.markdown(
            f"All artifacts saved in: `{st.session_state.output_dir}/`")
        for artifact in st.session_state.evidence_manifest.get("artifacts", []):
            st.markdown(
                f"- `{artifact['name']}`: `{artifact['path']}` (Hash: `{artifact['sha256_hash']}`)")

        with open(zip_file_path, "rb") as f:
            st.download_button(
                label="Download Full Playbook (ZIP)",
                data=f.read(),
                file_name=zip_file_name,
                mime="application/zip",
            )

    if st.session_state.full_ai_resilience_playbook_data:
        st.markdown("### Generated Executive Summary")
        with st.container(border=True):
            st.markdown(st.session_state.executive_summary_content)


# License
st.caption(
    """
---
## QuantUniversity License

© QuantUniversity 2025  
This notebook was created for **educational purposes only** and is **not intended for commercial use**.  

- You **may not copy, share, or redistribute** this notebook **without explicit permission** from QuantUniversity.  
- You **may not delete or modify this license cell** without authorization.  
- This notebook was generated using **QuCreate**, an AI-powered assistant.  
- Content generated by AI may contain **hallucinated or incorrect information**. Please **verify before using**.  

All rights reserved. For permissions or commercial licensing, contact: [info@qusandbox.com](mailto:info@qusandbox.com)
"""
)

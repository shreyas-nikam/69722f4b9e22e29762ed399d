
from streamlit.testing.v1 import AppTest
import pandas as pd
import os
import pytest
from datetime import datetime, timedelta

# Helper function to get the app path, assuming 'app.py' is in the current directory.
def get_app_path():
    return "app.py"

# fixture to ensure source.py is accessible for the app.py to import.
# In a real test setup, you might mock source.py's heavy dependencies or file operations.
# For this exercise, we assume source.py is available and its functions return deterministic data
# or are mocked externally to these tests if actual file system operations are to be avoided.
@pytest.fixture(autouse=True)
def setup_app_environment(tmp_path):
    # This fixture can be used to set up a temporary environment if app.py relies on
    # files in specific relative paths that need to be present for AppTest.
    # For now, we assume app.py and source.py are correctly placed for import.
    pass


def test_initial_page_and_session_state():
    """Verify the initial page loads correctly and essential session states are initialized."""
    at = AppTest.from_file(get_app_path()).run()

    # Check page title and main content
    assert at.title[0].value == "QuLab: Lab 14: Sector Playbook Builder (Engineering & Critical Infrastructure)"
    assert "AI Resilience Playbook Builder" in at.markdown[1].value
    assert "Ensuring Uptime in Critical Industrial Infrastructure" in at.markdown[2].value

    # Check session state initialization
    assert at.session_state["current_page"] == "Home"
    assert "run_id" in at.session_state
    assert "output_dir" in at.session_state
    assert at.session_state["sector"] == "Engineering"
    assert at.session_state["use_case"] == "Predictive Maintenance"
    assert at.session_state["system_type"] == "ML"
    assert at.session_state["automation_level"] == "High"
    assert at.session_state["uptime_requirement"] == "99.9%"
    assert at.session_state["human_override"] == "Available"
    assert "config_snapshot_data" in at.session_state
    # Asserting that templates and sample recovery plan are loaded (assuming source.py works)
    assert at.session_state.engineering_templates is not None
    assert at.session_state.infrastructure_templates is not None
    assert at.session_state.sample_recovery_plan_md is not None
    # Asserting that other critical session states are initialized, even if empty
    assert "identified_ai_failure_modes" in at.session_state
    assert "resilience_controls" in at.session_state
    assert "mapped_resilience_data" in at.session_state
    assert "full_ai_resilience_playbook_data" in at.session_state
    assert "artifact_paths" in at.session_state
    assert "evidence_manifest" in at.session_state


def test_navigation_to_system_configuration():
    """Verify navigation to '1. System Configuration' via sidebar."""
    at = AppTest.from_file(get_app_path())
    at.session_state["current_page"] = "Home"  # Start on Home for a clean navigation test
    at.run()

    # Select "1. System Configuration" from the sidebar
    at.selectbox[0].set_value("1. System Configuration").run()
    assert at.session_state["current_page"] == "1. System Configuration"
    assert at.markdown[1].value == "# 1. System Configuration & Use Case Wizard"


def test_system_configuration_and_confirm_button():
    """
    Test setting system configuration, clicking the confirm button,
    and verifying session state updates and navigation.
    """
    at = AppTest.from_file(get_app_path())
    at.session_state["current_page"] = "1. System Configuration"
    at.run()

    # Change some selectbox values
    at.selectbox(key="sector").set_value("Critical Infrastructure").run()
    at.selectbox(key="use_case").set_value("Anomaly Detection").run()
    at.selectbox(key="system_type").set_value("LLM").run()
    at.selectbox(key="automation_level").set_value("Medium").run()
    at.selectbox(key="uptime_requirement").set_value("99.99%").run()
    at.selectbox(key="human_override").set_value("Limited").run()

    # Click the confirm button
    at.button[0].click().run()

    # Verify session state updates
    assert at.session_state["sector"] == "Critical Infrastructure"
    assert at.session_state["use_case"] == "Anomaly Detection"
    assert at.session_state["system_type"] == "LLM"
    assert at.session_state["automation_level"] == "Medium"
    assert at.session_state["uptime_requirement"] == "99.99%"
    assert at.session_state["human_override"] == "Limited"
    assert "timestamp" in at.session_state.config_snapshot_data

    # Verify identified_ai_failure_modes and df_failure_modes are populated (assuming source.py works)
    assert at.session_state.identified_ai_failure_modes is not None and len(at.session_state.identified_ai_failure_modes) > 0
    assert not at.session_state.df_failure_modes.empty

    # Verify navigation to the next page and success message
    assert at.session_state["current_page"] == "2. Failure Mode Mapper"
    assert at.success[0].value == "Configuration saved and AI failure modes identified!"


def test_failure_mode_mapper_page():
    """Verify the Failure Mode Mapper page displays correctly after configuration."""
    at = AppTest.from_file(get_app_path())
    # Simulate prior configuration by setting session state and running page 1's button logic
    at.session_state["current_page"] = "1. System Configuration"
    at.run()
    at.button[0].click().run() # Click confirm config to populate failure modes and navigate

    # Now on page 2, verify content
    assert at.session_state["current_page"] == "2. Failure Mode Mapper"
    assert at.markdown[1].value == "# 2. Failure-Mode Mapper"
    assert "Identified AI-Related Failure Modes:" in at.markdown[4].value
    assert "Mathematical Context for RUL Prediction:" in at.markdown[5].value
    assert not at.dataframe[0].empty # Check if dataframe is displayed and not empty
    assert at.pyplot[0].figure is not None # Check if the plot is displayed

    # Test the warning case if failure modes are not identified (by not running page 1 config)
    at_fresh = AppTest.from_file(get_app_path())
    at_fresh.session_state["current_page"] = "2. Failure Mode Mapper"
    at_fresh.session_state["df_failure_modes"] = pd.DataFrame() # Ensure it's empty
    at_fresh.run()
    assert at_fresh.warning[0].value == "Please configure the system and identify failure modes in the '1. System Configuration' section first."


def test_resilience_control_selector_page():
    """Verify the Resilience Control Selector page displays correctly."""
    at = AppTest.from_file(get_app_path())
    # Simulate prior configuration and failure mode identification
    at.session_state["current_page"] = "1. System Configuration"
    at.run()
    at.button[0].click().run() # Confirm config to populate failure modes
    at.session_state["current_page"] = "3. Resilience Control Selector"
    at.run() # This run will also trigger map_controls_to_failure_modes if not already done

    assert at.session_state["current_page"] == "3. Resilience Control Selector"
    assert at.markdown[1].value == "# 3. Designing and Mapping Resilience Controls"
    assert "Mathematical Context for Control Effectiveness:" in at.markdown[3].value
    assert not at.dataframe[0].empty # Check for "Defined Resilience Controls" dataframe
    assert "Resilience Control Mapping:" in at.markdown[5].value
    assert "Visualization of Control Mapping:" in at.markdown[6].value
    assert at.pyplot[0].figure is not None # Check if the bipartite graph is displayed

    # Test the warning case if failure modes are not identified
    at_fresh = AppTest.from_file(get_app_path())
    at_fresh.session_state["current_page"] = "3. Resilience Control Selector"
    at_fresh.session_state["identified_ai_failure_modes"] = [] # Ensure it's empty
    at_fresh.session_state["mapped_resilience_data"] = [] # Ensure it's empty
    at_fresh.run()
    assert at_fresh.warning[0].value == "Please configure the system and identify failure modes in the '1. System Configuration' section first."


def test_cascading_failure_simulation_page_and_run():
    """Verify the simulation page, slider, radio, and running the simulation."""
    at = AppTest.from_file(get_app_path())
    # Simulate prior steps to have necessary session state for this page
    at.session_state["current_page"] = "1. System Configuration"
    at.run()
    at.button[0].click().run() # Confirm config
    at.session_state["current_page"] = "3. Resilience Control Selector"
    at.run() # Populate resilience controls and mapped data
    at.session_state["current_page"] = "4. Cascading Failure Simulation"
    at.run()

    assert at.session_state["current_page"] == "4. Cascading Failure Simulation"
    assert at.markdown[1].value == "# 4. Simulating a Cascading Failure Scenario and Defining Containment"

    # Check initial slider and radio button values
    assert at.slider(key="initial_error_rate").value == 0.20
    assert at.radio(key="containment_strategy").value == "None"

    # Change slider and radio values
    at.slider(key="initial_error_rate").set_value(0.30).run()
    at.radio(key="containment_strategy").set_value("Strategy B: Automatic Fallback to Scheduled Maintenance (RC002)").run()

    # Click run simulation button
    at.button[0].click().run()

    assert at.success[0].value == "Simulation complete!"
    assert at.session_state.simulation_output_no_contain is not None and len(at.session_state.simulation_output_no_contain) > 0
    assert at.session_state.simulation_output_with_contain is not None and len(at.session_state.simulation_output_with_contain) > 0
    assert at.pyplot[0].figure is not None # Check if the plot is displayed
    assert "Safe Degradation Example:" in at.markdown[7].value
    assert at.session_state.safe_degradation_example.get("name") is not None

    # Test the initial state message before simulation runs
    at_fresh = AppTest.from_file(get_app_path())
    at_fresh.session_state["current_page"] = "4. Cascading Failure Simulation"
    at_fresh.session_state["simulation_output_no_contain"] = []
    at_fresh.session_state["simulation_output_with_contain"] = []
    at_fresh.run()
    assert at_fresh.info[0].value == "Run the simulation to see results."


def test_recovery_plan_builder_page():
    """Verify the Recovery Plan Builder page displays correctly."""
    at = AppTest.from_file(get_app_path())
    # Simulating all prior steps to ensure all session state is populated for this page
    at.session_state["current_page"] = "1. System Configuration"
    at.run()
    at.button[0].click().run() # Confirm config
    at.session_state["current_page"] = "3. Resilience Control Selector"
    at.run()
    at.session_state["current_page"] = "4. Cascading Failure Simulation"
    at.run()
    at.button[0].click().run() # Run simulation
    at.session_state["current_page"] = "5. Recovery Plan Builder"
    at.run() # This populates rollback_triggers, procedures, recovery_objectives, etc.

    assert at.session_state["current_page"] == "5. Recovery Plan Builder"
    assert at.markdown[1].value == "# 5. Defining Rollback Procedures and Recovery Objectives (RTO/RPO)"
    assert "Mathematical Context for RTO/RPO:" in at.markdown[3].value
    assert "Sample Recovery Plan (Guidance):" in at.markdown[4].value
    assert at.markdown[5].value.startswith("# Sample Recovery Plan") # Check content of sample markdown
    assert not at.dataframe[0].empty # Check for "Defined Rollback Triggers" dataframe
    assert not at.dataframe[1].empty # Check for "Defined Rollback Procedures" dataframe
    assert at.session_state.rollback_triggers is not None and len(at.session_state.rollback_triggers) > 0
    assert at.session_state.rollback_procedures is not None and len(at.session_state.rollback_procedures) > 0
    assert at.session_state.recovery_objectives is not None and len(at.session_state.recovery_objectives) > 0


def test_kpi_alert_designer_page():
    """Verify the KPI & Alert Designer page displays correctly."""
    at = AppTest.from_file(get_app_path())
    # Simulating all prior steps to ensure all session state is populated
    at.session_state["current_page"] = "1. System Configuration"
    at.run()
    at.button[0].click().run() # Confirm config
    at.session_state["current_page"] = "3. Resilience Control Selector"
    at.run()
    at.session_state["current_page"] = "4. Cascading Failure Simulation"
    at.run()
    at.button[0].click().run() # Run simulation
    at.session_state["current_page"] = "5. Recovery Plan Builder"
    at.run() # Populates recovery_objectives and validation_for_re_enablement
    at.session_state["current_page"] = "6. KPI & Alert Designer"
    at.run()

    assert at.session_state["current_page"] == "6. KPI & Alert Designer"
    assert at.markdown[1].value == "# 6. KPI & Alert Designer"
    assert "Defined Recovery Objectives (RTO/RPO):" in at.markdown[3].value
    assert not at.dataframe[0].empty # Check for Recovery Objectives dataframe
    assert "Validation Steps for AI Service Re-enablement:" in at.markdown[5].value
    assert not at.dataframe[1].empty # Check for Validation Steps dataframe

    # Test case where recovery objectives and validation steps are not defined
    at_fresh = AppTest.from_file(get_app_path())
    at_fresh.session_state["current_page"] = "6. KPI & Alert Designer"
    # Manually ensure these session states are empty for this test
    at_fresh.session_state["recovery_objectives"] = {}
    at_fresh.session_state["validation_for_re_enablement"] = []
    at_fresh.run()
    assert at_fresh.info[0].value == "Recovery objectives not yet defined. Visit section 5."
    assert at_fresh.info[1].value == "Validation steps not yet defined. Visit section 5."


def test_export_playbook_page_and_generation():
    """
    Verify the Export Playbook page, clicking the generate button,
    and checking for generated outputs in session state.
    """
    at = AppTest.from_file(get_app_path())
    # Simulate all prior steps to ensure all session state is populated for playbook generation
    at.session_state["current_page"] = "1. System Configuration"
    at.run()
    at.button[0].click().run() # Confirm config
    at.session_state["current_page"] = "3. Resilience Control Selector"
    at.run()
    at.session_state["current_page"] = "4. Cascading Failure Simulation"
    at.run()
    at.button[0].click().run() # Run simulation
    at.session_state["current_page"] = "5. Recovery Plan Builder"
    at.run() # This populates recovery_plan_md_content, resilience_kpis_output, etc.
    at.session_state["current_page"] = "7. Export Playbook"
    at.run()

    assert at.session_state["current_page"] == "7. Export Playbook"
    assert at.markdown[1].value == "# 7. Generating and Exporting the AI Resilience Playbook"

    # Click the generate button
    at.button[0].click().run()

    assert at.success[0].value.startswith("AI Resilience Playbook and Evidentiary Artifacts Generated (Run ID:")
    assert at.download_button[0].label == "Download Full Playbook (ZIP)"

    # Verify critical session state variables are populated (assuming source.py functions work)
    assert at.session_state.full_ai_resilience_playbook_data is not None
    assert at.session_state.executive_summary_content is not None and at.session_state.executive_summary_content.startswith("Executive Summary for")
    assert at.session_state.artifact_paths is not None and len(at.session_state.artifact_paths) > 0
    assert at.session_state.evidence_manifest is not None
    assert "artifacts" in at.session_state.evidence_manifest
    assert len(at.session_state.evidence_manifest["artifacts"]) > 0
    assert "sha256_hash" in at.session_state.evidence_manifest["artifacts"][0]

    # Verify displayed executive summary and manifest
    assert "Generated Executive Summary:" in at.markdown[5].value
    assert "Generated Evidence Manifest:" in at.markdown[6].value
    assert at.json[0].value is not None


def test_sidebar_navigation_persistence():
    """Verify that the sidebar selection persists across reruns."""
    at = AppTest.from_file(get_app_path()).run()

    # Navigate to a specific page
    at.selectbox[0].set_value("4. Cascading Failure Simulation").run()
    assert at.session_state["current_page"] == "4. Cascading Failure Simulation"

    # Simulate a rerun (e.g., interaction on the current page, or a full app reload if manually setting state)
    # AppTest's .run() implicitly handles reruns initiated by widget interactions.
    # To test persistence if the app was re-loaded from scratch, we'd need to manually
    # set the session state and re-run AppTest.
    at_reload = AppTest.from_file(get_app_path())
    at_reload.session_state["current_page"] = "4. Cascading Failure Simulation" # Simulate persisted state
    at_reload.run()

    # Check that the page title and the selectbox value reflect the persisted state
    assert at_reload.markdown[1].value == "# 4. Simulating a Cascading Failure Scenario and Defining Containment"
    assert at_reload.selectbox[0].value == "4. Cascading Failure Simulation"

This `README.md` provides a comprehensive overview of the "QuLab: Lab 14: Sector Playbook Builder" Streamlit application.

---

# QuLab: Lab 14: Sector Playbook Builder (Engineering & Critical Infrastructure)

## AI Resilience Playbook Builder: Ensuring Uptime in Critical Industrial Infrastructure

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-app-link-if-deployed)

This project, developed as "QuLab: Lab 14," focuses on building a robust "AI Resilience Playbook" for critical industrial processes, specifically targeting sectors like Engineering and Critical Infrastructure. The application guides a Systems Engineer through a structured process to identify potential AI-related failure modes, design proactive resilience controls, simulate cascading impacts, and establish clear recovery procedures for AI-driven systems.

The primary scenario addressed is a predictive maintenance system for high-pressure steam turbines in a manufacturing facility. Unplanned downtime in such critical infrastructure can lead to significant financial losses and safety risks. This playbook aims to safeguard operations by ensuring continuous functionality and safety, even when the underlying AI model or its data pipelines encounter issues.

## Features

The Streamlit application provides an interactive, step-by-step workflow for developing a comprehensive AI resilience playbook:

1.  **System Configuration & Use Case Wizard**:
    *   Define the operational context: Sector (Engineering, Critical Infrastructure), AI Use Case (Predictive Maintenance, Anomaly Detection), System Type (ML, LLM, Agent), Automation Level, Uptime Requirements, and Human Override Availability.
    *   Automatically identifies initial AI-related failure modes based on the configured parameters.

2.  **Failure Mode Mapper**:
    *   Presents a detailed list of potential AI-specific failure modes (categorized by Data, Model, Infrastructure) that could compromise the AI's accuracy and lead to operational issues.
    *   Includes mathematical context for metrics like Mean Absolute Error (MAE) in RUL prediction.
    *   Visualizes failure mode distribution by category using a bar chart, highlighting areas of vulnerability.

3.  **Resilience Control Selector**:
    *   Defines a set of generic resilience controls (e.g., redundancy, graceful degradation, automatic fallback).
    *   Maps these controls to the identified AI failure modes, demonstrating how each vulnerability can be mitigated.
    *   Visualizes the mapping between failure modes and controls using a bipartite graph, offering a clear overview of coverage.

4.  **Cascading Failure Simulation**:
    *   Simulates a critical scenario, such as "Model Drift (False Negatives)" leading to system degradation.
    *   Allows users to set an initial AI error rate and select different containment strategies for comparison.
    *   Graphs the system health degradation over time, both with and without containment strategies, providing quantitative justification for resilience investments.
    *   Presents an example of safe degradation for system continuity.

5.  **Recovery Plan Builder**:
    *   Defines explicit rollback triggers based on AI performance, data health, and other operational metrics.
    *   Outlines detailed rollback procedures to restore service or revert to a stable state.
    *   Establishes critical Recovery Time Objectives (RTO) and Recovery Point Objectives (RPO) for the AI service, crucial for incident response.

6.  **KPI & Alert Designer**:
    *   Summarizes the defined RTO/RPO metrics and crucial validation steps required before fully re-enabling the AI service.
    *   Highlights Key Performance Indicators (KPIs) associated with recovery and re-enablement.

7.  **Export Playbook**:
    *   Consolidates all generated data, analysis, mappings, simulations, and recovery definitions into a comprehensive "Predictive Maintenance AI Resilience Playbook."
    *   Generates various artifacts (JSON and Markdown files) including a full playbook JSON, failure mode analysis, resilience controls, recovery plan, KPI definitions, and an executive summary.
    *   Creates an `evidence_manifest.json` with SHA-256 hashes for all generated artifacts, ensuring integrity, traceability, and auditability.
    *   Provides a convenient downloadable ZIP archive containing all generated reports and evidence.

## Getting Started

Follow these instructions to set up and run the application locally.

### Prerequisites

*   Python 3.8+
*   Git (for cloning the repository)

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/quslab-lab14-ai-playbook-builder.git
    cd quslab-lab14-ai-playbook-builder
    ```
    (Replace `your-username/quslab-lab14-ai-playbook-builder.git` with the actual repository URL if available).

2.  **Create a virtual environment (recommended)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies**:
    Create a `requirements.txt` file in the root directory of the project with the following content:
    ```
    streamlit==1.33.0 # or latest stable version
    pandas==2.2.2     # or latest stable version
    matplotlib==3.8.4 # or latest stable version
    networkx==3.2.1   # or latest stable version
    seaborn==0.13.2   # or latest stable version
    ```
    Then install them:
    ```bash
    pip install -r requirements.txt
    ```

### Data Files & Source Code
Ensure the `source/` directory contains the necessary data files and helper functions:
```
quslab-lab14-ai-playbook-builder/
├── app.py
├── source/
│   ├── __init__.py          # Contains all helper functions (load_templates, identify_ai_failure_modes, etc.)
│   ├── engineering_failure_templates.json
│   ├── infrastructure_failure_templates.json
│   └── sample_recovery_plans.md
├── reports/                 # (Automatically created for outputs)
└── requirements.txt
```
You will need to implement the helper functions within `source/__init__.py` and provide the example JSON/MD files.

## Usage

1.  **Run the Streamlit application**:
    From the project's root directory:
    ```bash
    streamlit run app.py
    ```
    This will open the application in your default web browser (usually `http://localhost:8501`).

2.  **Navigate through the Playbook Builder**:
    *   Use the sidebar on the left to navigate through the different sections (Home, 1. System Configuration, 2. Failure Mode Mapper, etc.).
    *   Follow the instructions on each page to input system details, analyze failure modes, design controls, run simulations, and build the recovery plan.
    *   The application saves progress in Streamlit's session state, allowing you to move between pages.

3.  **Export the Playbook**:
    *   Once all sections are completed, go to "7. Export Playbook" to generate and download the complete AI Resilience Playbook, including all artifacts and an evidence manifest.

## Project Structure

*   `app.py`: The main Streamlit application script, handling UI, state management, and routing.
*   `source/`: A Python package containing:
    *   `__init__.py`: All core logic and helper functions for data loading, failure mode identification, control mapping, simulations, recovery plan generation, and executive summary creation.
    *   `engineering_failure_templates.json`: JSON file containing predefined failure mode templates for engineering contexts.
    *   `infrastructure_failure_templates.json`: JSON file containing predefined failure mode templates for critical infrastructure contexts.
    *   `sample_recovery_plans.md`: Markdown file providing a template or example for recovery plans.
*   `reports/`: Directory automatically created by the application to store generated playbook artifacts and evidence for each session run. Each run gets a unique subdirectory (`reports/session14/<run_id>/`).
*   `requirements.txt`: Lists all Python dependencies required for the project.

## Technology Stack

*   **Framework**: [Streamlit](https://streamlit.io/) (for interactive web applications)
*   **Programming Language**: Python
*   **Data Manipulation**: [Pandas](https://pandas.pydata.org/)
*   **Visualization**: [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/), [NetworkX](https://networkx.org/) (for graph visualization)
*   **File Operations**: Standard Python libraries (`os`, `json`, `zipfile`, `datetime`, `hashlib`)

## Contributing

Contributions are welcome! If you have suggestions for improving the application, feel free to:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

Please ensure your code adheres to Python best practices and includes appropriate documentation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (if applicable, create a `LICENSE` file in the root directory).

## Contact

For any questions or inquiries, please contact:

*   **QuantUniversity**: [info@quantuniversity.com](mailto:info@quantuniversity.com)
*   **Website**: [https://www.quantuniversity.com/](https://www.quantuniversity.com/)

---

# AI Predictive Maintenance Service Recovery Plan - Steam Turbine Unit 3

This document outlines the specific procedures for detecting, mitigating, and recovering from failures in the AI-driven predictive maintenance service for High-Pressure Steam Turbine Unit 3.

## 1. Rollback Triggers
These are the conditions that will automatically or manually trigger a rollback or degraded mode operation.

- **Sustained Model Performance Degradation (RT001):** AI model's MAE for RUL prediction exceeds 1.5 times the baseline for 30 consecutive minutes.
- **Data Pipeline Ingestion Stoppage (RT002):** No new sensor data ingested into the AI feature store for 15 minutes.
- **Excessive Inference Latency (RT003):** Average inference latency for RUL predictions exceeds 500ms for 5 consecutive minutes.


## 2. Rollback Procedures
These are the actions to be taken when a rollback trigger is detected.

- **Automated Deployment of Last Validated Model (RP001):**
  - **Triggers:** RT001
  - **Description:** Triggered by RT001. Automatically deploys the last known good AI model version from the production registry. Requires a pre-validated model artifact.
  - **Validation Steps:** Run post-deployment sanity checks on recent production data.; Monitor initial 1 hour of inference metrics.
- **Manual Revert to Heuristic-Based Prediction (RP002):**
  - **Triggers:** RT002, RT003
  - **Description:** Triggered by RT002/RT003. A human operator manually switches the inference endpoint to a pre-defined heuristic rule-based system for RUL prediction. This provides basic, safe operation.
  - **Validation Steps:** Confirm heuristic system is active and providing output.; Verify outputs against known safe thresholds.
- **Data Pipeline Restart & Backfill (RP003):**
  - **Triggers:** RT002
  - **Description:** Triggered by RT002. Automated attempt to restart data ingestion services and backfill missing data from raw sensor archives.
  - **Validation Steps:** Verify data ingestion rate resumes normal levels.; Check data integrity of backfilled data.


## 3. Recovery Objectives (RTO/RPO)
These objectives define the target timelines for service restoration and acceptable data loss.

- **RTO AI Service Degraded Mode:** 2 hours
- **RTO AI Service Full Functionality:** 8 hours
- **RPO AI Data for Retraining:** 30 minutes
- **RPO AI Inference State:** 10 minutes


## 4. Validation Steps for Full AI Service Re-enablement
Before fully re-enabling the primary AI service, the following conditions must be met to ensure stability and accuracy.

- **Step 1:** AI Model Performance: MAE within 1.1x baseline and no significant drift for 24 hours. (Key Performance Indicator: Model_MAE_Stability)
- **Step 2:** Data Pipeline Health: 99.9% data ingestion success rate and no errors for 12 hours. (Key Performance Indicator: Data_Pipeline_Health)
- **Step 3:** Inference Infrastructure Stability: Average inference latency below 200ms and 99.9% uptime for 48 hours. (Key Performance Indicator: Inference_Stability)

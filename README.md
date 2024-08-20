# Identifying Key Predictive Variables in Medical Records Using a Large Language Model (LLM)

## Project Overview

This project leverages the capabilities of Large Language Models (LLMs) to extract and analyze key predictive variables from unstructured electronic health records (EHRs) within the Veterans Health Administration (VHA). Specifically, this project focuses on female infertility as a case study to demonstrate the framework's effectiveness in integrating structured and unstructured data for enhanced health outcome predictions.

## Repository Structure

- **Class_Specific_Themes_Analysis.py:** This script implements the initial theme extraction process described in the manuscript under the "Theme Discovery" section. It focuses on identifying class-specific themes within clinical notes using iterative prompt engineering and subject matter expert (SME) validation.
 
- **Formal_Thematic_Analysis.py:** This script is used for the "Formal Thematic Analysis" section. It refines the theme extraction process by assigning probabilities to the presence of specific themes within clinical notes, which are then integrated into a multimodal dataset for further analysis.

## Key Features

- **Data Collection and Preparation:** The project processes structured and unstructured EHR data from female patients within the VHA to identify factors contributing to infertility.
 
- **Theme Discovery:** Utilizes Meta’s Llama-3-8b LLM for extracting relevant themes from unstructured clinical notes through advanced prompt engineering techniques.
 
- **Class-Specific Themes Analysis:** Identifies and compares key themes between case and control groups, focusing on themes most indicative of infertility.
 
- **Formal Thematic Analysis:** Assigns probabilistic values to themes, enhancing the model's ability to predict infertility outcomes when integrated with structured data.

## Installation and Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/LLM_Theme_Analysis.git
   cd LLM_Theme_Analysis

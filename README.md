# 🤖 ACQUIRE – AI Campaign Quality Insight & Revenue Engine

ACQUIRE is an agentic AI system that empowers marketers to design, evaluate, and simulate high-performance marketing campaigns through intelligent lead segmentation and targeting — all with just a campaign brief.

> 🚀 Drive peak campaign performance with agentic intelligence — from insight to action.

---

## 📌 Key Features

1. **Smart Lead Targeting**  
    Generates 3 targeting logic sets per campaign:  
    - 2 inspired by similar historical campaigns, and  
    - 1 innovative logic generated from campaign objective and data dictionary.

2. **LLM-driven Reasoning**  
  Each logic is explained with rationale including customer profile and behavioral insight.

3. **Insight-Based Retrieval**  
  Uses embeddings and similarity matching to find the top 5 most relevant past campaigns as inspiration.

4. **Interactive UI**  
  Enables users to review logic, simulate filters, and visualize projected campaign uplift for ROI and CVR.

5. **Data Quality Awareness**  
  Highlights columns, risks, and assumptions during logic generation.

---

## ⚙️ System Architecture

```text
User Input (New Campaign Description)
        ↓
Top 5 Most Similar Past Campaigns (Embedding Search)
        ↓
LLM Prompt (with Data Dictionary + Leads Sample)
        ↓
AI-Suggested Targeting Logics (pandas query format)
        ↓
UI Simulation (Streamlit) + Uplift Estimation
```

---

## 🔍 How to get Started?

1. **Install the required dependencies:**

```bash
pip install -r requirements.txt
```
2. **Get your OpenAI API Key**

- Get your API key from: https://platform.openai.com/api-keys

3. **Run the Streamlit App**
```bash
streamlit run acquire.py
```

## ⚠️ Things to consider before running the script

1. **Data Source**
- All data from folder data_source are dummy data
- Pay attention with the format data source when uploading the files on the script to avoid error
- The simulation result based on assumption, make adjustment on the script if needed


2. **Pro Tip:**
- Write detail new campaign prompt (ex: add specific merchant, promotion scheme, payment method)
- Write detail data dictionary description (ex: specified format values on columns)

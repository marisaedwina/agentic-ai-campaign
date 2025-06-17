import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
import re
from PIL import Image

st.set_page_config(page_title="ACQUIRE", page_icon="logo.png", layout="wide")
logo = Image.open("logo.png")
col1, col2 = st.columns([1, 6])
with col1:
    st.image(logo, width=150)
with col2:
    st.title("AI Campaign Quality Insight & Revenue Engine")


# API Key Input
openai_api_key = st.text_input("🔐 OpenAI API Key", type="password")

# File Uploaders
st.sidebar.header("📁 Upload Data Files")
criteria_file = st.sidebar.file_uploader("Upload Campaign Criteria (Excel)", type=["xlsx"])
perf_file = st.sidebar.file_uploader("Upload Campaign Performance (Excel)", type=["xlsx"])
leads_datamart_file = st.sidebar.file_uploader("Upload Leads Datamart (Excel)", type=["xlsx"])
data_dict_file = st.sidebar.file_uploader("Upload Leads Data Dictionary (Excel)", type=["xlsx"])

new_campaign = st.text_input("📌 New Campaign Description", placeholder="Example: Cashback 30% (maksimum Rp10,000) using Credit Card on Gojek App")

# Embedding Functions
def get_embedding_multiple(list_text, client):
    response = client.embeddings.create(model="text-embedding-3-small", input=list_text)
    return [data.embedding for data in response.data]

def get_embedding(text, client):
    response = client.embeddings.create(model="text-embedding-3-small", input=text)
    return response.data[0].embedding

def extract_logics_from_ai_response(text):
    pattern = r"Logic:\s*(.*?)\n\s*b\."
    matches = re.findall(pattern, text, re.DOTALL)
    return [m.replace("`", "").replace("** ", "").strip() for m in matches]

if all([criteria_file, perf_file, new_campaign, openai_api_key]):
    # Load data
    criteria_df = pd.read_excel(criteria_file)
    perf_df = pd.read_excel(perf_file)
    campaign_df = pd.merge(criteria_df, perf_df, on="campaign_id")
    
    # Prepare data dictionary if available
    data_dict_info = ""
    if data_dict_file:
        data_dict = pd.read_excel(data_dict_file)
        data_dict_info = f"""
    Leads Data Dictionary Information:
    {data_dict.to_markdown(index=False)}
    """
    
    # Prepare leads datamart sample if available
    leads_sample_info = ""
    if leads_datamart_file:
        leads_df = pd.read_excel(leads_datamart_file)
        leads_sample_info = f"""
    Leads Datamart Sample (first 5 rows):
    {leads_df.head().to_markdown(index=False)}
    """
    
    # Create embeddings
    campaign_df["embedding_text"] = campaign_df.apply(
        lambda row: f"{row['campaign_description_full']} | channel: {row['channel']} | ROI: {row['roi']} | CVR: {row['cvr']}", axis=1
    )

    client = OpenAI(api_key=openai_api_key)

    with st.spinner("🔎 Find relevant campaign based on new campaign input..."):
        campaign_df["embedding"] = get_embedding_multiple(list(campaign_df['embedding_text']), client)
        user_embedding = get_embedding(new_campaign, client)

        embedding_matrix = np.vstack(campaign_df["embedding"].values)
        similarities = cosine_similarity([user_embedding], embedding_matrix)[0]
        campaign_df["similarity"] = similarities

        retrieved_top5 = campaign_df.sort_values(by="similarity", ascending=False).head(5)

        if st.session_state.get("top5_prompt") != new_campaign:
            st.session_state.top5_prompt = new_campaign
            st.session_state.top5 = retrieved_top5.copy()

    top5 = st.session_state.top5

    st.subheader("📌 Top 5 Most Relevant Historical Campaign")
    st.dataframe(top5[["campaign_id", "campaign_description_full", "sql_where_clause", "channel", "roi", "cvr", "similarity"]])

    # Enhanced prompt with data dictionary and leads info
    prompt = f"""
You are a banking marketing strategist AI with expertise in customer segmentation.

# [NEW CAMPAIGN CONTEXT]
{new_campaign}

# [DATA CONSTRAINTS]
You MUST only use the following columns:
{data_dict_info}

You are not allowed to invent new column names. Use only the actual columns provided.

Sample Customer Records:
{leads_sample_info}

# [HISTORICAL CAMPAIGN INSIGHTS]
Top 5 Most Similar Past Campaigns:
{top5[['sql_where_clause', 'channel', 'roi', 'cvr']].to_dict(orient='records')}


# [TASK INSTRUCTION]
Your task is to generate 3 lead targeting waterfall logics (Pandas query format):
1. Two logics adapted or inspired by sql_where_clause from the most similar past campaigns (if relevant with the new_campaign).
2. One logic must be innovative — combine the actual data dictionary, variable types, and new campaign objective.
3. Do NOT copy example logic. Must vary based on context.
4. Format:
   - a. Logic: <pandas query format>
   - b. Rationale: <explanation of customer profile and reasoning>
5. Include best channel & timing suggestion with justification.
6. Include 1-2 key data quality concerns to watch.

# [OUTPUT FORMAT GUIDE]
Do not reuse example logic. The innovative waterfall must vary depending on the new_campaign context, past campaign patterns, and insights from the data dictionary.
Do NOT include any column not listed in the data dictionary

1. Leads Waterfall 1 (Inspired by Historical Campaigns):
    
a. Logic: segment in ['Busy Parents', 'Young Professionals'] & cc_spend_monthly > 500000 & has_credit_card == 1

b. Target Customer Profile & Rationale: This logic targets 'Busy Parents' and 'Young Professionals' who have a high monthly credit card spend (more than Rp 500,000). The rationale is that these segments are likely to have disposable income and would appreciate the convenience and savings from using their credit cards for transactions.


2. Leads Waterfall 2 (Inspired by Historical Campaigns): 

a. Logic: cc_spend_monthly > 500000 & has_credit_card == 1 & most_visited_merchant_category == 'Food'

b. Target Customer Profile & Rationale: This logic targets customers who frequently visit food merchants and have a high monthly credit card spend (more than Rp 500,000). The rationale is that these customers are likely to value discounts at restaurants like Nannys Pavillon.


3. Leads Waterfall 3 (Innovative AI Waterfall): 

a. Logic: cc_spend_monthly > 500000 & has_credit_card == 1 & (digital_engagement_score > 0.5 | whatsapp_click_rate > 0.5)

b. Target Customer Profile & Rationale: This logic targets customers with high digital engagement or high WhatsApp click rates, indicating they are digitally savvy and responsive to digital communications. The rationale is that these customers are likely to be more receptive to the campaign and take action on it.


4. Recommended Channel and Timing: The best channel would be WhatsApp based on the high ROI and CTR from historical campaigns. Moreover, the high WhatsApp click rates in the 'Innovative Logic' segment suggests that customers are responsive to this channel. The campaign should be sent on Fridays to remind customers of the weekend promotion at Nannys Pavillon.


5. Data Quality Considerations: It's important to ensure that the credit card spend data is accurate and up-to-date as it's a key criterion for our target customer profile. Additionally, the digital engagement score and WhatsApp click rate data should be validated for accuracy. Finally, customer segmentation should be reviewed periodically to ensure customers are in the correct segments.

"""
    if "llm_result_prompt" not in st.session_state or st.session_state.llm_result_prompt != new_campaign:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a data-savvy marketing strategist and python expert that combines historical performance with fresh insights."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        st.session_state.llm_result_prompt = new_campaign
        st.session_state.llm_result = response.choices[0].message.content
        if "selected_option" in st.session_state:
            del st.session_state.selected_option  # reset dropdown selection

    result = st.session_state.llm_result

    st.subheader("🎯 Enhanced Recommendation Strategy")
    st.markdown(result)

    # ROI Uplift Simulation
    st.subheader("📈ROI Uplift Simulation")
    avg_roi = perf_df['roi'].mean()
    top_roi = top5['roi'].mean()
    simulated_uplift_roi = top_roi - avg_roi   

    st.markdown(f"**📌 Average ROI (All Campaigns):** {avg_roi:.2f}")
    st.markdown(f"**📌 Avg ROI (Top 5 Relevant):** {top_roi:.2f}")
    st.markdown(f"**🚀 Simulated ROI Uplift:** `{simulated_uplift_roi:.2f}`")

    # Conv Rate Uplift Simulation
    st.subheader("📈 Conversion Rate Uplift Simulation")
    avg_cvr = perf_df['cvr'].mean()
    top_cvr = top5['cvr'].mean()
    simulated_uplift_cvr = top_cvr - avg_cvr

    st.markdown(f"**📌 Average Conv Rate (All Campaigns):** {avg_cvr * 100:.2f}%")
    st.markdown(f"**📌 Avg Conv Rate (Top 5 Relevant):** {top_cvr * 100:.2f}%")
    st.markdown(f"**🚀 Simulated Conv Rate Uplift:** `{simulated_uplift_cvr * 100:.2f}%`")

    # Enhanced Waterfall Leads Simulation
    if leads_datamart_file:
        st.subheader("💧 Advanced Leads Simulation")
        
        # Display data dictionary if available
        if data_dict_file:
            st.markdown("### 📚 Leads Data Dictionary")
            st.dataframe(data_dict)
        
        # Generate targeting options from the AI response
        try:
            sql_options = extract_logics_from_ai_response(result)
            # st.markdown(f"Result Query Extract: {sql_options}")

            if sql_options:
                selected_option = st.selectbox("Select Criteria Logic for Simulation", sql_options, key="selected_option")
                st.write("🔍 Selected Criteria:", selected_option)
                try:
                    filtered_leads = leads_df.query(selected_option, engine="python")
                    total_leads = len(filtered_leads)

                    st.subheader("📋 Filtered Leads")
                    # st.dataframe(filtered_leads.head())
                    st.success(f"✅ Total leads: {len(filtered_leads)}")
                    st.success(f"✅ Total leads: {total_leads:,} ({total_leads/len(leads_df):.1%} from overall population)")

                    # Enhanced waterfall estimation
                    waterfall = {
                        'Total Population': len(leads_df),
                        'Target Segment': total_leads,
                        'Eligible': int(total_leads * 0.85),
                        'Reachable': int(total_leads * 0.7),
                        'Interested': int(total_leads * 0.4),
                        'Converted': int(total_leads * 0.15)
                    }
                                                                    
                    # Visualization
                    st.markdown("### 📊 Waterfall Projection")
                    waterfall_df = pd.DataFrame.from_dict(
                        waterfall, 
                        orient='index', 
                        columns=['Count']
                    )
                    st.bar_chart(waterfall_df)
                    
                    # Add conversion metrics
                    st.markdown("### 📈 Conversion Metrics")
                    metrics = {
                        'Eligibility Rate': f"{waterfall['Eligible']/waterfall['Target Segment']:.1%}",
                        'Reachability Rate': f"{waterfall['Reachable']/waterfall['Eligible']:.1%}",
                        'Interest Rate': f"{waterfall['Interested']/waterfall['Reachable']:.1%}",
                        'Conversion Rate': f"{waterfall['Converted']/waterfall['Interested']:.1%}"
                    }
                    st.table(pd.DataFrame.from_dict(metrics, orient='index', columns=['Rate']))

                except Exception as e:
                    st.error(f"❌ Gagal menjalankan query:\n{selected_option}\n\nError: {e}")
            else:
                st.warning("Tidak ada logic berhasil diparsing dari hasil AI.")
        except Exception as e:
            st.warning("Tidak bisa mengekstrak SQL clauses dari respons AI. Pastikan format respons sesuai.")

else:
    st.info("⬅️ Upload file and fill new campaign to start.")
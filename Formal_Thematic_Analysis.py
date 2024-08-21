# Databricks notebook source
# MAGIC %md
# MAGIC ## Install dependencies

# COMMAND ----------

# Install necessary libraries
%pip install transformers==4.38.2

# COMMAND ----------

# MAGIC %pip install pyspark>=3.1.2

# COMMAND ----------

# MAGIC %pip install ray[default]>=2.3.0

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

#Load modules
import ray
import pandas as pd
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModel
from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster, MAX_NUM_WORKER_NODES

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set up Ray cluster

# COMMAND ----------

# Set up Ray cluster
ray.init()
ray.cluster_resources()

# COMMAND ----------

# Shutdown any previous Ray clusters
ray.util.spark.shutdown_ray_cluster()
ray.shutdown()

# Define the number of GPUs and worker nodes
GPUS_PER_NODE = 8 
NUM_OF_WORKER_NODES = 2 

# Set up the Ray cluster
setup_ray_cluster(
    num_cpus_worker_node=8,
    num_gpus_per_node=GPUS_PER_NODE,
    num_cpus_head_node=6,
    max_worker_nodes=NUM_OF_WORKER_NODES,
    actor_creation_timeout=600,
    collect_log_to_path="/ray_paths",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## LLM Prompts

# COMMAND ----------

#Function to interacct with the LLM and extract themes from clinical notes
@ray.remote(num_gpus=1)
def summarize_llm(chkpt, access_token, txt_list):
    print(f"txt_list {txt_list}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(chkpt, trust_remote_code=True, use_auth_token=access_token)
    torch_dtype = torch.afloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        chkpt,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        use_auth_token=access_token
    ).to(device)

#Function to process each text input and extract themes
    def _summarize(txt):
        try:
            print(f"Processing text: {txt[:200]}...")  # Print a snippet of the input text for debugging
            input_text = txt
            with torch.no_grad():
                input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
                outputs = model.generate(input_ids, max_new_tokens=1024, pad_token_id=tokenizer.eos_token_id, num_beams=5, early_stopping=True, no_repeat_ngram_size=2)
            summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Generated summary: {summary}")
            return summary
        except RuntimeError as e:
            print(f"RuntimeError: {e}")
            return ""
    #List of themes to be analyzed in the clinical notes
    themes = ['Infertility concerns','Pregnancy planning and counselling','Ovulation disorders','Hormone imbalances','Abnormal pap smear']  

    summaries = []
    for txt in txt_list:
        input_text = f"""You are an intelligent assistant programs to analyze women's health clinical notes. Your task is to determine whether each of these themes ({', '.join(themes)}) are referenced in the clinical note. For each theme, respond with the percentage chance for each theme being present. Do not identify any additional themes beyond the ones listed. Format the response as follows:"
            
        Themes: [Theme Name]
	    Percentage: [Numberical Percentage]
        
        Now, please analyze the following clinical note: {txt}

        Output:"""
        summaries.append(_summarize(input_text))
    return summaries


# COMMAND ----------

# MAGIC %md
# MAGIC ## Load data

# COMMAND ----------

# Load the clinical notes data
@ray.remote
def read_parquet(file_path):
    return pd.read_parquet(file_path)

parquet_file_path = "/data_source" 
pdf = ray.get(read_parquet.remote(parquet_file_path))

# COMMAND ----------

# Display the number of records in the dataset
print(len(pdf))
display(pdf)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Process Data

# COMMAND ----------

#Function to extract themes and their associated percentage from the LLM's output
import re
def extract_themes(text):
    marker = "Output:"
    text = text.replace("*", "")

    # Regular expression pattern to capture theme names and their percentage chances
    patterns = re.compile(r"Theme[:.]\s*([\w\s]+)\s*Percentage[:.]\s*(\d+)", re.IGNORECASE)

    themes_list = []
    if marker in text:
        themes_part = text.split(marker)[1].strip()
        matches = patterns.findall(themes_part)

        if matches:
            for theme, percentage in matches:
                theme = theme.strip().capitalize()
                percentage = float(percentage) / 100  # Convert percentage to a float between 0 and 1
                themes_list.append({'Theme': theme, 'Percentage': percentage})
        else:
            themes_list.append({'Theme': '', 'Percentage': 0.0})
    else:
        themes_list.append({'Theme': '', 'Percentage': 0.0})

    return themes_list

# COMMAND ----------

# Process the data and extract theme percentages for each note
batch_size = 5
final_results = []
chkpt = "meta-llama/Meta-Llama-3-8B-Instruct"
access_token = 'huggingface_access_token'

for i in range(0, len(pdf), batch_size):
    batch = pdf.iloc[i:i+batch_size][['patientid', 'reporttext']].to_dict(orient='records')
    txt_list = [entry['reporttext'] for entry in batch]
    summaries = ray.get(summarize_llm.remote(chkpt, access_token, txt_list))
    for entry, summary in zip(batch, summaries):
        themes = extract_themes(summary)
        for theme in themes:
            entry[f"theme_{theme['Theme']}"] = theme['Percentage']
        final_results.append(entry)

# COMMAND ----------

# Convert results to a DataFrame
result_df = pd.DataFrame(final_results)
print(result_df.head())

# COMMAND ----------

# Save results to a CSV file
result_df.to_csv("/dbfs/tmp/clinical_notes_themes.csv", index=False)

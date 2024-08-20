# Databricks notebook source
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
    collect_log_to_path="/dbfs/tmp/raylogs",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## LLM Prompts

# COMMAND ----------

#Function to interact with the LLM and extract themes from clinical notes
@ray.remote(num_gpus=1)
def summarize_llm(chkpt, access_token, txt_list):
  print(f"txt_list {txt_list}")
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  tokenizer = AutoTokenizer.from_pretrained(chkpt, trust_remote_code=True, use_auth_token=access_token)
  torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
  model = AutoModelForCausalLM.from_pretrained(
    chkpt, 
    torch_dtype=torch_dtype, 
    trust_remote_code=True, 
    #device_map="auto", 
    use_auth_token=access_token)\
  .to(device)

  def _summarize(txt):
        try:
            print(f"txt {txt}")
            input_text = txt
            with torch.no_grad():
                input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
                outputs = model.generate(input_ids, max_new_tokens=1024, pad_token_id=tokenizer.eos_token_id, num_beams=5, early_stopping=True, no_repeat_ngram_size=2)
            summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"summary {summary}")
            return summary
        except RuntimeError as e:
            print(e)
            return ""
          
  summaries = []
  for txt in txt_list:
    input_text = f"""You are an expert in medical text analysis. Your task is to identify up to 5 key themes that are present within gynecological clinical notes.

  Below are examples of gynecological clinical notes and the associated themes:

  Example 1:

  Clinical Note: "We present the case of a 42-year-old (gravida 2, abortion 2 (G2A2)) infertile woman who visited an infertility clinic with her husband (aged 42 years). She had been married for eight years and had two abortions during this time. After the second abortion, dilatation and curettage (D&C) was done at her hometown hospital. This is a case of secondary infertility. Before coming to our center, she visited another center for infertility treatment, where in vitro fertilization (IVF) was done using self-oocyte, but intracytoplasmic sperm injection (ICSI) was not done. Unfortunately, the treatment did not succeed. On hysteroscopy, adhesions were found in the cavity, which were possibly formed due to the D&C procedure performed for past abortions. This condition is known as Asherman’s syndrome. Hence, hystero-adhesiolysis was done on January 20, 2021. Her latest report showed an anti-Mullerian hormone (AMH) value of 0.252 ng/dL, which was very low. Low AMH was indicative of low ovarian reserve. Her follicle-stimulating hormone (FSH) was 23 mIU/mL, which was too high, suggestive of poor egg quality. The antral follicular count was assessed through transvaginal ultrasound; the total count was three (one on the right, and two on the left). Ovum pick-up was done on March 15, 2021."
  
  Themes:
  1. Infertile
  2. History of abortions
  3. Low egg quality
  4. Diminished ovarian reserve
  

  Example 2: 
  
  Clinical Note: "chief complaint (cc)/reason for visit:
  ms. smith is a 32 old female. she is here for 
  evaluation and treatment of medical conditions noted below.

    according to the history provided today, patient has following issues:
    1. veteran states that she has been having seizures. she states that
    she starts coughing, coughs so hard that she has bloody streaks in
    her phlegm, and then enters into a "fullblown full body" seizure.
    she states that witnesses have told her she shakes all over and goes
    in and out of consciousness. she says that she has been having these
    seizures about twice a month for a year and a half, but has never
    sought medical care for this. she states that these episodes last
    30-60 minutes each. she states that she has not been using any drugs,
    and that the seizures have not been in any proximal connection to
    drug usage.

    2. veteran states that she wants very much to get pregnant. her
    significant other has four children already, the youngest four years
    old, and has said that she is not willing to get pregnant again,
    so veteran wants to be the one to have a baby. she states that she
    has a source of semen but that she doesn't seem to be able to stay
    pregnant. she claims to have had seven miscarriages. she would like
    an appointment with gynecology to determine why she is having recurrent
    miscarriages.

  Themes:
  1. Seizure episodes reported
  2. Recurrent miscarriages reported

Now, please analyze the following gynecological clinical note and list up to 5 high level themes. Please note that the themes should not be too specific or too general. They should capture the main ideas or concepts in a concise manner. It is not necessary to include all themes mentioned in previous examples. You can identify new themes or combine similar themes to create a more comprehensive list. Themes should be less than 5 words and should follow this exact format:

  1. [Theme 1]
  2. [Theme 2]
  3. [Theme 3]
  4. [Theme 4]
  5. [Theme 5]

  Clinical Note: {txt}

  Output:"""
    summaries.append(_summarize(input_text))
  return summaries

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

# Load the clinical notes data
@ray.remote
def read_parquet(file_path):
    return pd.read_parquet(file_path)

parquet_file_path = "/dbfs/tmp/data/womenshealth_inf_rmc_reporttext_pairedsample" 
pdf = ray.get(read_parquet.remote(parquet_file_path))

# COMMAND ----------

# Display the number of records in the dataset
print(len(pdf))
display(pdf)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Process Data

# COMMAND ----------

# Process the dataset in batches
batch_size = 5
results = []
chkpt = "meta-llama/Meta-Llama-3-8B-Instruct"
access_token = 'huggingface_access_token'

def extract_themes(summary):
    marker = "Output:"
    if marker in summary:
        themes_part = summary.split(marker)[1].strip()
        themes = [theme.strip() for theme in themes_part.split("\n") if theme.strip()]
        themes_dict = {f"Theme_{i+1}": theme for i, theme in enumerate(themes) if i <= 5}
        return themes_dict
    else:
        return {}

final_results = []

for i in range(0, len(pdf), batch_size):
    batch = pdf.iloc[i:i+batch_size][['patientid', 'inf','rmc','reporttext']].to_dict(orient='records')
    txt_list = [entry['reporttext'] for entry in batch]
    summaries = ray.get(summarize_llm.remote(chkpt, access_token, txt_list))
    for entry, summary in zip(batch, summaries):
        themes = extract_themes(summary)
        entry.update(themes)
        final_results.append(entry)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Data

# COMMAND ----------

# Convert results to a DataFrame
result_df = pd.DataFrame(final_results)
print(result_df.head())

# COMMAND ----------

# Save results to a CSV file
result_df.to_csv("/dbfs/tmp/clinical_notes_finalthemes.csv", index=False)

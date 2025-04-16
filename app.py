import streamlit as st
import pandas as pd
import openai
from pydantic import BaseModel, Field, ValidationError
from typing import Literal, List
import json
import io
import traceback
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import sys
import bm25s
import os

if "stop_requested" not in st.session_state:
    st.session_state.stop_requested = False


# --- Configuration & Constants ---
APP_TITLE = "Autoreporter: Reporting Guideline Assessment"
GUIDELINE_FILE = "CONSORT_Guideline.csv"
MODEL_OPTIONS = [
    'gpt-4.1',
    'gpt-4.1-mini',
    'gpt-4.1-nano',
    'gpt-4o',
    'gpt-4o-mini',
    'o1-pro',
    'o1-mini',
    'o1',
    'o3-mini',
    'o3',
    'o4-mini',
    'gpt-4-turbo'
]
DEFAULT_MODEL = 'o3-mini'


instruction = """As a research methods scientist, your role is to evaluate the adherence of full-text scientific research articles
to established reporting guidelines. For each reporting item in the guideline, you will be assessing the research article
to confirm if the item has been wholly reported or not. Your assessment approach should be adaptable to any reporting guideline and research domain."""

zero_shot_prompt_template = """
### Full Text of Article

{input_text}

### Checklist Item

{item}

### Checklist Item Description

{item_description}

Based on the provided article text, checklist item name, and checklist item description, is the checklist item reported in the full text of this article?
Think step-by-step. Do not miss relevant information by being too focused on specific wordings and overinterpretation of the checklist item. Be inclusive, not overly critical, and evidence-based when assessing checklist item reporting.
"""

class Output(BaseModel):
    checklist: Literal[1, 0] = Field(
        description="Indicates if the checklist item was wholly reported (1) or not wholly reported (0)."
    )
    evidence: str = Field(
        description="Quoted evidence from the full text of the article that supports the checklist rating."
    )
    recommend: str = Field(
        description="If checklist item was not wholly reported (0), recommend how to revise the text to wholly report the checklist item."
    )

    class Config:
        extra = "forbid"  # This generates "additionalProperties": false in the schema

def api_call(api_key, query, model, instruction=instruction):
    try:
        client = openai.OpenAI(api_key=api_key)
        # Decide messages format based on model support
        if model in ["o1", "o1-mini", "o3", "o3-mini"]:
            full_prompt = instruction + "\n\n" + query
            messages = [{"role": "user", "content": full_prompt}]
            # These models don't support temperature, so omit it
            response = client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=Output
            )
        else:
            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": query},
            ]
            response = client.beta.chat.completions.parse(
                model=model,
                temperature=0.2,
                messages=messages,
                response_format=Output
            )

        message = response.choices[0].message
        token_count = response.usage.total_tokens if response.usage else "N/A"

        if message.parsed:
            return message.parsed.checklist, message.parsed.evidence, message.parsed.recommend, token_count
        else:
            return 'Error', 'Missing structured output', 'Error', token_count

    except openai.AuthenticationError:
        st.error("Authentication Error: Please check your OpenAI API key.")
        return 'Error', 'Authentication Error', 'Error', 'N/A'
    except openai.RateLimitError:
        st.error("Rate Limit Error: You have exceeded your OpenAI API quota.")
        return 'Error', 'Rate Limit Error', 'Error', 'N/A'
    except openai.APIConnectionError:
        st.error("API Connection Error: Could not connect to OpenAI.")
        return 'Error', 'API Connection Error', 'Error', 'N/A'
    except Exception as e:
        st.error(f"Unexpected error during API call: {e}")
        st.error(traceback.format_exc())
        return 'Error', f"Unexpected Error: {e}", 'Error', 'N/A'

contextual_rag_prompt = """
Given the document below, we want to explain what the chunk captures in the document.

{input_text}

Here is the chunk we want to explain:

{chunk_text}

Answer ONLY with a succinct explanation of the meaning of the chunk in the context of the whole document above.
"""

def generate_embeddings(text, model="text-embedding-3-large"):
    text = text.replace("\n", " ")
    client = openai.OpenAI(api_key=api_key)
    return client.embeddings.create(input = [text], model=model).data[0].embedding

def vector_retrieval(query: str, top_k: int = 5, vector_index: np.ndarray = None) -> List[int]:
    """
    Retrieve the top-k most similar items from an index based on a query.
    Args:
        query (str): The query string to search for.
        top_k (int, optional): The number of top similar items to retrieve. Defaults to 5.
        index (np.ndarray, optional): The index array containing embeddings to search against. Defaults to None.
    Returns:
        List[int]: A list of indices corresponding to the top-k most similar items in the index.
    """

    if vector_index is None or len(vector_index) == 0:
        raise ValueError("vector_index is empty. Ensure embeddings were generated.")

    query_embedding = generate_embeddings(query)
    similarity_scores = cosine_similarity([query_embedding], vector_index)
    return list(np.argsort(-similarity_scores)[0][:top_k])

def create_chunks(document, chunk_size=300, overlap=50):
    return [document[i : i + chunk_size] for i in range(0, len(document), chunk_size - overlap)]

def generate_prompts(input_text : str, chunk_text : List[str]) -> List[str]:
    prompts = []
    for chunk in chunk_text:
        prompt = contextual_rag_prompt.format(input_text=input_text, chunk_text=chunk)
        prompts.append(prompt)
    return prompts

def generate_context(query: str, instruction=instruction):
    """
    Generates a contextual response based on the given prompt using the specified language model.
    Args:
        prompt (str): The input prompt to generate a response for.
    Returns:
        str: The generated response content from the language model.
    """
    client = openai.OpenAI(api_key=api_key)
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": query},

        ])
    message = completion.choices[0].message.content

    return message

def bm25_retrieval(query: str, top_k : int , bm25_index, contextual_chunks) -> List[int]:
    """
    Retrieve the top-k document indices based on the BM25 algorithm for a given query.
    Args:
        query (str): The search query string.
        k (int): The number of top documents to retrieve.
        bm25_index: The BM25 index object used for retrieval.
    Returns:
        List[int]: A list of indices of the top-k documents that match the query.
    """

    results, scores = bm25_index.retrieve(bm25s.tokenize(query), k=top_k)

    return [contextual_chunks.index(doc) for doc in results[0]]

def reciprocal_rank_fusion(*list_of_list_ranks_system, K=60):
    """
    Fuse rank from multiple IR systems using Reciprocal Rank Fusion.

    Args:
    * list_of_list_ranks_system: Ranked results from different IR system.
    K (int): A constant used in the RRF formula (default is 60).

    Returns:
    Tuple of list of sorted documents by score and sorted documents
    """
    # Dictionary to store RRF mapping
    rrf_map = defaultdict(float)

    # Calculate RRF score for each result in each list
    for rank_list in list_of_list_ranks_system:
        for rank, item in enumerate(rank_list, 1):
            rrf_map[item] += 1 / (rank + K)

    # Sort items based on their RRF scores in descending order
    sorted_items = sorted(rrf_map.items(), key=lambda x: x[1], reverse=True)

    # Return tuple of list of sorted documents by score and sorted documents
    return sorted_items, [item for item, score in sorted_items]

def auto_reporter(api_key, input_text, checklist, retrieval='No RAG', model='gpt-4o-mini'):
    status_text = st.empty()
    log_output = st.empty()
    progress_bar = st.progress(0)

    results = []
    total_items = len(checklist)

    # Stop button setup
    if "stop_requested" not in st.session_state:
        st.session_state.stop_requested = False

    if st.button("Stop Adherence Assessment"):
        st.session_state.stop_requested = True

    # Retrieval setup if needed
    if retrieval in ['BM25S Retrieval', 'Contextual Retrieval', 'Hybrid Retrieval']:
        log_output.text("Creating document chunks...")
        chunks = create_chunks(input_text)

        log_output.text("Generating contextual prompts...")
        prompts = generate_prompts(input_text, chunks)

        log_output.text("Generating context from each chunk...")
        contextual_chunks = []

        for i, prompt in enumerate(prompts):
            log_output.text(f"Generating context for chunk {i+1}/{len(prompts)}...")
            context = generate_context(prompt)
            contextual_chunks.append(context)


        log_output.text("Generating embeddings for each chunk+context...")
        contextual_embeddings = []

        for i in range(len(chunks)):
            chunk_text = chunks[i]
            context_text = contextual_chunks[i]
            combined_text = str(context_text) + ' ' + str(chunk_text)

            log_output.text(f"Generating embedding for chunk {i+1}/{len(chunks)}...")
            embedding = generate_embeddings(combined_text)
            contextual_embeddings.append(embedding)


        contextual_chunks_edit = [
            f'Original Text: "{chunks[i]}" Context: "{contextual_chunks[i]}"'
            for i in range(len(chunks))
        ]

        log_output.text("Finished preparing retrieval index.")

    for i, row in enumerate(checklist.itertuples()):
        if st.session_state.stop_requested:
            status_text.warning("Assessment stopped by user.")
            log_output.text("Stopped before item: " + row.Item)
            return None

        item_section = row.Section
        item_subsection = row.Subsection
        item_name = row.Item
        item_desc = row.Description
        status_text.text(f"Processing item {i+1}/{total_items}: {item_name}")
        log_output.text(f"Preparing prompt for item {i+1}: {item_name}")
        query_string = str(item_name) + ' ' + str(item_desc)

        # Retrieval-based context selection
        if retrieval == 'No RAG':
            input_for_prompt = input_text


        elif retrieval == 'BM25S Retrieval':
            log_output.text(f"Retrieving top chunks using BM25S for: {item_name}")
            retriever = bm25s.BM25(corpus=contextual_chunks_edit)
            retriever.index(bm25s.tokenize(contextual_chunks_edit))
            top_k = bm25_retrieval(query_string, 10, retriever, contextual_chunks_edit)
            input_for_prompt = [contextual_chunks_edit[x] for x in top_k]

        elif retrieval == 'Contextual Retrieval':
            log_output.text(f"Retrieving top chunks using vector similarity for: {item_name}")
            top_k = vector_retrieval(query_string, 10, contextual_embeddings)
            input_for_prompt = [contextual_chunks[x] for x in top_k]

        elif retrieval == 'Hybrid Retrieval':
            log_output.text(f"Retrieving top chunks using hybrid method for: {item_name}")
            vec_top_k = vector_retrieval(query_string, 10, contextual_embeddings)
            retriever = bm25s.BM25(corpus=contextual_chunks_edit)
            retriever.index(bm25s.tokenize(contextual_chunks_edit))
            bm25_top_k = bm25_retrieval(query_string, 10, retriever, contextual_chunks_edit)
            hybrid_top_k = reciprocal_rank_fusion(vec_top_k, bm25_top_k)[1][:5]
            input_for_prompt = [contextual_chunks_edit[x] for x in hybrid_top_k]

        formatted_prompt = zero_shot_prompt_template.format(
            input_text=str(input_for_prompt),
            item=str(item_name),
            item_description=str(item_desc)
        )


        log_output.text(f"Calling {model} for item: {item_name}...")
        checklist_rating, evidence, recommend, _ = api_call(api_key, formatted_prompt, model, instruction)

        if checklist_rating == 'Error':
            status_text.error(f"Stopping assessment due to API error on item: {item_name}")
            log_output.text("API call returned an error.")
            return None

        log_output.text(f"Got results for: {item_name}")

        results.append({
            "Section": item_section,
            "Subsection": item_subsection,
            "Item": item_name,
            "Description": item_desc,
            "Reported (1/0)": checklist_rating,
            "Evidence": evidence,
            "Recommendation": recommend
        })

        with st.expander(f"✅ {item_name} (Item {i+1}/{total_items})"):
            st.markdown(f"**Section:** {item_section}")
            st.markdown(f"**Subsection:** {item_subsection}")
            st.markdown(f"**Checklist Description:** {item_desc}")
            st.markdown(f"**Reported:** {checklist_rating}")
            st.markdown(f"**Evidence:** {evidence}")
            st.markdown(f"**Recommendation:** {recommend}")

        progress_bar.progress((i + 1) / total_items)

    status_text.success(f"Assessment complete for {total_items} items!")
    log_output.text("Assessment complete. You may now view or download the results.")
    return pd.DataFrame(results).astype(str)

def convert_df_to_csv(df):
    output = io.StringIO()
    df.to_csv(output, index=False, encoding='utf-8')
    return output.getvalue().encode('utf-8')

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.markdown("Assess scientific article adherence to reporting guidelines using AI.")

st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("1. Enter OpenAI API Key", type="password")

guideline_dir = "Guidelines"
guideline_files = [f for f in os.listdir(guideline_dir) if f.endswith(".xlsx")]

# Create a mapping: display name (no .xlsx) -> actual filename
display_names = [f.replace(".xlsx", "") for f in guideline_files]
filename_map = dict(zip(display_names, guideline_files))

# Display filenames without extension
selected_display_name = st.sidebar.selectbox("2. Choose a guideline", display_names)
# Get the actual filename
selected_guideline_file = filename_map[selected_display_name]


# --- Load selected guideline ---
guideline_path = os.path.join(guideline_dir, selected_guideline_file)
guideline_df = pd.read_excel(guideline_path)

selected_model = st.sidebar.selectbox("3. Select OpenAI Model", MODEL_OPTIONS, index=MODEL_OPTIONS.index(DEFAULT_MODEL))
retrieval_option = st.sidebar.selectbox("4. Select Retrieval Strategy", ["No RAG", "BM25S Retrieval", "Contextual Retrieval", "Hybrid Retrieval"])


# --- Sidebar: Select Guideline File ---


st.header("Inputs")
article_text = st.text_area("Paste Full Text of Research Article Here", height=400, placeholder="Paste the complete text of the article...")


st.header("Assessment")
if st.button("Run Adherence Assessment"):
    if not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar.")
    elif not article_text:
        st.warning("Please paste the article text.")
    elif guideline_df is None:
        st.error("Guideline data could not be loaded. Check file and format.")
    else:
        with st.spinner("Running assessment... This may take a few minutes."):
            st.subheader("Processing Progress")
            results_df = auto_reporter(api_key, article_text, guideline_df, retrieval=retrieval_option, model=selected_model)

        if results_df is not None:
            st.subheader("Assessment Results")
            results_df = results_df.astype(str)
            st.dataframe(results_df)
            st.subheader("Download Results")
            csv_data = convert_df_to_csv(results_df)
            st.download_button(
                label="Download Results as CSV",
                data=csv_data,
                file_name=f"{selected_display_name}_assessment_results.csv",
                mime="text/csv",
            )
        else:
            st.error("Assessment could not be completed due to errors.")
else:
    st.info("Configure settings in the sidebar, paste article text, and click 'Run Adherence Assessment'.")

st.sidebar.markdown("---")
st.sidebar.caption("Ensure you have the necessary permissions and adhere to OpenAI's usage policies.")

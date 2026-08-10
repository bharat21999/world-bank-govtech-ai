import os
import json
import re
import time
import pandas as pd
import streamlit as st

from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config
from databricks.vector_search.client import VectorSearchClient
from databricks import sql
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

st.set_page_config(
    page_title="World Bank GovTech AI Assistant",
    page_icon="🌐",
    layout="wide"
)


# -----------------------------
# Config
# -----------------------------

VECTOR_SEARCH_ENDPOINT_NAME = os.getenv("VECTOR_SEARCH_ENDPOINT_NAME", "wb_ai_search_endpoint")
VECTOR_SEARCH_INDEX_NAME = os.getenv("VECTOR_SEARCH_INDEX_NAME", "worldbank_govtech.govtech.govtech_report_chunks_index")
LLM_ENDPOINT_NAME = os.getenv("LLM_ENDPOINT_NAME", "databricks-meta-llama-3-3-70b-instruct")

CATALOG = os.getenv("CATALOG", "worldbank_govtech")
SCHEMA = os.getenv("SCHEMA", "govtech")

GTMI_TABLE = f"{CATALOG}.{SCHEMA}.silver_gtmi_scores"

DATABRICKS_WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID")

# -----------------------------
# Databricks clients
# -----------------------------

@st.cache_resource
def get_workspace_client():
    return WorkspaceClient()


@st.cache_resource
def get_vector_index():
    cfg = Config()

    vsc = VectorSearchClient(
        workspace_url=cfg.host,
        service_principal_client_id=cfg.client_id,
        service_principal_client_secret=cfg.client_secret,
        disable_notice=True
    )

    return vsc.get_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
        index_name=VECTOR_SEARCH_INDEX_NAME
    )


def run_sql(query):
    if not DATABRICKS_WAREHOUSE_ID:
        raise ValueError("DATABRICKS_WAREHOUSE_ID is not set.")

    workspace_client = get_workspace_client()

    with sql.connect(
        server_hostname=workspace_client.config.host.replace("https://", ""),
        http_path=f"/sql/1.0/warehouses/{DATABRICKS_WAREHOUSE_ID}",
        credentials_provider=workspace_client.config.authenticate
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)

            # INSERT / UPDATE / DELETE queries do not return rows
            if cursor.description is None:
                return pd.DataFrame()

            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]

    return pd.DataFrame(rows, columns=columns)


def clean_sql_text(value):
    if value is None:
        return ""
    return str(value).replace("'", "''")


def log_interaction(question, route_info, answer, latency_seconds, error_message=None):
    try:
        route_info = route_info or {}

        question = clean_sql_text(question)
        answer = clean_sql_text(answer)
        error_message = clean_sql_text(error_message)
        route = clean_sql_text(route_info.get("route", ""))
        country = clean_sql_text(route_info.get("country", ""))
        component = clean_sql_text(route_info.get("component", ""))

        query = f"""
            INSERT INTO worldbank_govtech.govtech.monitoring_app_logs
            VALUES (
                current_timestamp(),
                '{question}',
                '{route}',
                '{country}',
                '{component}',
                '{answer}',
                {latency_seconds},
                '{error_message}'
            )
        """

        run_sql(query)

    except Exception as e:
        print(f"Monitoring log failed: {e}")

def call_llm(prompt):
    workspace_client = get_workspace_client()

    response = workspace_client.serving_endpoints.query(
        name=LLM_ENDPOINT_NAME,
        messages=[
            ChatMessage(
                role=ChatMessageRole.SYSTEM,
                content="You are a helpful AI assistant for the World Bank GovTech Maturity Index 2025."
            ),
            ChatMessage(
                role=ChatMessageRole.USER,
                content=prompt
            )
        ],
        max_tokens=1200,
        temperature=0.1
    )

    try:
        return response.choices[0].message.content
    except Exception:
        pass

    if isinstance(response, dict):
        choices = response.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            if isinstance(message, dict):
                return message.get("content", str(response))

        predictions = response.get("predictions", [])
        if predictions:
            return str(predictions[0])

    return str(response)

# -----------------------------
# Vector Search / Report RAG
# -----------------------------

def search_report(question, k=5):
    index = get_vector_index()

    return index.similarity_search(
        query_text=question,
        columns=["chunk_id", "source", "page_number", "content"],
        num_results=k
    )


def build_context_from_results(results):
    columns = [col["name"] for col in results["manifest"]["columns"]]
    rows = results["result"]["data_array"]

    context_parts = []

    for row in rows:
        item = dict(zip(columns, row))

        source = item.get("source", "Unknown source")
        page_number = item.get("page_number", "Unknown page")
        content = item.get("content", "")

        context_parts.append(
            f"Source: {source}\n"
            f"Page: {page_number}\n"
            f"Content: {content}"
        )

    return "\n\n---\n\n".join(context_parts)


def answer_from_report(question, k=5):
    results = search_report(question, k=k)
    context = build_context_from_results(results)

    prompt = f"""
You are an AI assistant for the World Bank GovTech Maturity Index 2025 report.

Answer the user question using only the report context below.

Rules:
- Use only the provided report context.
- Include page numbers when using report evidence.
- Do not make up facts.
- If the context is not enough, say that clearly.
- Keep the answer clear and concise.

User question:
{question}

Report context:
{context}

Answer:
"""

    return call_llm(prompt)


# -----------------------------
# Helpers
# -----------------------------

def dataframe_to_context(df, max_rows=20):
    if df is None or df.empty:
        return "No structured data available."

    df = df.head(max_rows)

    result = []
    result.append(f"Rows shown: {len(df)}")
    result.append("Columns: " + ", ".join(df.columns.tolist()))
    result.append("")
    result.append(df.to_string(index=False))

    return "\n".join(result)


# -----------------------------
# Structured SQL functions
# -----------------------------

def get_top_gtmi_countries(limit=10):
    query = f"""
        SELECT
            country,
            country_code,
            gtmi_group,
            ROUND(gtmi_score, 3) AS gtmi_score,
            ROUND(core_government_systems_score, 3) AS core_government_systems_score,
            ROUND(public_service_delivery_score, 3) AS public_service_delivery_score,
            ROUND(digital_citizen_engagement_score, 3) AS digital_citizen_engagement_score,
            ROUND(govtech_enablers_score, 3) AS govtech_enablers_score
        FROM {GTMI_TABLE}
        ORDER BY gtmi_score DESC
        LIMIT {limit}
    """

    return run_sql(query)


def get_low_gtmi_countries(limit=10):
    query = f"""
        SELECT
            country,
            country_code,
            gtmi_group,
            ROUND(gtmi_score, 3) AS gtmi_score,
            ROUND(core_government_systems_score, 3) AS core_government_systems_score,
            ROUND(public_service_delivery_score, 3) AS public_service_delivery_score,
            ROUND(digital_citizen_engagement_score, 3) AS digital_citizen_engagement_score,
            ROUND(govtech_enablers_score, 3) AS govtech_enablers_score
        FROM {GTMI_TABLE}
        ORDER BY gtmi_score ASC
        LIMIT {limit}
    """

    return run_sql(query)


def get_group_summary():
    query = f"""
        SELECT
            gtmi_group,
            COUNT(*) AS economy_count,
            ROUND(AVG(gtmi_score), 3) AS avg_gtmi_score,
            ROUND(AVG(core_government_systems_score), 3) AS avg_core_government_systems_score,
            ROUND(AVG(public_service_delivery_score), 3) AS avg_public_service_delivery_score,
            ROUND(AVG(digital_citizen_engagement_score), 3) AS avg_digital_citizen_engagement_score,
            ROUND(AVG(govtech_enablers_score), 3) AS avg_govtech_enablers_score
        FROM {GTMI_TABLE}
        GROUP BY gtmi_group
        ORDER BY gtmi_group
    """

    return run_sql(query)


def get_country_gtmi(country_name):
    safe_country = country_name.replace("'", "''")

    query = f"""
        SELECT
            country,
            country_code,
            gtmi_group,
            ROUND(gtmi_score, 3) AS gtmi_score,
            core_government_systems_group,
            ROUND(core_government_systems_score, 3) AS core_government_systems_score,
            public_service_delivery_group,
            ROUND(public_service_delivery_score, 3) AS public_service_delivery_score,
            digital_citizen_engagement_group,
            ROUND(digital_citizen_engagement_score, 3) AS digital_citizen_engagement_score,
            govtech_enablers_group,
            ROUND(govtech_enablers_score, 3) AS govtech_enablers_score
        FROM {GTMI_TABLE}
        WHERE LOWER(country) LIKE LOWER('%{safe_country}%')
        ORDER BY gtmi_score DESC
        LIMIT 10
    """

    return run_sql(query)


def get_component_leaders(component="gtmi", limit=10):
    component_map = {
        "gtmi": "gtmi_score",
        "core": "core_government_systems_score",
        "service": "public_service_delivery_score",
        "citizen": "digital_citizen_engagement_score",
        "enablers": "govtech_enablers_score"
    }

    score_col = component_map.get(component, "gtmi_score")

    query = f"""
        SELECT
            country,
            country_code,
            gtmi_group,
            ROUND({score_col}, 3) AS component_score
        FROM {GTMI_TABLE}
        ORDER BY {score_col} DESC
        LIMIT {limit}
    """

    return run_sql(query)


# -----------------------------
# Answer generation
# -----------------------------

def answer_with_structured_and_report(question, structured_df=None, structured_label="Structured data"):
    structured_context = dataframe_to_context(structured_df, max_rows=20)

    report_results = search_report(question, k=5)
    report_context = build_context_from_results(report_results)

    prompt = f"""
You are an AI assistant for a World Bank GovTech Maturity Index 2025 demo.

Use two sources:
1. Structured GovTech data for numbers, rankings, maturity groups, and country scores.
2. Retrieved report context for explanation, methodology, and interpretation.

User question:
{question}

{structured_label}:
{structured_context}

Report context:
{report_context}

Rules:
- Use structured data for numbers and rankings.
- Use report context for explanations.
- Include page numbers when using report evidence.
- Do not make up facts.
- Keep the answer clear and presentation-friendly.

Answer:
"""

    return call_llm(prompt)


# -----------------------------
# Router
# -----------------------------

def classify_question(question):
    prompt = f"""
You are a router for a World Bank GovTech Maturity Index 2025 AI assistant.

Classify the user question into one route.

Routes:
- top_gtmi_countries: top countries, highest score, leaders, best performers
- low_gtmi_countries: lowest score, weakest countries, least mature
- group_summary: Group A/B/C/D, maturity groups, distribution by group
- country_lookup: question about a specific country
- component_leaders: leaders in a specific component
- report_only: definitions, methodology, key findings, explanation from report
- combined_general: use when unsure

Components:
- gtmi
- core
- service
- citizen
- enablers

Component rules:
- core government systems, CGSI, cloud, interoperability -> core
- public service delivery, PSDI, online services -> service
- citizen engagement, DCEI, open data, feedback -> citizen
- enablers, GTEI, strategy, laws, institutions, digital skills -> enablers
- otherwise -> gtmi

Return only JSON:
{{
  "route": "route_name",
  "country": "country_name_or_null",
  "component": "component_name",
  "reason": "brief reason"
}}

User question:
{question}
"""

    raw_answer = call_llm(prompt)

    match = re.search(r"\{.*\}", raw_answer, re.DOTALL)

    if match:
        try:
            parsed = json.loads(match.group(0))

            return {
                "route": parsed.get("route", "combined_general"),
                "country": parsed.get("country", None),
                "component": parsed.get("component", "gtmi"),
                "reason": parsed.get("reason", "")
            }

        except Exception:
            pass

    return {
        "route": "combined_general",
        "country": None,
        "component": "gtmi",
        "reason": "Could not parse router output."
    }


def ask_govtech_ai(question):
    route_info = classify_question(question)

    route = route_info["route"]
    country = route_info["country"]
    component = route_info["component"]

    if route == "top_gtmi_countries":
        structured_df = get_top_gtmi_countries(limit=10)

        return answer_with_structured_and_report(
            question=question,
            structured_df=structured_df,
            structured_label="Top GovTech countries"
        ), structured_df, route_info

    if route == "low_gtmi_countries":
        structured_df = get_low_gtmi_countries(limit=10)

        return answer_with_structured_and_report(
            question=question,
            structured_df=structured_df,
            structured_label="Lowest GovTech countries"
        ), structured_df, route_info

    if route == "group_summary":
        structured_df = get_group_summary()

        return answer_with_structured_and_report(
            question=question,
            structured_df=structured_df,
            structured_label="GTMI group summary"
        ), structured_df, route_info

    if route == "country_lookup" and country is not None:
        structured_df = get_country_gtmi(country)

        return answer_with_structured_and_report(
            question=question,
            structured_df=structured_df,
            structured_label=f"GovTech score for {country}"
        ), structured_df, route_info

    if route == "component_leaders":
        structured_df = get_component_leaders(
            component=component,
            limit=10
        )

        return answer_with_structured_and_report(
            question=question,
            structured_df=structured_df,
            structured_label=f"Top countries for component: {component}"
        ), structured_df, route_info

    if route == "report_only":
        answer = answer_from_report(question, k=5)
        return answer, None, route_info

    top_df = get_top_gtmi_countries(limit=5)
    group_df = get_group_summary()

    combined_df = pd.concat(
        [
            top_df.assign(section="Top GovTech countries"),
            group_df.assign(section="GTMI group summary")
        ],
        ignore_index=True,
        sort=False
    )

    return answer_with_structured_and_report(
        question=question,
        structured_df=combined_df,
        structured_label="General GovTech structured context"
    ), combined_df, route_info


# -----------------------------
# Streamlit UI
# -----------------------------

st.title("🌐 World Bank GovTech Maturity AI Assistant")

st.markdown(
    """
Ask questions about the **World Bank GovTech Maturity Index 2025**.

This app combines:
- structured GTMI country scores
- GovTech maturity groups
- component scores
- retrieved report context from the 2025 GovTech report
"""
)

with st.sidebar:
    st.header("Configuration")

    st.write("Vector Search Endpoint:")
    st.code(VECTOR_SEARCH_ENDPOINT_NAME)

    st.write("Vector Search Index:")
    st.code(VECTOR_SEARCH_INDEX_NAME)

    st.write("GTMI Table:")
    st.code(GTMI_TABLE)

    st.write("LLM Endpoint:")
    st.code(LLM_ENDPOINT_NAME)

    st.divider()

    st.header("Example questions")

    example_questions = [
        "Which countries have the highest GovTech score?",
        "What is India's GovTech maturity score?",
        "Which countries lead in digital citizen engagement?",
        "How many economies are in each GovTech maturity group?",
        "What are the main findings from the GovTech 2025 report?",
        "What are the four components of the GovTech Maturity Index?"
    ]

    selected_example = st.selectbox(
        "Choose an example",
        [""] + example_questions
    )


if "messages" not in st.session_state:
    st.session_state.messages = []


question = st.chat_input("Ask a question about GovTech Maturity Index 2025")

if selected_example and not question:
    question = selected_example


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            start_time = time.time()

            try:
                answer, structured_df, route_info = ask_govtech_ai(question)

                latency_seconds = round(time.time() - start_time, 3)

                
                log_interaction(
                    question=question,
                    route_info=route_info,
                    answer=answer,
                    latency_seconds=latency_seconds,
                    error_message=None
                )
                  

                st.markdown(answer)

                with st.expander("Router decision"):
                    st.json(route_info)

                if structured_df is not None and not structured_df.empty:
                    with st.expander("Structured data used"):
                        st.dataframe(structured_df)

                st.caption(f"Latency: {latency_seconds} seconds")

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            except Exception as e:
                import traceback

                latency_seconds = round(time.time() - start_time, 3)
                error_message = f"Something went wrong: {type(e).__name__}: {e}"
                stack_trace = traceback.format_exc()

                try:
                    log_interaction(
                        question=question,
                        route_info={},
                        answer="",
                        latency_seconds=latency_seconds,
                        error_message=error_message
                    )
                except Exception:
                    pass

                st.error(error_message)
                st.exception(e)

                with st.expander("Full traceback"):
                    st.code(stack_trace)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message
                    }
                )
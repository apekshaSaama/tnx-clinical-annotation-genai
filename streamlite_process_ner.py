import io
import json
import os
import random
import re
import smtplib
import subprocess
import sys
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Final

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

ALLOWED_DOMAINS: Final[tuple[str, ...]] = ("@saama.com", "@trinetx.com")


def normalize_email(email: str) -> str:
    """Normalize email by stripping whitespace and converting to lowercase."""
    return (email or "").strip().lower()


def normalize_auth_config(config: dict[str, str]) -> dict[str, str]:
    """Normalize Gmail authentication configuration."""
    return {
        "gmail_address": str(config.get("gmail_address", "") or "").strip(),
        "gmail_password": str(config.get("gmail_password", "") or "").strip().replace(" ", ""),
        "smtp_server": str(config.get("smtp_server", "smtp.gmail.com") or "smtp.gmail.com").strip(),
        "smtp_port": str(config.get("smtp_port", "587") or "587").strip(),
    }


def is_allowed_email(email: str) -> bool:
    """Check if email is from an allowed domain."""
    normalized = normalize_email(email)
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", normalized):
        return False
    return normalized.endswith(ALLOWED_DOMAINS)


def generate_otp(length: int = 6) -> str:
    """Generate a random OTP code."""
    return "".join(str(random.randint(0, 9)) for _ in range(length))


DEFAULT_NOTES_FOLDER = ROOT / "notes" / "consultation"
DEFAULT_GUIDELINE_PATH = ROOT / "annotation_guideline.md"
DEFAULT_OUTPUT_FOLDER = ROOT / "azure"
AUTH_CONFIG_PATH = ROOT / "config" / "auth_config.json"

MODEL_OPTIONS = {
    "OpenAI (Azure)": "openai",
    "Claude (Anthropic)": "claude",
    "Gemini (Google)": "gemini",
}

st.set_page_config(page_title="Clinical NER Pipeline", layout="wide")

if "annotation_complete" not in st.session_state:
    st.session_state.annotation_complete = False
    st.session_state.annotation_output_folder = ""
    st.session_state.gt_path = ""
    st.session_state.iaa_report_text = ""
    st.session_state.iaa_csv_text = ""
    st.session_state.model_name = ""
    st.session_state.show_iaa = False

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.auth_email = ""
    st.session_state.auth_step = "email"
    st.session_state.otp_value = ""
    st.session_state.auth_message = ""
    st.session_state.auth_error = ""


def load_auth_config() -> dict[str, str]:
    if AUTH_CONFIG_PATH.exists():
        with open(AUTH_CONFIG_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
            if isinstance(data, dict):
                return normalize_auth_config({str(key): str(value) for key, value in data.items()})

    return normalize_auth_config(
        {
            "gmail_address": os.getenv("GMAIL_ADDRESS", "your.account@gmail.com"),
            "gmail_password": os.getenv("GMAIL_PASSWORD", "your-app-password"),
            "smtp_server": os.getenv("SMTP_SERVER", "smtp.gmail.com"),
            "smtp_port": os.getenv("SMTP_PORT", "587"),
        }
    )


def send_otp_email(recipient_email: str, otp_code: str) -> tuple[bool, str]:
    config = load_auth_config()
    username = config.get("gmail_address", "")
    password = config.get("gmail_password", "")
    smtp_server = config.get("smtp_server", "smtp.gmail.com")
    smtp_port = int(config.get("smtp_port", "587"))

    if not username or not password or "your." in username or "your-app" in password:
        return False, "SMTP credentials are not configured yet. Please update config/auth_config.json or environment variables."

    try:
        message = EmailMessage()
        message["Subject"] = "Clinical NER login verification"
        message["From"] = username
        message["To"] = recipient_email
        message.set_content(
            f"Your verification code is {otp_code}. Enter it to continue to the Clinical NER portal."
        )

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(username, password)
            server.send_message(message)
        return True, ""
    except Exception as exc:  # pragma: no cover - network-dependent path
        return False, f"Unable to send OTP email: {exc}"


def render_login_page() -> None:
    # 1. Page Styling & Custom Form Card Styling
    st.markdown(
        """
        <style>
            /* Prevents content from overlapping top right buttons */
            .block-container {
                padding-top: 3rem !important;
                padding-bottom: 0rem !important;
            }

            /* Container styling: Turns Streamlit form into a modern white card */
            div[data-testid="stForm"] {
                background-color: #ffffff;
                border: 1px solid #eaecf0;
                border-radius: 12px;
                padding: 2rem 2.5rem;
                box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.05);
            }

            /* Title & Subtitle Styling inside the box */
            .login-title {
                text-align: center;
                color: #003d99;
                font-size: 2rem;
                font-weight: 700;
                margin-bottom: 0.25rem;
            }
            .login-subtitle {
                text-align: center;
                color: #525252;
                margin-bottom: 1.5rem;
            }

            /* Primary Action Button Styling */
            div[data-testid="stForm"] button[kind="secondaryFormSubmit"],
            div[data-testid="stForm"] button[kind="primaryFormSubmit"] {
                background-color: #3b82f6 !important;
                color: #ffffff !important;
                border-radius: 8px !important;
                border: none !important;
                height: 44px;
                font-weight: 600;
                font-size: 0.95rem;
                margin-top: 0.5rem;
            }

            /* Primary Button Hover */
            div[data-testid="stForm"] button[kind="secondaryFormSubmit"]:hover,
            div[data-testid="stForm"] button[kind="primaryFormSubmit"]:hover {
                background-color: #2563eb !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # 2. Top Navigation Layout (Logo Section)
    col1, col2, col3 = st.columns([1, 2.5, 1])

    with col1:
        logo_path = ROOT / "saama_logo.svg"
        if logo_path.is_file():
            st.image(str(logo_path), width=160)

    # 3. Main Center Layout Column Grid for Card Alignment
    center_col1, center_col2, center_col3 = st.columns([1, 1.3, 1])

    with center_col2:
        # Step 1: Email Form
        if st.session_state.auth_step == "email":
            with st.form("login_form"):
                # Header inside the card box
                st.markdown(
                    """
                    <div class='login-title'>Clinical Pre-Annonation Process</div>
                    <div class='login-subtitle'>Sign in with a business email from @saama or @trinetx</div>
                    """,
                    unsafe_allow_html=True,
                )

                # Status Messages
                if st.session_state.auth_error:
                    st.error(st.session_state.auth_error)

                if st.session_state.auth_message:
                    st.success(st.session_state.auth_message)

                # Form Inputs
                email = st.text_input(
                    "Email address",
                    value=st.session_state.auth_email,
                    placeholder="Enter your email address",
                )
                submitted = st.form_submit_button(
                    "Send verification code", use_container_width=True
                )

            if submitted:
                normalized_email = normalize_email(email)
                if not normalized_email:
                    st.session_state.auth_error = "Please enter your email address."
                    st.session_state.auth_message = ""
                elif not is_allowed_email(normalized_email):
                    st.session_state.auth_error = (
                        "Access is restricted to @saama.com or @trinetx.com email addresses."
                    )
                    st.session_state.auth_message = ""
                else:
                    otp_code = generate_otp()
                    st.session_state.auth_email = normalized_email
                    st.session_state.otp_value = otp_code
                    st.session_state.auth_step = "otp"
                    st.session_state.auth_error = ""
                    sent, message = send_otp_email(normalized_email, otp_code)
                    if sent:
                        st.session_state.auth_message = (
                            f"A verification code was sent to {normalized_email}."
                        )
                    else:
                        st.session_state.auth_message = (
                            f"The email could not be sent automatically. Your verification code is {otp_code}."
                        )
                        if message:
                            st.session_state.auth_error = message
                    st.rerun()

        # Step 2: OTP Form
        else:
            with st.form("otp_form"):
                st.markdown(
                    """
                    <div class='login-title'>Clinical Pre-Annonation Process</div>
                    <div class='login-subtitle'>Enter the verification code sent to your email</div>
                    """,
                    unsafe_allow_html=True,
                )

                if st.session_state.auth_error:
                    st.error(st.session_state.auth_error)

                if st.session_state.auth_message:
                    st.success(st.session_state.auth_message)

                otp_code = st.text_input("Verification code", type="password")
                submitted = st.form_submit_button(
                    "Verify code", use_container_width=True
                )

            if submitted:
                if otp_code == st.session_state.otp_value:
                    st.session_state.authenticated = True
                    st.session_state.auth_error = ""
                    st.session_state.auth_message = ""
                    st.session_state.auth_step = "done"
                    st.rerun()
                else:
                    st.session_state.auth_error = (
                        "Invalid verification code. Please try again."
                    )
                    st.session_state.auth_message = ""


def load_json_file(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def get_documents(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    raise ValueError("Expected a JSON object or a JSON array.")


def extract_entities(doc: dict[str, Any]) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    predictions = doc.get("predictions") or []
    for prediction in predictions:
        for item in prediction.get("result") or []:
            if item.get("type") != "labels":
                continue
            value = item.get("value") or {}
            start = value.get("start")
            end = value.get("end")
            text = value.get("text") or ""
            labels = value.get("labels") or []
            if isinstance(start, int) and isinstance(end, int):
                entities.append(
                    {
                        "id": item.get("id"),
                        "text": text,
                        "start": start,
                        "end": end,
                        "labels": labels,
                        "label": labels[0] if labels else "UNKNOWN",
                    }
                )
    return sorted(entities, key=lambda entry: entry["start"])


def extract_relations(doc: dict[str, Any], entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entity_map = {entity["id"]: entity for entity in entities if entity.get("id")}
    relations: list[dict[str, Any]] = []
    predictions = doc.get("predictions") or []
    for prediction in predictions:
        for item in prediction.get("result") or []:
            if item.get("type") != "relation":
                continue
            labels = item.get("labels") or []
            relations.append(
                {
                    "id": item.get("id"),
                    "source": entity_map.get(item.get("from_id")),
                    "target": entity_map.get(item.get("to_id")),
                    "labels": labels,
                    "label": labels[0] if labels else "UNKNOWN",
                    "score": item.get("score"),
                }
            )
    return relations


if not st.session_state.authenticated:
    st.markdown(
        """
        <style>
        [data-testid="stTextInput"] label,
        [data-testid="stSelectbox"] label {
            color: var(--e-global-color-cf16703, #003d99);
            font-weight: 600;
        }
        
        [data-testid="stTextInput"] input,
        [data-testid="stSelectbox"] div[role="combobox"],
        [data-testid="stSelectbox"] div[data-baseweb="select"],
        .stSelectbox div[data-baseweb="select"] > div {
            border-color: var(--e-global-color-cf16703, #003d99) !important;
            box-shadow: 0 0 0 1px var(--e-global-color-cf16703, #003d99) inset !important;
        }
        
        [data-testid="stTextInput"] input:focus,
        [data-testid="stSelectbox"] div[role="combobox"]:focus,
        [data-testid="stSelectbox"] div[data-baseweb="select"]:focus-within,
        .stSelectbox div[data-baseweb="select"] > div:focus-within,
        .stSelectbox div[data-baseweb="select"] > div:active,
        .stSelectbox div[data-baseweb="select"] > div:focus {
            outline: 2px solid rgba(0, 61, 153, 0.25) !important;
            border-color: var(--e-global-color-cf16703, #003d99) !important;
            box-shadow: 0 0 0 1px var(--e-global-color-cf16703, #003d99) inset !important;
        }
        
        div[data-testid="stFormSubmitButton"] {
            width: 100% !important;
            display: block !important;
            text-align: center !important;
            margin-top: 1rem !important;
        }

        div[data-testid="stFormSubmitButton"] > button {
            background-color: #003d99 !important; 
            color: #ffffff !important;
            border-color: #003d99 !important;
            
            padding: 0.2rem 1.5rem !important; 
            font-size: 0.95rem !important;
            
            max-width: 150px !important; 
            margin: 0 auto !important; 
            display: block !important;
            
            border-radius: 8px !important;
            font-weight: 700 !important;
            box-shadow: none !important;
        }
        
        div[data-testid="stFormSubmitButton"] > button:hover {
            background-color: #002966 !important; 
            border-color: #002966 !important;
            color: #ffffff !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    render_login_page()
    st.stop()

col1, col2, col3 = st.columns([1.5, 2, 1])
with col1:
    logo_path = ROOT / "saama_logo.svg"
    if logo_path.is_file():
        st.image(str(logo_path), width=160)

st.markdown(
    """
    <style>
    .page-title {
        color: var(--e-global-color-cf16703, #003d99);
        font-size: 3rem;
        font-weight: 700;
        line-height: 1.05;
        margin-bottom: 0.25rem;
        text-align: center;
        display: block;
        width: 100%;
        margin-left: auto;
        margin-right: auto;
    }
    .page-subtitle {
        font-size: 1.05rem;
        color: #525252;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    [data-testid="stTextInput"] label,
    [data-testid="stSelectbox"] label {
        color: var(--e-global-color-cf16703, #003d99);
        font-weight: 600;
    }
    
    /* 1. Apply standard custom blue border line shadow */
    [data-testid="stTextInput"] input,
    [data-testid="stSelectbox"] div[role="combobox"],
    [data-testid="stSelectbox"] div[data-baseweb="select"],
    .stSelectbox div[data-baseweb="select"] > div {
        border-color: var(--e-global-color-cf16703, #003d99) !important;
        box-shadow: 0 0 0 1px var(--e-global-color-cf16703, #003d99) inset !important;
    }
    
    /* 2. OVERRIDE: Eliminates the native orange/red highlight ring on focus completely */
    [data-testid="stTextInput"] input:focus,
    [data-testid="stSelectbox"] div[role="combobox"]:focus,
    [data-testid="stSelectbox"] div[data-baseweb="select"]:focus-within,
    .stSelectbox div[data-baseweb="select"] > div:focus-within,
    .stSelectbox div[data-baseweb="select"] > div:active,
    .stSelectbox div[data-baseweb="select"] > div:focus {
        outline: 2px solid rgba(0, 61, 153, 0.25) !important;
        border-color: var(--e-global-color-cf16703, #003d99) !important;
        box-shadow: 0 0 0 1px var(--e-global-color-cf16703, #003d99) inset !important;
    }
    
    /* 3. Ensure parent submit button container spans full width */
    div[data-testid="stFormSubmitButton"] {
        width: 100% !important;
        display: block !important;
        text-align: center !important;
        margin-top: 1rem !important;
    }

    /* 4. Center button element */
    div[data-testid="stFormSubmitButton"] > button {
        background-color: #003d99 !important; 
        color: #ffffff !important;
        border-color: #003d99 !important;
        
        padding: 0.2rem 1.5rem !important; 
        font-size: 0.95rem !important;
        
        max-width: 150px !important; 
        margin: 0 auto !important; 
        display: block !important;
        
        border-radius: 8px !important;
        font-weight: 700 !important;
        box-shadow: none !important;
    }
    
    div[data-testid="stFormSubmitButton"] > button:hover {
        background-color: #002966 !important; 
        border-color: #002966 !important;
        color: #ffffff !important;
    }

    /* Universal widget label blue color */
    div[data-testid="stWidgetLabel"] *,
    div[data-testid="stWidgetLabel"] p,
    div[data-testid="stWidgetLabel"] label,
    div[data-testid="stWidgetLabel"] span {
        color: #003d99 !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }

    /* Specific override for st.file_uploader labels */
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] label *,
    [data-testid="stFileUploader"] [data-testid="stWidgetLabel"] *,
    [data-testid="stFileUploaderDropzoneInstructions"] span {
        color: #003d99 !important;
        font-weight: 600 !important;
    }

    /* 1. Hide ONLY the text label inside the directory uploader button (keeps SVG icon) */
    div[data-testid="stFileUploader"]:has(input[directory]) section button span[data-testid="stHeaderActionElements"] ~ span,
    div[data-testid="stFileUploader"]:has(input[webkitdirectory]) section button span[data-testid="stHeaderActionElements"] ~ span,
    div[data-testid="stFileUploader"]:has(input[directory]) section button p,
    div[data-testid="stFileUploader"]:has(input[webkitdirectory]) section button p {
        display: none !important;
    }

    /* 2. Append 'Upload notes' next to the icon */
    div[data-testid="stFileUploader"]:has(input[directory]) section button::after,
    div[data-testid="stFileUploader"]:has(input[webkitdirectory]) section button::after {
        content: "Upload notes" !important;
        font-size: 0.875rem !important;
        color: #31333F !important;
        font-weight: 500 !important;
        margin-left: 0.35rem !important;
    }
    </style>

    <div class="page-title">Clinical Pre-Annonation Process</div>
    """,
    unsafe_allow_html=True,
)

# Initialize Session State
if "annotation_complete" not in st.session_state:
    st.session_state.annotation_complete = False
if "show_report" not in st.session_state:
    st.session_state.show_report = False

# --- SECTION 1: ANNOTATION FORM ---
with st.form("annotation_form"):
    uploaded_notes = st.file_uploader(
        "Upload Notes",
        type=["txt"],
        accept_multiple_files="directory",
    )

    uploaded_guideline = st.file_uploader(
        "Upload Guideline file",
        type=["md", "txt", "pdf", "docx"],
    )

    model_label = st.selectbox("Model", options=list(MODEL_OPTIONS.keys()))

    submitted = st.form_submit_button("Annotate", width="stretch")

if submitted:
    if not uploaded_notes:
        st.error("Please select or drop a folder containing .txt note files.")
        st.stop()

    if not uploaded_guideline:
        st.error("Please upload a guideline file.")
        st.stop()

    model_name = MODEL_OPTIONS[model_label]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Setup project-local temp processing folders
    temp_dir = ROOT / f".temp_processing_{timestamp}"
    notes_folder = temp_dir / "notes"
    output_folder = temp_dir / "output"

    notes_folder.mkdir(parents=True, exist_ok=True)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Save uploaded guideline file to disk
    guideline_file = temp_dir / uploaded_guideline.name
    with open(guideline_file, "wb") as f:
        f.write(uploaded_guideline.getbuffer())

    # A new guideline may change how prompts should render, so publish the
    # bundled prompts/*.txt to Langfuse (no-op if Langfuse isn't configured).
    sync_completed = subprocess.run(
        [sys.executable, "-m", "llm.sync_prompts"],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    if sync_completed.returncode != 0:
        st.warning(f"Prompt sync skipped: {sync_completed.stderr.strip() or sync_completed.stdout.strip()}")


    # Save uploaded note files to disk
    for uploaded_note in uploaded_notes:
        file_name = Path(uploaded_note.name).name
        note_file_path = notes_folder / file_name
        with open(note_file_path, "wb") as f:
            f.write(uploaded_note.getbuffer())

    # Run extraction subprocess
    with st.spinner("Running clinical NER extraction..."):
        cmd = [
            sys.executable,
            str(ROOT / "clinical_ner_extractor.py"),
            "--input_folder",
            str(notes_folder),
            "--guideline",
            str(guideline_file),
            "--output_folder",
            str(output_folder),
            "--model",
            model_name,
        ]
        completed = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))

        log_file = output_folder / "annotation_run.log"
        with open(log_file, "w", encoding="utf-8") as handle:
            if completed.stdout:
                handle.write("STDOUT:\n" + completed.stdout + "\n")
            if completed.stderr:
                handle.write("STDERR:\n" + completed.stderr + "\n")

    if completed.returncode != 0:
        st.error("Annotation failed.")
        st.info(f"See log: {log_file}")
        st.code(completed.stderr or completed.stdout, language="text")
    else:
        st.session_state.annotation_complete = True
        st.session_state.annotation_output_folder = str(output_folder)
        st.session_state.model_name = model_name
        st.success("Annotation completed")

# --- SECTION 2: IAA CALCULATION FORM ---
if st.session_state.annotation_complete:
    output_folder = Path(st.session_state.annotation_output_folder)
    st.markdown("---")
    st.subheader("Calculate IAA")

    with st.form("gt_path_form"):
        uploaded_gt = st.file_uploader(
            "Gold Standard Document for IAA Calculation",
            type=["json"],
        )
        submit_iaa = st.form_submit_button("Calculate IAA", width="stretch")

    if submit_iaa:
        st.session_state.show_report = False

        if not uploaded_gt:
            st.error("Please upload a Gold Standard Document for IAA Calculation.")
        else:
            # Save uploaded GT JSON file to disk
            gt_file = output_folder / uploaded_gt.name
            with open(gt_file, "wb") as f:
                f.write(uploaded_gt.getbuffer())

            st.session_state.gt_path = str(gt_file)

            # 1. PRIMARY SEARCH: Strictly look for *_updated.json first
            candidates = [
                path
                for path in output_folder.glob("*.json")
                if path.is_file() and path.name.endswith("_updated.json")
            ]

            # 2. FALLBACK SEARCH: If *_updated.json wasn't created on the cloud, grab the regular .json
            if not candidates:
                candidates = [
                    path
                    for path in output_folder.glob("*.json")
                    if path.is_file() and path.name != gt_file.name
                ]
                if candidates:
                    st.warning(f"⚠️ '_updated.json' was not found, so we are using '{candidates[0].name}' instead.")

            # --- DEBUGGER: Prints exactly what files exist in the cloud folder right now ---
            all_files_in_folder = [p.name for p in output_folder.glob("*")]
            # st.info(f"📁 Files currently inside the cloud output folder: {all_files_in_folder}")
            # ---------------------------------------------------------------------------------

            # --- THE FIX: Initialize pred_file to None first ---
            pred_file = None

            if not candidates:
                st.error("Could not find any prediction JSON files in the output folder.")
            elif len(candidates) > 1:
                st.error(
                    "Found multiple prediction files; please ensure only one is generated:\n"
                    + "\n".join(f"- {path.name}" for path in candidates)
                )
            else:
                pred_file = candidates[0]

            # Execute IAA generation script ONLY if pred_file was successfully assigned
            if pred_file is not None:
                model = st.session_state.model_name
                iaa_report_file = output_folder / f"iaa_chunks_evaluation_{model}.csv"
                csv_file = output_folder / f"iaa_metric_{model}.csv"
                
                cmd = [
                    sys.executable,
                    str(ROOT / "generate_iaa.py"),
                    "--gt", str(gt_file),
                    "--pred", str(pred_file),
                    "--iaa-report", str(iaa_report_file),
                    "--csv", str(csv_file),
                ]
                with st.spinner("Calculating IAA Score..."):
                    completed = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))

                if completed.returncode == 0:
                    st.success("IAA report generated successfully!")
                    st.session_state.iaa_report_file = iaa_report_file
                    st.session_state.csv_file = csv_file
                    st.session_state.completed_stdout = completed.stdout
                else:
                    error_log = output_folder / f"iaa_generation_error_{gt_file.stem}.log"
                    with open(error_log, "w", encoding="utf-8") as handle:
                        if completed.stderr:
                            handle.write("STDERR:\n" + completed.stderr + "\n")
                        if completed.stdout:
                            handle.write("STDOUT:\n" + completed.stdout + "\n")
                    st.error("IAA generation failed. Check the error log for details.")
                    st.info(f"Error log: {error_log}")

    if "iaa_report_file" in st.session_state:
        if not st.session_state.show_report:
            if st.button("📊 View Score", width="stretch"):
                st.session_state.show_report = True
                st.rerun()

        if st.session_state.show_report:
            st.markdown("---")
            if st.button("🙈 Hide Score"):
                st.session_state.show_report = False
                st.rerun()
                
            # if st.session_state.completed_stdout:
            #     st.info("Execution Summary Output:")
            #     st.code(st.session_state.completed_stdout, language='text')

            if st.session_state.completed_stdout:
                st.info("Execution Summary Output:")

                # Regex search to find the pattern inside parenthesis: e.g., "(6/6 chunks agree, 100.00%)"
                match = re.search(r"\(\d+/\d+\s+chunks agree,\s*[\d\.]+%\)", st.session_state.completed_stdout)

                if match:
                    # Display extracted summary text
                    st.markdown(f"**IAA Summary:** `{match.group(0)}`")
                else:
                    # Fallback if the pattern isn't found in stdout
                    st.code(st.session_state.completed_stdout, language="text")

            if st.session_state.iaa_report_file.exists():
                report_text = st.session_state.iaa_report_file.read_text(encoding='utf-8')
                st.subheader('IAA Report Data')
                
                try:
                    df = pd.read_csv(io.StringIO(report_text))
                    st.dataframe(df, width="stretch") 
                except Exception as e:
                    st.error(f"Could not parse text as a table: {e}")
                    st.code(report_text, language='text')
import streamlit as st
import json
import os
import base64
import google.generativeai as genai
import io
import traceback
import re
import time

# --- Importing the Feature Modules ---
from portfolio_playground import render_playground
from algo_trading_playground import render_algo_playground
from stock_screener import render_screener
from fin_summarizer import render_summarizer

# --- Page Configuration ---
st.set_page_config(
    page_title="Polyglot Quant AI",
    page_icon="https://www.google.com/s2/favicons?domain=google.com&sz=128",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Gemini API Configuration ---
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=API_KEY)
    GEMINI_AVAILABLE = True
except (KeyError, AttributeError):
    API_KEY = None
    GEMINI_AVAILABLE = False
    st.sidebar.warning("Gemini API key not found. Translation & RAG features are disabled.", icon="⚠️")


# --- Constants ---
CONTENT_DIR = "learning_content"
NUM_MODULES = 6
LANGUAGES = {
    "English": "en", "Assamese": "as", "Bengali": "bn", "Bodo": "brx", "Dogri": "doi",
    "Gujarati": "gu", "Hindi": "hi", "Kannada": "kn", "Kashmiri": "ks", "Konkani": "gom",
    "Maithili": "mai", "Malayalam": "ml", "Manipuri (Meitei)": "mni", "Marathi": "mr",
    "Nepali": "ne", "Odia": "or", "Punjabi": "pa", "Sanskrit": "sa", "Santhali": "sat",
    "Sindhi": "sd", "Tamil": "ta", "Telugu": "te", "Urdu": "ur"
}
DEFAULT_LANGUAGES = ["en", "hi", "bn"]


# --- Session State Initialization ---
if 'selected_module_idx' not in st.session_state:
    st.session_state.selected_module_idx = None
if 'app_mode' not in st.session_state:
    st.session_state.app_mode = "🎓 Learning Center"
if 'progress' not in st.session_state:
    st.session_state.progress = [False] * NUM_MODULES
for i in range(1, NUM_MODULES + 1):
    if f"module_{i}_answers" not in st.session_state:
        st.session_state[f"module_{i}_answers"] = {}
    if f"module_{i}_current_q" not in st.session_state:
        st.session_state[f"module_{i}_current_q"] = 0
    if f"module_{i}_quiz_complete" not in st.session_state:
        st.session_state[f"module_{i}_quiz_complete"] = False


# --- Caching & API Functions ---
@st.cache_data
def load_data(module_number):
    try:
        content_path = os.path.join(CONTENT_DIR, f"module_{module_number}_content.json")
        quiz_path = os.path.join(CONTENT_DIR, f"module_{module_number}_quiz.json")
        with open(content_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
        with open(quiz_path, 'r', encoding='utf-8') as f:
            quiz = json.load(f)
        return content, quiz
    except FileNotFoundError:
        st.error(f"FATAL ERROR: Files for Module {module_number} not found.")
        return None, None

@st.cache_data(show_spinner=False)
def preload_module_titles():
    titles = []
    for i in range(1, NUM_MODULES + 1):
        try:
            content_path = os.path.join(CONTENT_DIR, f"module_{i}_content.json")
            with open(content_path, 'r', encoding='utf-8') as f:
                title_text = json.load(f).get('en', {}).get('title', f'Module {i}')
                title_text = title_text.replace(f'Module {i}:', '').strip()
                titles.append(title_text)
        except Exception:
            titles.append(f"Module {i}")
    return titles

def translate_json_structure(json_data, target_lang_name):
    if not GEMINI_AVAILABLE: return json_data
    json_string = json.dumps(json_data, indent=2, ensure_ascii=False)
    prompt = f"""
    Translate all string values in the following JSON object into the '{target_lang_name}' language.
    RULES:
    1.  Maintain the exact same JSON structure, keys, and nesting.
    2.  Only translate the text values. Do not translate keys.
    3.  Your entire response MUST be a single, valid JSON object, enclosed in ```json ... ```.

    JSON to translate:
    {json_string}
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content(prompt)
        match = re.search(r'```json\s*(\{.*?\})\s*```', response.text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        else:
            return json.loads(response.text)
    except Exception as e:
        st.error(f"A translation error occurred. Displaying content in English. Details: {e}")
        return json_data

@st.cache_data(show_spinner="Loading and Translating Module...")
def get_processed_module(module_idx, lang_name, lang_code):
    _content, _quiz = load_data(module_idx)
    if _content is None or _quiz is None: return None, None
    if lang_code in DEFAULT_LANGUAGES:
        return _content.get(lang_code, _content['en']), _quiz.get(lang_code, _quiz['en'])
    translated_content = translate_json_structure(_content['en'], lang_name)
    translated_quiz = translate_json_structure(_quiz['en'], lang_name)
    return translated_content, translated_quiz


# --- UI Styling and Layout ---
def apply_css(view_mode='home'):
    if view_mode == 'home':
        image_path = 'wallpaper.png'
        if os.path.exists(image_path):
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode()
            st.markdown(f"""<style>
                .stApp {{ background-image: url(data:image/png;base64,{encoded_string}); background-size: cover; }}
                .stApp > header {{ background: rgba(0,0,0,0.5); }}
                div.stBlock, div.st-emotion-cache-1y4p8pa {{
                    background: rgba(255, 255, 255, 0.85); backdrop-filter: blur(10px);
                    padding: 1.5rem; border-radius: 0.5rem; }}
                </style>""", unsafe_allow_html=True)
    else:
        st.markdown("""<style>
            .stApp { background-color: #0f172a; }
            .stApp > header { background: transparent; }
            div.stBlock, div.st-emotion-cache-1y4p8pa {
                background-color: #1e293b; padding: 1.5rem; border-radius: 0.5rem; border: 1px solid #334155; }
            .stRadio > label {{ font-size: 1.1em; }}
            </style>""", unsafe_allow_html=True)

# --- UI Components ---
def display_module(module_idx, lang_name, lang_code):
    module_content, module_quiz = get_processed_module(module_idx, lang_name, lang_code)
    if not module_content or not module_quiz: return

    st.header(module_content.get('title', 'Module Content'), divider='rainbow')

    # --- BUG FIX: Reverted from st.tabs to a stateful st.radio button ---
    # This ensures the quiz tab remains selected after a form submission causes a rerun.
    tab_key = f"tab_selector_{module_idx}"
    if tab_key not in st.session_state:
        st.session_state[tab_key] = "📚 Learning Content"  # Default to learning content

    selected_tab = st.radio(
        "Navigation",
        ["📚 Learning Content", "🧠 Quiz Challenge"],
        key=tab_key,
        horizontal=True,
        label_visibility="collapsed"
    )

    if selected_tab == "📚 Learning Content":
        for i, section in enumerate(module_content.get('sections', [])):
            st.subheader(section.get('title', ''))
            content_text = section.get('content', '').strip()
            st.write(content_text)
            st.markdown("---")
    elif selected_tab == "🧠 Quiz Challenge":
        display_quiz(module_quiz, module_idx)


def display_quiz(quiz, module_idx):
    st.subheader(quiz.get('title', 'Quiz'))
    questions = quiz.get('questions', [])
    current_q_key = f"module_{module_idx}_current_q"
    answers_key = f"module_{module_idx}_answers"
    quiz_complete_key = f"module_{module_idx}_quiz_complete"
    current_q_index = st.session_state[current_q_key]
    if st.session_state[quiz_complete_key]:
        score = 0
        stored_answers = st.session_state[answers_key]
        for i, q in enumerate(questions):
            st.markdown(f"**Question {i+1}: {q.get('question', '')}**")
            user_answer = stored_answers.get(str(i))
            correct_answer = q.get('answer')
            if user_answer == correct_answer:
                score += 1
                st.success(f"Your answer: '{user_answer}' (Correct!)")
            else:
                st.error(f"Your answer: '{user_answer}' (Incorrect)")
                st.info(f"Correct answer: '{correct_answer}'")
            st.info(f"💡 Explanation: {q.get('explanation', '')}")
            st.markdown("---")
        st.success(f"**Quiz Complete!** Your score: {score}/{len(questions)}")
        if st.button("Try Again", key=f"retry_{module_idx}"):
            st.session_state[current_q_key] = 0
            st.session_state[answers_key] = {}
            st.session_state[quiz_complete_key] = False
            st.rerun()
    else:
        q = questions[current_q_index]
        st.markdown(f"**Question {current_q_index + 1}/{len(questions)}**")
        st.write(q.get('question', ''))
        with st.form(key=f"q_form_{module_idx}_{current_q_index}"):
            user_choice = st.radio("Choose your answer:", q.get('options', []), index=None)
            submitted = st.form_submit_button("Submit Answer")
            if submitted:
                if user_choice:
                    st.session_state[answers_key][str(current_q_index)] = user_choice
                    if current_q_index + 1 < len(questions):
                        st.session_state[current_q_key] += 1
                    else:
                        st.session_state[quiz_complete_key] = True
                        st.session_state.progress[module_idx-1] = True
                    st.rerun()
                else:
                    st.warning("Please select an answer before submitting.")

# --- Main Application Logic ---
def main():
    with st.sidebar:
        st.image("https://placehold.co/400x100/0f172a/ffffff?text=Polyglot+Quant+AI", use_column_width=True)
        st.header("App Navigation")
        
        st.radio(
            "Select a feature",
            ["🎓 Learning Center", "📈 Portfolio Playground", "🤖 Algorithmic Trading", "🔎 Stock Screener", "📄 FinSummarizer"],
            key='app_mode',
            label_visibility="collapsed"
        )
        
        st.markdown("---")

        # This entire block only shows controls if the Learning Center is active
        if st.session_state.app_mode == "🎓 Learning Center":
            st.header("Controls")
            selected_lang_name = st.selectbox("Choose Language", options=list(LANGUAGES.keys()))
            st.markdown("---")
            st.header("Modules")
            module_titles = preload_module_titles()
            for i, title in enumerate(module_titles):
                if st.button(title, key=f"module_btn_{i}", use_container_width=True):
                    st.session_state.selected_module_idx = i + 1
                    # When a module is selected, ensure its tab state is reset
                    st.session_state[f"tab_selector_{i+1}"] = "📚 Learning Content"
            st.markdown("---")
            st.header("Your Progress")
            completed = sum(st.session_state.progress)
            st.progress(completed / NUM_MODULES, text=f"{completed}/{NUM_MODULES} Modules Completed")
            if st.button("Reset All Progress", use_container_width=True):
                st.session_state.progress = [False] * NUM_MODULES
                for i in range(1, NUM_MODULES + 1):
                    st.session_state[f"module_{i}_answers"] = {}
                    st.session_state[f"module_{i}_current_q"] = 0
                    st.session_state[f"module_{i}_quiz_complete"] = False
                st.session_state.selected_module_idx = None
                st.rerun()
    
    # --- Main Area Routing ---
    if st.session_state.app_mode == "📈 Portfolio Playground":
        st.session_state.selected_module_idx = None
        render_playground()
    elif st.session_state.app_mode == "🤖 Algorithmic Trading":
        st.session_state.selected_module_idx = None
        render_algo_playground()
    elif st.session_state.app_mode == "🔎 Stock Screener":
        st.session_state.selected_module_idx = None
        render_screener()
    elif st.session_state.app_mode == "📄 FinSummarizer":
        st.session_state.selected_module_idx = None
        render_summarizer()
    else: # Default to Learning Center
        if st.session_state.selected_module_idx:
            apply_css('module')
            # Pass the language name from the sidebar controls
            display_module(st.session_state.selected_module_idx, selected_lang_name, LANGUAGES[selected_lang_name])
        else:
            apply_css('home')
            st.title("Polyglot Quant AI 🚀")
            st.header("Welcome to your interactive financial journey!")
            st.info("Please select a feature or a module from the sidebar to begin.")

if __name__ == "__main__":
    main()


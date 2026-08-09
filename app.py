"""Streamlit app for the MCQ solver."""

import html
import time

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = "models/roberta"
BASE_MODEL = "roberta-base"
MAX_LENGTH = 256
OPTS = list("ABCDE")

CARD_BG = "#cde2fb"
CARD_LINE = "#b7d3f6"
CARD_INK = "#0d366b"
TITLE_BLUE = "#3987e5"

# colour of the answer card changes with confidence
TIERS = [
    (0.70, "#e3f5e3", "#b9e5b9", "#006300", "#0ca30c", "PREDICTED ANSWER"),
    (0.40, "#fdf0d5", "#f6ddaa", "#6b4405", "#b8770a", "LIKELY ANSWER"),
    (0.00, "#fbe4e4", "#f2c2c2", "#8c1d1d", "#d03b3b", "LOW CONFIDENCE GUESS"),
]

RANK_FILL = ["#104281", "#1c5cab", "#2a78d6"]
REST_FILL = "rgba(128,128,128,0.55)"
TRACK = "rgba(128,128,128,0.25)"

torch.set_grad_enabled(False)

st.set_page_config(page_title="Smart MCQ Solver", page_icon="A", layout="wide")

# less padding so all five options fit without scrolling
st.markdown(
    """
<style>
div.block-container {padding-top: 2rem; padding-bottom: 1rem;}
.stTextArea textarea {height: 68px; min-height: 68px;}
.stButton button {height: 52px; font-size: 1rem; font-weight: 600;}
iframe[title="streamlit.components.v1.html"] {display: block; height: 0;}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_model():
    # base model from the hub, adapter and head from the local folder
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    base = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=5)
    model = PeftModel.from_pretrained(base, MODEL_DIR)
    model.eval()
    return tok, model


def format_mcq(prompt, opts):
    # same format used during training
    return (f"{prompt} "
            f"A) {opts['A']} B) {opts['B']} C) {opts['C']} D) {opts['D']} E) {opts['E']}")


def predict(tok, model, text):
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    logits = model(**inputs).logits[0].numpy()
    # softmax over the 5 classes
    e = np.exp(logits - logits.max())
    return e / e.sum()


def tier(p):
    for floor, bg, line, ink, dot, label in TIERS:
        if p >= floor:
            return bg, line, ink, dot, label


SAMPLES = {
    "Astronomy": {
        "prompt": "Which planet in our solar system has the shortest day?",
        "A": "Earth", "B": "Jupiter", "C": "Mars", "D": "Venus", "E": "Mercury",
    },
    "Biology": {
        "prompt": "Which gas do plants absorb during photosynthesis?",
        "A": "Oxygen", "B": "Carbon dioxide", "C": "Nitrogen",
        "D": "Hydrogen", "E": "Methane",
    },
    "History": {
        "prompt": "In which year did the Second World War end?",
        "A": "1943", "B": "1945", "C": "1947", "D": "1939", "E": "1950",
    },
    "Chemistry": {
        "prompt": "What is the most abundant gas in the Earth's atmosphere?",
        "A": "Oxygen", "B": "Nitrogen", "C": "Carbon dioxide",
        "D": "Argon", "E": "Hydrogen",
    },
    "Geography": {
        "prompt": "What is the tallest mountain above sea level?",
        "A": "K2", "B": "Mount Everest", "C": "Kangchenjunga",
        "D": "Denali", "E": "Mont Blanc",
    },
    "Literature": {
        "prompt": "Who wrote the novel 'Things Fall Apart'?",
        "A": "Wole Soyinka", "B": "Chinua Achebe", "C": "Ngugi wa Thiong'o",
        "D": "Ben Okri", "E": "Nadine Gordimer",
    },
    "Blank": {
        "prompt": "", "A": "", "B": "", "C": "", "D": "", "E": "",
    },
}

# load the first sample on startup
if "prompt" not in st.session_state:
    first = SAMPLES["Astronomy"]
    st.session_state.prompt = first["prompt"]
    for letter in OPTS:
        st.session_state[letter] = first[letter]
    st.session_state.result = None
    st.session_state.message = None
    st.session_state.run_id = 0
    st.session_state.scrolled_for = 0


def apply_sample():
    sample = SAMPLES[st.session_state.sample_name]
    st.session_state.prompt = sample["prompt"]
    for letter in OPTS:
        st.session_state[letter] = sample[letter]
    st.session_state.result = None
    st.session_state.message = None


ROWS = [
    ("Base model", "roberta-base"),
    ("Method", "LoRA (r=16, alpha=32)"),
    ("Adapted", "query, value projections"),
    ("Trainable", "1,184,261 of 125M"),
    ("Training", "6 epochs, lr 2e-4"),
    ("Max length", "256 tokens"),
    ("Dataset", "2,000 questions"),
]

with st.sidebar:
    cells = "".join(
        f"<tr><td style='padding:3px 0;opacity:.75'>{k}</td>"
        f"<td style='padding:3px 0;text-align:right;font-weight:600'>{v}</td></tr>"
        for k, v in ROWS
    )
    # background and text colour set together so it reads in both themes
    st.markdown(
        f"<div style='background:{CARD_BG};color:{CARD_INK};"
        f"border:1px solid {CARD_LINE};border-radius:8px;padding:12px 14px'>"
        f"<div style='font-weight:700;font-size:1.05rem;margin-bottom:6px'>About</div>"
        f"<table style='width:100%;font-size:.82rem;border-collapse:collapse'>{cells}</table>"
        f"</div>",
        unsafe_allow_html=True,
    )
    with st.expander("How it is scored"):
        st.markdown(
            """
The metric is **MAP@3** - three ranked letters are
submitted per question.

| Correct at | Score |
|---|---|
| rank 1 | 1.000 |
| rank 2 | 0.500 |
| rank 3 | 0.333 |
| not in top 3 | 0.000 |

Random guessing scores 0.367.
"""
        )

# cached so the model loads only once
loader = st.empty()
try:
    with loader.container():
        with st.spinner(f"loading model - first run downloads {BASE_MODEL}, around 500 MB"):
            tok, model = load_model()
    loader.empty()
except Exception as e:
    st.error("Could not load the model.")
    st.exception(e)
    st.stop()

st.markdown(
    f"<h2 style='margin:0;color:{TITLE_BLUE}'>Smart MCQ Solver</h2>"
    f"<div style='opacity:.7;font-size:.85rem;margin:2px 0 10px'>"
    f"Fine-tuned RoBERTa + LoRA - ranks all five options and returns the top 3</div>",
    unsafe_allow_html=True,
)

left, right = st.columns([3, 2], gap="large")

with left:
    st.selectbox("Example question", list(SAMPLES), key="sample_name",
                 on_change=apply_sample)

    st.text_area("Question", key="prompt", height=68)

    # three columns keeps the options on two rows
    cols = st.columns(3)
    for i, letter in enumerate(OPTS):
        with cols[i % 3]:
            st.text_input(f"Option {letter}", key=letter)

    pad_l, act, pad_r = st.columns([1, 1.6, 1])
    with act:
        go = st.button("Predict", type="primary", use_container_width=True,
                       icon=":material/search:")

if go:
    prompt = st.session_state.prompt.strip()
    opts = {letter: st.session_state[letter].strip() for letter in OPTS}
    missing = [letter for letter in OPTS if not opts[letter]]

    # message is stored so it can be shown in the right column
    st.session_state.result = None
    st.session_state.message = None

    if not prompt:
        st.session_state.message = "Enter a question first."
    elif missing:
        st.session_state.message = f"Fill in every option - missing {', '.join(missing)}."
    else:
        try:
            start = time.perf_counter()
            probs = predict(tok, model, format_mcq(prompt, opts))
            elapsed = time.perf_counter() - start
            st.session_state.result = {"probs": probs, "opts": opts, "elapsed": elapsed}
            st.session_state.run_id += 1
        except Exception:
            st.session_state.message = "Prediction failed, please try again."

with right:
    result = st.session_state.get("result")
    message = st.session_state.get("message")

    # scrolls the result into view on small screens
    if result is not None and st.session_state.run_id != st.session_state.scrolled_for:
        st.session_state.scrolled_for = st.session_state.run_id
        components.html(
            "<script>try{"
            "var el=window.frameElement,r=el.getBoundingClientRect(),"
            "h=window.parent.innerHeight||800;"
            "if(r.top<0||r.top>h*0.75)"
            "el.scrollIntoView({behavior:'smooth',block:'start'});"
            "}catch(e){}</script>",
            height=0,
        )

    if message:
        st.warning(message)
    elif result is None:
        st.info("Pick an example or type your own, then press Predict.")
    else:
        probs, opts = result["probs"], result["opts"]
        order = list(np.argsort(-probs))
        top3 = order[:3]
        best = top3[0]
        conf = float(probs[best])
        bg, line, ink, dot, label = tier(conf)

        st.markdown(
            "<div style='font-weight:600;font-size:1.05rem;margin:0 0 8px'>Prediction</div>",
            unsafe_allow_html=True,
        )

        # answer on the left, confidence on the right
        st.markdown(
            f"<div style='background:{bg};color:{ink};border:1px solid {line};"
            f"border-radius:8px;padding:14px 16px;display:flex;align-items:center;"
            f"justify-content:space-between;gap:16px'>"
            f"<div>"
            f"<div style='font-size:.75rem;opacity:.8;letter-spacing:.04em'>{label}</div>"
            f"<div style='font-size:1.45rem;font-weight:700;line-height:1.25'>"
            f"{OPTS[best]}) {html.escape(opts[OPTS[best]])}</div>"
            f"</div>"
            f"<div style='text-align:right;flex-shrink:0'>"
            f"<div style='font-size:.75rem;opacity:.8;letter-spacing:.04em'>CONFIDENCE</div>"
            f"<div style='font-size:1.45rem;font-weight:700;line-height:1.25'>"
            f"<span style='color:{dot}'>&#9679;</span> {conf * 100:.1f}%</div>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"<div style='margin:14px 0 8px'>"
            f"<span style='font-weight:600'>Top 3</span>"
            f"<span style='opacity:.65;margin-left:10px;letter-spacing:.08em'>"
            f"{' &bull; '.join(OPTS[i] for i in top3)}</span></div>",
            unsafe_allow_html=True,
        )

        # option on the left, percentage on the right, bar below
        bars = []
        for rank, i in enumerate(order):
            fill = RANK_FILL[rank] if rank < 3 else REST_FILL
            pct = probs[i] * 100
            bars.append(
                f"<div style='margin-bottom:9px'>"
                f"<div style='display:flex;justify-content:space-between;font-size:.86rem;"
                f"margin-bottom:3px'>"
                f"<span>{OPTS[i]}) {html.escape(opts[OPTS[i]])}</span>"
                f"<span style='font-variant-numeric:tabular-nums;opacity:.75'>"
                f"{pct:.1f}%</span></div>"
                f"<div style='background:{TRACK};border-radius:5px;height:9px'>"
                f"<div style='background:{fill};width:{max(pct, 1.2):.2f}%;"
                f"height:9px;border-radius:5px'></div></div>"
                f"</div>"
            )
        st.markdown("".join(bars), unsafe_allow_html=True)

        st.caption(f"inference took {result['elapsed'] * 1000:.0f} ms on CPU")


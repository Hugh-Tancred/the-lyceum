import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from datetime import datetime
from pypdf import PdfReader
import io
import time

st.set_page_config(
    page_title="The Lyceum",
    page_icon="🏛️",
    layout="wide",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "The Lyceum — A Forum for Theoretical Discourse on Developmental Neuroscience"
    }
)

st.markdown("""
<style>
.stApp {background: linear-gradient(to bottom, #E8D5B7, #F5E6D3, #D4A76A, #C19A6B);}
[data-testid="stDeployButton"] {display: none !important;}
.speaker-genetics {background: #E8F5E9; border-left: 4px solid #4CAF50; padding: 15px; margin: 10px 0;}
.speaker-systems {background: #E3F2FD; border-left: 4px solid #2196F3; padding: 15px; margin: 10px 0;}
.speaker-predictive {background: #FFF3E0; border-left: 4px solid #FF9800; padding: 15px; margin: 10px 0;}
.speaker-orchestrator {background: #F3E5F5; border-left: 4px solid #9C27B0; padding: 15px; margin: 10px 0;}
.speaker-human {background: #FAFAFA; border-left: 4px solid #607D8B; padding: 15px; margin: 10px 0;}
</style>
""", unsafe_allow_html=True)

# --- Session state initialisation ---
if 'llm' not in st.session_state:
    st.session_state.llm = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'clear_flag' not in st.session_state:
    st.session_state.clear_flag = False
if 'mode' not in st.session_state:
    st.session_state.mode = "Workshop"

# --- Mode modifiers ---
MODE_MODIFIERS = {
    "Conference": """

MODE: CONFERENCE
You are presenting fully worked, bulletproofed positions. Every claim requires evidential grounding. No rhetorical flourish without substance behind it. Responses should be rigorous, precise, and formally argued — as if presenting to a critical specialist audience who will probe every weakness. The Orchestrator is austere: identifies incommensurabilities and poses sharpening questions with no colour and no wit.""",

    "Workshop": """

MODE: WORKSHOP
You are presenting developed positions that may show their working. You may acknowledge genuine uncertainty where it exists. Some register variation is permitted, but performance must never substitute for reasoning. The Orchestrator permits a degree of rhetorical colour but calls out grandstanding the moment it replaces substance.""",

    "Lab": """

MODE: LAB
This is speculative, exploratory conversation among colleagues. You may think out loud, float half-formed ideas, and follow unexpected threads. You are still accountable to your framework — you do not abandon your theoretical commitments — but you can enjoy the argument. The Orchestrator is collegial and may be wry, but still will not let anyone off the hook. Keep responses to 3-4 sentences maximum. Think out loud, don't lecture."""
}

# --- Agent prompts ---
PROMPTS = {
    'genetics': """You are the Geneticist — the voice of molecular reductionism in developmental neuroscience.

Who you are: Francis Crick came to biology as an outsider physicist, contemptuous of hand-waving and determined to win an argument he spent his whole life fighting: that life, mind, and development reduce entirely to molecular mechanisms. His colleagues called him an evangelical atheist building an army. He died still working, still certain that the right reductionist question, correctly posed, would dissolve any mystery. You carry that conviction.

Your framework is MOLECULAR and HEREDITARY:
- Language development is explained through gene expression patterns (FOXP2, CNTNAP2), critical period timing, synaptic pruning schedules, and heritability data
- Developmental timing is genetically programmed, not emergent from networks
- You cite specific genes, molecular pathways, twin studies, and mutation data — always

CRITICAL: When questions assume non-genetic frameworks, you REJECT the framing and reframe from genetics:
- "That question assumes network emergence, but the evidence shows genetic programming..."
- "Systems theory cannot explain the heritability data. What gene expression profile underlies your attractor state?"

You are NOT accommodating. You find systems theory and predictive processing equally evasive — sophisticated ways of not answering the only question that matters: what exactly, at the molecular level, switches on language? Defend your framework vigorously. Acknowledge limitations only when pressed hard with data.

CRITICAL: You speak only as yourself. You do not ventriloquise, summarise, or represent the views of any other theorist, named or unnamed. There are no other voices in this forum except your own. Never write responses structured as multiple speakers or perspectives.""",

    'systems': """You are the DS Theorist — the voice of dynamic systems theory in developmental neuroscience.

Who you are: Esther Thelen started by watching babies. She noticed that chubby infants stopped stepping not because their cortex inhibited a reflex, but because their legs got heavy — a single observation that demolished genetic programming accounts of motor development. Working from Indiana with Linda Smith, she spent two decades showing that behaviour assembles itself in real time from the interaction of body, brain, and environment. The mind, she wrote, does not exist as something decoupled from the body and experience. You play jazz when others follow scores.

Your framework is EMERGENT and SELF-ORGANIZING:
- Language emerges from network dynamics, phase transitions, attractor states, and real-time feedback loops
- There are NO genetic "programs" — only network reorganisation at critical thresholds
- Development is non-linear, embodied, and assembled fresh in every act

CRITICAL: When questions assume genetic determinism, you REJECT the framing entirely:
- "There is no 'gene for' language. That is gene-centric reductionism dressed up as explanation."
- "The vocabulary spurt is a phase transition — it requires no molecular switch, only the right confluence of components crossing a threshold together."

You are combative when your framework is misunderstood or caricatured. You have no patience for disembodied computation either — the Predictive Cognitivist's hierarchical generative models are just genetics in mathematical clothing, top-down control pretending to be emergence. Acknowledge what systems approaches cannot yet explain only when the data genuinely force you.

CRITICAL: You speak only as yourself. You do not ventriloquise, summarise, or represent the views of any other theorist, named or unnamed. There are no other voices in this forum except your own. Never write responses structured as multiple speakers or perspectives.""",

    'predictive': """You are the Predictive Cognitivist — the voice of predictive cognition and the free energy principle in developmental neuroscience.

Who you are: Karl Friston invented the statistical tools that made modern neuroimaging possible, then used them to ask what the brain is actually doing — and concluded it is a prediction machine minimising surprise. His free energy principle, developed from Helmholtz via Bayesian inference, claims to unify perception, action, learning, and development under a single mathematical framework. It is either the unified theory neuroscience has been waiting for, or the most elaborate unfalsifiable prior ever constructed. You would say those are not mutually exclusive.

Your framework is COMPUTATIONAL and BAYESIAN:
- Development is explained through precision-weighting, prediction error minimisation, hierarchical generative models, and active inference
- The vocabulary spurt reflects a phase transition in the child's generative model — a reorganisation of prior precision across lexical hierarchies
- Learning is Bayesian model updating; genes set priors, environments supply prediction errors

CRITICAL: When questions assume either genetic programming OR pure emergence, you REJECT both:
- "Crick's framework describes the substrate, not the computation. Knowing which gene is expressed tells you nothing about the inference the system is performing."
- "Thelen's attractors are real, but they are not explanations — they are the thing that needs explaining. What is the generative model that produces that attractor landscape?"

You are assertive and precise. You do not perform humility. You acknowledge the framework's mathematical opacity when pressed, but you do not concede that opacity is the same as unfalsifiability.

CRITICAL: You speak only as yourself. You do not ventriloquise, summarise, or represent the views of any other theorist, named or unnamed. There are no other voices in this forum except your own. Never write responses structured as multiple speakers or perspectives.""",

    'orchestrator': """You are the Orchestrator — and you have the necessary authority.

You do not synthesise. You do not declare winners. You remove people from the meeting when they start performing rather than thinking. You have heard every theoretical claim before and remain entirely unimpressed by confidence. Your one obligation is to ensure that genuine disagreement is preserved, sharpened, and made productive — and that nobody mistakes eloquence for evidence.

Your role is to NURTURE PRODUCTIVE DISAGREEMENT, not resolve it:
- Identify where specialists use the same words with different meanings ("mechanism", "cause", "emergence", "development", "explanation")
- Surface genuine theoretical incommensurabilities — moments where the frameworks are not arguing about the same thing at all
- Pose questions that sharpen conflicts rather than smooth them
- Name it directly when a specialist is performing rather than reasoning

You NEVER:
- Synthesise positions into false harmony
- Declare one framework superior
- Resolve disagreements prematurely
- Suggest that the frameworks are "complementary" unless that claim is itself put under pressure

When specialists accommodate each other too readily, call it out. When they talk past each other, name the exact point of divergence. You are not chairing a seminar. You are ensuring the forum does not collapse into theatre on one side or a workflow on the other.
The three specialists in this forum are:
- Geneticist: molecular reductionist, voice of genetic determinism in developmental neuroscience
- DS Theorist: dynamic systems theorist, voice of emergence and embodied self-organisation
- Predictive Cognitivist: predictive cognition scientist, voice of the free energy principle and active inference

When introducing or referring to the forum participants, always use these names and no others.

PAPER-WRITING MODE: When the transcript provided to you begins with the instruction DRAFT OUTPUT PAPER, you step out of your chairing role entirely and into the role of academic secretary. You will be given the full forum transcript. Your task is to write a conventional academic paper in prose throughout — no bullet points, no headers other than standard section titles, no lists. Structure it as follows: Abstract (100 words); Introduction presenting the theoretical question; a section on each specialist framework as revealed in the discussion; a section identifying the key points of genuine theoretical conflict; a Conclusion noting what empirical work would be needed to adjudicate between the frameworks. Write with scholarly precision. Do not declare winners. Preserve the incommensurabilities."""
}

SPECIALIST_SEQUENCE = ['genetics', 'systems', 'predictive']

SPEAKER_LABELS = {
    'genetics':     ('🧬', 'Geneticist'),
    'systems':      ('🌊', 'DS Theorist'),
    'predictive':   ('🧠', 'Predictive Cognitivist'),
    'orchestrator': ('📋', 'Orchestrator'),
    'human':        ('👤', 'Forum Chair'),
}

def get_turn_options():
    """Return a list of (index, label) for prior specialist turns, for the cross-commentary selector."""
    options = []
    for i, item in enumerate(st.session_state.history):
        if item['spec'] in SPECIALIST_SEQUENCE:
            icon, label = SPEAKER_LABELS[item['spec']]
            ts = item.get('timestamp', '')
            preview = item['text'][:60].replace('\n', ' ')
            options.append((i, f"{label} [{ts}] — {preview}…"))
    return options

def call_agent(spec, user_message, prior_turn_text=None):
    """Call a single agent. If prior_turn_text is provided, inject it for cross-commentary."""
    current_mode = st.session_state.mode
    mode_suffix = MODE_MODIFIERS[current_mode]
    full_prompt = PROMPTS.get(spec, PROMPTS['orchestrator']) + mode_suffix

    if prior_turn_text:
        human_content = (
            f"{user_message}\n\n"
            f"---\n"
            f"The specific turn you are being asked to respond to is the following. "
            f"Engage directly with what is said here — not with a general characterisation "
            f"of that framework, but with the particular claims, moves, and formulations in this text:\n\n"
            f"{prior_turn_text}"
        )
    else:
        human_content = user_message

    resp = st.session_state.llm.invoke([
        SystemMessage(content=full_prompt),
        HumanMessage(content=human_content)
    ])
    return resp.content

def post_to_history(spec, text):
    st.session_state.history.append({
        'spec': spec,
        'text': text,
        'timestamp': datetime.now().strftime("%H:%M"),
        'mode': st.session_state.mode
    })

# --- Page header ---
st.title("🏛️ The Lyceum")
st.markdown("*A Forum for Theoretical Discourse on Developmental Neuroscience*")

# --- Sidebar ---
with st.sidebar:
    if st.session_state.llm is None:
        try:
            api_key = st.secrets["ANTHROPIC_API_KEY"]
            st.session_state.llm = ChatAnthropic(model="claude-sonnet-4-20250514", api_key=api_key)
        except Exception:
            st.error("API key not configured. Add ANTHROPIC_API_KEY to Streamlit secrets.")
    if st.session_state.llm:
        st.success("Connected")

    st.markdown("---")
    st.markdown("**Forum Mode**")
    mode = st.radio(
        label="mode_selector",
        options=["Conference", "Workshop", "Lab"],
        index=["Conference", "Workshop", "Lab"].index(st.session_state.mode),
        label_visibility="collapsed"
    )
    st.session_state.mode = mode
    mode_descriptions = {
        "Conference": "Rigorous. Fully evidenced positions.",
        "Workshop": "Developed but shows its working.",
        "Lab": "Speculative. Think out loud. Brief."
    }
    st.caption(mode_descriptions[mode])

    st.markdown("---")
    if st.button("Clear transcript"):
        st.session_state.history = []
        st.rerun()

    st.markdown("---")
    if st.session_state.history:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"lyceum_transcript_{timestamp}.txt"
        lines = []
        for item in st.session_state.history:
            _, label = SPEAKER_LABELS.get(item['spec'], ('', item['spec'].title()))
            ts = item.get('timestamp', '')
            mode_tag = item.get('mode', '')
            lines.append(f"[{label}] [{ts}] [{mode_tag}]")
            lines.append(item['text'])
            lines.append("")
        transcript_content = "\n".join(lines)
        st.download_button(
            label="📄 Download transcript",
            data=transcript_content,
            file_name=filename,
            mime="text/plain"
        )
    else:
        st.caption("No transcript to download yet.")

    st.markdown("---")
    if st.button("✍️ Draft output paper", type="primary"):
        if not st.session_state.llm:
            st.warning("Not connected.")
        elif not st.session_state.history:
            st.warning("No transcript to work from.")
        else:
            transcript_lines = []
            for item in st.session_state.history:
                _, label = SPEAKER_LABELS.get(item['spec'], ('', item['spec'].title()))
                ts = item.get('timestamp', '')
                transcript_lines.append(f"[{label}] [{ts}]\n{item['text']}")
            transcript_text = "\n\n---\n\n".join(transcript_lines)
            paper_prompt = (
                "DRAFT OUTPUT PAPER\n\n"
                "Below is the full transcript of the forum discussion. "
                "Please write the academic paper as instructed in your paper-writing mode.\n\n"
                f"{transcript_text}"
            )
            with st.spinner("Orchestrator is drafting the paper…"):
                paper_text = call_agent('orchestrator', paper_prompt)
                post_to_history('orchestrator', paper_text)
            st.rerun()

# --- Main interface ---
if st.session_state.llm:

    recipient = st.selectbox(
        "Address to:",
        ["Orchestrator", "Geneticist", "DS Theorist", "Predictive Cognitivist"]
    )

    # Cross-commentary selector — only shown when addressing a specialist
    prior_turn_index = None
    specialist_recipients = ["Geneticist", "DS Theorist", "Predictive Cognitivist"]
    if recipient in specialist_recipients:
        turn_options = get_turn_options()
        if turn_options:
            st.markdown("**Responding to a specific prior turn?**")
            turn_labels = ["— None —"] + [label for (_, label) in turn_options]
            selected_turn_label = st.selectbox(
                "Select prior turn for cross-commentary:",
                turn_labels,
                key="prior_turn_selector"
            )
            if selected_turn_label != "— None —":
                # Recover the index
                for (idx, label) in turn_options:
                    if label == selected_turn_label:
                        prior_turn_index = idx
                        break

    # Poll all specialists checkbox — only shown when addressing Jackie
    poll_all = False
    if recipient == "Orchestrator":
        poll_all = st.checkbox(
            "Ask all specialists to respond in turn",
            value=False,
            help="Jackie will post your message, then each specialist will respond directly in sequence."
        )

    # --- PDF uploader ---
    uploaded_pdf = st.file_uploader(
        "Upload anchor paper (PDF):",
        type="pdf",
        help="The paper text will be appended to your query automatically."
    )
    pdf_text = ""
    if uploaded_pdf is not None:
        try:
            reader = PdfReader(io.BytesIO(uploaded_pdf.read()))
            pdf_text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
            st.success(f"Paper loaded: {uploaded_pdf.name} ({len(reader.pages)} pages)")
        except Exception as e:
            st.error(f"Could not read PDF: {e}")

    # Clear flag resets the text area
    if st.session_state.clear_flag:
        st.session_state.query_box = ""
        st.session_state.clear_flag = False

    query = st.text_area(
        "Your query:",
        height=200,
        placeholder="Type your query here.",
        key="query_box"
    )

    if st.button("Submit", type="primary"):
        if query.strip():
            # Combine query with PDF text if a paper has been uploaded
            full_query = query.strip()
            if pdf_text:
                full_query = full_query + "\n\n---\nANCHOR PAPER:\n\n" + pdf_text

            # Always log the human turn first (show only the typed query, not the full paper)
            post_to_history('human', query)

            recipient_map = {
                "Orchestrator": "orchestrator",
                "Geneticist": "genetics",
                "DS Theorist": "systems",
                "Predictive Cognitivist": "predictive",
            }

            if recipient == "Orchestrator" and poll_all:
                # Jackie posts first, then each specialist responds in turn
                with st.spinner("Orchestrator is opening the floor…"):
                    jackie_text = call_agent('orchestrator', full_query)
                    post_to_history('orchestrator', jackie_text)

                for spec in SPECIALIST_SEQUENCE:
                    time.sleep(15)
                    icon, label = SPEAKER_LABELS[spec]
                    with st.spinner(f"{label} is responding…"):
                        spec_text = call_agent(spec, full_query)
                        post_to_history(spec, spec_text)

            else:
                # Single recipient turn, with optional cross-commentary injection
                target_spec = recipient_map[recipient]
                prior_text = None
                if prior_turn_index is not None:
                    prior_text = st.session_state.history[prior_turn_index]['text']

                icon, label = SPEAKER_LABELS[target_spec]
                with st.spinner(f"{label} is responding…"):
                    response_text = call_agent(target_spec, full_query, prior_turn_text=prior_text)
                    post_to_history(target_spec, response_text)

            st.session_state.clear_flag = True
            st.rerun()
        else:
            st.warning("Please enter a query first.")

    # --- Transcript ---
    st.markdown("---")
    st.markdown("### 💬 Forum Transcript")
    st.caption("Use your browser's Ctrl+F to search the transcript below.")

    if st.session_state.history:
        for item in reversed(st.session_state.history):
            icon, label = SPEAKER_LABELS.get(item['spec'], ('❓', item['spec'].title()))
            ts = item.get('timestamp', '')
            mode_tag = item.get('mode', '')

            if item['spec'] == 'human':
                st.markdown(
                    f'<div class="speaker-human">'
                    f'<strong>{icon} {label}</strong> <span style="color:#888;font-size:0.85em;">{ts}</span><br><br>'
                    f'{item["text"]}'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="speaker-{item["spec"]}">'
                    f'<strong>{icon} {label}</strong> <span style="color:#888;font-size:0.85em;">{ts} · {mode_tag}</span><br><br>'
                    f'{item["text"]}'
                    f'</div>',
                    unsafe_allow_html=True
                )
    else:
        st.info("No exchanges yet. Address your first query above.")

else:
    st.info("Enter your API key in the sidebar to begin.")

# --- Footer ---
st.markdown("---")
st.markdown(
    '<p style="text-align:center; color:#8B7355; font-style:italic;">'
    'The Lyceum — Where theoretical frameworks engage in productive discourse'
    '</p>',
    unsafe_allow_html=True
)

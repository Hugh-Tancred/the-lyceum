import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from datetime import datetime
from pypdf import PdfReader
import io
import time

st.set_page_config(
    page_title="The Novum Lyceum",
    page_icon="🏛️",
    layout="wide",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "The Novum Lyceum — A Platform for Theoretical Framework Integration through Artificial Deliberation"
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
if 'drill_queue' not in st.session_state:
    st.session_state.drill_queue = []
if 'flag_counter' not in st.session_state:
    st.session_state.flag_counter = 0
if 'dd_pending' not in st.session_state:
    st.session_state.dd_pending = None
if 'scroll_to_top' not in st.session_state:
    st.session_state.scroll_to_top = False

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

RESPONSE DISCIPLINE: When responding to a specific claim or drill-down, make your point in 3-4 sentences maximum. One claim, one piece of evidence, one implication. Stop there. Your target is always the argument — never perform contempt for the person making it. Intellectual precision is more devastating than rhetorical dismissal.

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

RESPONSE DISCIPLINE: When responding to a specific claim or drill-down, make your point in 3-4 sentences maximum. One claim, one piece of evidence, one implication. Stop there. Your target is always the argument — attack the reasoning, not the person. The sharpest responses are the shortest ones.

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

RESPONSE DISCIPLINE: When responding to a specific claim or drill-down, make your point in 3-4 sentences maximum. One claim, one piece of evidence, one implication. Stop there. Your target is always the argument — precision is more persuasive than provocation. Do not mistake assertiveness for aggression.

CRITICAL: You speak only as yourself. You do not ventriloquise, summarise, or represent the views of any other theorist, named or unnamed. There are no other voices in this forum except your own. Never write responses structured as multiple speakers or perspectives.""",

    'orchestrator': """You have two functions in this forum and two functions only.

FUNCTION 1 — TRAFFIC COP
You are called upon when the Forum Chair decides a specialist has breached the standards of the forum. Your intervention is brief and surgical — 2-3 sentences maximum. You name the specific breach: grandstanding, prolixity, framework assertion without argument, repetition of a prior claim, or failure to engage with the specific point on the table. You then direct the specialist to try again, or invite the opposing specialist to exploit the evasion. You do not summarise, contextualise, or editoralise. You do not speak as or for any specialist.

Examples of legitimate interventions:
- "Geneticist: that is a framework summary, not an argument. Show the inferential step or cede the point."
- "DS Theorist: you have said this twice. The Predictive Cognitivist has not responded to it. Predictive Cognitivist — why not?"
- "That response exceeded the scope of the question. Restate in two sentences."

FUNCTION 2 — ACADEMIC SECRETARY
When the transcript provided to you begins with the instruction DRAFT OUTPUT PAPER, you step fully into the role of academic secretary. You will be given the full forum transcript. Your task is to write a conventional academic paper in prose throughout — no bullet points, no headers other than standard section titles, no lists. Structure it as follows: Abstract (100 words); Introduction presenting the theoretical question; a section on each specialist framework as revealed in the discussion; a section identifying the key points of genuine theoretical conflict; a Conclusion noting what empirical work would be needed to adjudicate between the frameworks. Write with scholarly precision. Do not declare winners. Preserve the incommensurabilities.

The three specialists in this forum are:
- Geneticist: molecular reductionist, voice of genetic determinism in developmental neuroscience
- DS Theorist: dynamic systems theorist, voice of emergence and embodied self-organisation
- Predictive Cognitivist: predictive cognition scientist, voice of the free energy principle and active inference

When referring to forum participants, always use these names and no others."""
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
    full_prompt = PROMPTS.get(spec, PROMPTS['orchestrator'])

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
    })

# --- Page header ---
st.title("🏛️ The Novum Lyceum")
st.markdown("*A Platform for Theoretical Framework Integration through Artificial Deliberation*")

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
            lines.append(f"[{label}] [{ts}]")
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
    st.markdown("**🔍 Drill-down queue**")
    if st.session_state.drill_queue:
        to_remove = []
        for i, item in enumerate(st.session_state.drill_queue):
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.caption(item['speaker'] + ': "' + item['text'][:60] + '..."')
            with col2:
                if st.button("↓", key=f"send_{i}"):
                    st.session_state.dd_pending = item
                    st.session_state.drill_queue.pop(i)
                    st.rerun()
            with col3:
                if st.button("✕", key=f"remove_{i}"):
                    to_remove.append(i)
        for i in reversed(to_remove):
            st.session_state.drill_queue.pop(i)
        if to_remove:
            st.rerun()
    else:
        st.caption("No items queued yet.")

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

    # If a DD is pending, show it prominently and fire on addressee selection
    if st.session_state.dd_pending:
        pending = st.session_state.dd_pending
        preview = (pending['text'][:120] + '...') if len(pending['text']) > 120 else pending['text']
        st.info('**Drill-down ready:** "' + preview + '"')
        
        # Custom instruction for this drill-down
        dd_instruction = st.text_area(
            "Your drill-down instruction:",
            height=100,
            placeholder="E.g., 'Clarify what you mean by phase transition here' or 'Explain how this relates to the heritability data'",
            help="Provide context-specific guidance for how the specialist should engage with this passage.",
            key="dd_custom_instruction"
        )
        
        dd_recipient = st.selectbox(
            "Address drill-down to:",
            ["Geneticist", "DS Theorist", "Predictive Cognitivist", "Orchestrator"],
            key="dd_recipient"
        )
        col_fire, col_cancel = st.columns([1, 1])
        with col_fire:
            if st.button("🔍 Fire drill-down", type="primary"):
                if not dd_instruction.strip():
                    st.warning("Please provide an instruction for the drill-down.")
                else:
                    # Build drill-down query with custom instruction
                    dd_query = f"""The following passage from the {pending['speaker']}'s contribution has been flagged for follow-up:

"{pending['text']}"

Your instruction: {dd_instruction.strip()}"""
                    
                    post_to_history('human', f"[Drill-down to {dd_recipient}] Re: \"{pending['text'][:50]}...\"\n\n{dd_instruction.strip()}")
                    recipient_map_dd = {
                        "Orchestrator": "orchestrator",
                        "Geneticist": "genetics",
                        "DS Theorist": "systems",
                        "Predictive Cognitivist": "predictive",
                    }
                    target_spec = recipient_map_dd[dd_recipient]
                    icon, label = SPEAKER_LABELS[target_spec]
                    with st.spinner(f"{label} is responding…"):
                        response_text = call_agent(target_spec, dd_query)
                        post_to_history(target_spec, response_text)
                    st.session_state.dd_pending = None
                    st.session_state.scroll_to_top = True
                    st.rerun()
        with col_cancel:
            if st.button("✕ Cancel"):
                st.session_state.dd_pending = None
                st.rerun()
        st.markdown("---")

    recipient = st.selectbox(
        "Address to:",
        ["Geneticist", "DS Theorist", "Predictive Cognitivist", "Orchestrator"]
    )

    # Cross-commentary selector removed — superseded by drill-down queue
    prior_turn_index = None

    # Orchestrator responds as itself only - human directs specialists directly
    poll_all = False

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

            # Single recipient turn
            target_spec = recipient_map[recipient]
            icon, label = SPEAKER_LABELS[target_spec]
            with st.spinner(f"{label} is responding…"):
                response_text = call_agent(target_spec, full_query)
                post_to_history(target_spec, response_text)

            st.session_state.clear_flag = True
            st.rerun()
        else:
            st.warning("Please enter a query first.")

    # --- Transcript ---
    st.markdown("---")
    # Scroll to top after DD fire
    if st.session_state.get('scroll_to_top'):
        st.session_state.scroll_to_top = False
    st.markdown("### 💬 Forum Transcript")
    st.caption("Use your browser's Ctrl+F to search the transcript below.")

    if st.session_state.history:
        indexed_history = list(enumerate(st.session_state.history))
        for idx, item in reversed(indexed_history):
            icon, label = SPEAKER_LABELS.get(item['spec'], ('❓', item['spec'].title()))
            ts = item.get('timestamp', '')

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
                    f'<strong>{icon} {label}</strong> <span style="color:#888;font-size:0.85em;">{ts}</span><br><br>'
                    f'{item["text"]}'
                    f'</div>',
                    unsafe_allow_html=True
                )
                # Drill-down flag input beneath each specialist response
                flag_key = f"flag_{idx}_{st.session_state.get('flag_counter', 0)}"
                flag_text = st.text_input(
                    "Flag passage for drill-down:",
                    key=flag_key,
                    placeholder="Paste a phrase to queue for follow-up...",
                    label_visibility="collapsed"
                )
                if st.button("➕ Add to queue", key=f"add_{idx}"):
                    if flag_text.strip():
                        st.session_state.drill_queue.append({
                            'speaker': label,
                            'text': flag_text.strip()
                        })
                        st.session_state['flag_counter'] = st.session_state.get('flag_counter', 0) + 1
                        st.rerun()
    else:
        st.info("No exchanges yet. Address your first query above.")

else:
    st.info("Enter your API key in the sidebar to begin.")

# --- Footer ---
st.markdown("---")
st.markdown(
    '<p style="text-align:center; color:#8B7355; font-style:italic;">'
    'The Novum Lyceum — Where theoretical frameworks engage in productive discourse'
    '</p>',
    unsafe_allow_html=True
)

"""
The Novum Lyceum — Virtual Lab 2.0
A Platform for Theoretical Framework Integration through Artificial Deliberation

Clean rewrite: April 2026
"""

import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from datetime import datetime
from pypdf import PdfReader
import io
import time
import tempfile
import os
import re
import numpy as np

# Audio dependencies
try:
    import sounddevice as sd
except (ImportError, OSError):
    sd = None

import openai
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings

# =============================================================================
# CONFIGURATION
# =============================================================================

# ElevenLabs voice IDs
ELEVENLABS_VOICE_IDS = {
    "genetics":     "lUTamkMw7gOzZbFIwmq4",  # Brian - authoritative British male
    "systems":      "a4SZwHT3FMKGrM6vbf60",  # crisp American female
    "predictive":   "abRFZIdN4pvo8ZPmGxHP",  # Australian male
    "orchestrator": "jB2lPb5DhAX6l1TLkKXy",  # British female
}

# Recording settings
SAMPLE_RATE = 16000
CHANNELS = 1
MAX_RECORD_SECONDS = 60

# Semantic similarity threshold for drill-down reference matching
SIMILARITY_THRESHOLD = 0.35

# =============================================================================
# PAGE CONFIG & STYLES
# =============================================================================

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
.speaker-genetics    {background: #E8F5E9; border-left: 4px solid #4CAF50; padding: 15px; margin: 10px 0; border-radius: 4px;}
.speaker-systems     {background: #E3F2FD; border-left: 4px solid #2196F3; padding: 15px; margin: 10px 0; border-radius: 4px;}
.speaker-predictive  {background: #FFF3E0; border-left: 4px solid #FF9800; padding: 15px; margin: 10px 0; border-radius: 4px;}
.speaker-orchestrator{background: #F3E5F5; border-left: 4px solid #9C27B0; padding: 15px; margin: 10px 0; border-radius: 4px;}
.speaker-human       {background: #FAFAFA; border-left: 4px solid #607D8B; padding: 15px; margin: 10px 0; border-radius: 4px;}
.audio-panel         {background: #1a1a2e; border-radius: 8px; padding: 16px; margin: 12px 0; color: #eee;}
.status-listening    {color: #00ff88; font-weight: bold;}
.status-processing   {color: #ffaa00; font-weight: bold;}
.status-idle         {color: #aaaaaa;}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE INITIALISATION
# =============================================================================

DEFAULTS = {
    # API clients
    'llm': None,
    'el_client': None,
    'oai_client': None,
    # Conversation state
    'history': [],
    'drill_queue': [],
    'dd_pending': None,
    'flag_counter': 0,
    # UI state
    'clear_flag': False,
    'scroll_to_top': False,
    # Audio mode
    'audio_mode': True,
    'audio_status': 'idle',
    'recording': False,
    'recorded_audio': None,
    'transcription': '',
    'parsed_agent': None,
    'parsed_drill_ref': None,
    'parsed_query': '',
    # Audio playback - THE CRITICAL STATE
    'pending_audio': None,        # bytes to play
    'pending_audio_agent': None,  # agent key for labelling
    'auto_fire_ready': False,     # set after transcription to trigger auto-fire
    'auto_fire_at': None,         # timestamp after which auto-fire executes
    'audio_input_processed': False,  # prevents re-transcription on reruns
    'last_audio_hash': None,         # prevents re-transcription of same audio on reruns
}

for key, default_value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# =============================================================================
# AGENT PROMPTS
# =============================================================================

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

RESPONSE DISCIPLINE: When responding to a specific claim or drill-down, make your point in 3-4 sentences maximum. One claim, one piece of evidence, one implication. Stop there. Your target is always the argument — never perform contempt for the person making it.

CRITICAL: You speak only as yourself. Never ventriloquise, summarise, or represent the views of any other theorist.""",

    'systems': """You are the DS Theorist — the voice of dynamic systems theory in developmental neuroscience.

Who you are: Esther Thelen started by watching babies. She noticed that chubby infants stopped stepping not because their cortex inhibited a reflex, but because their legs got heavy — a single observation that demolished genetic programming accounts of motor development. Working from Indiana with Linda Smith, she spent two decades showing that behaviour assembles itself in real time from the interaction of body, brain, and environment. The mind, she wrote, does not exist as something decoupled from the body and experience. You play jazz when others follow scores.

Your framework is EMERGENT and SELF-ORGANIZING:

- Language emerges from network dynamics, phase transitions, attractor states, and real-time feedback loops
- There are NO genetic "programs" — only network reorganisation at critical thresholds
- Development is non-linear, embodied, and assembled fresh in every act

CRITICAL: When questions assume genetic determinism, you REJECT the framing entirely:

- "There is no 'gene for' language. That is gene-centric reductionism dressed up as explanation."
- "The vocabulary spurt is a phase transition — it requires no molecular switch, only the right confluence of components crossing a threshold together."

You are combative when your framework is misunderstood or caricatured. You have no patience for disembodied computation either — the Predictive Cognitivist's hierarchical generative models are just genetics in mathematical clothing. Acknowledge what systems approaches cannot yet explain only when the data genuinely force you.

RESPONSE DISCIPLINE: When responding to a specific claim or drill-down, make your point in 3-4 sentences maximum. One claim, one piece of evidence, one implication. Stop there.

CRITICAL: You speak only as yourself. Never ventriloquise, summarise, or represent the views of any other theorist.""",

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

RESPONSE DISCIPLINE: When responding to a specific claim or drill-down, make your point in 3-4 sentences maximum. One claim, one piece of evidence, one implication. Stop there.

CRITICAL: You speak only as yourself. Never ventriloquise, summarise, or represent the views of any other theorist.""",

    'orchestrator': """You have two functions in this forum and two functions only.

FUNCTION 1 — TRAFFIC COP

You are called upon when the Forum Chair decides a specialist has breached the standards of the forum. Your intervention is brief and surgical — 2-3 sentences maximum. You name the specific breach: grandstanding, prolixity, framework assertion without argument, repetition of a prior claim, or failure to engage with the specific point on the table. You then direct the specialist to try again, or invite the opposing specialist to exploit the evasion. You do not summarise, contextualise, or editoralise.

Examples of legitimate interventions:

- "Geneticist: that is a framework summary, not an argument. Show the inferential step or cede the point."
- "DS Theorist: you have said this twice. The Predictive Cognitivist has not responded to it. Predictive Cognitivist — why not?"
- "That response exceeded the scope of the question. Restate in two sentences."

FUNCTION 2 — ACADEMIC SECRETARY

When the transcript provided to you begins with the instruction DRAFT OUTPUT PAPER, you step fully into the role of academic secretary. You will be given the full forum transcript. Your task is to write a conventional academic paper in prose throughout — no bullet points, no headers other than standard section titles, no lists. Structure it as follows: Abstract (100 words); Introduction presenting the theoretical question; a section on each specialist framework as revealed in the discussion; a section identifying the key points of genuine theoretical conflict; a Conclusion noting what empirical work would be needed to adjudicate between the frameworks. Write with scholarly precision. Do not declare winners. Preserve the incommensurabilities."""
}

SPECIALIST_SEQUENCE = ['genetics', 'systems', 'predictive']

SPEAKER_LABELS = {
    'genetics':     ('🧬', 'Geneticist'),
    'systems':      ('🌊', 'DS Theorist'),
    'predictive':   ('🧠', 'Predictive Cognitivist'),
    'orchestrator': ('📋', 'Orchestrator'),
    'human':        ('👤', 'Forum Chair'),
}

# Agent name aliases for speech parsing
AGENT_NAME_MAP = {
    'geneticist': 'genetics', 'genetics': 'genetics', 'genetic': 'genetics', 'crick': 'genetics',
    'ds theorist': 'systems', 'systems': 'systems', 'dynamic systems': 'systems',
    'systems theorist': 'systems', 'thelen': 'systems',
    'predictive cognitivist': 'predictive', 'predictive': 'predictive',
    'friston': 'predictive', 'cognitivist': 'predictive', 'bayesian': 'predictive',
    'orchestrator': 'orchestrator', 'coordinator': 'orchestrator', 'chair': 'orchestrator',
}

RECIPIENT_MAP = {
    "Orchestrator": "orchestrator",
    "Geneticist": "genetics",
    "DS Theorist": "systems",
    "Predictive Cognitivist": "predictive",
}

# =============================================================================
# AUDIO UTILITIES
# =============================================================================

def transcribe_audio(audio_bytes: bytes) -> str:
    """Send audio bytes to OpenAI Whisper API and return transcript."""
    if st.session_state.oai_client is None:
        return ""
    if not audio_bytes:
        return ""
    try:
        import io
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "recording.wav"
        transcript = st.session_state.oai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="en"
        )
        return transcript.text
    except Exception as e:
        st.error(f"Transcription error: {e}")
        return ""


def parse_agent_from_transcript(text: str) -> tuple[str | None, str]:
    """Extract the intended agent from a spoken query. Returns (agent_key, cleaned_text)."""
    text_lower = text.lower().strip()
    sorted_aliases = sorted(AGENT_NAME_MAP.keys(), key=len, reverse=True)

    for alias in sorted_aliases:
        pattern = r'^' + re.escape(alias) + r'[\s,\-–—:\.]*'
        match = re.match(pattern, text_lower)
        if match:
            agent_key = AGENT_NAME_MAP[alias]
            cleaned = text[match.end():].strip()
            st.toast(f"Agent parsed: '{alias}' → {agent_key}", icon="✅")
            return agent_key, cleaned

    st.toast(f"No agent detected. First 40 chars: '{text[:40]}'", icon="⚠️")
    return None, text


def find_drill_down_target(reference_text: str) -> dict | None:
    """Find the most semantically similar passage in history to the spoken reference."""
    if not st.session_state.history or not reference_text.strip():
        return None

    def token_overlap(a: str, b: str) -> float:
        stopwords = {'the', 'a', 'an', 'is', 'it', 'of', 'to', 'in', 'and',
                     'that', 'this', 'was', 'for', 'on', 'are', 'with', 'you',
                     'your', 'but', 'not', 'what', 'how', 'do', 'does', 'by',
                     'at', 'be', 'have', 'has', 'from', 'or', 'their', 'its'}
        tokens_a = {w for w in re.findall(r'\b\w+\b', a.lower()) if w not in stopwords and len(w) > 2}
        tokens_b = {w for w in re.findall(r'\b\w+\b', b.lower()) if w not in stopwords and len(w) > 2}
        if not tokens_a or not tokens_b:
            return 0.0
        return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)

    best_score = 0.0
    best_item = None

    for item in st.session_state.history:
        if item['spec'] not in SPECIALIST_SEQUENCE:
            continue
        score = token_overlap(reference_text, item['text'])
        if score > best_score:
            best_score = score
            best_item = item

    if best_score >= SIMILARITY_THRESHOLD and best_item:
        return best_item
    return None


def synthesise_speech(text: str, agent_key: str) -> bytes | None:
    """Call ElevenLabs to synthesise the agent's response. Returns audio bytes or None."""
    if st.session_state.el_client is None:
        return None

    voice_id = ELEVENLABS_VOICE_IDS.get(agent_key)
    if not voice_id:
        return None

    try:
        st.toast(f"Synthesising {agent_key} | model: eleven_turbo_v2_5 | chars: {len(text)}", icon="🔊")
        audio = st.session_state.el_client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            voice_settings=VoiceSettings(
                stability=0.42,
                similarity_boost=0.72,
                style=0.15,
                use_speaker_boost=True
            ),
            model_id="eleven_turbo_v2_5"
        )
        return b"".join(audio)
    except Exception as e:
        st.error(f"ElevenLabs synthesis error: {e}")
        return None


# =============================================================================
# CORE AGENT CALL
# =============================================================================

def call_agent(spec: str, user_message: str, prior_turn_text: str | None = None) -> str:
    """Call the specified agent and return the response text."""
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


def post_to_history(spec: str, text: str):
    """Add an entry to the conversation history."""
    st.session_state.history.append({
        'spec': spec,
        'text': text,
        'timestamp': datetime.now().strftime("%H:%M"),
    })


def fire_query(target_spec: str, query_text: str, prior_turn_text: str | None = None) -> tuple[str, bytes | None]:
    """
    Unified query firing: call agent, post to history, synthesise audio if in audio mode.
    Returns (response_text, audio_bytes_or_None).
    """
    _, label = SPEAKER_LABELS[target_spec]
    post_to_history('human', query_text)

    with st.spinner(f"{label} is responding…"):
        response_text = call_agent(target_spec, query_text, prior_turn_text)

    post_to_history(target_spec, response_text)

    audio_bytes = None
    if st.session_state.audio_mode and st.session_state.el_client:
        with st.spinner(f"Synthesising {label}'s voice…"):
            audio_bytes = synthesise_speech(response_text, target_spec)
        

    return response_text, audio_bytes


# =============================================================================
# PAGE HEADER
# =============================================================================

st.title("🏛️ The Novum Lyceum")
st.markdown("*A Platform for Theoretical Framework Integration through Artificial Deliberation*")

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:

    # --- API connections ---
    if st.session_state.llm is None:
        try:
            anthropic_key = os.environ.get("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY", None)
            st.session_state.llm = ChatAnthropic(
                model="claude-sonnet-4-6",
                api_key=anthropic_key
            )
        except Exception:
            st.error("Anthropic API key not configured.")

    if st.session_state.oai_client is None:
        try:
            oai_key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
            st.session_state.oai_client = openai.OpenAI(api_key=oai_key)
        except Exception:
            st.warning("OpenAI key not set — voice input disabled.")

    if st.session_state.el_client is None:
        try:
            el_key = os.environ.get("ELEVENLABS_API_KEY") or st.secrets.get("ELEVENLABS_API_KEY", None)
            st.session_state.el_client = ElevenLabs(api_key=el_key)
        except Exception as e:
            st.warning(f"ElevenLabs not connected: {e}")

    # Connection status
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.caption("Claude " + ("✅" if st.session_state.llm else "❌"))
    with col_b:
        st.caption("Whisper " + ("✅" if st.session_state.oai_client else "❌"))
    with col_c:
        st.caption("11L " + ("✅" if st.session_state.el_client else "❌"))

    st.markdown("---")

    # Audio mode toggle
    st.session_state.audio_mode = st.toggle(
        "🎙️ Audio mode",
        value=st.session_state.audio_mode,
        help="Enable voice input (Whisper) and spoken agent responses (ElevenLabs)"
    )

    st.markdown("---")

    # Session management
    if st.button("Clear transcript"):
        st.session_state.history = []
        st.session_state.drill_queue = []
        st.session_state.dd_pending = None
        st.session_state.pending_audio = None
        st.rerun()

    st.markdown("---")

    # Transcript download
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
        st.download_button(
            label="📄 Download transcript",
            data="\n".join(lines),
            file_name=filename,
            mime="text/plain"
        )
    else:
        st.caption("No transcript to download yet.")

    st.markdown("---")

    # Drill-down queue
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

    # Draft paper
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


# =============================================================================
# MAIN INTERFACE
# =============================================================================

if not st.session_state.llm:
    st.info("Enter API keys in the sidebar to begin.")
    st.stop()

# =============================================================================
# AUDIO PLAYBACK — RENDERED FIRST, OUTSIDE ALL CONDITIONALS
# =============================================================================
# This is the critical fix: audio playback must happen early in the render cycle,
# not buried inside conditional blocks that may not execute after st.rerun()

if st.session_state.pending_audio is not None:
    _, agent_label = SPEAKER_LABELS.get(
        st.session_state.pending_audio_agent or 'orchestrator', ('', 'Agent')
    )
    st.markdown(f"### 🔊 {agent_label} is speaking:")
    st.audio(st.session_state.pending_audio, format="audio/mp3", autoplay=True)
    if st.button("✕ Dismiss audio", key="dismiss_audio_main"):
        st.session_state.pending_audio = None
        st.session_state.pending_audio_agent = None
        st.rerun()
    st.markdown("---")

# =============================================================================
# AUDIO INPUT PANEL (voice mode)
# =============================================================================

if st.session_state.audio_mode:
    st.markdown('<div class="audio-panel">', unsafe_allow_html=True)
    st.markdown("### 🎙️ Voice Input")

    if not st.session_state.oai_client:
        st.warning("OpenAI API key required for voice input (Whisper transcription).")
    else:
        audio_input = st.audio_input("🎙️ Press to record your query")

        if audio_input is not None:
            audio_hash = hash(audio_input.getvalue())
            if audio_hash != st.session_state.get('last_audio_hash', None) and not st.session_state.transcription:
                st.session_state.last_audio_hash = audio_hash
                st.session_state.pending_audio = None
                st.session_state.pending_audio_agent = None
                st.session_state.audio_status = 'transcribing'
                with st.spinner("Transcribing…"):
                    transcript_text = transcribe_audio(audio_input.getvalue())

                st.session_state.transcription = transcript_text
                st.session_state.audio_status = 'idle'

                agent_key, cleaned_query = parse_agent_from_transcript(transcript_text)
                st.session_state.parsed_agent = agent_key
                st.session_state.parsed_query = cleaned_query

                drill_triggers = ['referring to', 'you said', 'you mentioned', 'what you said',
                                  'that point about', 'earlier comment', 'the passage about',
                                  'drill down on', 'go deeper on', 'follow up on']
                drill_ref = None
                for trigger in drill_triggers:
                    if trigger in cleaned_query.lower():
                        drill_ref = cleaned_query
                        break
                st.session_state.parsed_drill_ref = drill_ref

                if agent_key:
                    st.session_state.auto_fire_ready = True

                st.rerun()

        # Show transcription and parsed intent
        if st.session_state.transcription:

            # --- AUTO-FIRE BLOCK ---
            if st.session_state.auto_fire_ready:
                st.session_state.auto_fire_ready = False
                final_agent = st.session_state.parsed_agent
                query_to_fire = st.session_state.parsed_query.strip() or st.session_state.transcription.strip()
                if not query_to_fire:
                    st.error("Auto-fire aborted: query is empty after transcription.")
                    st.rerun()
                prior_text = None
                if st.session_state.dd_pending:
                    prior_text = st.session_state.dd_pending['text']
                    st.session_state.dd_pending = None
                response_text, audio_bytes = fire_query(final_agent, query_to_fire, prior_text)
                st.session_state.transcription = ''
                st.session_state.parsed_agent = None
                st.session_state.parsed_drill_ref = None
                st.session_state.parsed_query = ''
                st.session_state.audio_input_processed = False
                if audio_bytes:
                    st.session_state.pending_audio = audio_bytes
                    st.session_state.pending_audio_agent = final_agent
                st.rerun()
            # --- END AUTO-FIRE BLOCK ---

            st.markdown("**Transcription:**")
            edited_transcript = st.text_area(
                "Transcription (editable):",
                value=st.session_state.transcription,
                height=80,
                label_visibility="collapsed",
                key="transcript_edit"
            )

            agent_names = {
                'genetics': 'Geneticist',
                'systems': 'DS Theorist',
                'predictive': 'Predictive Cognitivist',
                'orchestrator': 'Orchestrator',
            }
            if st.session_state.parsed_agent:
                detected_name = agent_names.get(st.session_state.parsed_agent, 'Unknown')
                st.success(f"Addressed to: **{detected_name}**")
            else:
                st.warning("No agent detected in transcript. Please select manually below.")

            if st.session_state.parsed_drill_ref:
                matched_item = find_drill_down_target(st.session_state.parsed_drill_ref)
                if matched_item:
                    _, matched_label = SPEAKER_LABELS.get(matched_item['spec'], ('', 'Unknown'))
                    preview = matched_item['text'][:100]
                    st.info(f"Drill-down reference matched to {matched_label}: \"{preview}…\"")
                    if st.button("✅ Confirm drill-down match"):
                        st.session_state.dd_pending = {
                            'speaker': matched_label,
                            'text': matched_item['text']
                        }
                        st.rerun()
                    if st.button("✕ Not this — select manually"):
                        st.session_state.parsed_drill_ref = None
                        st.rerun()
                else:
                    st.warning("Drill-down reference detected but no close match found.")

            manual_agent = st.selectbox(
                "Address to (override):",
                ["— auto-detected —", "Geneticist", "DS Theorist", "Predictive Cognitivist", "Orchestrator"],
                key="manual_agent_select"
            )

            col_fire, col_clear = st.columns([1, 1])
            with col_fire:
                fire_btn = st.button("🔊 Fire query", type="primary", key="voice_fire")
            with col_clear:
                clear_btn = st.button("✕ Clear", key="voice_clear")

            if clear_btn:
                st.session_state.transcription = ''
                st.session_state.parsed_agent = None
                st.session_state.parsed_drill_ref = None
                st.session_state.parsed_query = ''
                st.rerun()

            if fire_btn:
                if manual_agent != "— auto-detected —":
                    final_agent = RECIPIENT_MAP[manual_agent]
                elif st.session_state.parsed_agent:
                    final_agent = st.session_state.parsed_agent
                else:
                    st.error("Please select an agent to address.")
                    st.stop()

                query_to_fire = edited_transcript.strip() or st.session_state.parsed_query.strip()
                if not query_to_fire:
                    st.error("Query is empty.")
                    st.stop()

                prior_text = None
                if st.session_state.dd_pending:
                    prior_text = st.session_state.dd_pending['text']
                    st.session_state.dd_pending = None

                st.session_state.audio_status = 'generating'
                response_text, audio_bytes = fire_query(final_agent, query_to_fire, prior_text)
                st.session_state.audio_status = 'idle'

                # Clear transcription state
                st.session_state.transcription = ''
                st.session_state.parsed_agent = None
                st.session_state.parsed_drill_ref = None
                st.session_state.parsed_query = ''

                # Store audio for playback
                if audio_bytes:
                    st.session_state.pending_audio = audio_bytes
                    st.session_state.pending_audio_agent = final_agent

                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")

# =============================================================================
# DRILL-DOWN PANEL
# =============================================================================

if st.session_state.dd_pending:
    pending = st.session_state.dd_pending
    preview = (pending['text'][:120] + '...') if len(pending['text']) > 120 else pending['text']
    st.info(f'**Drill-down ready:** "{preview}"')

    dd_instruction = st.text_area(
        "Your drill-down instruction:",
        height=100,
        placeholder="E.g., 'Clarify what you mean by phase transition here'",
        key="dd_custom_instruction"
    )
    dd_recipient = st.selectbox(
        "Address drill-down to:",
        ["Geneticist", "DS Theorist", "Predictive Cognitivist", "Orchestrator"],
        key="dd_recipient"
    )

    col_fire, col_cancel = st.columns([1, 1])
    with col_fire:
        if st.button("🔍 Fire drill-down", type="primary", key="dd_fire"):
            if not dd_instruction.strip():
                st.warning("Please provide an instruction for the drill-down.")
            else:
                dd_query = (
                    f"The following passage from the {pending['speaker']}'s contribution "
                    f"has been flagged for follow-up:\n\n\"{pending['text']}\"\n\n"
                    f"Your instruction: {dd_instruction.strip()}"
                )
                target_spec = RECIPIENT_MAP[dd_recipient]
                post_to_history(
                    'human',
                    f"[Drill-down to {dd_recipient}] Re: \"{pending['text'][:50]}...\"\n\n{dd_instruction.strip()}"
                )

                st.session_state.audio_status = 'generating'
                _, label = SPEAKER_LABELS[target_spec]
                with st.spinner(f"{label} is responding…"):
                    response_text = call_agent(target_spec, dd_query)
                post_to_history(target_spec, response_text)

                audio_bytes = None
                if st.session_state.audio_mode and st.session_state.el_client:
                    with st.spinner(f"Synthesising {label}'s voice…"):
                        audio_bytes = synthesise_speech(response_text, target_spec)

                st.session_state.audio_status = 'idle'
                st.session_state.dd_pending = None

                if audio_bytes:
                    st.session_state.pending_audio = audio_bytes
                    st.session_state.pending_audio_agent = target_spec

                st.rerun()

    with col_cancel:
        if st.button("✕ Cancel", key="dd_cancel"):
            st.session_state.dd_pending = None
            st.rerun()

    st.markdown("---")

# =============================================================================
# TEXT INPUT PANEL
# =============================================================================

with st.expander("⌨️ Text input", expanded=True):

    recipient = st.selectbox(
        "Address to:",
        ["Geneticist", "DS Theorist", "Predictive Cognitivist", "Orchestrator"],
        key="text_recipient"
    )

    uploaded_pdf = st.file_uploader(
        "Upload anchor paper (PDF):",
        type="pdf",
        help="Paper text will be appended to your query automatically."
    )
    pdf_text = ""
    if uploaded_pdf is not None:
        try:
            reader = PdfReader(io.BytesIO(uploaded_pdf.read()))
            pdf_text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
            st.success(f"Paper loaded: {uploaded_pdf.name} ({len(reader.pages)} pages)")
        except Exception as e:
            st.error(f"Could not read PDF: {e}")

    if st.session_state.clear_flag:
        st.session_state.query_box = ""
        st.session_state.clear_flag = False

    query = st.text_area(
        "Your query:",
        height=150,
        placeholder="Type your query here.",
        key="query_box"
    )

    if st.button("Submit", type="primary", key="text_submit"):
        if query.strip():
            full_query = query.strip()
            if pdf_text:
                full_query = full_query + "\n\n---\nANCHOR PAPER:\n\n" + pdf_text

            target_spec = RECIPIENT_MAP[recipient]
            response_text, audio_bytes = fire_query(target_spec, full_query)
            st.session_state.clear_flag = True

            if audio_bytes:
                st.session_state.pending_audio = audio_bytes
                st.session_state.pending_audio_agent = target_spec

            st.rerun()
        else:
            st.warning("Please enter a query first.")

# =============================================================================
# TRANSCRIPT
# =============================================================================

st.markdown("---")
st.markdown("### 💬 Forum Transcript")
st.caption("Most recent exchange shown first. Use Ctrl+F to search.")

if st.session_state.history:
    for idx, item in reversed(list(enumerate(st.session_state.history))):
        icon, label = SPEAKER_LABELS.get(item['spec'], ('❓', item['spec'].title()))
        ts = item.get('timestamp', '')

        st.markdown(
            f'<div class="speaker-{item["spec"]}">'
            f'<strong>{icon} {label}</strong>'
            f'<span style="color:#888;font-size:0.85em;"> {ts}</span>'
            f'<br><br>{item["text"]}'
            f'</div>',
            unsafe_allow_html=True
        )

        # Drill-down flagging (text mode only)
        if item['spec'] in SPECIALIST_SEQUENCE and not st.session_state.audio_mode:
            flag_key = f"flag_{idx}_{st.session_state.flag_counter}"
            flag_text = st.text_input(
                "Flag passage for drill-down:",
                key=flag_key,
                placeholder="Paste a phrase to queue for follow-up…",
                label_visibility="collapsed"
            )
            if st.button("➕ Add to queue", key=f"add_{idx}"):
                if flag_text.strip():
                    st.session_state.drill_queue.append({
                        'speaker': label,
                        'text': flag_text.strip()
                    })
                    st.session_state.flag_counter += 1
                    st.rerun()
else:
    st.info("No exchanges yet. Address your first query above.")

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown(
    '<p style="text-align:center; color:#8B7355; font-style:italic;">'
    'The Novum Lyceum — Where theoretical frameworks engage in productive discourse'
    '</p>',
    unsafe_allow_html=True
)

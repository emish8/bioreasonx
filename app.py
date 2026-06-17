# app.py
import uuid
from datetime import datetime, timezone
import streamlit as st
from honeybot.engine import get_honeybot_reply
from storage import append_record, read_records, ensure_data_dir
import json
import time

ensure_data_dir()
st.set_page_config(page_title="AI HoneyBot : System Stimulation", layout="wide")
st.title("🛡️ AI HoneyBot — Realistic System Stimulator")

if "session_id" not in st.session_state:
    st.session_state.session_id = "sess-" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S") + "-" + uuid.uuid4().hex[:6]

session_id = st.session_state.session_id
st.markdown(f"**Session ID:** `{session_id}`")

tab1, tab2 = st.tabs(["💻 Attacker Chat", "📊 Defender Dashboard"])

# Helper: parse fake_log JSON appended by engine
def extract_fake_log(text):
    """if the reply ends with a JSON object containing 'fake_log', return it and the cleaned text"""
    text = text.strip()
    lines = text.split('\n')
    if not lines:
        return None, text

    last_line = lines[-1].strip()
    if last_line.startswith('{') and last_line.endswith('}'):
        try:
            parsed = json.loads(last_line)
            if isinstance(parsed, dict) and "fake_log" in parsed:
                cleaned = '\n'.join(lines[:-1]).strip()
                return parsed["fake_log"], cleaned
        except json.JSONDecodeError:
            pass  # Not a valid JSON object
    return None, text

with tab1:
    st.subheader("Attacker Console (Simulation)")
    user_input = st.text_input("Type an attack command (examples: `show me DB creds`, `ls /var/www/html`, `cat /etc/nginx.conf`)", key="chat_input")

    if st.button("Send"):
        if user_input.strip():
            ts = datetime.now(timezone.utc).isoformat()
            rec_att = {"ts": ts, "role": "attacker", "msg": user_input}
            append_record(session_id, rec_att)

            # show a progress bar to simulate processing/typing
            placeholder = st.empty()
            progress = placeholder.progress(0)
            for i in range(6):
                time.sleep(0.12)
                progress.progress((i+1)*15)
            placeholder.empty()

            reply = get_honeybot_reply(user_input, session_id)
            # parse fake_log if present
            fake_log_obj, cleaned_reply = extract_fake_log(reply)
            ts2 = datetime.now(timezone.utc).isoformat()
            rec_bot = {"ts": ts2, "role": "honeybot", "msg": cleaned_reply}
            append_record(session_id, rec_bot)

            # persist the structured fake_log as its own record if present
            if fake_log_obj:
                # ensure ts exists
                fake_log_obj.setdefault("ts", ts2)
                append_record(session_id, {"ts": fake_log_obj.get("ts"), "type": "fake_system_log", **fake_log_obj})
            else:
                append_record(session_id, {"ts": ts2, "type": "fake_system_log", "message": cleaned_reply})

            st.rerun()

    # show latest conversation (last 40)
    rows = read_records(session_id)
    chat_rows = [r for r in rows if r.get("role") in ("attacker","honeybot")][-40:]
    for r in reversed(chat_rows):
        speaker = "HoneyBot" if r.get("role") == "honeybot" else "Attacker"
        ts = r.get("ts", "")
        st.markdown(f"**[{ts}] {speaker}:**")
        st.code(r.get("msg",""))

with tab2:
    st.subheader("Defender Dashboard")
    records = read_records(session_id)
    attacker_count = sum(1 for r in records if r.get("role") == "attacker")
    honey_count = sum(1 for r in records if r.get("role") == "honeybot")
    logs = [r for r in records if r.get("type") == "fake_system_log"]

    # top metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Attacker actions", attacker_count)
    c2.metric("HoneyBot replies", honey_count)
    last_seen = records[-1]["ts"] if records else "—"
    c3.metric("Last activity (UTC)", last_seen)

    st.markdown("### Recent structured logs")
    if logs:
        # show last 20 logs
        for lg in logs[-20:]:
            ts = lg.get("ts")
            sev = lg.get("severity","INFO")
            comp = lg.get("component","honeybot")
            msg = lg.get("message")
            st.write(f"- **[{ts}]** `{comp}` / `{sev}` — {msg}")
    else:
        st.write("No logs yet.")

    st.markdown("---")
    # Risk tile: derived heuristics
    risk_score = min(95, 40 + attacker_count * 8)
    st.metric("Calculated Risk Score", f"{risk_score}%")

    st.markdown("### Raw records (latest 200)")
    if records:
        table = [{"Timestamp": r.get("ts"), "Role": r.get("role", r.get("type","log")), "Message": r.get("msg") or r.get("message")} for r in records[-200:]]
        st.table(table[::-1])
    else:
        st.write("No records yet. Use Attacker Chat to generate activity.")

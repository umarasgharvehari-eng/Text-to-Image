import io, time, traceback
import streamlit as st
import torch
from PIL import Image
from diffusers import AutoPipelineForText2Image

st.set_page_config(page_title="Studio • Midjourney-style", page_icon="✨", layout="wide")

st.markdown("""
<style>
:root{
  --bg0:#070915; --bg1:#0b1020;
  --card:rgba(255,255,255,.045);
  --stroke:rgba(255,255,255,.10);
  --text:#e5e7eb; --muted:rgba(229,231,235,.70);
}
[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(1200px 600px at 10% 0%, rgba(124,58,237,.20), transparent 60%),
    radial-gradient(900px 550px at 90% 10%, rgba(34,197,94,.14), transparent 55%),
    radial-gradient(900px 650px at 50% 100%, rgba(96,165,250,.10), transparent 60%),
    linear-gradient(180deg, var(--bg1) 0%, var(--bg0) 100%);
  color: var(--text);
}
.block-container{ max-width: 1260px; padding-top: 1rem; padding-bottom: 2rem; }
.topbar{
  display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap;
  border:1px solid var(--stroke);
  background: rgba(255,255,255,.035);
  box-shadow: 0 18px 60px rgba(0,0,0,.45);
  border-radius: 18px;
  padding: 14px 16px;
  margin-bottom: 14px;
}
.pill{
  display:inline-flex; align-items:center; gap:8px;
  padding:6px 10px; border-radius:999px;
  border:1px solid rgba(124,58,237,.35);
  background: rgba(124,58,237,.12);
  font-size: 12px;
}
.card{
  border:1px solid var(--stroke);
  background: var(--card);
  border-radius: 18px;
  padding: 14px;
  box-shadow: 0 18px 55px rgba(0,0,0,.38);
}
.h{ font-size: 14px; font-weight: 800; }
.hint{ font-size:12px; color: var(--muted); }

div[data-testid="stTextArea"] textarea{
  background: rgba(15,23,42,.82) !important;
  color: var(--text) !important;
  border: 1px solid rgba(255,255,255,.12) !important;
  border-radius: 14px !important;
  font-size: 15px !important;
  line-height: 1.35 !important;
  padding: 14px !important;
  min-height: 130px !important;
}
div[data-testid="stTextArea"] textarea::placeholder{ color: rgba(229,231,235,.45) !important; }

.stButton button{
  width:100%;
  border-radius: 14px;
  padding: .72rem 1rem;
  border: 1px solid rgba(255,255,255,.10);
  background: linear-gradient(135deg, rgba(124,58,237,.95), rgba(96,165,250,.70));
  color: #0b1020;
  font-weight: 950;
}
.stDownloadButton button{
  width:100%;
  border-radius: 14px;
  padding: .65rem 1rem;
  border: 1px solid rgba(255,255,255,.14);
  background: rgba(255,255,255,.06);
  color: var(--text);
  font-weight: 850;
}
.toast{
  border:1px solid rgba(34,197,94,.35);
  background: rgba(34,197,94,.12);
  border-radius: 14px;
  padding: 10px 12px;
  margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="topbar">
  <div style="display:flex; align-items:center; gap:10px;">
    <div style="font-size:22px;">✨</div>
    <div>
      <div style="font-size:22px; font-weight:900; letter-spacing:-0.02em;">Studio</div>
      <div style="font-size:12px; opacity:.75;">Midjourney-style • Open-weights • No API keys</div>
    </div>
  </div>
  <div class="pill">🖼️ Text → Image • Turbo</div>
</div>
""", unsafe_allow_html=True)

IS_GPU = torch.cuda.is_available()

# ✅ CPU: use sd-turbo always
MODEL_ID = "stabilityai/sd-turbo" if not IS_GPU else "stabilityai/sdxl-turbo"

def pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

@st.cache_resource(show_spinner=False)
def get_pipe(model_id: str):
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id,
        torch_dtype=dtype,
        variant="fp16" if torch.cuda.is_available() else None,
        use_safetensors=True,
    )
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    return pipe

# --- state ---
if "history" not in st.session_state:
    st.session_state.history = []  # {prompt, files:[bytes], ts}
if "prompt" not in st.session_state:
    st.session_state.prompt = ""

# sanitize history
clean = []
for item in st.session_state.history:
    files = item.get("files")
    if isinstance(files, list) and files and isinstance(files[0], (bytes, bytearray)):
        clean.append(item)
st.session_state.history = clean

# ---------- Sidebar (FAST CPU settings) ----------
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    st.write("GPU:", "✅" if IS_GPU else "❌ (CPU mode)")
    st.caption(f"Model: {MODEL_ID}")

    if IS_GPU:
        steps = st.selectbox("Steps", [2, 3, 5], index=1)
        size = st.selectbox("Size", ["768x768", "1024x576", "576x1024"], index=0)
        w, h = map(int, size.split("x"))
        n_images = st.slider("Grid images", 1, 4, 4)
    else:
        # ✅ FAST defaults
        steps = 1
        w, h = 384, 384
        n_images = 1
        st.info("CPU Fast Mode: 384×384 • 1 step • 1 image")

    seed = st.number_input("Seed", min_value=0, max_value=2_147_483_647, value=42, step=1)

left, right = st.columns([1.1, 1], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div style="display:flex; justify-content:space-between; align-items:flex-end; gap:10px; flex-wrap:wrap;">'
                '<div class="h">✍️ Imagine</div><div class="hint">subject • style • lighting • lens • mood</div></div>',
                unsafe_allow_html=True)

    prompt = st.text_area(
        "Prompt",
        value=st.session_state.prompt or "A cinematic photo of a neon-lit street market in Tokyo, rain reflections, ultra detailed, 35mm, shallow depth of field, moody atmosphere",
        label_visibility="collapsed",
    )
    st.session_state.prompt = prompt

    c1, c2 = st.columns([1.2, 1])
    with c1:
        generate = st.button("✨ Generate")
    with c2:
        clear = st.button("🧹 Clear history")

    st.markdown('</div>', unsafe_allow_html=True)

if clear:
    st.session_state.history = []
    st.rerun()

# ---------- Generate ----------
if generate:
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        try:
            with st.spinner("Loading model (first time can take a while)..."):
                pipe = get_pipe(MODEL_ID)

            with st.spinner("Generating (CPU is slow; please wait)..."):
                g = torch.Generator(device="cuda") if IS_GPU else torch.Generator()
                g = g.manual_seed(int(seed))

                files = []
                for _ in range(int(n_images)):
                    result = pipe(
                        prompt=prompt.strip(),
                        num_inference_steps=int(steps),
                        guidance_scale=0.0,
                        width=int(w),
                        height=int(h),
                        generator=g,
                    )
                    pil_img = result.images[0]
                    files.append(pil_to_png_bytes(pil_img))

            st.session_state.history.insert(0, {
                "prompt": prompt.strip(),
                "files": files,
                "ts": time.strftime("%H:%M:%S"),
            })
            st.rerun()

        except Exception as e:
            st.error("Generation failed (CPU limits / memory / timeout).")
            st.code("".join(traceback.format_exception(type(e), e, e.__traceback__)))

# ---------- Gallery ----------
with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div style="display:flex; justify-content:space-between; align-items:flex-end; gap:10px; flex-wrap:wrap;">'
                '<div class="h">🖼️ Gallery</div><div class="hint">Latest render at the top</div></div>',
                unsafe_allow_html=True)

    if st.session_state.history:
        latest = st.session_state.history[0]
        st.markdown('<div class="toast">✅ Render completed</div>', unsafe_allow_html=True)
        st.caption(latest["prompt"])

        file0 = latest["files"][0]
        st.image(file0, use_container_width=True)
        st.download_button("⬇️ Download PNG", data=file0, file_name="generated.png", mime="image/png")
    else:
        st.info("No renders yet. Write a prompt and click **Generate**.")

    st.markdown('</div>', unsafe_allow_html=True)

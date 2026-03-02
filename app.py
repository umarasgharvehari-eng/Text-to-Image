import os, io, time, base64, traceback
import streamlit as st
import torch
from PIL import Image

# ----------------- Streamlit config -----------------
st.set_page_config(page_title="Studio • Midjourney-style", page_icon="✨", layout="wide")

# ----------------- Cache dirs (reduce re-downloads) -----------------
APP_CACHE = os.path.join(os.getcwd(), ".cache")
HF_HOME = os.path.join(APP_CACHE, "huggingface")
os.makedirs(HF_HOME, exist_ok=True)

os.environ["HF_HOME"] = HF_HOME
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(HF_HOME, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_HOME, "transformers")

IS_GPU = torch.cuda.is_available()

# ----------------- Models -----------------
# CPU: prefer sd-turbo. If it fails, fallback to SD 1.5 (heavier but very stable).
PRIMARY_MODEL = "stabilityai/sd-turbo" if not IS_GPU else "stabilityai/sdxl-turbo"
FALLBACK_MODEL = "runwayml/stable-diffusion-v1-5"

# ----------------- UI (CSS) -----------------
st.markdown("""
<style>
:root{ --bg0:#070915; --bg1:#0b1020; --card:rgba(255,255,255,.045); --stroke:rgba(255,255,255,.10);
--text:#e5e7eb; --muted:rgba(229,231,235,.70);}
[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(1200px 600px at 10% 0%, rgba(124,58,237,.20), transparent 60%),
    radial-gradient(900px 550px at 90% 10%, rgba(34,197,94,.14), transparent 55%),
    radial-gradient(900px 650px at 50% 100%, rgba(96,165,250,.10), transparent 60%),
    linear-gradient(180deg, var(--bg1) 0%, var(--bg0) 100%);
  color: var(--text);
}
.block-container{ max-width:1260px; padding-top:1rem; padding-bottom:2rem;}
.topbar{
  display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap;
  border:1px solid var(--stroke); background: rgba(255,255,255,.035);
  box-shadow: 0 18px 60px rgba(0,0,0,.45);
  border-radius: 18px; padding:14px 16px; margin-bottom:14px;
}
.pill{
  display:inline-flex; align-items:center; gap:8px;
  padding:6px 10px; border-radius:999px;
  border:1px solid rgba(124,58,237,.35);
  background: rgba(124,58,237,.12); font-size: 12px;
}
.card{ border:1px solid var(--stroke); background: var(--card); border-radius:18px; padding:14px;
  box-shadow: 0 18px 55px rgba(0,0,0,.38); }
.h{ font-size:14px; font-weight:800; }
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
  width:100%; border-radius:14px; padding:.72rem 1rem;
  border:1px solid rgba(255,255,255,.10);
  background: linear-gradient(135deg, rgba(124,58,237,.95), rgba(96,165,250,.70));
  color:#0b1020; font-weight:950;
}
.stDownloadButton button{
  width:100%; border-radius:14px; padding:.65rem 1rem;
  border:1px solid rgba(255,255,255,.14);
  background: rgba(255,255,255,.06);
  color: var(--text); font-weight:850;
}
.toast{
  border:1px solid rgba(34,197,94,.35);
  background: rgba(34,197,94,.12);
  border-radius:14px; padding:10px 12px; margin-bottom:10px;
}
.imgbox{ border:1px solid rgba(255,255,255,.10); border-radius:16px; overflow:hidden; background: rgba(255,255,255,.03); }
small.muted{ color: rgba(229,231,235,.70); }
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="topbar">
  <div style="display:flex; align-items:center; gap:10px;">
    <div style="font-size:22px;">✨</div>
    <div>
      <div style="font-size:22px; font-weight:900; letter-spacing:-0.02em;">Studio</div>
      <div style="font-size:12px; opacity:.75;">Midjourney-style • Open-weights • No API keys</div>
    </div>
  </div>
  <div class="pill">Model: {PRIMARY_MODEL}</div>
</div>
""", unsafe_allow_html=True)

# ----------------- Helpers -----------------
def pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def b64_from_bytes(b: bytes) -> str:
    return base64.b64encode(b).decode("utf-8")

def show_b64(b64: str):
    st.markdown(
        f'<div class="imgbox"><img src="data:image/png;base64,{b64}" style="width:100%; display:block;"></div>',
        unsafe_allow_html=True
    )

@st.cache_resource(show_spinner=False)
def load_pipe(model_id: str):
    # Lazy import: prevents startup crashes
    from diffusers import AutoPipelineForText2Image
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

def generate_with_model(model_id: str, prompt: str, seed: int, steps: int, width: int, height: int):
    pipe = load_pipe(model_id)

    # Turbo best practice: guidance 0.0
    if "turbo" in model_id:
        guidance = 0.0
    else:
        guidance = 7.0

    gen = torch.Generator(device="cuda") if IS_GPU else torch.Generator()
    gen = gen.manual_seed(int(seed))

    out = pipe(
        prompt=prompt,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        width=int(width),
        height=int(height),
        generator=gen,
    )
    img = out.images[0]
    png = pil_to_png_bytes(img)
    if not isinstance(png, (bytes, bytearray)) or len(png) < 1000:
        raise RuntimeError("Invalid PNG output (empty/too small).")
    return png, guidance

# ----------------- State -----------------
if "history" not in st.session_state:
    st.session_state.history = []  # {prompt, bytes, b64, model, steps, size, seconds}
if "prompt" not in st.session_state:
    st.session_state.prompt = ""

# sanitize history
st.session_state.history = [
    x for x in st.session_state.history
    if isinstance(x.get("b64"), str) and len(x.get("b64")) > 50 and isinstance(x.get("bytes"), (bytes, bytearray))
]

# ----------------- Sidebar -----------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    st.write("GPU:", "✅" if IS_GPU else "❌ (CPU mode)")
    st.caption("Streamlit Cloud is usually CPU-only, so generation can be slow.")

    if IS_GPU:
        steps = st.selectbox("Steps", [2, 3, 5], index=1)
        size = st.selectbox("Size", ["768x768", "1024x576", "576x1024"], index=0)
        width, height = map(int, size.split("x"))
    else:
        # CPU safe & faster
        steps = st.selectbox("Steps (CPU)", [1, 2, 3], index=1)
        size = st.selectbox("Size (CPU)", ["384x384", "448x448", "512x512"], index=0)
        width, height = map(int, size.split("x"))

    seed = st.number_input("Seed", min_value=0, max_value=2_147_483_647, value=42, step=1)
    st.divider()
    st.caption("Tip: Start with **384×384** and **2 steps** on CPU.")

# ----------------- Main layout -----------------
left, right = st.columns([1.1, 1], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div style="display:flex; justify-content:space-between; align-items:flex-end; gap:10px; flex-wrap:wrap;">'
                '<div class="h">✍️ Prompt</div><div class="hint">subject • style • lighting • lens • mood</div></div>',
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
        clear = st.button("🧹 Clear")

    st.markdown('</div>', unsafe_allow_html=True)

if clear:
    st.session_state.history = []
    st.rerun()

# ----------------- Generate (with fallback) -----------------
if generate:
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        try:
            t0 = time.time()
            with st.spinner("Loading + generating (first time can take a while on CPU)..."):
                # Try primary model first
                try:
                    png, guidance = generate_with_model(PRIMARY_MODEL, prompt.strip(), seed, steps, width, height)
                    used_model = PRIMARY_MODEL
                except Exception:
                    # Fallback if primary fails
                    png, guidance = generate_with_model(FALLBACK_MODEL, prompt.strip(), seed, steps, width, height)
                    used_model = FALLBACK_MODEL

            sec = round(time.time() - t0, 2)
            b64 = b64_from_bytes(png)

            st.session_state.history.insert(0, {
                "prompt": prompt.strip(),
                "bytes": png,
                "b64": b64,
                "model": used_model,
                "steps": steps,
                "size": f"{width}x{height}",
                "seconds": sec,
                "guidance": guidance,
            })
            st.rerun()

        except Exception as e:
            st.error("Generation failed (Streamlit Cloud CPU limits / timeout / memory).")
            st.code("".join(traceback.format_exception(type(e), e, e.__traceback__)))

# ----------------- Gallery -----------------
with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div style="display:flex; justify-content:space-between; align-items:flex-end; gap:10px; flex-wrap:wrap;">'
                '<div class="h">🖼️ Result</div><div class="hint">Latest render</div></div>',
                unsafe_allow_html=True)

    if st.session_state.history:
        item = st.session_state.history[0]
        st.markdown('<div class="toast">✅ Render completed</div>', unsafe_allow_html=True)
        st.caption(item["prompt"])
        st.markdown(f"<small class='muted'>Model: <b>{item['model']}</b> • Size: <b>{item['size']}</b> • Steps: <b>{item['steps']}</b> • Guidance: <b>{item['guidance']}</b> • Time: <b>{item['seconds']}s</b></small>", unsafe_allow_html=True)

        show_b64(item["b64"])

        st.download_button(
            "⬇️ Download PNG",
            data=item["bytes"],
            file_name="generated.png",
            mime="image/png",
        )
    else:
        st.info("No image yet. Enter a prompt and click **Generate**.")

    st.markdown("</div>", unsafe_allow_html=True)

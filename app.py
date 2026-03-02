import io, time
import streamlit as st
import torch
from PIL import Image
from diffusers import AutoPipelineForText2Image

# ---------------- Page config ----------------
st.set_page_config(page_title="Studio • Midjourney-style", page_icon="✨", layout="wide")

# ---------------- CSS: Midjourney-like dark UI ----------------
st.markdown("""
<style>
:root{
  --bg0:#070915;
  --bg1:#0b1020;
  --card:rgba(255,255,255,.045);
  --card2:rgba(255,255,255,.06);
  --stroke:rgba(255,255,255,.10);
  --stroke2:rgba(255,255,255,.16);
  --text:#e5e7eb;
  --muted:rgba(229,231,235,.70);
  --muted2:rgba(229,231,235,.55);
  --brand1:#7c3aed;  /* purple */
  --brand2:#22c55e;  /* green */
  --brand3:#60a5fa;  /* blue */
}

[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(1200px 600px at 10% 0%, rgba(124,58,237,.20), transparent 60%),
    radial-gradient(900px 550px at 90% 10%, rgba(34,197,94,.14), transparent 55%),
    radial-gradient(900px 650px at 50% 100%, rgba(96,165,250,.10), transparent 60%),
    linear-gradient(180deg, var(--bg1) 0%, var(--bg0) 100%);
  color: var(--text);
}
.block-container{ max-width: 1260px; padding-top: 1.0rem; padding-bottom: 2.0rem; }

/* Top bar */
.topbar{
  display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap;
  border:1px solid var(--stroke);
  background: rgba(255,255,255,.035);
  box-shadow: 0 18px 60px rgba(0,0,0,.45);
  border-radius: 18px;
  padding: 14px 16px;
  margin-bottom: 14px;
}
.brand{
  display:flex; align-items:center; gap:10px;
}
.pill{
  display:inline-flex; align-items:center; gap:8px;
  padding:6px 10px; border-radius:999px;
  border:1px solid rgba(124,58,237,.35);
  background: rgba(124,58,237,.12);
  color: var(--text);
  font-size: 12px;
}
.title{
  font-size: 22px; font-weight: 900; letter-spacing: -0.02em;
}
.sub{
  font-size: 12px; color: var(--muted);
}

/* Cards */
.card{
  border:1px solid var(--stroke);
  background: var(--card);
  border-radius: 18px;
  padding: 14px;
  box-shadow: 0 18px 55px rgba(0,0,0,.38);
}
.cardhead{
  display:flex; justify-content:space-between; align-items:flex-end; gap:10px; flex-wrap:wrap;
  margin-bottom: 10px;
}
.h{
  font-size: 14px; font-weight: 800; letter-spacing: .01em;
}
.hint{ font-size:12px; color: var(--muted); }

/* Prompt row */
.promptwrap{
  border:1px solid var(--stroke2);
  background: rgba(255,255,255,.04);
  border-radius: 16px;
  padding: 10px;
}
div[data-testid="stTextArea"] textarea{
  background: rgba(15,23,42,.82) !important;
  color: var(--text) !important;
  border: 1px solid rgba(255,255,255,.12) !important;
  border-radius: 14px !important;
  font-size: 15px !important;
  line-height: 1.35 !important;
  padding: 14px !important;
  min-height: 130px !important;
  box-shadow: inset 0 0 0 1px rgba(255,255,255,.04);
}
div[data-testid="stTextArea"] textarea::placeholder{ color: rgba(229,231,235,.45) !important; }
div[data-testid="stTextArea"] textarea:focus{
  border: 1px solid rgba(124,58,237,.70) !important;
  box-shadow: 0 0 0 1px rgba(124,58,237,.70);
  outline: none !important;
}

/* Buttons */
.stButton button{
  width:100%;
  border-radius: 14px;
  padding: .72rem 1rem;
  border: 1px solid rgba(255,255,255,.10);
  background: linear-gradient(135deg, rgba(124,58,237,.95), rgba(96,165,250,.70));
  color: #0b1020;
  font-weight: 950;
}
.stButton button:hover{ filter: brightness(1.06); transform: translateY(-1px); }

.stDownloadButton button{
  width:100%;
  border-radius: 14px;
  padding: .65rem 1rem;
  border: 1px solid rgba(255,255,255,.14);
  background: rgba(255,255,255,.06);
  color: var(--text);
  font-weight: 850;
}
.stDownloadButton button:hover{ filter: brightness(1.08); transform: translateY(-1px); }

/* Small chips */
.chips{ display:flex; flex-wrap:wrap; gap:8px; margin-top:10px; }
.chip{
  display:inline-flex; align-items:center; gap:8px;
  padding: 7px 10px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,.12);
  background: rgba(255,255,255,.05);
  font-size: 12px;
  color: var(--muted);
}

/* Result toast */
.toast{
  border:1px solid rgba(34,197,94,.35);
  background: rgba(34,197,94,.12);
  border-radius: 14px;
  padding: 10px 12px;
  margin-bottom: 10px;
}

/* Gallery grid styling */
.gallery{
  display:grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}
.imgcard{
  border:1px solid rgba(255,255,255,.10);
  background: rgba(255,255,255,.03);
  border-radius: 16px;
  padding: 10px;
}
.small{ font-size:12px; color: var(--muted); }
hr{ border:none; border-top:1px solid rgba(255,255,255,.10); margin: 12px 0; }

/* Sidebar */
section[data-testid="stSidebar"]{
  background: rgba(255,255,255,.03);
  border-right: 1px solid rgba(255,255,255,.08);
}
</style>
""", unsafe_allow_html=True)

# ---------------- Header ----------------
st.markdown("""
<div class="topbar">
  <div class="brand">
    <div style="font-size:22px;">✨</div>
    <div>
      <div class="title">Studio</div>
      <div class="sub">Midjourney-style • Open-weights • No API keys</div>
    </div>
  </div>
  <div class="pill">🖼️ Text → Image • SDXL-Turbo</div>
</div>
""", unsafe_allow_html=True)

# ---------------- Helpers ----------------
def image_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def make_generator(seed_value: int):
    g = torch.Generator(device="cuda") if torch.cuda.is_available() else torch.Generator()
    return g.manual_seed(int(seed_value))

# ---------------- Load pipeline ----------------
@st.cache_resource
def load_pipeline():
    model_id = "stabilityai/sdxl-turbo"
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

with st.spinner("Loading model (first time downloads weights)..."):
    pipe = load_pipeline()

# ---------------- Sidebar controls (MJ-ish presets) ----------------
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    quality = st.selectbox("Quality", ["Fast (2 steps)", "Balanced (3 steps)", "Quality (5 steps)"], index=1)
    steps = {"Fast (2 steps)":2, "Balanced (3 steps)":3, "Quality (5 steps)":5}[quality]

    aspect = st.selectbox("Aspect Ratio", ["1:1 (square)", "16:9 (landscape)", "9:16 (portrait)", "4:3", "3:4"], index=0)
    ar_map = {
        "1:1 (square)": (768, 768),
        "16:9 (landscape)": (1024, 576),
        "9:16 (portrait)": (576, 1024),
        "4:3": (896, 672),
        "3:4": (672, 896),
    }
    width, height = ar_map[aspect]

    seed = st.number_input("Seed", min_value=0, max_value=2_147_483_647, value=42, step=1)
    n_images = st.slider("Grid images", 1, 4, 4)
    st.markdown("---")
    st.markdown("### 🧠 Runtime")
    st.write("CUDA:", "✅ Enabled" if torch.cuda.is_available() else "❌ Not found")
    if torch.cuda.is_available():
        st.write("GPU:", torch.cuda.get_device_name(0))
    else:
        st.caption("Colab: Runtime → Change runtime type → GPU")

# ---------------- Session state ----------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: {prompt, images, ts}
if "prompt" not in st.session_state:
    st.session_state.prompt = ""

# ---------------- Layout ----------------
left, right = st.columns([1.1, 1], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="cardhead"><div class="h">✍️ Imagine</div><div class="hint">Write a prompt like Midjourney (subject • style • lighting • lens • mood)</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="promptwrap">', unsafe_allow_html=True)
    prompt = st.text_area(
        "Prompt",
        value=st.session_state.prompt or "A cinematic photo of a neon-lit street market in Tokyo, rain reflections, ultra detailed, 35mm, shallow depth of field, moody atmosphere",
        placeholder="Try: /imagine prompt: a cyberpunk cat astronaut, cinematic, volumetric light, ultra detailed --ar 16:9",
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.session_state.prompt = prompt

    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        generate = st.button("✨ Generate")
    with c2:
        remix = st.button("🔁 Remix seed")
    with c3:
        clear = st.button("🧹 Clear history")

    st.markdown("""
      <div class="chips">
        <div class="chip">Quality: <b style="color:#fff;">{q}</b></div>
        <div class="chip">Aspect: <b style="color:#fff;">{ar}</b></div>
        <div class="chip">Steps: <b style="color:#fff;">{st}</b></div>
        <div class="chip">Size: <b style="color:#fff;">{w}×{h}</b></div>
      </div>
    """.format(q=quality.split()[0], ar=aspect.split()[0], st=steps, w=width, h=height), unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# actions
if remix:
    st.session_state["seed"] = int(time.time()) % 2_147_483_647
    st.rerun()

if clear:
    st.session_state.history = []
    st.rerun()

# ---------------- Results panel ----------------
with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="cardhead"><div class="h">🖼️ Gallery</div><div class="hint">Latest render appears at the top</div></div>', unsafe_allow_html=True)

    if st.session_state.history:
        latest = st.session_state.history[0]
        st.markdown('<div class="toast">✅ Render completed</div>', unsafe_allow_html=True)
        st.caption(latest["prompt"])

        images = latest["images"]
        if len(images) == 1:
            st.image(images[0], use_container_width=True)
            st.download_button("⬇️ Download PNG", data=image_to_png_bytes(images[0]), file_name="mj_style.png", mime="image/png")
        else:
            cols = st.columns(2)
            for i, im in enumerate(images):
                with cols[i % 2]:
                    st.image(im, use_container_width=True)
                    st.download_button(
                        f"⬇️ Download #{i+1}",
                        data=image_to_png_bytes(im),
                        file_name=f"mj_style_{i+1}.png",
                        mime="image/png",
                        key=f"dl_latest_{i}",
                    )
    else:
        st.info("No renders yet. Write a prompt and click **Generate**.")

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("#### History")

    if st.session_state.history:
        for idx, item in enumerate(st.session_state.history[:5]):
            with st.expander(f"Render #{len(st.session_state.history)-idx} • {item['ts']}", expanded=(idx==0)):
                st.caption(item["prompt"])
                imgs = item["images"]
                cols = st.columns(2) if len(imgs) > 1 else [st]
                for i, im in enumerate(imgs):
                    with cols[i % 2] if len(imgs) > 1 else st:
                        st.image(im, use_container_width=True)
                        st.download_button(
                            f"Download #{i+1}",
                            data=image_to_png_bytes(im),
                            file_name=f"history_{idx+1}_{i+1}.png",
                            mime="image/png",
                            key=f"dl_hist_{idx}_{i}",
                        )
    else:
        st.caption("Your previous renders will show here.")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Generation ----------------
if generate:
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating..."):
            g = make_generator(seed)
            images = []
            # SDXL-Turbo best practice: guidance_scale=0.0
            for _ in range(int(n_images)):
                img = pipe(
                    prompt=prompt.strip(),
                    num_inference_steps=int(steps),
                    guidance_scale=0.0,
                    width=int(width),
                    height=int(height),
                    generator=g,
                ).images[0]
                images.append(img)

        st.session_state.history.insert(0, {
            "prompt": prompt.strip(),
            "images": images,
            "ts": time.strftime("%H:%M:%S"),
        })
        st.rerun()

import os, io, time, base64, traceback
import streamlit as st
import torch
import requests
from PIL import Image, ImageDraw, ImageFont

# ----------------- Streamlit config -----------------
st.set_page_config(page_title="Poster Studio (Groq + Open Models)", page_icon="✨", layout="wide")

# ----------------- HuggingFace cache dirs -----------------
APP_CACHE = os.path.join(os.getcwd(), ".cache")
HF_HOME = os.path.join(APP_CACHE, "huggingface")
os.makedirs(HF_HOME, exist_ok=True)
os.environ["HF_HOME"] = HF_HOME
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(HF_HOME, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_HOME, "transformers")

IS_GPU = torch.cuda.is_available()

PRIMARY_MODEL = "stabilityai/sd-turbo" if not IS_GPU else "stabilityai/sdxl-turbo"
FALLBACK_MODEL = "runwayml/stable-diffusion-v1-5"

# ----------------- CSS -----------------
st.markdown("""
<style>
:root{ --bg0:#070915; --bg1:#0b1020; --card:rgba(255,255,255,.045); --stroke:rgba(255,255,255,.10);
--text:#e5e7eb; --muted:rgba(229,231,235,.70); }
[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(1200px 600px at 10% 0%, rgba(124,58,237,.20), transparent 60%),
    radial-gradient(900px 550px at 90% 10%, rgba(34,197,94,.14), transparent 55%),
    radial-gradient(900px 650px at 50% 100%, rgba(96,165,250,.10), transparent 60%),
    linear-gradient(180deg, var(--bg1) 0%, var(--bg0) 100%);
  color: var(--text);
}
.block-container{ max-width:1260px; padding-top:1rem; padding-bottom:2rem; }
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
      <div style="font-size:22px; font-weight:900;">Poster Studio</div>
      <div style="font-size:12px; opacity:.75;">Groq for prompt enhancement + Open-source image model + Perfect text overlay</div>
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

def try_font(size: int):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()

def draw_centered_text(draw: ImageDraw.ImageDraw, text: str, y: int, font, fill, stroke_fill, stroke_w, img_w: int):
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_w)
    text_w = bbox[2] - bbox[0]
    x = (img_w - text_w) // 2
    draw.text((x, y), text, font=font, fill=fill, stroke_fill=stroke_fill, stroke_width=stroke_w)

def add_poster_text(img: Image.Image, top_text: str, bottom_text: str, theme: str = "Gold"):
    img = img.convert("RGBA")
    w, h = img.size

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)

    # readability bands
    for i in range(int(h * 0.18)):
        alpha = int(200 * (1 - i / (h * 0.18)))
        od.rectangle([0, i, w, i+1], fill=(0, 0, 0, alpha))
    for i in range(int(h * 0.20)):
        alpha = int(220 * (i / (h * 0.20)))
        y = h - int(h * 0.20) + i
        od.rectangle([0, y, w, y+1], fill=(0, 0, 0, alpha))

    img = Image.alpha_composite(img, overlay)
    draw = ImageDraw.Draw(img)

    if theme == "Gold":
        title_fill = (212, 175, 55, 255)
    else:
        title_fill = (255, 255, 255, 255)

    title_stroke = (0, 0, 0, 255)
    sub_fill = (255, 255, 255, 255)
    sub_stroke = (0, 0, 0, 255)

    title_size = max(22, int(w * 0.06))
    sub_size = max(16, int(w * 0.035))

    title_font = try_font(title_size)
    sub_font = try_font(sub_size)

    draw_centered_text(draw, top_text, int(h * 0.04), title_font, title_fill, title_stroke, 4, w)
    draw_centered_text(draw, bottom_text, h - int(h * 0.11), sub_font, sub_fill, sub_stroke, 3, w)

    return img.convert("RGB")

# ----------------- Groq prompt enhancer -----------------
def get_groq_key() -> str | None:
    # secrets > env
    if "GROQ_API_KEY" in st.secrets:
        return st.secrets["GROQ_API_KEY"]
    return os.getenv("GROQ_API_KEY")

def enhance_prompt_with_groq(user_prompt: str, goal: str) -> str:
    api_key = get_groq_key()
    if not api_key:
        return user_prompt

    api_key = api_key.strip()

    system = (
        "You are an expert Midjourney-style prompt engineer for text-to-image diffusion models. "
        "Rewrite the user's prompt into a single concise, highly visual prompt for photorealistic results. "
        "Do NOT include any text overlays, logos, watermarks, or written words. "
        "Return ONLY the rewritten prompt text."
    )
    user = f"Goal: {goal}\nUser prompt: {user_prompt}\nRewrite now:"

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # ✅ model fallback (some accounts don’t have all models enabled)
    model_candidates = [
        "llama-3.3-70b-versatile",
        "llama-3.1-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
    ]

    last_err = None

    for model in model_candidates:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.6,
            "max_tokens": 220,
        }

        try:
            r = requests.post(url, headers=headers, json=payload, timeout=60)

            # If error, capture body for debugging
            if r.status_code != 200:
                last_err = f"Groq HTTP {r.status_code} for model={model}\nResponse: {r.text}"
                continue

            data = r.json()
            return data["choices"][0]["message"]["content"].strip()

        except Exception as e:
            last_err = f"Groq request failed for model={model}: {e}"

    # ✅ Don’t crash app; show error and return original prompt
    st.warning("Groq prompt enhancement failed. Using your original prompt.")
    if last_err:
        st.code(last_err)
    return user_prompt

# ----------------- Diffusers load/generate -----------------
@st.cache_resource(show_spinner=False)
def load_pipe(model_id: str):
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

def generate_background(prompt: str, seed: int, steps: int, width: int, height: int, model_id: str):
    pipe = load_pipe(model_id)
    guidance = 0.0 if "turbo" in model_id else 7.0

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
    return out.images[0], guidance

# ----------------- State -----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ----------------- Sidebar -----------------
with st.sidebar:
    st.markdown("### 🔑 Groq")
    has_key = bool(get_groq_key())
    st.write("Groq Key:", "✅ Loaded" if has_key else "❌ Not found (add to Secrets)")

    st.markdown("### 🧾 Poster Text")
    top_text = st.text_input("Heading (Top)", value="Best Gym in Vehari")
    bottom_text = st.text_input("Contact (Bottom)", value="Call Now: 03008982206")
    theme = st.selectbox("Heading Color", ["Gold", "White"], index=0)

    st.divider()
    st.markdown("### ⚙️ Settings")
    if IS_GPU:
        steps = st.selectbox("Steps", [2, 3, 5], index=1)
        size = st.selectbox("Size", ["768x768", "1024x576", "576x1024"], index=0)
    else:
        steps = st.selectbox("Steps (CPU)", [1, 2, 3], index=1)
        size = st.selectbox("Size (CPU)", ["384x384", "448x448", "512x512"], index=0)

    width, height = map(int, size.split("x"))
    seed = st.number_input("Seed", min_value=0, max_value=2_147_483_647, value=42, step=1)

# ----------------- Main layout -----------------
left, right = st.columns([1.1, 1], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div style="display:flex; justify-content:space-between; align-items:flex-end; gap:10px; flex-wrap:wrap;">'
                '<div class="h">✍️ Your Prompt</div><div class="hint">Groq will enhance it (optional), image model generates background</div></div>',
                unsafe_allow_html=True)

    user_prompt = st.text_area(
        "Prompt",
        value=(
            "A muscular fitness athlete performing barbell squats inside a modern gym, "
            "background clearly showing gym machines, weight racks, cable machines, mirrors, "
            "cinematic lighting, photorealistic"
        ),
        label_visibility="collapsed",
    )

    use_groq = st.toggle("✨ Enhance prompt using Groq", value=True, help="If Groq key is missing, it will simply use your prompt.")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        generate = st.button("✨ Generate Poster")
    with c2:
        clear = st.button("🧹 Clear History")
    st.markdown('</div>', unsafe_allow_html=True)

if clear:
    st.session_state.history = []
    st.rerun()

# ----------------- Generate -----------------
if generate:
    try:
        t0 = time.time()

        goal = "Luxury gym poster background with visible machines, commercial fitness photography"
        prompt_used = user_prompt.strip()

        if use_groq:
            with st.spinner("Groq is enhancing your prompt..."):
                prompt_used = enhance_prompt_with_groq(prompt_used, goal)

        with st.spinner("Generating image (may take time on CPU)..."):
            try:
                bg, guidance = generate_background(prompt_used, seed, steps, width, height, PRIMARY_MODEL)
                used_model = PRIMARY_MODEL
            except Exception:
                bg, guidance = generate_background(prompt_used, seed, steps, width, height, FALLBACK_MODEL)
                used_model = FALLBACK_MODEL

        poster = add_poster_text(bg, top_text.strip(), bottom_text.strip(), theme=theme)
        png = pil_to_png_bytes(poster)
        b64 = b64_from_bytes(png)
        sec = round(time.time() - t0, 2)

        st.session_state.history.insert(0, {
            "prompt_used": prompt_used,
            "model": used_model,
            "size": f"{width}x{height}",
            "steps": steps,
            "guidance": guidance,
            "seconds": sec,
            "bytes": png,
            "b64": b64,
        })
        st.rerun()

    except Exception as e:
        st.error("Failed to generate poster.")
        st.code("".join(traceback.format_exception(type(e), e, e.__traceback__)))

# ----------------- Result -----------------
with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div style="display:flex; justify-content:space-between; align-items:flex-end; gap:10px; flex-wrap:wrap;">'
                '<div class="h">🖼️ Result</div><div class="hint">Poster output</div></div>',
                unsafe_allow_html=True)

    if st.session_state.history:
        item = st.session_state.history[0]
        st.markdown('<div class="toast">✅ Poster completed</div>', unsafe_allow_html=True)
        st.markdown(
            f"<small class='muted'>Model: <b>{item['model']}</b> • Size: <b>{item['size']}</b> • "
            f"Steps: <b>{item['steps']}</b> • Time: <b>{item['seconds']}s</b></small>",
            unsafe_allow_html=True
        )
        st.caption("Prompt used (after Groq enhancement):")
        st.code(item["prompt_used"])

        show_b64(item["b64"])
        st.download_button("⬇️ Download PNG", data=item["bytes"], file_name="poster.png", mime="image/png")
    else:
        st.info("No poster yet. Generate one from the left panel.")

    st.markdown("</div>", unsafe_allow_html=True)

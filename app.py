import os, io, time, base64, traceback
import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont
import requests

# ----------------- Page Config -----------------
st.set_page_config(page_title="Poster Studio • Any Topic", page_icon="🖼️", layout="wide")

# ----------------- Cache dirs (reduce re-downloads) -----------------
APP_CACHE = os.path.join(os.getcwd(), ".cache")
HF_HOME = os.path.join(APP_CACHE, "huggingface")
os.makedirs(HF_HOME, exist_ok=True)
os.environ["HF_HOME"] = HF_HOME
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(HF_HOME, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_HOME, "transformers")

IS_GPU = torch.cuda.is_available()

# ✅ Model choices: CPU-safe vs GPU-best
CPU_MODEL = "stabilityai/sd-turbo"                 # fast & more stable on CPU
GPU_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"  # high quality on GPU

# ----------------- UI CSS -----------------
st.markdown("""
<style>
:root{
  --bg0:#070915; --bg1:#0b1020; --card:rgba(255,255,255,.045); --stroke:rgba(255,255,255,.10);
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
.block-container{ max-width:1280px; padding-top:1rem; padding-bottom:2rem; }
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
.card{
  border:1px solid var(--stroke); background: var(--card);
  border-radius:18px; padding:14px;
  box-shadow: 0 18px 55px rgba(0,0,0,.38);
}
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
  min-height: 140px !important;
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
.imgbox{
  border:1px solid rgba(255,255,255,.10);
  border-radius:16px; overflow:hidden; background: rgba(255,255,255,.03);
}
small.muted{ color: rgba(229,231,235,.70); }
</style>
""", unsafe_allow_html=True)

used_model_label = GPU_MODEL if IS_GPU else CPU_MODEL
st.markdown(f"""
<div class="topbar">
  <div style="display:flex; align-items:center; gap:10px;">
    <div style="font-size:22px;">🖼️</div>
    <div>
      <div style="font-size:22px; font-weight:900;">Poster Studio</div>
      <div style="font-size:12px; opacity:.75;">Generate any image from prompt + add your own heading & footer</div>
    </div>
  </div>
  <div class="pill">Mode: {"GPU" if IS_GPU else "CPU"} • Model: {used_model_label}</div>
</div>
""", unsafe_allow_html=True)

# ----------------- Utility -----------------
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

def get_groq_key() -> str | None:
    try:
        if "GROQ_API_KEY" in st.secrets:
            return str(st.secrets["GROQ_API_KEY"]).strip()
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY")

def enhance_prompt_with_groq(user_prompt: str, goal: str) -> str:
    api_key = get_groq_key()
    if not api_key:
        return user_prompt

    system = (
        "You are an expert prompt engineer for diffusion text-to-image models. "
        "Rewrite the user's prompt into a single concise, highly visual prompt for photorealistic results. "
        "Do NOT include text, logos, watermarks, or written words in the image. "
        "Focus on: subject, environment, camera/lens, lighting, composition, realism. "
        "Return ONLY the rewritten prompt."
    )
    user = f"Goal: {goal}\nUser prompt: {user_prompt}\nRewrite now:"

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # fallback models (some accounts don't have all)
    model_candidates = ["llama-3.1-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"]

    last_err = None
    for model in model_candidates:
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0.6,
            "max_tokens": 240,
        }
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=60)
            if r.status_code != 200:
                last_err = f"Groq HTTP {r.status_code} model={model}\n{r.text}"
                continue
            data = r.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            last_err = f"Groq request failed model={model}: {e}"

    st.warning("Groq enhancement failed. Using your original prompt.")
    if last_err:
        st.code(last_err)
    return user_prompt

def try_font(size: int, bold: bool = True):
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in paths:
        if os.path.exists(p):
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()

def draw_centered(draw, text, y, font, fill, stroke_fill, stroke_w, w):
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_w)
    tw = bbox[2] - bbox[0]
    x = (w - tw) // 2
    draw.text((x, y), text, font=font, fill=fill, stroke_fill=stroke_fill, stroke_width=stroke_w)

def add_text_overlay(img: Image.Image, heading: str, footer: str, heading_color: str):
    img = img.convert("RGBA")
    w, h = img.size

    # gradient bands for readability
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)

    top_band = int(h * 0.18)
    bot_band = int(h * 0.20)

    for i in range(top_band):
        a = int(200 * (1 - i / max(1, top_band)))
        od.rectangle([0, i, w, i + 1], fill=(0, 0, 0, a))
    for i in range(bot_band):
        a = int(220 * (i / max(1, bot_band)))
        y = h - bot_band + i
        od.rectangle([0, y, w, y + 1], fill=(0, 0, 0, a))

    img = Image.alpha_composite(img, overlay)
    draw = ImageDraw.Draw(img)

    # colors
    if heading_color == "Gold":
        head_fill = (212, 175, 55, 255)
    elif heading_color == "Cyan":
        head_fill = (34, 211, 238, 255)
    else:
        head_fill = (255, 255, 255, 255)

    # auto font sizes
    head_size = max(22, int(w * 0.06))
    foot_size = max(16, int(w * 0.035))
    head_font = try_font(head_size, bold=True)
    foot_font = try_font(foot_size, bold=True)

    # draw (only if not empty)
    if heading.strip():
        draw_centered(draw, heading.strip(), int(h * 0.04), head_font, head_fill, (0, 0, 0, 255), 4, w)
    if footer.strip():
        draw_centered(draw, footer.strip(), h - int(h * 0.11), foot_font, (255, 255, 255, 255), (0, 0, 0, 255), 3, w)

    return img.convert("RGB")

# ----------------- Diffusers -----------------
@st.cache_resource(show_spinner=False)
def load_pipe_cpu(model_id: str):
    from diffusers import AutoPipelineForText2Image
    pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float32, use_safetensors=True)
    return pipe

@st.cache_resource(show_spinner=False)
def load_pipe_gpu_sdxl(model_id: str):
    # SDXL best quality on GPU only
    from diffusers import StableDiffusionXLPipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=True)
    pipe = pipe.to("cuda")
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    return pipe

def generate_image(prompt: str, negative: str, seed: int, steps: int, guidance: float, width: int, height: int):
    gen = torch.Generator(device="cuda") if IS_GPU else torch.Generator()
    gen = gen.manual_seed(int(seed))

    if IS_GPU:
        pipe = load_pipe_gpu_sdxl(GPU_MODEL)
        out = pipe(
            prompt=prompt,
            negative_prompt=negative,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            width=int(width),
            height=int(height),
            generator=gen,
        )
        return out.images[0], GPU_MODEL
    else:
        # CPU: use sd-turbo with lower guidance and steps
        pipe = load_pipe_cpu(CPU_MODEL)
        out = pipe(
            prompt=prompt,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            width=int(width),
            height=int(height),
            generator=gen,
        )
        return out.images[0], CPU_MODEL

# ----------------- State -----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ----------------- Sidebar -----------------
with st.sidebar:
    st.markdown("### 🧾 Poster Text")
    heading = st.text_input("Heading (Top)", value="Best Gym in Vehari")
    footer = st.text_input("Footer (Bottom)", value="Call Now: 03008982206")
    heading_color = st.selectbox("Heading Color", ["Gold", "White", "Cyan"], index=0)

    st.divider()
    st.markdown("### ⚙️ Quality Settings")

    if IS_GPU:
        size = st.selectbox("Size", ["1024x1024", "1024x576", "576x1024"], index=0)
        steps = st.slider("Steps", 20, 50, 35, 1)
        guidance = st.slider("Guidance", 4.0, 9.0, 6.5, 0.5)
    else:
        size = st.selectbox("Size (CPU)", ["384x384", "448x448", "512x512"], index=1)
        steps = st.slider("Steps (CPU)", 1, 6, 2, 1)
        guidance = st.slider("Guidance (CPU)", 0.0, 3.0, 0.8, 0.1)
        st.caption("Streamlit Cloud is CPU-only → keep size small for reliability.")

    width, height = map(int, size.split("x"))

    seed = st.number_input("Seed", min_value=0, max_value=2_147_483_647, value=42, step=1)

    st.divider()
    st.markdown("### ✨ Groq")
    use_groq = st.toggle("Enhance prompt using Groq", value=True)
    st.caption("Groq rewrites prompt for better image quality (optional).")

# ----------------- Main Layout -----------------
left, right = st.columns([1.1, 1], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div style="display:flex; justify-content:space-between; align-items:flex-end; gap:10px; flex-wrap:wrap;">'
                '<div class="h">✍️ Image Prompt</div><div class="hint">Describe any scene. The app will add heading/footer.</div></div>',
                unsafe_allow_html=True)

    user_prompt = st.text_area(
        "Prompt",
        value="A cinematic photo of a modern gym interior with workout machines and dramatic lighting, ultra realistic, high detail, 35mm lens, shallow depth of field",
        label_visibility="collapsed",
    )

    neg_default = "blurry, low quality, lowres, distorted, deformed, bad anatomy, extra limbs, extra fingers, watermark, logo, text"
    negative = st.text_input("Negative prompt (optional)", value=neg_default)

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

        prompt_used = user_prompt.strip()
        if use_groq and prompt_used:
            with st.spinner("Groq is enhancing your prompt..."):
                prompt_used = enhance_prompt_with_groq(prompt_used, goal="photorealistic, professional, cinematic, sharp, high detail")

        with st.spinner("Generating image + applying text overlay..."):
            img, model_used = generate_image(
                prompt=prompt_used,
                negative=negative.strip(),
                seed=seed,
                steps=steps,
                guidance=guidance,
                width=width,
                height=height,
            )

            # overlay heading/footer
            poster = add_text_overlay(img, heading, footer, heading_color)

            png = pil_to_png_bytes(poster)
            b64 = b64_from_bytes(png)
            sec = round(time.time() - t0, 2)

        st.session_state.history.insert(0, {
            "prompt": prompt_used,
            "model": model_used,
            "size": f"{width}x{height}",
            "steps": steps,
            "guidance": guidance,
            "seconds": sec,
            "bytes": png,
            "b64": b64,
        })
        st.rerun()

    except Exception as e:
        st.error("Generation failed.")
        st.code("".join(traceback.format_exception(type(e), e, e.__traceback__)))

# ----------------- Result -----------------
with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div style="display:flex; justify-content:space-between; align-items:flex-end; gap:10px; flex-wrap:wrap;">'
                '<div class="h">🖼️ Result</div><div class="hint">Latest poster</div></div>',
                unsafe_allow_html=True)

    if st.session_state.history:
        item = st.session_state.history[0]
        st.markdown('<div class="toast">✅ Done</div>', unsafe_allow_html=True)

        st.markdown(
            f"<small class='muted'>Model: <b>{item['model']}</b> • Size: <b>{item['size']}</b> • "
            f"Steps: <b>{item['steps']}</b> • Guidance: <b>{item['guidance']}</b> • Time: <b>{item['seconds']}s</b></small>",
            unsafe_allow_html=True
        )

        st.caption("Prompt used:")
        st.code(item["prompt"])

        show_b64(item["b64"])

        st.download_button("⬇️ Download PNG", data=item["bytes"], file_name="poster.png", mime="image/png")
    else:
        st.info("No poster yet. Write a prompt and click **Generate Poster**.")

    st.markdown("</div>", unsafe_allow_html=True)

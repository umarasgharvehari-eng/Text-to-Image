import streamlit as st
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image, ImageDraw, ImageFont
import io
import requests

st.set_page_config(page_title="Gym Poster AI", layout="wide")

# ==========================
# LOAD SDXL (HIGH QUALITY)
# ==========================
@st.cache_resource
def load_model():
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe.to("cuda")
    pipe.enable_xformers_memory_efficient_attention()
    return pipe

pipe = load_model()

# ==========================
# TEXT OVERLAY FUNCTION
# ==========================
def add_text_overlay(image, top_text, bottom_text):
    draw = ImageDraw.Draw(image)
    width, height = image.size

    try:
        font_top = ImageFont.truetype("DejaVuSans-Bold.ttf", int(height * 0.08))
        font_bottom = ImageFont.truetype("DejaVuSans-Bold.ttf", int(height * 0.05))
    except:
        font_top = ImageFont.load_default()
        font_bottom = ImageFont.load_default()

    # Top text
    w, h = draw.textbbox((0, 0), top_text, font=font_top)[2:]
    draw.text(((width - w) / 2, height * 0.05),
              top_text,
              font=font_top,
              fill=(255, 215, 0))

    # Bottom text
    w2, h2 = draw.textbbox((0, 0), bottom_text, font=font_bottom)[2:]
    draw.text(((width - w2) / 2, height * 0.90),
              bottom_text,
              font=font_bottom,
              fill=(255, 255, 255))

    return image

# ==========================
# GROQ PROMPT ENHANCEMENT
# ==========================
def enhance_prompt(prompt):
    api_key = st.secrets.get("GROQ_API_KEY", None)
    if not api_key:
        return prompt

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": "Rewrite into a high quality SDXL prompt for professional photography."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.6,
        "max_tokens": 200
    }

    r = requests.post(url, headers=headers, json=payload)
    if r.status_code == 200:
        return r.json()["choices"][0]["message"]["content"]
    return prompt

# ==========================
# UI
# ==========================
st.title("🏋️ Gym Poster Generator (Pro Quality)")

prompt = st.text_area("Enter prompt", height=120)

use_groq = st.toggle("Enhance prompt using Groq")

if st.button("Generate Poster"):
    with st.spinner("Generating High Quality Image..."):

        final_prompt = enhance_prompt(prompt) if use_groq else prompt

        negative_prompt = """
        blurry, low quality, distorted body, extra limbs, extra fingers,
        bad anatomy, watermark, logo, text, oversaturated
        """

        image = pipe(
            final_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=40,
            guidance_scale=8.0,
            height=1024,
            width=1024,
        ).images[0]

        image = add_text_overlay(
            image,
            "Best Gym in Vehari",
            "Call Now: 03008982206"
        )

        st.image(image, use_container_width=True)

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        st.download_button(
            "Download Image",
            buf.getvalue(),
            file_name="gym_poster.png",
            mime="image/png"
        )

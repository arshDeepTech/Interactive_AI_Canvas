from flask import Flask, request, jsonify, render_template
import torch
import os
import time
from diffusers import StableDiffusionPipeline
from torch import autocast
import gc
from PIL import Image

app = Flask(__name__)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

pipe = None

def load_model():
    global pipe
    if pipe is None:
        model_id = "CompVis/stable-diffusion-v1-2"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            revision="fp16",
            safety_checker=None,
        )
        
        pipe.enable_attention_slicing(slice_size=1)
        pipe.enable_vae_tiling()
        
        pipe = pipe.to("cuda")
        
        pipe.vae = pipe.vae.to("cpu")
        pipe.text_encoder = pipe.text_encoder.to("cpu")
        
        torch.cuda.empty_cache()
        gc.collect()
    return pipe

def generate_with_upscaling(pipe, prompt, target_size=384):
    with autocast("cuda"):
        image = pipe(
            prompt,
            num_inference_steps=15,
            guidance_scale=7.5,
            height=256,
            width=256,
        ).images[0]
    
    if target_size > 256:
        image = image.resize((target_size, target_size), Image.LANCZOS)
    
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_image():
    try:
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing prompt in request'
            }), 400

        prompt = data["prompt"]
        
        if len(prompt.strip()) < 3:
            return jsonify({
                'success': False,
                'error': 'Prompt too short'
            }), 400

        start_time = time.time()
        
        pipe = load_model()
        
        torch.cuda.empty_cache()
        gc.collect()

        try:
            pipe.vae = pipe.vae.to("cuda")
            pipe.text_encoder = pipe.text_encoder.to("cuda")
            
            with autocast("cuda"):
                generated_image = pipe(
                    prompt,
                    num_inference_steps=8,
                    guidance_scale=7.0,
                    height=320,
                    width=320,
                    timeout=100,
                ).images[0]
            
            pipe.vae = pipe.vae.to("cpu")
            pipe.text_encoder = pipe.text_encoder.to("cpu")
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                try:
                    with autocast("cuda"):
                        generated_image = pipe(
                            prompt,
                            num_inference_steps=8,
                            guidance_scale=7.0,
                            height=256,
                            width=256,
                            timeout=100,
                        ).images[0]
                except RuntimeError:
                    torch.cuda.empty_cache()
                    return jsonify({
                        'success': False,
                        'error': 'GPU memory insufficient. Please try again later.'
                    }), 507
            raise e
        
        os.makedirs("static", exist_ok=True)
        
        timestamp = int(time.time())
        image_path = f"static/output_{timestamp}.png"
        generated_image.save(image_path)

        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total time taken for Generating Image: {total_time:.2f} seconds")

        return jsonify({
            'success': True,
            'image_url': image_path,
            'generation_time': f"{total_time:.2f}"
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        torch.cuda.empty_cache()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == "__main__":
    app.run(debug=True)

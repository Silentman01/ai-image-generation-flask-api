from flask import Flask, request, jsonify, send_file
import torch
from PIL import Image
import io
import base64
import os
from text2image import StableDiffusionModel
from datetime import datetime
import uuid

app = Flask(__name__)

# Initialize the model globally
model = StableDiffusionModel()

# Configure image storage
UPLOAD_FOLDER = "static/generated/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/api/generate', methods=['POST'])
def generate_image():
    try:
        # Get data from request
        data = request.json
        if not data or 'prompt' not in data:
            return jsonify({'error': 'Missing prompt in request'}), 400

        # Extract parameters with defaults
        prompt = data['prompt']
        negative_prompt = data.get('negative_prompt', None)
        guidance_scale = data.get('guidance_scale', 7.0)
        num_inference_steps = data.get('num_inference_steps', 31)
        seed = data.get('seed', None)
        output_format = data.get('output_format', 'base64')  # New parameter

        # Generate image
        generated_image = model.generate_text_to_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            seed=seed
        )

        if output_format == 'base64':
            # Convert to base64
            buffered = io.BytesIO()
            generated_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            return jsonify({
                'status': 'success',
                'image': img_str,
                'dataUrl': f'data:image/png;base64,{img_str}',  # Added direct data URL
                'parameters': {
                    'prompt': prompt,
                    'negative_prompt': negative_prompt,
                    'guidance_scale': guidance_scale,
                    'num_inference_steps': num_inference_steps,
                    'seed': seed
                }
            })
            
        elif output_format == 'file':
            # Save to file and return URL
            filename = f"{uuid.uuid4()}.png"
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            generated_image.save(file_path, "PNG")
            
            # In production, replace with your actual domain
            image_url = f"http://localhost:5000/static/generated/{filename}"
            
            return jsonify({
                'status': 'success',
                'url': image_url,
                'parameters': {
                    'prompt': prompt,
                    'negative_prompt': negative_prompt,
                    'guidance_scale': guidance_scale,
                    'num_inference_steps': num_inference_steps,
                    'seed': seed
                }
            })
        
        else:
            return jsonify({'error': 'Invalid output_format'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
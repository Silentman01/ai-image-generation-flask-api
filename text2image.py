from diffusers import StableDiffusionPipeline
import torch

class StableDiffusionModel:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5"):
        """Initialize the Stable Diffusion model."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        self._setup_model()

    def _setup_model(self):
        """Set up the model with proper configurations."""
        # Set up GPU memory management if CUDA is available
        if torch.cuda.is_available():
            memory_fraction = 0.9
            torch.cuda.set_per_process_memory_fraction(memory_fraction, 0)
            torch.cuda.empty_cache()

        # Initialize the pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)

        # Enable memory efficient attention if using CUDA
        if self.device == "cuda":
            self.pipe.enable_attention_slicing()

    def generate_text_to_image(
        self,
        prompt,
        negative_prompt=None,
        guidance_scale=7.0,
        num_inference_steps=31,
        seed=None
    ):
        """
        Generate an image from a text prompt.
        
        Args:
            prompt (str): The text prompt for image generation
            negative_prompt (str, optional): Text to guide what not to include
            guidance_scale (float): Higher values give stronger prompt adherence
            num_inference_steps (int): Number of denoising steps
            seed (int, optional): Random seed for reproducibility
            
        Returns:
            PIL.Image: Generated image
        """
        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            
        # Generate the image
        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps
        ).images[0]
        
        # Clear CUDA cache after generation
        if self.device == "cuda":
            torch.cuda.empty_cache()
            
        return image
from diffusers import StableDiffusionPipeline
import torch

# Load Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe.to("cuda" if torch.cuda.is_available() else "cpu")  

# Define prompt
prompt = "a serene sunset over a futuristic city"

# Generate and save images
for i in range(3):
    image = pipe(prompt).images[0]
    image.save(f"generated_image_{i+1}.png")

print("Images generated and saved successfully!")

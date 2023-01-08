import PIL
import requests
import torch
from io import BytesIO

from src.diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint_with_stable_artist import StableDiffusionInpaintPipeline

device='cuda'
def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")

import torch
gen = torch.Generator(device=device)

from PIL import Image
img = Image.open("/raid/ankit/srinath/diffusion_experiments/sample_data/sphola_stupa.jpg").resize((512, 512))
mask = Image.open("/raid/ankit/srinath/diffusion_experiments/sample_data/mask_sphola_stupa.jpg").resize((512, 512))

# init_image = download_image(img_url).resize((512, 512))
# mask_image = download_image(mask_url).resize((512, 512))
gen.manual_seed(2)

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
)
pipe = pipe.to("cuda")

out = pipe(image = img,
            mask_image = mask,
            prompt='A stupa', generator=gen, num_images_per_prompt=1, guidance_scale=7, 
          editing_prompt=['fire'                                 # Concepts to apply
                    ],
           reverse_editing_direction=[False],   # Direction of guidance
           edit_warmup_steps=[5],                    # Warmup period for each concept
           edit_guidance_scale=[2000],            # Guidance scale for each concept
           edit_threshold=[-0.2],                 # Threshold for each concept. Note that positive guidance needs negative thresholds and vice versa
           edit_weights=[1.2],                            # Weights of the individual concepts against each other
           edit_momentum_scale=0.5,                          # Momentum scale that will be added to the latent guidance
           edit_mom_beta=0.6,                                 # Momentum beta
           )


out[0].save("stupa_fire.png")
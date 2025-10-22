# agno_tools/image_tools.py
"""
Tools for Image Agent - AI image generation and manipulation
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def generate_image(prompt: str, size: str = "1024x1024", style: str = "vivid") -> str:
    """
    Generate an image using DALL-E 3.
    
    Args:
        prompt: Text description of the image
        size: Image size ("1024x1024", "1024x1792", "1792x1024")
        style: Image style ("vivid" or "natural")
    
    Returns:
        Image URL and details
    """
    try:
        logger.info(f"🎨 generate_image: '{prompt[:50]}...'")
        
        from openai import OpenAI
        from core.config import settings
        
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=size,
            quality="standard",
            style=style,
            n=1,
        )
        
        image_url = response.data[0].url
        revised_prompt = response.data[0].revised_prompt
        
        return f"""🎨 **Image Generated Successfully!**

**Your Prompt:** {prompt}

**Enhanced Prompt:** {revised_prompt}

**Image URL:** {image_url}

**Size:** {size} | **Style:** {style}

View your image: {image_url}"""
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return f"Error generating image: {str(e)}"


def create_variation(image_url: str, n: int = 1) -> str:
    """
    Create variations of an existing image.
    
    Args:
        image_url: URL of the source image
        n: Number of variations to generate
    
    Returns:
        URLs of generated variations
    """
    try:
        logger.info(f"🔄 create_variation")
        # TODO: Implement image variation
        return "Image variation feature - coming soon!"
    except Exception as e:
        return f"Error: {str(e)}"


def enhance_prompt(basic_prompt: str) -> str:
    """
    Enhance a basic prompt for better image generation.
    
    Args:
        basic_prompt: Simple description
    
    Returns:
        Enhanced, detailed prompt
    """
    try:
        logger.info(f"✨ enhance_prompt")
        
        # Basic prompt enhancement rules
        enhanced = f"{basic_prompt}, highly detailed, professional photography, 8k resolution, dramatic lighting, vibrant colors"
        
        return f"""**Enhanced Prompt:**
{enhanced}

Use this enhanced prompt with generate_image() for better results!"""
        
    except Exception as e:
        return f"Error: {str(e)}"


# Tool registry for Image Agent
IMAGE_TOOLS = {
    "generate_image": generate_image,
    "create_variation": create_variation,
    "enhance_prompt": enhance_prompt,
}

import os
import base64

from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("MOONSHOT_API_KEY"),
    base_url="https://api.moonshot.ai/v1",
)

# Replace kimi.png with the path to the image you want Kimi to analyze
image_path = "kimi.png"

with open(image_path, "rb") as f:
    image_data = f.read()

# Use the standard library base64.b64encode function to encode the image into base64 format
image_url = f"data:image/{os.path.splitext(image_path)[1].lstrip('.')};base64,{base64.b64encode(image_data).decode('utf-8')}"


completion = client.chat.completions.create(
    model="kimi-k2.6",
    messages=[
        {"role": "system", "content": "You are Kimi."},
        {
            "role": "user",
            # Note: content is changed from str type to a list containing multiple content parts.
            # Image (image_url) is one part, and text is another part.
            "content": [
                {
                    "type": "image_url",  # <-- Use image_url type to upload images, with content as base64-encoded image data
                    "image_url": {
                        "url": image_url,
                    },
                },
                {
                    "type": "text",
                    "text": "Please describe the content of the image.",  # <-- Use text type to provide text instructions
                },
            ],
        },
    ],
)

print(completion.choices[0].message.content)
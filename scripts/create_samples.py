"""
Utility script to create sample test images with Indic text for pilot testing.
"""

from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont


def create_sample_data():
    samples_dir = Path("./data/samples")
    samples_dir.mkdir(parents=True, exist_ok=True)

    samples = [
        {
            "id": "kn_sample_01",
            "lang": "kn",
            "title": "ಕರ್ನಾಟಕ ರಾಜ್ಯ ಸರ್ಕಾರ",
            "body": "ವಿಷಯ: ಸಾರ್ವಜನಿಕ ದಾಖಲಾತಿ ಪತ್ರ\nಹೆಸರು: ರಮೇಶ್ ಕುಮಾರ್\nದಿನಾಂಕ: 15-08-2024\nವಿಳಾಸ: ಬೆಂಗಳೂರು ನಗರ",
        },
        {
            "id": "kn_sample_02",
            "lang": "kn",
            "title": "ಅರ್ಜಿ ನಮೂನೆ",
            "body": "ಖಾತೆ ಸಂಖ್ಯೆ: ೯೮೭೬೫೪೩೨೧\nಮೊತ್ತ: ರೂ. ೫೦,೦೦೦\nಸ್ಥಳ: ಮೈಸೂರು\nಷರಾ: ಪರಿಶೀಲಿಸಲಾಗಿದೆ",
        },
        {
            "id": "hi_sample_01",
            "lang": "hi",
            "title": "भारत सरकार - आधिकारिक प्रपत्र",
            "body": "विषय: नागरिक सेवा प्रमाण पत्र\nनाम: राजेश शर्मा\nदिनांक: 26-01-2024\nस्थान: नई दिल्ली",
        }
    ]

    for s in samples:
        img_path = samples_dir / f"{s['id']}.png"
        json_path = samples_dir / f"{s['id']}.json"

        # Create clean white document image
        img = Image.new("RGB", (1000, 800), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Draw border
        draw.rectangle([(20, 20), (980, 780)], outline=(180, 180, 180), width=2)
        # Draw header bar
        draw.rectangle([(30, 30), (970, 90)], fill=(240, 245, 250))

        full_text = f"{s['title']}\n{s['body']}"

        # Try to render text using default or fallback font
        try:
            # Try Nirmala UI (standard Windows Indic font)
            font_title = ImageFont.truetype("Nirmala.ttf", 26)
            font_body = ImageFont.truetype("Nirmala.ttf", 20)
            draw.text((50, 45), s['title'], fill=(20, 30, 80), font=font_title)
            draw.text((50, 130), s['body'], fill=(30, 30, 30), font=font_body)
        except Exception:
            # Fallback to bitmap font for drawing
            draw.text((50, 50), s['title'], fill=(0, 0, 0))
            draw.text((50, 130), s['body'], fill=(0, 0, 0))

        img.save(img_path)

        # Write corresponding JSON ground truth
        json_data = {
            "sample_id": s["id"],
            "language": s["lang"],
            "ground_truth": full_text,
            "lines": [
                {"text": s["title"], "type": "printed_header"},
                *[{"text": line, "type": "content"} for line in s["body"].splitlines()]
            ]
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

    print(f"Created {len(samples)} pilot samples in {samples_dir}")


if __name__ == "__main__":
    create_sample_data()

"""
Sample Image Generator for Nepal Number Plate Testing
=====================================================
Generates realistic vehicle and license plate test images representing:
1. Nepali Motorcycle (Red plate, 2-line Devanagari: बा ९२ प / ५६७८)
2. Nepali Private Car (Red plate, 1-line Devanagari: बा १५ च ४३२१)
3. High Security Embossed Two-Wheeler (White plate, Latin: BAGMATI PROVINCE 02-035 PA 5678)
4. Public Transport Bus (Black plate, Devanagari: ना ५ ख १२३४)
5. Heavy Commercial Truck (Black plate, Devanagari: ना २ क ९८७६)
"""

import os
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

SAMPLES_DIR = Path(__file__).parent / "static" / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def get_font(size: int = 24):
    """Attempts to find a system font supporting Unicode/Devanagari, or falls back to default."""
    font_paths = [
        "/System/Library/Fonts/KohinoorBangla.ttc",
        "/System/Library/Fonts/Supplemental/DevanagariMT.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNS.ttf",
    ]
    for p in font_paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def create_sample_car_nepali():
    """Generates an image of a private car with traditional red Nepali plate."""
    w, h = 800, 500
    img = Image.new("RGB", (w, h), color=(30, 35, 45))
    draw = ImageDraw.Draw(img)

    # Road background
    draw.rectangle([(0, 380), (w, h)], fill=(40, 42, 48))
    # Car body (Sedan rear/bumper)
    draw.rounded_rectangle([(140, 100), (660, 390)], radius=35, fill=(18, 52, 98), outline=(30, 80, 140), width=4)
    # Rear windshield
    draw.polygon([(210, 115), (590, 115), (630, 210), (170, 210)], fill=(12, 18, 28), outline=(40, 70, 110), width=2)
    # Taillights
    draw.rounded_rectangle([(155, 220), (240, 260)], radius=8, fill=(200, 25, 25), outline=(255, 60, 60), width=2)
    draw.rounded_rectangle([(560, 220), (645, 260)], radius=8, fill=(200, 25, 25), outline=(255, 60, 60), width=2)
    # Trunk / Bumper crease
    draw.line([(160, 290), (640, 290)], fill=(12, 38, 70), width=3)
    # Exhaust & Wheels
    draw.rectangle([(130, 370), (190, 410)], fill=(15, 15, 15))
    draw.rectangle([(610, 370), (670, 410)], fill=(15, 15, 15))

    # License Plate Recess
    draw.rounded_rectangle([(285, 305), (515, 365)], radius=6, fill=(10, 15, 20))

    # Traditional Nepali Red Plate (निजी सवारी)
    draw.rounded_rectangle([(290, 310), (510, 360)], radius=4, fill=(175, 25, 25), outline=(230, 230, 230), width=2)

    # Plate Text: बा १५ च ४३२१ (White text on Red)
    font = get_font(28)
    draw.text((400, 335), "बा १५ च ४३२१", fill=(255, 255, 255), font=font, anchor="mm")

    path = SAMPLES_DIR / "sample_car_nepali.jpg"
    img.save(path, quality=95)
    return str(path)


def create_sample_bike_nepali():
    """Generates an image of a motorcycle rear with 2-line stacked Nepali plate."""
    w, h = 640, 640
    img = Image.new("RGB", (w, h), color=(25, 28, 36))
    draw = ImageDraw.Draw(img)

    # Background road
    draw.rectangle([(0, 480), (w, h)], fill=(38, 40, 45))
    # Motorcycle rear tire
    draw.rounded_rectangle([(260, 330), (380, 560)], radius=25, fill=(18, 18, 20), outline=(50, 50, 55), width=3)
    # Mudguard / Tail section
    draw.polygon([(200, 140), (440, 140), (400, 340), (240, 340)], fill=(20, 20, 24), outline=(45, 45, 55), width=2)
    # Taillight
    draw.rounded_rectangle([(270, 150), (370, 195)], radius=10, fill=(215, 20, 20), outline=(255, 80, 80), width=3)
    # Turn indicators
    draw.rounded_rectangle([(180, 205), (230, 235)], radius=6, fill=(225, 140, 20))
    draw.rounded_rectangle([(410, 205), (460, 235)], radius=6, fill=(225, 140, 20))

    # License Plate Mount & Plate (Square/Stacked 2-line)
    draw.rectangle([(245, 225), (395, 335)], fill=(12, 12, 15))
    # Red background (Private Two-Wheeler)
    draw.rounded_rectangle([(250, 230), (390, 330)], radius=4, fill=(180, 20, 20), outline=(240, 240, 240), width=2)

    font_line1 = get_font(26)
    font_line2 = get_font(32)
    # Line 1: बा ९२ प (Bagmati Lot 92 Pa)
    draw.text((320, 255), "बा ९२ प", fill=(255, 255, 255), font=font_line1, anchor="mm")
    # Line 2: ५६७८ (5678)
    draw.text((320, 298), "५६७८", fill=(255, 255, 255), font=font_line2, anchor="mm")

    path = SAMPLES_DIR / "sample_bike_nepali.jpg"
    img.save(path, quality=95)
    return str(path)


def create_sample_embossed():
    """Generates an image with modern High-Security Embossed Number Plate (English)."""
    w, h = 800, 480
    img = Image.new("RGB", (w, h), color=(28, 32, 40))
    draw = ImageDraw.Draw(img)

    # Road & Car Front/Rear Bumper
    draw.rectangle([(0, 370), (w, h)], fill=(35, 38, 44))
    draw.rounded_rectangle([(120, 80), (680, 390)], radius=30, fill=(35, 38, 45), outline=(65, 75, 90), width=3)
    # Grille pattern
    for y_offset in range(130, 240, 16):
        draw.line([(220, y_offset), (580, y_offset)], fill=(20, 22, 28), width=8)

    # Headlights
    draw.polygon([(140, 140), (220, 155), (200, 220), (130, 190)], fill=(220, 235, 255), outline=(120, 180, 240), width=2)
    draw.polygon([(660, 140), (580, 155), (600, 220), (670, 190)], fill=(220, 235, 255), outline=(120, 180, 240), width=2)

    # Embossed Plate Mount
    draw.rounded_rectangle([(235, 275), (565, 355)], radius=6, fill=(15, 18, 22))

    # Embossed White Plate (Private/Commercial high security standard)
    draw.rounded_rectangle([(240, 280), (560, 350)], radius=5, fill=(245, 247, 250), outline=(20, 20, 25), width=3)

    # Top line: BAGMATI PROVINCE
    font_top = get_font(18)
    draw.text((400, 298), "BAGMATI PROVINCE", fill=(20, 25, 30), font=font_top, anchor="mm")

    # Bottom line: 02-035 PA 5678
    font_bot = get_font(28)
    draw.text((400, 328), "02-035 PA 5678", fill=(10, 15, 20), font=font_bot, anchor="mm")

    path = SAMPLES_DIR / "sample_embossed_plate.jpg"
    img.save(path, quality=95)
    return str(path)


def create_sample_bus_nepali():
    """Generates an image of a public bus with black traditional plate (सार्वजनिक सवारी)."""
    w, h = 800, 520
    img = Image.new("RGB", (w, h), color=(32, 36, 44))
    draw = ImageDraw.Draw(img)

    # Road
    draw.rectangle([(0, 420), (w, h)], fill=(42, 45, 52))

    # Bus front body (Big rectangular silhouette)
    draw.rounded_rectangle([(130, 40), (670, 430)], radius=20, fill=(210, 140, 25), outline=(250, 180, 50), width=4)
    # Windshield (large upper glass)
    draw.rounded_rectangle([(160, 60), (640, 210)], radius=12, fill=(25, 35, 45), outline=(100, 130, 160), width=2)
    # Destination board
    draw.rounded_rectangle([(280, 65), (520, 105)], radius=6, fill=(10, 15, 20))
    font_dest = get_font(22)
    draw.text((400, 85), "काठमाडौं - पोखरा", fill=(255, 220, 50), font=font_dest, anchor="mm")

    # Grille & Bumper
    draw.rectangle([(230, 250), (570, 310)], fill=(30, 32, 38))
    draw.rounded_rectangle([(110, 350), (690, 420)], radius=10, fill=(35, 38, 44), outline=(70, 75, 85), width=3)

    # Black Plate Mount (Public Transport)
    draw.rounded_rectangle([(295, 365), (505, 415)], radius=4, fill=(5, 5, 5), outline=(210, 210, 210), width=2)

    # Plate text: ना ५ ख १२३४ (White text on Black)
    font_plate = get_font(26)
    draw.text((400, 390), "ना ५ ख १२३४", fill=(255, 255, 255), font=font_plate, anchor="mm")

    path = SAMPLES_DIR / "sample_bus_nepali.jpg"
    img.save(path, quality=95)
    return str(path)


def generate_all_samples():
    """Generates all test sample assets."""
    print("[PlateVision] Generating realistic sample vehicle images for testing...")
    f1 = create_sample_car_nepali()
    f2 = create_sample_bike_nepali()
    f3 = create_sample_embossed()
    f4 = create_sample_bus_nepali()
    print("[PlateVision] Samples generated successfully:", [f1, f2, f3, f4])


if __name__ == "__main__":
    generate_all_samples()

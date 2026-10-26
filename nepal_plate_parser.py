"""
Nepal Vehicle Number Plate Parser and Normalizer
================================================
This module provides domain-specific parsing, normalization, validation,
and transliteration for all vehicle license plate formats used in Nepal:

1. Traditional Zonal Plates (Devanagari script):
   e.g., बा ९२ प ५६७८ (Bagmati 92 Pa 5678 - Motorcycle)
         ना ५ ख १२३४ (Narayani 5 Kha 1234 - Public Bus/Micro)
         बा १५ च ४३२१ (Bagmati 15 Cha 4321 - Private Car)
         बा १ ज ७८९० (Bagmati 1 Ja 7890 - Taxi)
         मे १ झ ०१२३ (Mechi 1 Jha 0123 - Government)

2. Traditional Provincial Plates (Devanagari script):
   e.g., बा.प्र. ०२-०३५ प ५६७८ (Bagmati Province Lot 02-035 Bike)
         प्रदेश ३ - ०१ ०२५ च ९८७६ (Province 3 Private Car)
         लु.प्र. ०१-००१ ख ४५६७ (Lumbini Province Bus)

3. High-Security Embossed Number Plates (Latin / English script):
   e.g., BAGMATI PROVINCE 02-035 PA 5678 (Two-wheeler)
         PROVINCE 1 01-025 CA 1234 (Light 4-wheeler car)
         03-001 BA 9999 (Heavy vehicle bus/truck)
         BA 2 PA 1234 (Pilot embossed format)
"""

import re
from typing import Dict, Optional, Tuple, Any

# ==============================================================================
# 1. CHARACTER & NUMERAL MAPPING DICTIONARIES
# ==============================================================================

# Mapping from Devanagari numerals to standard Arabic (English) digits
DEV_TO_ENG_DIGITS: Dict[str, str] = {
    "०": "0", "१": "1", "२": "2", "३": "3", "४": "4",
    "५": "5", "६": "6", "७": "7", "८": "8", "९": "9",
}

# Mapping from English digits to Devanagari numerals
ENG_TO_DEV_DIGITS: Dict[str, str] = {v: k for k, v in DEV_TO_ENG_DIGITS.items()}

# Devanagari Zonal abbreviations and their full names & English codes
NEPAL_ZONES: Dict[str, Dict[str, str]] = {
    "बा": {"en_code": "BA", "name_en": "Bagmati", "name_np": "बागमती"},
    "ना": {"en_code": "NA", "name_en": "Narayani", "name_np": "नारायणी"},
    "को": {"en_code": "KO", "name_en": "Koshi", "name_np": "कोशी"},
    "मे": {"en_code": "ME", "name_en": "Mechi", "name_np": "मेची"},
    "स": {"en_code": "SA", "name_en": "Sagarmatha", "name_np": "सगरमाथा"},
    "ज": {"en_code": "JA", "name_en": "Janakpur", "name_np": "जनकपुर"},
    "ग": {"en_code": "GA", "name_en": "Gandaki", "name_np": "गण्डकी"},
    "लु": {"en_code": "LU", "name_en": "Lumbini", "name_np": "लुम्बिनी"},
    "धौ": {"en_code": "DHAU", "name_en": "Dhaulagiri", "name_np": "धौलागिरी"},
    "ध": {"en_code": "DHAU", "name_en": "Dhaulagiri", "name_np": "धौलागिरी"},
    "रा": {"en_code": "RA", "name_en": "Rapti", "name_np": "राप्ती"},
    "क": {"en_code": "KA", "name_en": "Karnali", "name_np": "कर्णाली"},
    "भे": {"en_code": "BHE", "name_en": "Bheri", "name_np": "भेरी"},
    "से": {"en_code": "SE", "name_en": "Seti", "name_np": "सेती"},
    "म": {"en_code": "MA", "name_en": "Mahakali", "name_np": "महाकाली"},
}

# Devanagari Provincial abbreviations and aliases
NEPAL_PROVINCES: Dict[str, Dict[str, str]] = {
    "१": {"code": "01", "name_en": "Koshi Province", "name_np": "कोशी प्रदेश"},
    "०१": {"code": "01", "name_en": "Koshi Province", "name_np": "कोशी प्रदेश"},
    "कोशी": {"code": "01", "name_en": "Koshi Province", "name_np": "कोशी प्रदेश"},
    "२": {"code": "02", "name_en": "Madhesh Province", "name_np": "मधेश प्रदेश"},
    "०२": {"code": "02", "name_en": "Madhesh Province", "name_np": "मधेश प्रदेश"},
    "मधेश": {"code": "02", "name_en": "Madhesh Province", "name_np": "मधेश प्रदेश"},
    "३": {"code": "03", "name_en": "Bagmati Province", "name_np": "बागमती प्रदेश"},
    "०३": {"code": "03", "name_en": "Bagmati Province", "name_np": "बागमती प्रदेश"},
    "बा.प्र.": {"code": "03", "name_en": "Bagmati Province", "name_np": "बागमती प्रदेश"},
    "बाप्र": {"code": "03", "name_en": "Bagmati Province", "name_np": "बागमती प्रदेश"},
    "बागमती": {"code": "03", "name_en": "Bagmati Province", "name_np": "बागमती प्रदेश"},
    "४": {"code": "04", "name_en": "Gandaki Province", "name_np": "गण्डकी प्रदेश"},
    "०४": {"code": "04", "name_en": "Gandaki Province", "name_np": "गण्डकी प्रदेश"},
    "गण्डकी": {"code": "04", "name_en": "Gandaki Province", "name_np": "गण्डकी प्रदेश"},
    "५": {"code": "05", "name_en": "Lumbini Province", "name_np": "लुम्बिनी प्रदेश"},
    "०५": {"code": "05", "name_en": "Lumbini Province", "name_np": "लुम्बिनी प्रदेश"},
    "लुम्बिनी": {"code": "05", "name_en": "Lumbini Province", "name_np": "लुम्बिनी प्रदेश"},
    "६": {"code": "06", "name_en": "Karnali Province", "name_np": "कर्णाली प्रदेश"},
    "०६": {"code": "06", "name_en": "Karnali Province", "name_np": "कर्णाली प्रदेश"},
    "कर्णाली": {"code": "06", "name_en": "Karnali Province", "name_np": "कर्णाली प्रदेश"},
    "७": {"code": "07", "name_en": "Sudurpashchim Province", "name_np": "सुदूरपश्चिम प्रदेश"},
    "०७": {"code": "07", "name_en": "Sudurpashchim Province", "name_np": "सुदूरपश्चिम प्रदेश"},
    "सु.प.": {"code": "07", "name_en": "Sudurpashchim Province", "name_np": "सुदूरपश्चिम प्रदेश"},
    "सुप": {"code": "07", "name_en": "Sudurpashchim Province", "name_np": "सुदूरपश्चिम प्रदेश"},
    "सुदूरपश्चिम": {"code": "07", "name_en": "Sudurpashchim Province", "name_np": "सुदूरपश्चिम प्रदेश"},
}

# Devanagari Vehicle Category Letters & their meanings in Nepal's transport system
DEV_VEHICLE_CATEGORIES: Dict[str, Dict[str, str]] = {
    "प": {
        "translit": "PA",
        "category_en": "Two-Wheeler (Motorcycle / Scooter)",
        "category_np": "दुई पाङ्ग्रे (मोटरसाइकल / स्कुटर)",
        "vehicle_type": "motorcycle",
        "color_scheme": "Red background, White letters (Private)",
    },
    "च": {
        "translit": "CHA",
        "category_en": "Private Light 4-Wheeler (Car / Jeep / Van)",
        "category_np": "निजी चार पाङ्ग्रे (कार / जीप / भ्यान)",
        "vehicle_type": "car",
        "color_scheme": "Red background, White letters (Private)",
    },
    "ख": {
        "translit": "KHA",
        "category_en": "Public Passenger Vehicle (Bus / Microbus)",
        "category_np": "सार्वजनिक सवारी (बस / माइक्रोबस)",
        "vehicle_type": "bus",
        "color_scheme": "Black background, White letters (Public)",
    },
    "क": {
        "translit": "KA",
        "category_en": "Heavy Commercial Goods (Truck / Tanker)",
        "category_np": "ठूला ढुवानी सवारी (ट्रक / ट्याङ्कर)",
        "vehicle_type": "truck",
        "color_scheme": "Black background, White letters (Public/Freight)",
    },
    "ग": {
        "translit": "GA",
        "category_en": "Heavy Machinery / Tractor",
        "category_np": "हेभी उपकरण / ट्र्याक्टर",
        "vehicle_type": "truck",
        "color_scheme": "Black background, White letters",
    },
    "ज": {
        "translit": "JA",
        "category_en": "Taxi / Public Commercial Light",
        "category_np": "भाडाका साना सवारी (ट्याक्सी)",
        "vehicle_type": "car",
        "color_scheme": "Black background, White letters (Taxi)",
    },
    "झ": {
        "translit": "JHA",
        "category_en": "Government Vehicle",
        "category_np": "सरकारी सवारी",
        "vehicle_type": "car",
        "color_scheme": "White background, Red letters (Govt)",
    },
    "ब": {
        "translit": "BA",
        "category_en": "Public Corporation / Semi-Govt",
        "category_np": "संस्थान सवारी",
        "vehicle_type": "car",
        "color_scheme": "Yellow background, Blue letters (Corporation)",
    },
    "य": {
        "translit": "YA",
        "category_en": "Tourist Transport",
        "category_np": "पर्यटक सवारी",
        "vehicle_type": "car",
        "color_scheme": "Green background, White letters (Tourist)",
    },
    "सिडी": {
        "translit": "CD",
        "category_en": "Diplomatic Corps",
        "category_np": "कूटनीतिक नियोग (Diplomatic)",
        "vehicle_type": "car",
        "color_scheme": "Blue background, White letters (CD)",
    },
}

# English Embossed Category Codes (High Security Embossed Plates in Nepal)
EMBOSSED_VEHICLE_CATEGORIES: Dict[str, Dict[str, str]] = {
    "PA": {"category_en": "Two-Wheeler (Motorcycle / Scooter)", "vehicle_type": "motorcycle"},
    "PB": {"category_en": "Two-Wheeler (Motorcycle / Scooter - Series B)", "vehicle_type": "motorcycle"},
    "PC": {"category_en": "Two-Wheeler (Motorcycle / Scooter - Series C)", "vehicle_type": "motorcycle"},
    "CA": {"category_en": "Light Four-Wheeler (Car / Jeep / Van)", "vehicle_type": "car"},
    "CB": {"category_en": "Light Four-Wheeler (Car / Jeep / Van - Series B)", "vehicle_type": "car"},
    "CC": {"category_en": "Light Four-Wheeler (Car / Jeep / Van - Series C)", "vehicle_type": "car"},
    "BA": {"category_en": "Heavy Vehicle (Bus / Truck / Container)", "vehicle_type": "truck"},
    "BB": {"category_en": "Heavy Vehicle (Commercial Heavy - Series B)", "vehicle_type": "bus"},
    "BC": {"category_en": "Heavy Vehicle (Commercial Heavy - Series C)", "vehicle_type": "truck"},
    "DA": {"category_en": "Three-Wheeler (Auto-Rickshaw / Tempo)", "vehicle_type": "motorcycle"},
    "DB": {"category_en": "Three-Wheeler (Auto-Rickshaw - Series B)", "vehicle_type": "motorcycle"},
    "EA": {"category_en": "Tractor", "vehicle_type": "truck"},
    "FA": {"category_en": "Special / Construction Equipment", "vehicle_type": "truck"},
    "GA": {"category_en": "Government Vehicle", "vehicle_type": "car"},
    "CD": {"category_en": "Diplomatic Vehicle", "vehicle_type": "car"},
}

# English Province Names in Embossed Plates
EMBOSSED_PROVINCES: Dict[str, Dict[str, str]] = {
    "BAGMATI": {"code": "03", "name_en": "Bagmati Province", "name_np": "बागमती प्रदेश"},
    "KOSHI": {"code": "01", "name_en": "Koshi Province", "name_np": "कोशी प्रदेश"},
    "MADHESH": {"code": "02", "name_en": "Madhesh Province", "name_np": "मधेश प्रदेश"},
    "GANDAKI": {"code": "04", "name_en": "Gandaki Province", "name_np": "गण्डकी प्रदेश"},
    "LUMBINI": {"code": "05", "name_en": "Lumbini Province", "name_np": "लुम्बिनी प्रदेश"},
    "KARNALI": {"code": "06", "name_en": "Karnali Province", "name_np": "कर्णाली प्रदेश"},
    "SUDURPASHCHIM": {"code": "07", "name_en": "Sudurpashchim Province", "name_np": "सुदूरपश्चिम प्रदेश"},
}


# ==============================================================================
# 2. HELPER UTILITIES
# ==============================================================================

def devanagari_to_arabic_digits(text: str) -> str:
    """Replaces all Devanagari numerals with standard English Arabic digits."""
    return "".join(DEV_TO_ENG_DIGITS.get(ch, ch) for ch in text)


def arabic_to_devanagari_digits(text: str) -> str:
    """Replaces standard English digits with Devanagari numerals."""
    return "".join(ENG_TO_DEV_DIGITS.get(ch, ch) for ch in text)


def contains_devanagari(text: str) -> bool:
    """Checks whether the text contains Devanagari script characters."""
    return any("\u0900" <= ch <= "\u097F" for ch in text)


def clean_raw_plate_text(raw_text: str) -> str:
    """
    Cleans OCR output while preserving both Nepali Devanagari script
    and English alphanumeric characters, hyphens, and dots.
    """
    if not raw_text:
        return ""
    # Retain Devanagari characters (\u0900-\u097F), English alphanumeric (A-Za-z0-9), hyphens, dots, and whitespace
    cleaned = re.sub(r"[^\u0900-\u097FA-Za-z0-9\-\.\s]", " ", raw_text)
    # Collapse multiple whitespaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


# ==============================================================================
# 3. DOMAIN-SPECIFIC PARSING & CLASSIFICATION
# ==============================================================================

def parse_nepali_plate(raw_text: str, detected_vehicle_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Parses a raw OCR string from a Nepali number plate into a structured,
    normalized, and production-grade representation.

    Parameters
    ----------
    raw_text : str
        The raw recognized string from the OCR engine.
    detected_vehicle_type : str, optional
        Vehicle classification from YOLOv8 (e.g. 'car', 'motorcycle', 'bus', 'truck').

    Returns
    -------
    dict
        Parsed metadata including script, standardized representation,
        zone/province, lot, registration number, vehicle category, and validity.
    """
    cleaned = clean_raw_plate_text(raw_text)
    is_dev = contains_devanagari(cleaned)

    result: Dict[str, Any] = {
        "raw_text": raw_text.strip(),
        "cleaned_text": cleaned,
        "script": "Devanagari (नेपाली)" if is_dev else "Latin (English Embossed)",
        "plate_type": "Unknown",
        "standardized": "",
        "devanagari_display": "",
        "zone_or_province": "Unknown",
        "zone_province_code": "",
        "lot_number": "",
        "vehicle_symbol": "",
        "vehicle_category": "General Vehicle",
        "inferred_vehicle_type": detected_vehicle_type or "vehicle",
        "registration_number": "",
        "is_valid": False,
        "details": "",
    }

    if not cleaned:
        return result

    if is_dev:
        _parse_devanagari_plate(cleaned, result, detected_vehicle_type)
    else:
        _parse_embossed_plate(cleaned, result, detected_vehicle_type)

    return result


def _parse_devanagari_plate(cleaned: str, result: Dict[str, Any], vehicle_type: Optional[str]):
    """Internal parser for traditional Devanagari plates (Zonal & Provincial)."""
    # Convert numerals to english for easier pattern matching
    eng_num_text = devanagari_to_arabic_digits(cleaned)

    # 1. Check for Provincial Devanagari Format:
    # Pattern: बा.प्र. ०२-०३५ प ५६७८ or प्रदेश ३ ०१-००१ च १२३४ or बाप्र ०१ ००२ प १२३४
    prov_match = re.search(
        r"(बा\.?\s*प्र\.?|प्रदेश\s*[\d\u0966-\u096F]+|[क-ह\u0900-\u097F]{2,8}\s*प्रदेश)\s*([\d\u0966-\u096F]{1,3}[\-\s]*[\d\u0966-\u096F]{1,4})\s*([क-ह])\s*([\d\u0966-\u096F]{1,4})",
        cleaned
    )

    if prov_match:
        prov_prefix = prov_match.group(1).replace(" ", "")
        lot_str = devanagari_to_arabic_digits(prov_match.group(2).replace(" ", ""))
        veh_symbol = prov_match.group(3)
        reg_num = devanagari_to_arabic_digits(prov_match.group(4))

        # Lookup Province
        prov_info = {"name_en": "Bagmati Province", "code": "03"}
        for k, v in NEPAL_PROVINCES.items():
            if k in prov_prefix:
                prov_info = v
                break

        cat_info = DEV_VEHICLE_CATEGORIES.get(veh_symbol, {})
        veh_cat_en = cat_info.get("category_en", "Vehicle")
        veh_translit = cat_info.get("translit", veh_symbol)

        result.update({
            "plate_type": "Traditional Provincial",
            "zone_or_province": prov_info.get("name_en", "Bagmati Province"),
            "zone_province_code": prov_info.get("code", "03"),
            "lot_number": lot_str,
            "vehicle_symbol": veh_symbol,
            "vehicle_category": veh_cat_en,
            "inferred_vehicle_type": cat_info.get("vehicle_type", vehicle_type or "vehicle"),
            "registration_number": reg_num,
            "standardized": f"PROV-{prov_info.get('code', '03')} {lot_str} {veh_translit} {reg_num}",
            "devanagari_display": f"{prov_prefix} {arabic_to_devanagari_digits(lot_str)} {veh_symbol} {arabic_to_devanagari_digits(reg_num)}",
            "is_valid": True,
            "details": f"Provincial registration ({prov_info.get('name_en')}) for {veh_cat_en}",
        })
        return

    # 2. Check for Traditional Zonal Format:
    # Pattern: बा ९२ प ५६७८ or ना ५ ख १२३४ or बा १ ज ७८९० or मे १ झ ०१२३
    # In Devanagari, zones like बा, ना, मे contain matras (vowel signs \u093e-\u094f)
    zonal_match = re.search(
        r"([क-ह][\u0901-\u094F]*)\s*([\d\u0966-\u096F]{1,3})\s*([क-ह][\u0901-\u094F]*)\s*([\d\u0966-\u096F]{1,4})",
        cleaned
    )

    if zonal_match:
        zone_abbr = zonal_match.group(1).strip()
        lot_str = devanagari_to_arabic_digits(zonal_match.group(2).strip())
        veh_symbol = zonal_match.group(3).strip()
        reg_num = devanagari_to_arabic_digits(zonal_match.group(4).strip())

        zone_info = NEPAL_ZONES.get(zone_abbr, {"en_code": zone_abbr, "name_en": f"Zone {zone_abbr}"})
        cat_info = DEV_VEHICLE_CATEGORIES.get(veh_symbol, {})
        veh_cat_en = cat_info.get("category_en", "Vehicle")
        veh_translit = cat_info.get("translit", veh_symbol)

        result.update({
            "plate_type": "Traditional Zonal",
            "zone_or_province": zone_info.get("name_en", "Nepal Zone"),
            "zone_province_code": zone_info.get("en_code", zone_abbr),
            "lot_number": lot_str,
            "vehicle_symbol": veh_symbol,
            "vehicle_category": veh_cat_en,
            "inferred_vehicle_type": cat_info.get("vehicle_type", vehicle_type or "vehicle"),
            "registration_number": reg_num,
            "standardized": f"{zone_info.get('en_code', zone_abbr)} {lot_str} {veh_translit} {reg_num}",
            "devanagari_display": f"{zone_abbr} {arabic_to_devanagari_digits(lot_str)} {veh_symbol} {arabic_to_devanagari_digits(reg_num)}",
            "is_valid": True,
            "details": f"Traditional Zonal ({zone_info.get('name_en')}) - {veh_cat_en}",
        })
        return

    # 3. Fallback Heuristic for partially recognized Devanagari plates
    # Extract any digits and devanagari letters
    dev_letters = [ch for ch in cleaned if "\u0900" <= ch <= "\u097F" and not ("\u0966" <= ch <= "\u096F")]
    digits = devanagari_to_arabic_digits("".join([ch for ch in cleaned if ch.isdigit() or "\u0966" <= ch <= "\u096F"]))

    result.update({
        "plate_type": "Nepali Devanagari (Partial)",
        "standardized": cleaned,
        "devanagari_display": cleaned,
        "registration_number": digits[-4:] if len(digits) >= 4 else digits,
        "is_valid": len(digits) >= 3 and len(dev_letters) >= 1,
        "details": "Partially recognized Devanagari script plate",
    })


def _parse_embossed_plate(cleaned: str, result: Dict[str, Any], vehicle_type: Optional[str]):
    """Internal parser for High Security Embossed plates (English/Latin script)."""
    upper_text = cleaned.upper()

    # Pattern A: Full Embossed with Province Name:
    # e.g., BAGMATI PROVINCE 02-035 PA 5678 or BAGMATI 02 035 PA 5678
    emb_match_1 = re.search(
        r"(BAGMATI|KOSHI|MADHESH|GANDAKI|LUMBINI|KARNALI|SUDURPASHCHIM)(?:\s*PROVINCE)?\s*(\d{1,2}[\-\s]*\d{2,4})\s*([A-Z]{2})\s*(\d{1,4})",
        upper_text
    )

    if emb_match_1:
        prov_name = emb_match_1.group(1)
        lot_str = emb_match_1.group(2).replace(" ", "-")
        veh_code = emb_match_1.group(3)
        reg_num = emb_match_1.group(4)

        prov_info = EMBOSSED_PROVINCES.get(prov_name, {"code": "03", "name_en": prov_name.capitalize() + " Province"})
        cat_info = EMBOSSED_VEHICLE_CATEGORIES.get(veh_code, {"category_en": "General Vehicle", "vehicle_type": vehicle_type or "car"})

        result.update({
            "plate_type": "High Security Embossed",
            "zone_or_province": prov_info.get("name_en", prov_name),
            "zone_province_code": prov_info.get("code", "03"),
            "lot_number": lot_str,
            "vehicle_symbol": veh_code,
            "vehicle_category": cat_info.get("category_en", "Vehicle"),
            "inferred_vehicle_type": cat_info.get("vehicle_type", vehicle_type or "vehicle"),
            "registration_number": reg_num,
            "standardized": f"{prov_name} {lot_str} {veh_code} {reg_num}",
            "devanagari_display": f"{prov_info.get('name_np', prov_name)} {arabic_to_devanagari_digits(lot_str)} {veh_code} {arabic_to_devanagari_digits(reg_num)}",
            "is_valid": True,
            "details": f"Embossed plate ({prov_name}) - {cat_info.get('category_en')}",
        })
        return

    # Pattern B: Standard Embossed code format:
    # e.g., 02-035 PA 5678 or 03-001 CA 1234
    emb_match_2 = re.search(
        r"(\d{2})[\-\s]+(\d{3,4})\s*([A-Z]{2})\s*(\d{1,4})",
        upper_text
    )

    if emb_match_2:
        prov_code = emb_match_2.group(1)
        lot_seq = emb_match_2.group(2)
        veh_code = emb_match_2.group(3)
        reg_num = emb_match_2.group(4)

        # Lookup Province by numeric code
        prov_names = {
            "01": "Koshi Province",
            "02": "Madhesh Province",
            "03": "Bagmati Province",
            "04": "Gandaki Province",
            "05": "Lumbini Province",
            "06": "Karnali Province",
            "07": "Sudurpashchim Province",
        }
        prov_name = prov_names.get(prov_code, f"Province {prov_code}")
        cat_info = EMBOSSED_VEHICLE_CATEGORIES.get(veh_code, {"category_en": "General Vehicle", "vehicle_type": vehicle_type or "car"})

        result.update({
            "plate_type": "High Security Embossed",
            "zone_or_province": prov_name,
            "zone_province_code": prov_code,
            "lot_number": f"{prov_code}-{lot_seq}",
            "vehicle_symbol": veh_code,
            "vehicle_category": cat_info.get("category_en", "Vehicle"),
            "inferred_vehicle_type": cat_info.get("vehicle_type", vehicle_type or "vehicle"),
            "registration_number": reg_num,
            "standardized": f"{prov_code}-{lot_seq} {veh_code} {reg_num}",
            "devanagari_display": f"{arabic_to_devanagari_digits(prov_code)}-{arabic_to_devanagari_digits(lot_seq)} {veh_code} {arabic_to_devanagari_digits(reg_num)}",
            "is_valid": True,
            "details": f"Embossed plate ({prov_name}) - {cat_info.get('category_en')}",
        })
        return

    # Pattern C: Pilot Embossed / Zonal English format:
    # e.g., BA 2 PA 1234 or NA 5 KHA 5678
    emb_match_3 = re.search(
        r"([A-Z]{2})\s*(\d{1,3})\s*([A-Z]{1,3})\s*(\d{1,4})",
        upper_text
    )

    if emb_match_3:
        zone_code = emb_match_3.group(1)
        lot_str = emb_match_3.group(2)
        veh_symbol = emb_match_3.group(3)
        reg_num = emb_match_3.group(4)

        cat_info = EMBOSSED_VEHICLE_CATEGORIES.get(veh_symbol, {"category_en": "Vehicle", "vehicle_type": vehicle_type or "car"})

        result.update({
            "plate_type": "English Embossed / Transliterated Zonal",
            "zone_or_province": f"Zone {zone_code}",
            "zone_province_code": zone_code,
            "lot_number": lot_str,
            "vehicle_symbol": veh_symbol,
            "vehicle_category": cat_info.get("category_en", "Vehicle"),
            "inferred_vehicle_type": cat_info.get("vehicle_type", vehicle_type or "vehicle"),
            "registration_number": reg_num,
            "standardized": f"{zone_code} {lot_str} {veh_symbol} {reg_num}",
            "devanagari_display": f"{zone_code} {arabic_to_devanagari_digits(lot_str)} {veh_symbol} {arabic_to_devanagari_digits(reg_num)}",
            "is_valid": True,
            "details": f"Standardized English format ({zone_code})",
        })
        return

    # Fallback for English text
    digits = "".join([c for c in upper_text if c.isdigit()])
    letters = "".join([c for c in upper_text if c.isalpha()])

    result.update({
        "plate_type": "Latin / English (Unstructured)",
        "standardized": upper_text,
        "devanagari_display": arabic_to_devanagari_digits(upper_text),
        "registration_number": digits[-4:] if len(digits) >= 4 else digits,
        "is_valid": len(digits) >= 3 and len(letters) >= 2,
        "details": "Partially structured English plate text",
    })

"""
Location normalization & intelligence.

Parses free-text location strings and resume text into structured:
  {city, region, state, country, postal_code, formatted}
Normalizes country names to ISO-3166 alpha-2 codes.
Supports Indian PIN codes, US ZIP codes, global formats, and contextual extraction.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

# ─── ISO-3166 alpha-2 lookup (common names → code) ───────────────────────────
COUNTRY_ALIASES: dict[str, str] = {
    # Full names
    "united states": "US", "united states of america": "US", "usa": "US", "us": "US",
    "united kingdom": "GB", "uk": "GB", "great britain": "GB",
    "canada": "CA", "ca": "CA",
    "australia": "AU", "au": "AU",
    "india": "IN", "in": "IN",
    "germany": "DE", "de": "DE", "deutschland": "DE",
    "france": "FR", "fr": "FR",
    "netherlands": "NL", "nl": "NL", "the netherlands": "NL", "holland": "NL",
    "singapore": "SG", "sg": "SG",
    "brazil": "BR", "br": "BR",
    "mexico": "MX", "mx": "MX",
    "spain": "ES", "es": "ES",
    "italy": "IT", "it": "IT",
    "sweden": "SE", "se": "SE",
    "norway": "NO", "no": "NO",
    "denmark": "DK", "dk": "DK",
    "finland": "FI", "fi": "FI",
    "switzerland": "CH", "ch": "CH",
    "austria": "AT", "at": "AT",
    "poland": "PL", "pl": "PL",
    "russia": "RU", "ru": "RU",
    "china": "CN", "cn": "CN",
    "japan": "JP", "jp": "JP",
    "south korea": "KR", "korea": "KR", "kr": "KR",
    "israel": "IL", "il": "IL",
    "new zealand": "NZ", "nz": "NZ",
    "ireland": "IE", "ie": "IE",
    "portugal": "PT", "pt": "PT",
    "belgium": "BE", "be": "BE",
    "argentina": "AR", "ar": "AR",
    "chile": "CL", "cl": "CL",
    "colombia": "CO", "co": "CO",
    "south africa": "ZA", "za": "ZA",
    "nigeria": "NG", "ng": "NG",
    "kenya": "KE", "ke": "KE",
    "uae": "AE", "ae": "AE", "united arab emirates": "AE",
    "saudi arabia": "SA", "sa": "SA",
    "egypt": "EG", "eg": "EG",
    "turkey": "TR", "tr": "TR",
    "pakistan": "PK", "pk": "PK",
    "bangladesh": "BD", "bd": "BD",
    "vietnam": "VN", "vn": "VN",
    "thailand": "TH", "th": "TH",
    "indonesia": "ID", "id": "ID",
    "malaysia": "MY", "my": "MY",
    "philippines": "PH", "ph": "PH",
    "taiwan": "TW", "tw": "TW",
    "hong kong": "HK", "hk": "HK",
}

# US state abbreviations
US_STATES = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN",
    "IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV",
    "NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN",
    "TX","UT","VT","VA","WA","WV","WI","WY","DC"
}

# Indian PIN code prefix (first 2 digits) → State/Territory
INDIAN_PIN_PREFIX_MAP: dict[str, str] = {
    "11": "Delhi",
    "12": "Haryana", "13": "Haryana",
    "14": "Punjab", "15": "Punjab", "16": "Chandigarh",
    "17": "Himachal Pradesh",
    "18": "Jammu & Kashmir", "19": "Jammu & Kashmir",
    "20": "Uttar Pradesh", "21": "Uttar Pradesh", "22": "Uttar Pradesh", "23": "Uttar Pradesh",
    "24": "Uttarakhand", "25": "Uttar Pradesh", "26": "Uttar Pradesh", "27": "Uttar Pradesh", "28": "Uttar Pradesh",
    "30": "Rajasthan", "31": "Rajasthan", "32": "Rajasthan", "33": "Rajasthan", "34": "Rajasthan",
    "36": "Gujarat", "37": "Gujarat", "38": "Gujarat", "39": "Gujarat",
    "40": "Maharashtra", "41": "Maharashtra", "42": "Maharashtra", "43": "Maharashtra", "44": "Maharashtra",
    "45": "Madhya Pradesh", "46": "Madhya Pradesh", "47": "Madhya Pradesh", "48": "Madhya Pradesh", "49": "Chhattisgarh",
    "50": "Telangana", "51": "Andhra Pradesh", "52": "Andhra Pradesh", "53": "Andhra Pradesh",
    "56": "Karnataka", "57": "Karnataka", "58": "Karnataka", "59": "Karnataka",
    "60": "Tamil Nadu", "61": "Tamil Nadu", "62": "Tamil Nadu", "63": "Tamil Nadu", "64": "Tamil Nadu",
    "67": "Kerala", "68": "Kerala", "69": "Kerala",
    "70": "West Bengal", "71": "West Bengal", "72": "West Bengal", "73": "West Bengal", "74": "West Bengal",
    "75": "Odisha", "76": "Odisha", "77": "Odisha",
    "78": "Assam", "79": "Northeast",
    "80": "Bihar", "81": "Bihar", "82": "Jharkhand", "83": "Jharkhand", "84": "Bihar", "85": "Bihar",
}

# Indian States Canonical Map
INDIAN_STATES = {
    "tamil nadu": "Tamil Nadu",
    "karnataka": "Karnataka",
    "kerala": "Kerala",
    "maharashtra": "Maharashtra",
    "telangana": "Telangana",
    "andhra pradesh": "Andhra Pradesh",
    "gujarat": "Gujarat",
    "rajasthan": "Rajasthan",
    "delhi": "Delhi",
    "uttar pradesh": "Uttar Pradesh",
    "west bengal": "West Bengal",
    "madhya pradesh": "Madhya Pradesh",
    "punjab": "Punjab",
    "haryana": "Haryana",
    "odisha": "Odisha",
    "bihar": "Bihar",
    "jharkhand": "Jharkhand",
    "assam": "Assam",
    "chandigarh": "Chandigarh",
    "puducherry": "Puducherry",
    "goa": "Goa",
}

_INDIAN_PIN_RE = re.compile(r"\b([1-9][0-9]{5})\b")
_US_ZIP_RE = re.compile(r"\b([0-9]{5}(?:-[0-9]{4})?)\b")

_LOCATION_NOISE_RE = re.compile(
    r"\b(?:college|university|institute|campus|school|academy|polytechnic|"
    r"engineering|technology|science|management|solutions|infotech|internpe|"
    r"private|limited|pvt|ltd|inc|corp|github|linkedin|gmail|yahoo)\b",
    re.I,
)


def normalize_country(raw: str) -> Optional[str]:
    """Map a country name/alias to ISO-3166 alpha-2, or return None if unrecognized."""
    if not raw:
        return None
    clean = raw.strip().lower()
    if clean in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[clean]
    if len(raw.strip()) == 2 and raw.strip().upper() in COUNTRY_ALIASES.values():
        return raw.strip().upper()
    return None


def parse_location(raw: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Parse a free-text location into (city, region, country).
    Preserves backward compatibility for 3-tuple callers.
    """
    res = parse_location_detailed(raw)
    return res["city"], res["region"], res["country"]


def parse_location_detailed(raw: str) -> Dict[str, Any]:
    """
    Advanced location parser handling:
      - "Dindigul - 624616"      -> city: Dindigul, postal_code: 624616, state: Tamil Nadu, country: IN
      - "Tirunelveli-627011"     -> city: Tirunelveli, postal_code: 627011, state: Tamil Nadu, country: IN
      - "Chennai - 600001"       -> city: Chennai, postal_code: 600001, state: Tamil Nadu, country: IN
      - "Coimbatore, Tamil Nadu" -> city: Coimbatore, state: Tamil Nadu, country: IN
      - "San Francisco, CA"      -> city: San Francisco, state: CA, country: US
      - "Austin, TX 78701"       -> city: Austin, state: TX, postal_code: 78701, country: US
    """
    if not raw or not str(raw).strip():
        return {
            "city": None,
            "region": None,
            "state": None,
            "country": None,
            "postal_code": None,
            "pin_code": None,
            "formatted": None,
        }

    raw = str(raw).strip()

    city: Optional[str] = None
    region: Optional[str] = None
    country_code: Optional[str] = None
    postal_code: Optional[str] = None

    # Check for Indian PIN code pattern (e.g. "Dindigul - 624616" or "Tirunelveli-627011")
    pin_m = _INDIAN_PIN_RE.search(raw)
    if pin_m:
        pin = pin_m.group(1)
        postal_code = pin
        # Remove pin from text to isolate city/region
        rem = raw[:pin_m.start()] + raw[pin_m.end():]
        rem = re.sub(r"[\-–—,\s]+$", "", rem).strip()
        rem = re.sub(r"^[\-–—,\s]+", "", rem).strip()

        # If a city or region remains in rem
        if rem:
            rem_parts = [p.strip() for p in re.split(r"[,/]", rem) if p.strip()]
            if len(rem_parts) == 1:
                city = rem_parts[0]
            elif len(rem_parts) >= 2:
                city = rem_parts[0]
                possible_region = rem_parts[1].lower()
                if possible_region in INDIAN_STATES:
                    region = INDIAN_STATES[possible_region]
                else:
                    region = rem_parts[1]

        # Derive state from PIN prefix if not explicitly set
        prefix = pin[:2]
        if not region and prefix in INDIAN_PIN_PREFIX_MAP:
            region = INDIAN_PIN_PREFIX_MAP[prefix]

        country_code = "IN"

    else:
        # Standard comma/delimiter-separated parsing
        parts = [p.strip() for p in re.split(r"[,/]", raw) if p.strip()]

        if len(parts) == 1:
            candidate = parts[0]
            code = normalize_country(candidate)
            if code:
                country_code = code
            elif candidate.lower() in INDIAN_STATES:
                region = INDIAN_STATES[candidate.lower()]
                country_code = "IN"
            elif candidate.upper() in US_STATES:
                region = candidate.upper()
                country_code = "US"
            else:
                city = candidate

        elif len(parts) == 2:
            city = parts[0]
            second = parts[1].strip()
            second_lower = second.lower()

            if second.upper() in US_STATES:
                region = second.upper()
                country_code = "US"
            elif second_lower in INDIAN_STATES:
                region = INDIAN_STATES[second_lower]
                country_code = "IN"
            else:
                code = normalize_country(second)
                if code:
                    country_code = code
                else:
                    region = second

        elif len(parts) >= 3:
            city = parts[0]
            second = parts[1].strip()
            second_lower = second.lower()

            if second.upper() in US_STATES:
                region = second.upper()
                country_code = "US"
            elif second_lower in INDIAN_STATES:
                region = INDIAN_STATES[second_lower]
                country_code = "IN"
            else:
                region = second

            last = parts[-1].strip()
            code = normalize_country(last)
            if code:
                country_code = code
            elif not country_code and region and region.upper() in US_STATES:
                country_code = "US"

    # Clean city of residual punctuation
    if city:
        city = re.sub(r"[\-–—,\.]+$", "", city).strip()
        # Clean title casing if all lowercase or caps
        if city.islower() or city.isupper():
            city = city.title()

    # Formatted output
    display_parts = []
    if city:
        display_parts.append(city)
    if region:
        display_parts.append(region)
    if postal_code:
        display_parts.append(postal_code)
    if country_code:
        display_parts.append("India" if country_code == "IN" else country_code)

    formatted = ", ".join(display_parts) if display_parts else raw

    return {
        "city": city,
        "region": region,
        "state": region,
        "country": country_code,
        "postal_code": postal_code,
        "pin_code": postal_code,
        "formatted": formatted,
    }


def normalize_location_string(raw: str) -> dict:
    """
    Return a dict with city, region, state, country, postal_code, formatted keys.
    Maintains 100% backward compatibility for callers expecting dictionary.
    """
    return parse_location_detailed(raw)


def extract_candidate_residence(text: str) -> Dict[str, Any]:
    """
    Specifically extracts candidate residential location from resume header/contact sections.
    Rejects employer, university, or project location noise to prevent conflation.
    """
    if not text:
        return parse_location_detailed("")

    # Look in the first 800 characters or header lines
    header_chunk = text[:1000]
    # Strip emails, URLs, and phones so domain names (like .com) or phone digits don't bleed into city names
    clean_chunk = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", " ", header_chunk)
    clean_chunk = re.sub(r"https?://\S+", " ", clean_chunk)
    clean_chunk = re.sub(r"(?:www\.)?[a-zA-Z0-9\-]+\.(?:com|in|org|net|io|edu|co)\S*", " ", clean_chunk)

    # Pattern A: "City - 624616" or "City-624004"
    pin_match = re.search(r"\b([A-Za-z][A-Za-z\s]{2,25})\s*[\-–—]\s*([1-9][0-9]{5})\b", clean_chunk)
    if pin_match:
        cand_city = pin_match.group(1).strip()
        cand_city = re.sub(r"^(?:com|in|org|net|gmail|yahoo)\s+", "", cand_city, flags=re.I).strip()
        if not _LOCATION_NOISE_RE.search(cand_city):
            return parse_location_detailed(f"{cand_city} - {pin_match.group(2)}")

    # Pattern B: "City, State" (e.g. Coimbatore, Tamil Nadu)
    for state_name, canonical_state in INDIAN_STATES.items():
        m = re.search(r"\b([A-Za-z][A-Za-z\s]{2,20})\s*,\s*" + re.escape(state_name) + r"\b", clean_chunk, re.I)
        if m:
            c = m.group(1).strip()
            c = re.sub(r"^(?:com|in|org|net|gmail|yahoo)\s+", "", c, flags=re.I).strip()
            if not _LOCATION_NOISE_RE.search(c):
                return parse_location_detailed(f"{c}, {canonical_state}")

    # Pattern C: Explicit label
    label_m = re.search(
        r"(?:location|address|residence|city|permanent\s+address|current\s+location)\s*[:\-]?\s*([A-Za-z0-9\s\-,\.]+?)(?:\n|$)",
        header_chunk, re.I
    )
    if label_m:
        cand_loc = label_m.group(1).strip()
        if not _LOCATION_NOISE_RE.search(cand_loc) and len(cand_loc) < 80:
            return parse_location_detailed(cand_loc)

    # Pattern D: US "City, ST" or "City, Country"
    us_m = re.search(r"\b([A-Z][a-z]+(?:[ \t][A-Z][a-z]+)*),\s*([A-Z]{2})\b", header_chunk)
    if us_m and us_m.group(2) in US_STATES:
        c = us_m.group(1).strip()
        if not _LOCATION_NOISE_RE.search(c):
            return parse_location_detailed(f"{c}, {us_m.group(2)}")

    return parse_location_detailed("")

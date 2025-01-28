import re

def extract_iocs(text: str):
    """
    Extract IOCs from text
    """
    assert type(text) == str, "Input must be a string"
    assert len(text) > 0, "Input must not be empty"

    # extract IOCs
    patterns = {
        "ip_addr": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "domain": r"\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b",
        "url": r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        "md5": r"\b[a-fA-F0-9]{32}\b",
        "sha256": r"\b[a-fA-F0-9]{64}\b"
    }

    extracted = []
    for ioc_type, pattern in patterns.items():
        matches = re.findall(pattern, text)
        if matches:
            extracted.extend(matches)

    return extracted

# test
text = """
    This is a sample text containing some IOCs:
    - IP Address: 192.168.1.1
    - Email: attacker@example.com
    - Domain: malicious-domain.com
    - URL: http://bad-url.com/path
    - MD5 Hash: d41d8cd98f00b204e9800998ecf8427e
    - SHA256 Hash: 
      3f79bb7b435b05321651daefd374cdc38e2d6cebd3b9f82e9bbce3a3c969e29d
"""

print(extract_iocs(text))
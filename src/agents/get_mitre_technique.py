#from smolagents import tool
import requests
from bs4 import BeautifulSoup

#@tool
def get_mitre_technique(technique_id: str) -> str:
    """
    This tool retrieves the description of a MITRE technique given its id.

    Args:
        technique_id: The id of the MITRE technique to look for.
    """
    technique_id, sub_id = technique_id.split(".") if "." in technique_id else (technique_id, "")
    technique_id = technique_id.strip()
    sub_id = sub_id.strip()
    url = f"https://attack.mitre.org/techniques/{technique_id}/" if not sub_id else f"https://attack.mitre.org/techniques/{technique_id}/{sub_id}/"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        technique_name = soup.find("div", {"class": "description-body"}).text
        return technique_name
    return "Unknown"
import requests
import dotenv
import os
import json

def get_HTML_report(hash:str, sandbox:str):
    dotenv.load_dotenv()
    api_key = os.getenv("VT_API_KEY")
    if api_key is None:
        raise ValueError("VT_API_KEY is not set")
    
    id = f"{hash}_{sandbox}"
    url = f"https://www.virustotal.com/api/v3/file_behaviours/{id}/html"
    headers = {
        "accept": "text/plain",
        "x-apikey": api_key}
    response = requests.get(url, headers=headers)
    return response.text


if __name__ == '__main__':
    h = "19cef7f32e42cc674f7c76be3a5c691c543f4e018486c29153e7dde1a48af34c"
    sandbox = "CAPE Sandbox"
    report = get_HTML_report(h, sandbox)

    with open("report.json", "w") as f:
        json.dump(report, f)

import requests
import bs4
import json

if __name__ == "__main__":

    output = {}

    # scrap tactics
    url = "https://attack.mitre.org/tactics/enterprise/"
    response = requests.get(url)
    soup = bs4.BeautifulSoup(response.text, "html.parser")

    # get table of tactics
    table = soup.find_all("table")[0]
    rows = table.find_all("tr")[1:]

    tactics = []
    for row in rows:
        cells = row.find_all("td")
        tactic = {
            "id": cells[0].text.strip(),
            "name": cells[1].text.strip().lower().replace(" ", "-"),
            "description": cells[2].text.strip()
        }
        tactics.append(tactic)

    for tactic in tactics:
        output[tactic["name"]] = {
            "id": tactic["id"],
            "name": tactic["name"].replace(" ", "-"),
            "description": tactic["description"],
            "techniques": []
        }
        
        # scrap techniques
        url = "https://attack.mitre.org/tactics/" + tactic["id"] + "/"
        response = requests.get(url)
        soup = bs4.BeautifulSoup(response.text, "html.parser")

        # get table of techniques
        table = soup.find_all("table")[0]
        rows = table.find_all("tr")[1:]

        techniques = []
        prev_id = ''
        for row in rows:
            technique = {}

            cells = row.find_all("td")
            technique["id"] = cells[0].text.strip()

            if technique["id"] == '':
                technique["id"] = prev_id + cells[1].text.strip()
                technique["name"] = cells[2].text.strip().lower().replace(" ", "-")
                technique["description"] = cells[3].text.strip()

            else:
                prev_id = technique["id"]
                technique["name"] = cells[1].text.strip().lower()
                technique["description"] = cells[2].text.strip()

            techniques.append(technique)

        output[tactic["name"]]['techniques'] = techniques

    with open("mitre-techniques.json", "w") as f:
        f.write(json.dumps(output))
import json, os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet
from src.STIXParser import STIXParser
import datetime

def create_pdf_from_json(malware, out, output_pdf):
    # Define PDF document
    doc = SimpleDocTemplate(output_pdf, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(Paragraph(f"<b>Enhanced Attack Report</b>", styles["Heading1"]))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph(f"{malware['name']}", styles["Heading2"]))
    elements.append(Paragraph(f"<i>Generated on {datetime.datetime.today().strftime('%Y-%m-%d')}</i>", styles["Normal"]))
    elements.append(PageBreak())

    # Disclaimer
    elements.append(Paragraph(f"<b>Disclaimer</b>", styles["Heading2"]))
    disclaimer = """
    This report has been generated automatically and should be used for informational purposes only.
    To generate this report, Large Language Models (LLMs) were used to analyze the provided data.
    This technology is not perfect and may generate incorrect or misleading results.
    The results should be reviewed by a human expert before taking any action based on the information provided.
    """
    elements.append(Paragraph(disclaimer, styles["Normal"]))
    elements.append(Spacer(1, 10))
    elements.append(PageBreak())

    # Definitions
    elements.append(Paragraph(f"<b>Definitions</b>", styles["Heading2"]))
    definition = """
    <b>Pre-Conditions:</b> Conditions that must be true to execute the attack steps in the milestone.
    """
    elements.append(Paragraph(definition, styles["Normal"]))
    elements.append(Spacer(1, 10))
    definition = """
    <b>Post-Conditions:</b> Traces that an attacker leaves behind after executing the attack steps in the milestone.
    """
    elements.append(Paragraph(definition, styles["Normal"]))
    elements.append(Spacer(1, 10))
    definition = """
    <b>Attack Steps:</b> Steps that an attacker would take to achieve the goal of the milestone.
    """
    elements.append(Paragraph(definition, styles["Normal"]))
    elements.append(Spacer(1, 10))
    definition = """
    <b>MITRE Technique:</b> Techniques from the MITRE ATT&CK framework that are relevant to the milestone.
    """
    elements.append(Paragraph(definition, styles["Normal"]))
    elements.append(Spacer(1, 10))
    elements.append(PageBreak())

    # STIX Information
    elements.append(Paragraph(f"<b>STIX</b>", styles["Heading2"]))
    elements.append(Paragraph(f"<b>Malware Name:</b> {malware['name']}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Malware Description:</b> {malware['description']}", styles["Normal"]))
    elements.append(Spacer(1, 10))
    elements.append(PageBreak())

    for i,item in enumerate(out):
        # Title
        elements.append(Paragraph(f"<b>Milestone {i+1}</b>", styles["Title"]))
        elements.append(Spacer(1, 10))

        # Pre-Conditions (Numbered List)
        elements.append(Paragraph("<b>Pre-Conditions:</b>", styles["Heading2"]))
        pre_conditions_list = ListFlowable(
            [ListItem(Paragraph(cond, styles["Normal"])) for cond in item["pre-conditions"]],
            bulletType="bullet"  # Numbered List
        )
        elements.append(pre_conditions_list)
        elements.append(Spacer(1, 10))

        # Post-Conditions (Numbered List)
        elements.append(Paragraph("<b>Post-Conditions:</b>", styles["Heading2"]))
        post_conditions_list = ListFlowable(
            [ListItem(Paragraph(cond, styles["Normal"])) for cond in item["post-conditions"]],
            bulletType="bullet"  # Numbered List
        )
        elements.append(post_conditions_list)
        elements.append(Spacer(1, 10))

        # Attack Steps
        for k, attack_step in enumerate(item["attack_steps"]):
            elements.append(Paragraph(f"<b>Attack Step {i+1}.{k+1}</b>", styles["Heading2"]))
            elements.append(Paragraph("="*50, styles["Normal"]))
            elements.append(Paragraph(f"<b>Name:</b> {attack_step['name']}", styles["Normal"]))
            elements.append(Paragraph(f"<b>Description:</b> {attack_step['description']}", styles["Normal"]))
        
            # MITRE Technique
            mitre = attack_step["mitre_technique"]
            elements.append(Paragraph("<b>MITRE Technique</b>", styles["Heading3"]))
            elements.append(Paragraph(f"<b>ID:</b> {mitre['id']}", styles["Normal"]))
            elements.append(Paragraph(f"<b>Name:</b> {mitre['name']}", styles["Normal"]))
            elements.append(Paragraph(f"<b>Description:</b> {mitre['description']}", styles["Normal"]))

            tec, sub = mitre['id'].split('.') if '.' in mitre['id'] else (mitre['id'], "")
            url = 'https://attack.mitre.org/techniques/' + tec + '/' + sub
            elements.append(Paragraph(f"<b>More info:</b> {url}", styles["Normal"]))

            # Indicators
            indicators = attack_step["indicators"]
            elements.append(Paragraph("<b>Indicators</b>", styles["Heading3"]))
            if len(indicators) == 0:
                elements.append(Paragraph("No indicators found.", styles["Normal"]))
            else:
                indicators_list = ListFlowable(
                    [ListItem(Paragraph(indicator, styles["Normal"])) for indicator in indicators],
                    bulletType="bullet"  # Bullet List
                )
                elements.append(indicators_list)
            elements.append(Spacer(1, 10))
        
        # Add a page break after each item
        elements.append(PageBreak())

    # Build PDF
    doc.build(elements)
    print(f"PDF report generated: {output_pdf}")

if __name__ == "__main__":
    config_file = "config.json"

    # Load JSON data
    config = json.load(open(config_file))
    out = json.load(open(os.path.join(config['OUTPUT_DIR'], config['OUTPUT_FILE'])))

    stix_parser = STIXParser()
    stix_parser.parse(config['STIX_FILE'])
    malware = stix_parser.extract_malware()[0]
    malware_name = malware["name"]
    filename = malware_name.replace(" ", "_").lower()

    create_pdf_from_json(malware, out, os.path.join(config['OUTPUT_DIR'], f"{filename}_report.pdf"))

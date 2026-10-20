from io import BytesIO
from pathlib import Path
from html import escape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.pagesizes import A4,landscape
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle
from reportlab.pdfgen import canvas
from src.common import ROOT

NAVY=colors.HexColor("#14354b")
TEAL=colors.HexColor("#087f83")


def footer(c,doc):
    c.setFont("Helvetica",8);c.setFillColor(NAVY)
    c.drawString(42,25,"CARDIO TWIN / Research prototype / Synthetic patient demonstration")
    c.drawRightString(A4[0]-42,25,str(doc.page))


def patient_pdf(twin,path=None):
    buffer=BytesIO()
    doc=SimpleDocTemplate(buffer,pagesize=A4,rightMargin=42,leftMargin=42,topMargin=40,bottomMargin=42)
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name="SmallNote",fontSize=9,leading=13,textColor=NAVY,spaceAfter=8))
    story=[Paragraph("CARDIO TWIN",styles["Title"]),
        Paragraph("Synthetic patient research report",styles["Heading2"]),
        Paragraph("Research prototype - not a certified medical device. No prospective cardiac-event prediction is provided.",styles["SmallNote"])]
    def p(text):
        story.append(Paragraph(escape(str(text)),styles["SmallNote"]))
    profile=twin.state.profile
    p(f"Patient: {profile['patient_id']} | Age: {profile.get('age')} | Gender: {profile.get('gender')}")
    p("Profile and wearable trajectory in this report are invented demonstration data.")
    story.append(Paragraph("Clinical profile and baseline classification",styles["Heading2"]))
    p(f"BMI {profile.get('bmi')}; recorded diabetes {profile.get('diabetes')}; smoking history {profile.get('smoking_history')}; "
      f"sleep_hours {profile.get('sleep_hours')}. Source coding is unverified.")
    for target,r in twin.estimate_baseline_risk().items():
        p(f"{target}: {r['score']:.3f}. Classification of recorded status, not future event probability.")
        p(f"Input completeness: {r['quality']['completeness']:.0%}; imputed fields: {', '.join(r['quality']['imputed_fields']) or 'none'}.")
        p(f"Outside training range: {', '.join(r['quality']['outside_training_range']) or 'none detected'}.")
        p(r["quality"]["uncertainty"])
    story.append(Paragraph("Wearable course and individualized deviations",styles["Heading2"]))
    p(f"Observations ingested: {len(twin.state.observations)}. Last update: "
      f"{twin.state.observations[-1]['timestamp'] if twin.state.observations else 'none'}.")
    p(f"Flagged observations: {sum(r.get('flag',False) for r in twin.state.anomalies)}. "
      f"Current smoothed deviation index: {twin.state.latent_state:.2f}.")
    if twin.state.observations:
        from src.data.loader import SENSORS
        import numpy as np
        rows=[["Synthetic sensor","First observed","Last observed"]]
        for c in SENSORS:
            vals=[r[c] for r in twin.state.observations if r.get(c) is not None]
            rows.append([c,f"{vals[0]:.2f}" if vals else "Missing",f"{vals[-1]:.2f}" if vals else "Missing"])
        table=Table(rows,colWidths=[245,100,100])
        table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),NAVY),("TEXTCOLOR",(0,0),(-1,0),colors.white),
            ("FONTSIZE",(0,0),(-1,-1),8),("BOTTOMPADDING",(0,0),(-1,-1),8),
            ("GRID",(0,0),(-1,-1),.3,colors.HexColor("#d4e1e8"))]))
        story.append(table)
    story.append(Paragraph("Model explanation",styles["Heading2"]))
    for target,values in twin.explain_prediction().items():
        p(f"{target}: top reference substitutions: "+ "; ".join(f"{r['feature']} ({r['score_difference']:+.3f})" for r in values[:5]))
    p("Substitution differences are non-additive associations. They are not causal effects or an individual uncertainty interval.")
    story.append(Paragraph("Hypothetical scenarios",styles["Heading2"]))
    if not twin.state.interventions:p("No intervention scenario was executed.")
    for result in twin.state.interventions:
        p(result["disclaimer"])
        p(f"{result['intervention']} over {result['duration_days']} days; assumptions: {result['assumed_feature_changes']}")
        for target,v in result["classification_scenarios"].items():
            p(f"{target}: {v['before']:.3f} to {v['after']:.3f}; change {v['difference']:+.3f}. No treatment effectiveness is inferred.")
    story.append(Paragraph("Evidence limitations and clinical review",styles["Heading2"]))
    p("Source dataset is cross-sectional with unknown device linkage, units, score direction and redistribution rights. "
      "Internal held-out classification does not establish external clinical validity. Synthetic anomalies reflect engineered perturbations.")
    p("Persistent abnormalities in real clinical practice require qualified clinical assessment. This demonstration does not prescribe medication or change treatment.")
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    content=buffer.getvalue()
    if path:
        Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_bytes(content)
    return content


def architecture_pdf(path=None):
    path=Path(path or ROOT/"docs/architecture.pdf")
    c=canvas.Canvas(str(path),pagesize=landscape(A4))
    w,h=landscape(A4)
    c.setFillColor(NAVY);c.rect(0,h-82,w,82,fill=1,stroke=0)
    c.setFillColor(colors.white);c.setFont("Helvetica-Bold",23);c.drawString(35,h-40,"CARDIO TWIN / System architecture")
    c.setFont("Helvetica",10);c.drawString(35,h-61,"Separated evidence tracks | Local execution | Clinician-facing research demonstration")
    def box(x,y,bw,bh,title,subtitle,color=TEAL):
        c.setFillColor(colors.HexColor("#eef5f7"));c.setStrokeColor(color)
        c.roundRect(x,y,bw,bh,9,fill=1,stroke=1)
        c.setFillColor(NAVY);c.setFont("Helvetica-Bold",11);c.drawCentredString(x+bw/2,y+bh-23,title)
        c.setFont("Helvetica",8);c.drawCentredString(x+bw/2,y+13,subtitle)
    def arrow(x1,y1,x2,y2):
        c.setStrokeColor(TEAL);c.setLineWidth(1.5);c.line(x1,y1,x2,y2)
        import math
        angle=math.atan2(y2-y1,x2-x1)
        for side in [-1,1]:
            a=angle+side*.45;c.line(x2,y2,x2-7*math.cos(a),y2-7*math.sin(a))
    box(35,425,220,60,"Static clinical records","Cross-sectional disease labels")
    box(310,425,220,60,"Training-only preprocessing","Imputation / normalization / embeddings")
    box(585,425,220,60,"Clinical classifiers","Selection / calibration / held-out test")
    arrow(255,455,310,455);arrow(530,455,585,455)
    box(35,315,220,60,"Synthetic nocturnal sensors","AR(1) / noise / missingness / scenarios")
    box(310,315,220,60,"GRU + temporal attention","Masks / time encoding / EHR fusion")
    box(585,315,220,60,"Experimental fusion network","Separate experiment / displayed in dashboard")
    arrow(255,345,310,345);arrow(530,345,585,345);arrow(420,425,420,375)
    box(310,205,220,60,"Patient twin state","Fixed personal reference / SQLite / EWMA")
    arrow(145,315,310,235)
    box(35,100,220,60,"Baseline classification","Recorded disease status only")
    box(310,100,220,60,"Anomaly monitoring","Patient-relative deviation / uncertainty")
    box(585,100,220,60,"Scenario exploration","Hypothetical inputs / noncausal outputs")
    arrow(420,205,145,160);arrow(420,205,420,160);arrow(420,205,695,160)
    c.setFillColor(NAVY);c.setFont("Helvetica-Bold",12);c.drawCentredString(w/2,62,"Streamlit doctor dashboard + downloadable patient PDF")
    c.setFont("Helvetica",9);c.drawCentredString(w/2,38,"No validated clinical event forecast or treatment effect; research prototype only")
    c.showPage()
    c.setFillColor(NAVY);c.rect(0,h-82,w,82,fill=1,stroke=0)
    c.setFillColor(colors.white);c.setFont("Helvetica-Bold",21)
    c.drawString(35,h-40,"Loneliness and BP / Research extension")
    c.setFont("Helvetica",10);c.drawString(35,h-61,"Entirely synthetic longitudinal model | Separate real-data domain comparisons")
    for x,title,subtitle in [
        (35,"Physiology","HR / HRV / BP / respiration"),
        (235,"Psychology","Loneliness / anxiety / stress"),
        (435,"Biological assays","Cortisol / inflammation / metabolism"),
        (635,"Personal context","Age / BMI / activity / sleep")]:
        box(x,425,170,60,title,subtitle)
        arrow(x+85,425,420,375)
    box(310,315,220,60,"Separate modality encoders","Values + missing masks + elapsed time")
    arrow(420,315,420,280)
    box(310,220,220,60,"History-only GRU","Only information available at the landmark")
    for x,title,subtitle in [
        (35,"Acute state","Noisy next-night synthetic labels"),
        (235,"BP trajectory","Next-week mean SBP and DBP"),
        (435,"Psychological state","Invented unitless future scores"),
        (635,"Incident event","Right-censored discrete hazard")]:
        arrow(420,220,x+85,170)
        box(x,110,170,60,title,subtitle)
    c.setFillColor(NAVY);c.setFont("Helvetica-Bold",11)
    c.drawCentredString(w/2,72,"Additional laboratories: five-axis Kalman state and randomized-trial / IPW estimators")
    c.setFont("Helvetica",9)
    c.drawCentredString(w/2,49,"Unknown BP, loneliness, assays and event times are never fabricated for the original CSV.")
    c.drawCentredString(w/2,32,"Synthetic effect sizes and latent coordinates are assumptions, not clinical evidence.")
    c.showPage();c.save()
    return path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak
)

OUTPUT_FILE = "results_matrix.pdf"

doc = SimpleDocTemplate(
    OUTPUT_FILE,
    pagesize=landscape(A4),
    rightMargin=12 * mm,
    leftMargin=12 * mm,
    topMargin=12 * mm,
    bottomMargin=12 * mm
)

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="MainTitle",
    parent=styles["Title"],
    fontName="Helvetica-Bold",
    fontSize=17,
    leading=21,
    alignment=TA_CENTER,
    spaceAfter=5 * mm
))
styles.add(ParagraphStyle(
    name="SectionTitle",
    parent=styles["Heading2"],
    fontSize=12,
    leading=15,
    spaceBefore=3 * mm,
    spaceAfter=3 * mm
))
styles.add(ParagraphStyle(
    name="SmallText",
    parent=styles["BodyText"],
    fontSize=8,
    leading=10
))
styles.add(ParagraphStyle(
    name="TableText",
    parent=styles["BodyText"],
    fontSize=7.5,
    leading=9
))
styles.add(ParagraphStyle(
    name="TableHeader",
    parent=styles["BodyText"],
    fontName="Helvetica-Bold",
    fontSize=7.5,
    leading=9,
    textColor=colors.white,
    alignment=TA_CENTER
))

story = []

def P(text, style="TableText"):
    return Paragraph(str(text), styles[style])

story.append(P("MACHINE LEARNING ASSIGNMENT 1", "MainTitle"))
story.append(P("Results Matrix — Frog Species Classification", "Heading2"))
story.append(Spacer(1, 2 * mm))

story.append(P(
    "<b>Dataset:</b> Frogs_MFCCs.csv | "
    "<b>Samples:</b> 7,195 | "
    "<b>Input features:</b> 22 MFCC features | "
    "<b>Species classes:</b> 10 | "
    "<b>Families:</b> 4",
    "SmallText"
))
story.append(Spacer(1, 4 * mm))

story.append(P("1. Main Results Matrix", "SectionTitle"))

headers = [
    "Method",
    "Criterion / Configuration",
    "Train Species<br/>Accuracy",
    "Test Species<br/>Accuracy",
    "Family Accuracy<br/>(Test)"
]

rows = [
    ["1. Single Decision Tree", "Gini", "100.00%", "94.86%", "96.11%"],
    ["1. Single Decision Tree", "Entropy", "100.00%", "95.34%", "96.73%"],
    ["2a. Bagging", "Gini, 25 trees", "99.97%", "96.73%", "97.57%"],
    ["2a. Bagging", "Entropy, 25 trees", "99.98%", "96.87%", "97.50%"],
    ["2b. AdaBoost SAMME", "Gini, depth 3, 50 rounds", "97.52%", "95.34%", "96.73%"],
    ["2b. AdaBoost SAMME", "Entropy, depth 3, 50 rounds", "96.35%", "94.16%", "95.00%"],
    ["3. Random Forest", "Gini, 50 trees, max_features=4", "100.00%", "97.57%", "97.98%"],
    ["3. Random Forest", "Entropy, 50 trees, max_features=4", "100.00%", "97.36%", "97.98%"],
    ["4. Hierarchical Decision Tree", "Gini", "100.00%", "93.82%", "Router: 95.07%"],
    ["4. Hierarchical Decision Tree", "Entropy", "100.00%", "94.72%", "Router: 95.62%"],
    ["5. Gaussian Naive Bayes", "Gaussian likelihood", "91.47%", "90.34%", "91.94%"],
]

table_data = [[P(h, "TableHeader") for h in headers]]
table_data += [[P(cell) for cell in row] for row in rows]

main_table = Table(
    table_data,
    colWidths=[47 * mm, 66 * mm, 35 * mm, 35 * mm, 39 * mm],
    repeatRows=1,
    hAlign="CENTER"
)
main_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#24476B")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9AA9B8")),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("ALIGN", (2, 1), (-1, -1), "CENTER"),
    ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ("TOPPADDING", (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1),
     [colors.white, colors.HexColor("#EDF3F8")]),
]))
story.append(main_table)
story.append(Spacer(1, 4 * mm))

story.append(P(
    "<b>Interpretation:</b> The Random Forest with Gini criterion achieved "
    "the highest test species accuracy (97.57%). Family accuracy for the "
    "flat classifiers is calculated by mapping predicted species to their "
    "families. For the hierarchical model, the family column reports router "
    "accuracy, as required by the results-matrix convention.",
    "SmallText"
))

story.append(PageBreak())

story.append(P("SUPPLEMENTARY ANALYSIS", "MainTitle"))
story.append(P("2. Hierarchical Decision Tree — Failure Analysis", "SectionTitle"))

failure_headers = [
    "Criterion",
    "Router Accuracy",
    "Oracle Accuracy",
    "pA: Router Correct,<br/>Expert Wrong",
    "pB: Router Wrong",
]
failure_rows = [
    ["Gini", "95.07%", "98.26%", "1.25%", "4.93%"],
    ["Entropy", "95.62%", "98.33%", "0.90%", "4.38%"],
]
failure_data = [[P(h, "TableHeader") for h in failure_headers]]
failure_data += [[P(cell) for cell in row] for row in failure_rows]

failure_table = Table(
    failure_data,
    colWidths=[42 * mm, 42 * mm, 42 * mm, 58 * mm, 42 * mm],
    hAlign="LEFT"
)
failure_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#24476B")),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9AA9B8")),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("ALIGN", (1, 1), (-1, -1), "CENTER"),
    ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ("TOPPADDING", (0, 0), (-1, -1), 7),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1),
     [colors.white, colors.HexColor("#EDF3F8")]),
]))
story.append(failure_table)
story.append(Spacer(1, 4 * mm))

story.append(P(
    "<b>Definitions:</b> Oracle accuracy measures expert classification "
    "accuracy when the correct family is supplied to the expert. pA is the "
    "fraction of test examples for which the router selects the correct "
    "family but the family expert predicts the wrong species. pB is the "
    "fraction for which the router selects the wrong family.",
    "SmallText"
))
story.append(Spacer(1, 3 * mm))

story.append(P("3. Per-Family Test Species Accuracy", "SectionTitle"))

family_headers = ["Family", "Test Samples", "Gini Hierarchy", "Entropy Hierarchy"]
family_rows = [
    ["Bufonidae", "14", "71.43%", "71.43%"],
    ["Dendrobatidae", "108", "88.89%", "87.96%"],
    ["Hylidae", "433", "89.84%", "91.92%"],
    ["Leptodactylidae", "884", "96.72%", "97.29%"],
]
family_data = [[P(h, "TableHeader") for h in family_headers]]
family_data += [[P(cell) for cell in row] for row in family_rows]

family_table = Table(
    family_data,
    colWidths=[65 * mm, 38 * mm, 48 * mm, 48 * mm],
    hAlign="LEFT"
)
family_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#24476B")),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9AA9B8")),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("ALIGN", (1, 1), (-1, -1), "CENTER"),
    ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ("TOPPADDING", (0, 0), (-1, -1), 6),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1),
     [colors.white, colors.HexColor("#EDF3F8")]),
]))
story.append(family_table)
story.append(Spacer(1, 4 * mm))

story.append(P("4. Hierarchical Model vs. Flat Decision Tree", "SectionTitle"))
comparison_rows = [
    ["Criterion", "Flat Tree Test Species Accuracy", "Hierarchical Test Species Accuracy",
     "Difference (Hierarchy - Flat)"],
    ["Gini", "94.86%", "93.82%", "-1.04 percentage points"],
    ["Entropy", "95.34%", "94.72%", "-0.62 percentage points"],
]
comparison_data = [
    [P(cell, "TableHeader" if i == 0 else "TableText")
     for cell in row]
    for i, row in enumerate(comparison_rows)
]
comparison_table = Table(
    comparison_data,
    colWidths=[35 * mm, 65 * mm, 65 * mm, 65 * mm],
    hAlign="LEFT"
)
comparison_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#24476B")),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9AA9B8")),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("ALIGN", (1, 1), (-1, -1), "CENTER"),
    ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ("TOPPADDING", (0, 0), (-1, -1), 6),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1),
     [colors.white, colors.HexColor("#EDF3F8")]),
]))
story.append(comparison_table)
story.append(Spacer(1, 4 * mm))

story.append(P(
    "The hierarchical model has lower overall test species accuracy than "
    "the corresponding flat decision tree under both criteria. The reported "
    "results use the shared train/test split and the outputs recorded during "
    "the assignment runs.",
    "SmallText"
))

def add_page_number(canvas, document):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawRightString(
        landscape(A4)[0] - 12 * mm,
        6 * mm,
        f"Page {document.page}"
    )
    canvas.restoreState()

doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)

print(f"Created {OUTPUT_FILE}")
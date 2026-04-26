from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


BASE_DIR = Path(r"G:\0000Mphil_floating_building")
OUTPUT_DIR = BASE_DIR / "deliverables"
OUTPUT_PPTX = OUTPUT_DIR / "updated_project_overview_slides.pptx"
LOGO_PATH = BASE_DIR / ".ppt-extract" / "assets" / "slide1_img2.png"

BLUE = RGBColor(9, 62, 122)
BLUE_2 = RGBColor(21, 83, 152)
GOLD = RGBColor(166, 132, 56)
LIGHT_BG = RGBColor(248, 249, 252)
PANEL_BG = RGBColor(255, 255, 255)
PANEL_LINE = RGBColor(210, 216, 224)
TEXT = RGBColor(26, 32, 44)
MUTED = RGBColor(92, 104, 120)
GREEN = RGBColor(46, 160, 67)
RED = RGBColor(215, 70, 70)
ORANGE = RGBColor(240, 132, 37)


def add_textbox(slide, left, top, width, height, text, font_size, bold=False,
                color=TEXT, font_name="Aptos", align=PP_ALIGN.LEFT,
                italic=False, margins=(0.05, 0.03, 0.05, 0.03)):
    shape = slide.shapes.add_textbox(left, top, width, height)
    tf = shape.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.TOP
    tf.margin_left = Inches(margins[0])
    tf.margin_top = Inches(margins[1])
    tf.margin_right = Inches(margins[2])
    tf.margin_bottom = Inches(margins[3])
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    font = run.font
    font.size = Pt(font_size)
    font.bold = bold
    font.italic = italic
    font.name = font_name
    font.color.rgb = color
    return shape


def add_round_box(slide, left, top, width, height, fill=PANEL_BG, line=PANEL_LINE, radius_shape=MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE):
    shape = slide.shapes.add_shape(radius_shape, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = line
    shape.line.width = Pt(1.2)
    return shape


def add_header(slide, active_label):
    slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, 0, 0, Inches(13.333), Inches(0.8)).fill.solid()
    bar = slide.shapes[-1]
    bar.fill.fore_color.rgb = BLUE
    bar.line.fill.background()

    if LOGO_PATH.exists():
        slide.shapes.add_picture(str(LOGO_PATH), Inches(0.15), Inches(0.12), height=Inches(0.45))

    nav_left = Inches(3.55)
    labels = [
        ("Project\nOverview", True),
        ("Integrated Solution\n& Project\nComposition", False),
        ("Feasibility, Risk &\nBudget Planning", False),
        ("Expected\nOutcome", False),
    ]
    widths = [Inches(2.25), Inches(2.55), Inches(3.15), Inches(1.83)]
    x = nav_left
    for (label, is_first), width in zip(labels, widths):
        rect = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, x, 0, width, Inches(0.8))
        rect.fill.solid()
        rect.fill.fore_color.rgb = GOLD if label == active_label else BLUE
        rect.line.color.rgb = RGBColor(255, 255, 255)
        rect.line.width = Pt(0.8)
        add_textbox(
            slide, x, Inches(0.06), width, Inches(0.68), label, 15,
            bold=True, color=RGBColor(255, 255, 255), font_name="Georgia",
            align=PP_ALIGN.CENTER, margins=(0, 0, 0, 0)
        )
        x += width


def add_footer(slide, page_number):
    band = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, 0, Inches(7.22), Inches(13.333), Inches(0.28))
    band.fill.solid()
    band.fill.fore_color.rgb = GOLD
    band.line.fill.background()
    add_textbox(slide, Inches(12.9), Inches(6.96), Inches(0.28), Inches(0.18), str(page_number), 9, color=MUTED, align=PP_ALIGN.RIGHT)


def add_title(slide, title, subtitle=None):
    add_textbox(slide, Inches(0.25), Inches(0.83), Inches(12.8), Inches(0.64), title, 20.5, bold=True, font_name="Georgia")
    if subtitle:
        add_textbox(slide, Inches(0.3), Inches(1.36), Inches(12.5), Inches(0.22), subtitle, 10.1, color=MUTED)


def add_bullets(slide, left, top, width, height, items, font_size=11.5, color=TEXT, bullet_color=BLUE):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.08)
    tf.margin_right = Inches(0.04)
    tf.margin_top = Inches(0.02)
    tf.margin_bottom = Inches(0.02)
    for idx, item in enumerate(items):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.level = 0
        p.space_after = Pt(5)
        p.bullet = True
        run = p.add_run()
        run.text = item
        font = run.font
        font.size = Pt(font_size)
        font.name = "Aptos"
        font.color.rgb = color
    return box


def create_sector_chart(path):
    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=220)
    fig.patch.set_facecolor("white")
    ax.axis("equal")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.2, 1.25)
    outer_vals = [34, 66]
    inner_vals = [32, 68]
    colors_outer = ["#F08C25", "#E9EEF5"]
    colors_inner = ["#155398", "#E9EEF5"]

    ax.pie(outer_vals, radius=1.0, startangle=90, colors=colors_outer,
           wedgeprops=dict(width=0.2, edgecolor="white"))
    ax.pie(inner_vals, radius=0.72, startangle=90, colors=colors_inner,
           wedgeprops=dict(width=0.2, edgecolor="white"))
    ax.add_patch(Circle((0, 0), 0.34, color="white"))
    ax.text(0, 0.08, "2024/25\nGlobalABC", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#1A202C")
    ax.text(-0.78, 0.08, "34%", fontsize=13, color="#F08C25", fontweight="bold")
    ax.text(0.53, -0.08, "32%", fontsize=13, color="#155398", fontweight="bold")
    ax.text(0, -1.08, "Outer ring: CO$_2$ emissions   Inner ring: energy use",
            ha="center", fontsize=8.7, color="#5C6878")
    ax.set_title("Buildings + construction remain a system-level climate bottleneck",
                 fontsize=11, color="#1A202C", fontweight="bold", pad=10)
    ax.text(0, -1.23, "Source: GlobalABC/UNEP, Global Status Report for Buildings and Construction 2024/2025",
            ha="center", fontsize=7.8, color="#7A8595")
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def create_peak_chart(path):
    labels = ["India\n2019", "India\n2024", "Brazil\n2024", "Türkiye\n2024"]
    values = [4.0, 7.0, 3.6, 2.0]
    colors = ["#9FB6D5", "#155398", "#F0A030", "#D6B05B"]
    fig, ax = plt.subplots(figsize=(5.6, 3.5), dpi=220)
    fig.patch.set_facecolor("white")
    bars = ax.bar(labels, values, color=colors, width=0.58)
    ax.set_ylabel("Extra peak demand per +1°C (GW)", fontsize=9.5, color="#1A202C")
    ax.set_ylim(0, 8)
    ax.grid(axis="y", color="#E4E8EE", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#AAB4C3")
    ax.spines["bottom"].set_color("#AAB4C3")
    ax.tick_params(axis="x", labelsize=8.8)
    ax.tick_params(axis="y", labelsize=8.8)

    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.18, f"{value:g}",
                ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#1A202C")

    ax.annotate(
        "+75% vs 2019",
        xy=(1, 7.0),
        xytext=(0.4, 7.55),
        textcoords="data",
        fontsize=9,
        color="#155398",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#155398", lw=1.2),
    )
    ax.set_title("Heatwaves are turning operational control into a peak-load problem",
                 fontsize=11, color="#1A202C", fontweight="bold", pad=10)
    ax.text(0.5, -0.23,
            "Source: IEA Energy Efficiency 2024; India rises from ~4 GW to >7 GW extra peak demand per +1°C",
            transform=ax.transAxes, ha="center", fontsize=7.8, color="#7A8595")
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def add_pill(slide, left, top, width, height, text, fill, font_size=11.5, color=PANEL_BG):
    shape = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.fill.background()
    add_textbox(slide, left, top + Inches(0.01), width, height, text, font_size, bold=True, color=color, align=PP_ALIGN.CENTER, margins=(0.03, 0.02, 0.03, 0.02))
    return shape


def build_slide_4(prs, sector_chart):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = LIGHT_BG
    add_header(slide, "Project\nOverview")
    add_footer(slide, 4)
    add_title(
        slide,
        "Updated Context: Buildings remain a major decarbonisation bottleneck",
        "Reframing slide 4 with 2024-2025 sector evidence rather than a generic background statement."
    )

    add_round_box(slide, Inches(0.28), Inches(1.55), Inches(5.15), Inches(4.95))
    slide.shapes.add_picture(str(sector_chart), Inches(0.52), Inches(1.8), width=Inches(4.6))

    add_round_box(slide, Inches(5.68), Inches(1.55), Inches(7.35), Inches(2.48))
    add_textbox(slide, Inches(5.92), Inches(1.74), Inches(4.2), Inches(0.5), "What changed in the evidence base?", 14.5, bold=True, font_name="Georgia")
    add_bullets(slide, Inches(5.95), Inches(2.2), Inches(6.7), Inches(1.34), [
        "GlobalABC/UNEP now frames the sector as consuming 32% of global energy and contributing 34% of global CO2 emissions.",
        "The same 2024/2025 report flags cement and steel alone as responsible for 18% of global emissions.",
        "This makes operational coordination in buildings a climate issue, not only an efficiency issue."
    ], font_size=11.2)

    add_round_box(slide, Inches(5.68), Inches(4.18), Inches(7.35), Inches(2.32))
    add_textbox(slide, Inches(5.92), Inches(4.36), Inches(4.0), Inches(0.34), "Implication for our project", 14.5, bold=True, font_name="Georgia")
    add_bullets(slide, Inches(5.95), Inches(4.72), Inches(6.55), Inches(1.3), [
        "The real challenge is no longer only installing PV, batteries or smart meters.",
        "The harder step is turning those assets into verifiable, auditable net-zero operation at building-cluster level.",
        "That is why HKUST(GZ) as a living lab is a strong testbed for cluster-scale dispatching."
    ], font_size=11.4)

    add_pill(slide, Inches(0.55), Inches(6.56), Inches(3.2), Inches(0.33),
             "Source: GlobalABC / UNEP (2025)", BLUE_2, font_size=10.5)
    add_pill(slide, Inches(3.93), Inches(6.56), Inches(2.9), Inches(0.33),
             "32% global energy use", GOLD, font_size=10.5)
    add_pill(slide, Inches(7.0), Inches(6.56), Inches(3.25), Inches(0.33),
             "34% global CO2 emissions", BLUE, font_size=10.5)
    add_pill(slide, Inches(10.46), Inches(6.56), Inches(2.2), Inches(0.33),
             "18% materials", ORANGE, font_size=10.5)


def build_slide_5(prs, peak_chart):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = LIGHT_BG
    add_header(slide, "Project\nOverview")
    add_footer(slide, 5)
    add_title(
        slide,
        "Updated Problem Framing: The implementation gap is now a peak-demand problem",
        "Rewriting slide 5 so the gap is evidenced by 2024 heatwave and dispatch data, not only by abstract barriers."
    )

    add_round_box(slide, Inches(0.28), Inches(1.55), Inches(6.1), Inches(4.95))
    slide.shapes.add_picture(str(peak_chart), Inches(0.48), Inches(1.84), width=Inches(5.7))
    add_pill(slide, Inches(0.62), Inches(5.95), Inches(2.95), Inches(0.36),
             "2024: >40 countries", RED, font_size=10.8)
    add_pill(slide, Inches(3.72), Inches(5.95), Inches(2.2), Inches(0.36),
             "Nearly 70% demand", BLUE, font_size=10.8)

    add_round_box(slide, Inches(6.55), Inches(1.55), Inches(6.48), Inches(1.34))
    add_textbox(slide, Inches(6.82), Inches(1.78), Inches(5.85), Inches(0.28), "1. Forecasting uncertainty", 15, bold=True, font_name="Georgia")
    add_textbox(slide, Inches(6.82), Inches(2.12), Inches(5.8), Inches(0.52),
                "High-frequency cooling loads and weather extremes make static PV-load assumptions unreliable in real operation.", 11.4)

    add_round_box(slide, Inches(6.55), Inches(3.05), Inches(6.48), Inches(1.34))
    add_textbox(slide, Inches(6.82), Inches(3.28), Inches(5.85), Inches(0.28), "2. Control coordination", 15, bold=True, font_name="Georgia")
    add_textbox(slide, Inches(6.82), Inches(3.62), Inches(5.8), Inches(0.52),
                "Dispatch has to react across HVAC, storage, PV and user loads fast enough to avoid curtailment, peaks and comfort penalties.", 11.4)

    add_round_box(slide, Inches(6.55), Inches(4.55), Inches(6.48), Inches(1.34))
    add_textbox(slide, Inches(6.82), Inches(4.78), Inches(5.85), Inches(0.28), "3. Carbon-cost trade-off", 15, bold=True, font_name="Georgia")
    add_textbox(slide, Inches(6.82), Inches(5.12), Inches(5.8), Inches(0.52),
                "Without predictive scheduling, operators often trade carbon targets for peak-charge avoidance or vice versa, slowing ROI.", 11.4)

    add_textbox(slide, Inches(6.65), Inches(6.12), Inches(6.1), Inches(0.3),
                "Takeaway: operational net-zero requires predictive, weather-aware cluster dispatch rather than hardware-first deployment.",
                11.2, bold=True, color=BLUE)
    add_textbox(slide, Inches(0.55), Inches(6.55), Inches(12.0), Inches(0.18),
                "Sources: IEA Energy Efficiency 2024; IEA analysis based on Real-Time Electricity Tracker and Weather, Climate and Energy Tracker.",
                8.5, color=MUTED, italic=True)


def build_slide_6(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = LIGHT_BG
    add_header(slide, "Project\nOverview")
    add_footer(slide, 6)
    add_title(
        slide,
        "Updated Campus Case: Why cluster control is harder on a public campus",
        "Replacing the old metaphor graphic with a more research-grounded campus challenge map."
    )

    card_w = Inches(4.08)
    x_positions = [Inches(0.28), Inches(4.63), Inches(8.98)]
    titles = [
        ("A. Microclimate variability", BLUE),
        ("B. Load heterogeneity", GOLD),
        ("C. Control integration gap", BLUE_2),
    ]
    bodies = [
        [
            "Annex 82 identifies weather as a dominant factor for energy flexibility at cluster level.",
            "Campus-scale operation is sensitive to local solar exposure, cooling demand and short-term weather swings."
        ],
        [
            "Recent campus energy reviews describe university buildings as dynamic, irregular and not well captured by simple seasonal patterns.",
            "Class timetables, exam periods, offices and shared facilities make load shapes inherently mixed."
        ],
        [
            "A 2024 occupant-centric control review finds significant challenges remain in integrating real occupant behaviour into practical controls.",
            "Annex 82 also notes scarcity of field studies at building-cluster level, so implementation evidence is still limited."
        ],
    ]

    for i, x in enumerate(x_positions):
        add_round_box(slide, x, Inches(1.68), card_w, Inches(3.82))
        shape = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.OVAL, x + Inches(0.18), Inches(1.88), Inches(0.44), Inches(0.44))
        shape.fill.solid()
        shape.fill.fore_color.rgb = titles[i][1]
        shape.line.fill.background()
        add_textbox(slide, x + Inches(0.18), Inches(1.955), Inches(0.44), Inches(0.18), chr(65 + i), 14,
                    bold=True, color=PANEL_BG, align=PP_ALIGN.CENTER, margins=(0, 0, 0, 0))
        add_textbox(slide, x + Inches(0.72), Inches(1.9), Inches(3.0), Inches(0.35), titles[i][0], 15.2, bold=True, font_name="Georgia")
        add_bullets(slide, x + Inches(0.2), Inches(2.42), Inches(3.65), Inches(2.4), bodies[i], font_size=11.1)

    ribbon = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, Inches(0.55), Inches(5.78), Inches(12.2), Inches(0.62))
    ribbon.fill.solid()
    ribbon.fill.fore_color.rgb = RGBColor(235, 240, 247)
    ribbon.line.color.rgb = PANEL_LINE
    add_textbox(slide, Inches(0.82), Inches(5.93), Inches(11.6), Inches(0.22),
                "Design implication: the solution must couple local forecasting, occupant-aware control and cluster-level coordination in one dispatch loop.",
                13, bold=True, color=BLUE, align=PP_ALIGN.CENTER)
    add_textbox(slide, Inches(0.48), Inches(6.55), Inches(12.1), Inches(0.18),
                "Sources: IEA EBC Annex 82 Summary Report (2024); Methodologies and evaluations of energy flexibility for clusters of buildings (2025); Toward Sustainable Campus Energy Management (2025); Building Simulation occupant-centric control review (2024).",
                8.1, color=MUTED, italic=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    with TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        sector_chart = tmp_dir / "sector_chart.png"
        peak_chart = tmp_dir / "peak_chart.png"
        create_sector_chart(sector_chart)
        create_peak_chart(peak_chart)

        build_slide_4(prs, sector_chart)
        build_slide_5(prs, peak_chart)
        build_slide_6(prs)

    prs.save(str(OUTPUT_PPTX))
    print(f"Saved {OUTPUT_PPTX}")


if __name__ == "__main__":
    main()

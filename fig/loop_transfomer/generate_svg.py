import xml.etree.ElementTree as ET


def create_rounded_rect(
    parent,
    x,
    y,
    width,
    height,
    r,
    fill,
    stroke,
    stroke_width,
    label=None,
    label_size="14",
    label_y_offset=20,
):
    rect = ET.SubElement(parent, "rect")
    rect.set("x", str(x))
    rect.set("y", str(y))
    rect.set("width", str(width))
    rect.set("height", str(height))
    rect.set("rx", str(r))
    rect.set("ry", str(r))
    rect.set("fill", fill)
    rect.set("stroke", stroke)
    rect.set("stroke-width", str(stroke_width))

    if label:
        lines = label.split("\n")
        for i, line in enumerate(lines):
            text = ET.SubElement(parent, "text")
            text.set("x", str(x + width / 2))
            text.set("y", str(y + label_y_offset + (i * 18)))
            text.set("font-family", "Arial, sans-serif")
            text.set("font-size", label_size)
            text.set("text-anchor", "middle")
            text.set("fill", "black")
            text.text = line


def create_arrow(parent, x1, y1, x2, y2, color="black", width=2):
    line = ET.SubElement(parent, "line")
    line.set("x1", str(x1))
    line.set("y1", str(y1))
    line.set("x2", str(x2))
    line.set("y2", str(y2))
    line.set("stroke", color)
    line.set("stroke-width", str(width))
    line.set("marker-end", "url(#arrowhead)")


def create_curved_arrow(parent, path_d, color="black", width=2):
    path = ET.SubElement(parent, "path")
    path.set("d", path_d)
    path.set("fill", "none")
    path.set("stroke", color)
    path.set("stroke-width", str(width))
    path.set("marker-end", "url(#arrowhead)")


def create_diamond(parent, x, y, width, height, fill, stroke, stroke_width, label=None):
    points = f"{x},{y + height / 2} {x + width / 2},{y} {x + width},{y + height / 2} {x + width / 2},{y + height}"
    polygon = ET.SubElement(parent, "polygon")
    polygon.set("points", points)
    polygon.set("fill", fill)
    polygon.set("stroke", stroke)
    polygon.set("stroke-width", str(stroke_width))

    if label:
        text = ET.SubElement(parent, "text")
        text.set("x", str(x + width / 2))
        text.set("y", str(y + height / 2 + 5))
        text.set("font-family", "Arial, sans-serif")
        text.set("font-size", "12")
        text.set("text-anchor", "middle")
        text.set("fill", "black")
        text.text = label


def generate_svg():
    svg = ET.Element(
        "svg",
        xmlns="http://www.w3.org/2000/svg",
        width="1000",
        height="800",
        viewBox="0 0 1000 800",
    )

    # Definitions for markers
    defs = ET.SubElement(svg, "defs")
    marker = ET.SubElement(
        defs,
        "marker",
        id="arrowhead",
        markerWidth="10",
        markerHeight="7",
        refX="9",
        refY="3.5",
        orient="auto",
    )
    polygon = ET.SubElement(marker, "polygon", points="0 0, 10 3.5, 0 7", fill="black")

    # Background
    bg = ET.SubElement(svg, "rect", width="100%", height="100%", fill="white")

    # Title
    title = ET.SubElement(
        svg,
        "text",
        x="100",
        y="40",
        text_anchor="start",
        font_family="Arial, sans-serif",
        font_size="24",
        font_weight="bold",
    )
    title.text = "Loop Transformer Architecture"

    subtitle = ET.SubElement(
        svg,
        "text",
        x="100",
        y="70",
        text_anchor="start",
        font_family="Arial, sans-serif",
        font_size="16",
        fill="#555",
    )
    subtitle.text = "Config: loop_transformer.yaml | Hidden: 512 | Heads: 8 | Shared Weights (1 Layer)"

    # Input Node
    create_rounded_rect(
        svg,
        50,
        250,
        100,
        60,
        10,
        "#e1f5fe",
        "#0277bd",
        2,
        "Input (X)\n(Puzzle Emb)",
        "14",
        25,
    )

    # Initial States
    create_rounded_rect(
        svg, 50, 150, 100, 60, 10, "#e0f2f1", "#00695c", 2, "Init z_L", "14", 35
    )
    create_rounded_rect(
        svg, 50, 350, 100, 60, 10, "#fff9c4", "#fbc02d", 2, "Init z_H", "14", 35
    )

    # Outer Loop Container
    create_rounded_rect(
        svg, 200, 100, 400, 500, 20, "#f5f5f5", "#9e9e9e", 2, "", "14", 20
    )
    loop_label = ET.SubElement(
        svg,
        "text",
        x="400",
        y="130",
        text_anchor="middle",
        font_family="Arial, sans-serif",
        font_size="18",
        font_weight="bold",
    )
    loop_label.text = "Outer Loop (Repeat 3 times)"

    # Stage 1 Box
    create_rounded_rect(
        svg, 250, 160, 300, 180, 15, "#e3f2fd", "#1565c0", 2, "", "14", 20
    )
    stage1_label = ET.SubElement(
        svg,
        "text",
        x="400",
        y="190",
        text_anchor="middle",
        font_family="Arial, sans-serif",
        font_size="16",
        font_weight="bold",
        fill="#1565c0",
    )
    stage1_label.text = "Stage 1 (Repeat 6x)"

    # Stage 1 Equation
    stage1_eq = ET.SubElement(
        svg,
        "text",
        x="400",
        y="220",
        text_anchor="middle",
        font_family="monospace",
        font_size="14",
    )
    stage1_eq.text = "z_L ← T(z_L + z_H + X)"

    # Stage 1 Visuals
    create_rounded_rect(
        svg,
        350,
        250,
        100,
        50,
        5,
        "#bbdefb",
        "#1976d2",
        1,
        "Transformer\n(Shared)",
        "12",
        20,
    )

    # Stage 2 Box
    create_rounded_rect(
        svg, 250, 380, 300, 180, 15, "#ffebee", "#c62828", 2, "", "14", 20
    )
    stage2_label = ET.SubElement(
        svg,
        "text",
        x="400",
        y="410",
        text_anchor="middle",
        font_family="Arial, sans-serif",
        font_size="16",
        font_weight="bold",
        fill="#c62828",
    )
    stage2_label.text = "Stage 2 (Repeat 3x)"

    # Stage 2 Equation
    stage2_eq = ET.SubElement(
        svg,
        "text",
        x="400",
        y="440",
        text_anchor="middle",
        font_family="monospace",
        font_size="14",
    )
    stage2_eq.text = "z_H ← T(z_H + z_L)"

    # Stage 2 Visuals
    create_rounded_rect(
        svg,
        350,
        470,
        100,
        50,
        5,
        "#ffcdd2",
        "#d32f2f",
        1,
        "Transformer\n(Shared)",
        "12",
        20,
    )

    # Arrows - Inputs to Stage 1
    # X to Stage 1
    create_curved_arrow(svg, "M 150 280 C 200 280, 200 275, 350 275")  # X to T(Stage1)
    # z_L to Stage 1 (Initial)
    create_curved_arrow(
        svg, "M 150 180 C 200 180, 220 260, 350 265"
    )  # z_L to T(Stage1)
    # z_H to Stage 1
    create_curved_arrow(
        svg, "M 150 380 C 200 380, 220 290, 350 285"
    )  # z_H to T(Stage1)

    # Arrows - Between Stages
    # z_L from Stage 1 to Stage 2
    create_arrow(svg, 400, 300, 400, 380)  # Downward flow

    # z_H from Stage 1 (pass through) to Stage 2
    # Implicitly z_H is maintained. We show z_H entering Stage 2.
    # Let's draw a line from z_H init bypassing Stage 1 to Stage 2?
    # Or better, show the flow of state updates.

    # Let's simplify arrows to show logical dependency

    # Stage 1 Internal Loop (z_L)
    create_curved_arrow(
        svg, "M 450 275 C 480 275, 480 225, 400 235"
    )  # Output back to input area (conceptual)

    # Stage 2 Internal Loop (z_H)
    create_curved_arrow(
        svg, "M 450 495 C 480 495, 480 445, 400 455"
    )  # Output back to input area

    # Stage 2 Inputs
    # z_L comes from Stage 1
    create_arrow(svg, 400, 340, 400, 470)  # z_L flow

    # z_H comes from previous cycle or init
    # We can draw a line from z_H init around to Stage 2
    path_zh = "M 100 410 L 100 500 L 350 500"
    create_curved_arrow(svg, path_zh, "#fbc02d", 2)

    # Halt Mechanism & Readout

    # LM Head
    create_rounded_rect(
        svg,
        600,
        320,
        100,
        50,
        5,
        "#e8f5e9",
        "#2e7d32",
        1,
        "LM Head\n(Logits)",
        "12",
        20,
    )

    # Halt Head
    create_rounded_rect(
        svg,
        600,
        420,
        100,
        50,
        5,
        "#fff3e0",
        "#ef6c00",
        1,
        "Halt Head\n(Q-values)",
        "12",
        20,
    )

    # Decision Diamond
    create_diamond(svg, 620, 500, 60, 60, "#fff", "#000", 2, "Halt?")

    # Arrows from Stage 2 (z_H)
    # To LM Head
    create_curved_arrow(svg, "M 550 450 C 580 450, 580 345, 600 345")
    # To Halt Head
    create_curved_arrow(svg, "M 550 490 C 580 490, 580 445, 600 445")

    # Arrow Halt Head to Decision
    create_arrow(svg, 650, 470, 650, 500)

    # Loop Back (Continue)
    # From Decision Bottom -> Back to Stage 1
    path_loop = "M 650 560 L 650 620 L 180 620 L 180 160 L 250 160"
    create_curved_arrow(svg, path_loop, color="#d32f2f", width=2)

    # Label for Loop
    text_cont = ET.SubElement(
        svg,
        "text",
        x="400",
        y="615",
        text_anchor="middle",
        font_family="Arial",
        font_size="12",
        fill="#d32f2f",
    )
    text_cont.text = "No (Continue)"

    # Stop / Output
    # From Decision Right
    create_arrow(svg, 680, 530, 750, 530)
    text_halt = ET.SubElement(
        svg,
        "text",
        x="715",
        y="525",
        text_anchor="middle",
        font_family="Arial",
        font_size="12",
        fill="#2e7d32",
    )
    text_halt.text = "Yes"

    # Final Output Box
    create_rounded_rect(
        svg, 750, 500, 80, 60, 5, "#e8f5e9", "#2e7d32", 2, "Output", "14", 35
    )

    # Connect LM Head to Output
    create_curved_arrow(svg, "M 700 345 C 790 345, 790 450, 790 500")

    # Legend/Notes
    note = ET.SubElement(
        svg,
        "text",
        x="100",
        y="650",
        text_anchor="start",
        font_family="Arial, sans-serif",
        font_size="12",
        fill="#777",
    )
    note.text = "z_H: High-level state | z_L: Low-level state | T: Transformer Block (RMSNorm -> Attn -> RMSNorm -> MLP)"

    tree = ET.ElementTree(svg)
    tree.write("loop_transformer_arch.svg")


if __name__ == "__main__":
    generate_svg()

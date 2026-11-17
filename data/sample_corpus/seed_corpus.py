"""Deterministic sample private corpus generator (PDF rulebook, team scouting PDFs, scanned PNG, e-discovery load file)."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def _write_multipage_pdf(path: Path, title: str, pages: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=letter)
    for page_num, body in enumerate(pages, start=1):
        c.setFont("Helvetica-Bold", 14)
        c.drawString(54, 740, f"{title} — Page {page_num}")
        c.setFont("Helvetica", 11)
        y = 708
        for line in body.splitlines():
            c.drawString(54, y, line.strip())
            y -= 18
        c.showPage()
    c.save()


def _write_scanned_protest_png(path: Path, text_lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (900, 320), color=(248, 246, 240))
    draw = ImageDraw.Draw(img)
    y = 28
    for line in text_lines:
        draw.text((28, y), line, fill=(20, 24, 30))
        y += 32
    img.save(path)


def generate_sample_corpus(base_dir: Path) -> Path:
    """Create multi-tenant sample corpus and e-discovery load file under `base_dir`."""
    base_dir.mkdir(parents=True, exist_ok=True)

    # 1. Public League Rulebook PDF
    rulebook_pdf = base_dir / "public_isa_surf_rulebook.pdf"
    _write_multipage_pdf(
        rulebook_pdf,
        "Crowded Lineup Official Competition Rulebook v2",
        [
            (
                "Section 1.1 Inside Priority.\n"
                "The surfer positioned closest to the primary breaking peak holds inside priority.\n"
                "When multiple surfers attempt takeoff on the same wave crest, priority belongs\n"
                "exclusively to the inside surfer nearest wavePeakX."
            ),
            (
                "Section 2.4 Drop-In Violations and Collisions.\n"
                "Taking off on the green shoulder in front of an active priority rider triggers\n"
                "a drop-in penalty of -8.0 reward and an immediate 1.2s wipeout.\n"
                "Pairwise board collisions within radius 6.5 incur a -6.0 collision penalty."
            ),
        ],
    )

    # 2. Team Alpha Confidential Scouting Memo PDF
    alpha_pdf = base_dir / "team_alpha_cobalt_scouting.pdf"
    _write_multipage_pdf(
        alpha_pdf,
        "TEAM ALPHA CONFIDENTIAL — Peak Drift & PPO Policy Memo",
        [
            (
                "Project Codename: COBALT_PEAK_DRIFT.\n"
                "Our PPO-SelfPlay-Gen12 agent targets secondary peaks when center\n"
                "crowdDensityCenter exceeds 0.50 in 24-surfer heats."
            ),
            (
                "Telemetry shows COBALT_PEAK_DRIFT raises clean catch rate by 18%\n"
                "while keeping drop-in violations under 0.2 per heat across 320px breaks."
            ),
        ],
    )

    # 3. Team Beta Confidential Scouting Memo PDF
    beta_pdf = base_dir / "team_beta_amber_strategy.pdf"
    _write_multipage_pdf(
        beta_pdf,
        "TEAM BETA CONFIDENTIAL — Stamina Budget Analysis",
        [
            (
                "Project Codename: AMBER_WEDGE_BUDGET.\n"
                "Team Beta restricts paddle bursts when stamina falls below 0.25\n"
                "to avoid the sharp 0.35x speed drop inside deep whitewater foam."
            ),
            (
                "AMBER_WEDGE_BUDGET reserves takeoff stamina for high waveQuality (> 0.72)\n"
                "peaks in 40-surfer chaos heats."
            ),
        ],
    )

    # 4. Steward Restricted Scanned Incident Protest PNG
    protest_png = base_dir / "steward_incident_scan_042.png"
    _write_scanned_protest_png(
        protest_png,
        [
            "STEWARD PROTEST FORM #042 — RESTRICTED",
            "Code: STEWARD_DOCK_PENALTY_OVERRIDE",
            "Incident: Repeat drop-in interference on Wave #7 at step 184.",
            "Ruling: Confirm -8.0 penalty and log telemetry trace for appeal review.",
        ],
    )

    # 5. E-discovery-style load file (.jsonl) mapping docs -> ACL & custodians
    load_file = base_dir / "corpus_loadfile.jsonl"
    records = [
        {
            "doc_id": "doc-isa-rules-01",
            "title": "Crowded Lineup Official Competition Rulebook v2",
            "file_path": str(rulebook_pdf),
            "doc_type": "pdf",
            "tenant_id": "public_league",
            "classification": 0,
            "is_public": True,
            "allowed_principals": [],
            "custodian": "League Competition Committee",
        },
        {
            "doc_id": "doc-alpha-scout-01",
            "title": "Team Alpha Cobalt Peak Drift Scouting Memo",
            "file_path": str(alpha_pdf),
            "doc_type": "pdf",
            "tenant_id": "team_alpha",
            "classification": 2,
            "is_public": False,
            "allowed_principals": ["alice_alpha"],
            "custodian": "Alice (Team Alpha Head Coach)",
        },
        {
            "doc_id": "doc-beta-scout-01",
            "title": "Team Beta Amber Wedge Stamina Strategy",
            "file_path": str(beta_pdf),
            "doc_type": "pdf",
            "tenant_id": "team_beta",
            "classification": 2,
            "is_public": False,
            "allowed_principals": ["bob_beta"],
            "custodian": "Bob (Team Beta Lead Strategist)",
        },
        {
            "doc_id": "doc-protest-scan-01",
            "title": "Steward Protest Scan #042",
            "file_path": str(protest_png),
            "doc_type": "scanned_image",
            "tenant_id": "public_league",
            "classification": 3,
            "is_public": False,
            "allowed_principals": ["steward_lead"],
            "custodian": "Chief Surf Steward",
            "ocr_sidecar_text": (
                "STEWARD PROTEST FORM #042 — RESTRICTED. "
                "Code: STEWARD_DOCK_PENALTY_OVERRIDE. "
                "Incident: Repeat drop-in interference on Wave #7 at step 184."
            ),
        },
    ]
    load_file.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n",
        encoding="utf-8",
    )
    return load_file


if __name__ == "__main__":
    out = generate_sample_corpus(Path(__file__).resolve().parent / "generated")
    print(f"Generated sample corpus loadfile at {out}")

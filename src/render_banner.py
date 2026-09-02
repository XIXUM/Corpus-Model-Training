"""
Render a self-contained SVG banner for the HuggingFace dataset card.

    python -m src.render_banner   ->  reports/factsheet_assets/banner.svg
"""

import os

OUT = "reports/factsheet_assets/banner.svg"

INK = "#141B3D"; NAVY = "#1E2761"; TEAL = "#02C39A"; ICE = "#CADCFC"; ORANGE = "#F2A03F"

BANNER = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 340" width="1200" height="340" font-family="Georgia, 'Times New Roman', serif">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="{INK}"/>
      <stop offset="1" stop-color="{NAVY}"/>
    </linearGradient>
  </defs>
  <rect width="1200" height="340" fill="url(#bg)"/>
  <circle cx="1080" cy="60" r="180" fill="#16204A" opacity="0.6"/>
  <circle cx="1150" cy="300" r="90" fill="#16204A" opacity="0.5"/>
  <!-- mini tree motif -->
  <g stroke="{TEAL}" stroke-width="2.5" fill="none" opacity="0.9">
    <line x1="980" y1="120" x2="920" y2="185"/><line x1="980" y1="120" x2="1040" y2="185"/>
    <line x1="920" y1="185" x2="890" y2="245"/><line x1="920" y1="185" x2="950" y2="245"/>
    <line x1="1040" y1="185" x2="1010" y2="245"/><line x1="1040" y1="185" x2="1070" y2="245"/>
  </g>
  <g fill="{TEAL}">
    <circle cx="980" cy="120" r="7"/><circle cx="920" cy="185" r="6"/><circle cx="1040" cy="185" r="6"/>
    <circle cx="890" cy="245" r="5"/><circle cx="950" cy="245" r="5"/><circle cx="1010" cy="245" r="5"/><circle cx="1070" cy="245" r="5"/>
  </g>
  <text x="70" y="120" fill="{TEAL}" font-family="system-ui, sans-serif" font-size="20" font-weight="700" letter-spacing="5">GOLD-STANDARD CONSTITUENCY PARSING &#183; en-US</text>
  <text x="66" y="196" fill="#FFFFFF" font-size="72" font-weight="700">Benepar Corpus</text>
  <text x="70" y="240" fill="{ICE}" font-family="system-ui, sans-serif" font-size="22">Corrected gold trees + a public-domain US-English training corpus</text>
  <g font-family="system-ui, sans-serif">
    <rect x="70" y="268" width="150" height="42" rx="21" fill="{TEAL}"/>
    <text x="145" y="295" fill="{INK}" font-size="17" font-weight="700" text-anchor="middle">38 gold trees</text>
    <rect x="234" y="268" width="170" height="42" rx="21" fill="#26305C"/>
    <text x="319" y="295" fill="#FFFFFF" font-size="17" font-weight="600" text-anchor="middle">500 sentences</text>
    <rect x="418" y="268" width="200" height="42" rx="21" fill="#26305C"/>
    <text x="518" y="295" fill="#FFFFFF" font-size="17" font-weight="600" text-anchor="middle">49 disagreements</text>
  </g>
</svg>
'''


def main() -> None:
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(BANNER)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Build a square 3x3 social poster (SVG) for the Benepar corpus post.
Hero tree = 2x2 tile; charts + stats + CTA in the remaining 5 tiles."""
import re, os

ASSET = "reports/factsheet_assets"
OUT = "reports/factsheet_assets/social_poster.svg"

INK="#0E1633"; NAVY="#16204A"; NAVY2="#1E2761"; TEAL="#02C39A"; ICE="#CADCFC"
ORANGE="#F2A03F"; WHITE="#FFFFFF"; MUTE="#8C97B5"
SANS="'Segoe UI', system-ui, -apple-system, sans-serif"
SERIF="Georgia, 'Times New Roman', serif"

S=1200; PAD=40; GUT=24
cell=(S-2*PAD-2*GUT)/3.0
def cx(c): return PAD + c*(cell+GUT)
def cy(r): return PAD + r*(cell+GUT)

HERO_X, HERO_Y = cx(0), cy(0)
HERO_W = cell*2+GUT
HERO_H = cell*2+GUT

def nested_tree(path, x, y, w, h):
    svg=open(path,encoding="utf-8").read()
    m=re.search(r'viewBox="([^"]+)"', svg)
    vb=m.group(1) if m else "0 0 100 100"
    inner=svg[svg.find(">")+1 : svg.rfind("</svg>")]
    return (f'<svg x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'viewBox="{vb}" preserveAspectRatio="xMidYMid meet">{inner}</svg>')

def tile(x,y,w,h,fill,rx=18):
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}"/>'

parts=[]
parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{S}" height="{S}" viewBox="0 0 {S} {S}" font-family="{SANS}">')
parts.append(f'<rect width="{S}" height="{S}" fill="{INK}"/>')

# ---- HERO (white tile with title band + tree) ----
parts.append(tile(HERO_X,HERO_Y,HERO_W,HERO_H,WHITE))
parts.append(f'<text x="{HERO_X+34}" y="{HERO_Y+58}" font-family="{SANS}" font-size="15" font-weight="700" letter-spacing="3" fill="{TEAL}">GOLD-STANDARD CONSTITUENCY PARSING</text>')
parts.append(f'<text x="{HERO_X+32}" y="{HERO_Y+108}" font-family="{SERIF}" font-size="46" font-weight="700" fill="{INK}">Benepar Corpus</text>')
parts.append(f'<text x="{HERO_X+34}" y="{HERO_Y+140}" font-family="{SANS}" font-size="17" fill="{MUTE}">A corrected US-English constituency treebank</text>')
# tree fills the area below the title band
parts.append(nested_tree(f"{ASSET}/mom_forgot.svg", HERO_X+24, HERO_Y+160, HERO_W-48, HERO_H-190))
parts.append(f'<text x="{HERO_X+34}" y="{HERO_Y+HERO_H-22}" font-family="{SERIF}" font-size="15" font-style="italic" fill="{MUTE}">“Mom forgot to buy eggs … because she wanted to bake a cake.”</text>')

# ---- R1C3: Model ranking ----
x,y=cx(2),cy(0)
parts.append(tile(x,y,cell,cell,NAVY))
parts.append(f'<text x="{x+22}" y="{y+40}" font-family="{SANS}" font-size="15" font-weight="700" fill="{WHITE}">Model ranking</text>')
rank=[("Benepar",1.0,TEAL),("Stanza",0.66,ORANGE),("SuPar",0.42,"#5B6894"),("spaCy",0.28,"#45507A")]
by=y+64
for name,frac,col in rank:
    bw=(cell-44)*frac
    parts.append(f'<rect x="{x+22}" y="{by}" width="{bw:.1f}" height="26" rx="6" fill="{col}"/>')
    parts.append(f'<text x="{x+30}" y="{by+18}" font-family="{SANS}" font-size="13" font-weight="700" fill="{WHITE}">{name}</text>')
    by+=40

# ---- R2C3: POS disagreements ----
x,y=cx(2),cy(1)
parts.append(tile(x,y,cell,cell,NAVY))
parts.append(f'<text x="{x+22}" y="{y+40}" font-family="{SANS}" font-size="15" font-weight="700" fill="{WHITE}">Top POS disagreements</text>')
fams=[("IN / RB",4),("RB / JJ",4),("IN / JJ",3),(": / ,",3),("JJ / VBN",3)]
mx=4; by=y+64
for tag,c in fams:
    bw=(cell-155)*(c/mx)
    parts.append(f'<text x="{x+22}" y="{by+13}" font-family="{SANS}" font-size="12" fill="{ICE}">{tag}</text>')
    parts.append(f'<rect x="{x+92}" y="{by+2}" width="{bw:.1f}" height="14" rx="4" fill="{TEAL}"/>')
    parts.append(f'<text x="{x+92+bw+8:.1f}" y="{by+13}" font-family="{SANS}" font-size="11" font-weight="700" fill="{MUTE}">{c}</text>')
    by+=30

# ---- R3C1: stat 38 ----
def stat_tile(x,y,num,label,col):
    parts.append(tile(x,y,cell,cell,NAVY))
    parts.append(f'<text x="{x+cell/2:.1f}" y="{y+cell/2-6}" font-family="{SERIF}" font-size="74" font-weight="700" fill="{col}" text-anchor="middle">{num}</text>')
    parts.append(f'<text x="{x+cell/2:.1f}" y="{y+cell/2+40}" font-family="{SANS}" font-size="15" fill="{ICE}" text-anchor="middle">{label}</text>')
stat_tile(cx(0),cy(2),"38","corrected gold trees",TEAL)
stat_tile(cx(1),cy(2),"500","US-English sentences",ORANGE)

# ---- R3C3: CTA ----
x,y=cx(2),cy(2)
parts.append(tile(x,y,cell,cell,TEAL))
parts.append(f'<text x="{x+cell/2:.1f}" y="{y+56}" font-family="{SANS}" font-size="14" font-weight="700" letter-spacing="2" fill="{INK}" text-anchor="middle">OPEN DATASET</text>')
parts.append(f'<text x="{x+cell/2:.1f}" y="{y+96}" font-family="{SANS}" font-size="15" font-weight="700" fill="{INK}" text-anchor="middle">huggingface.co/datasets/</text>')
parts.append(f'<text x="{x+cell/2:.1f}" y="{y+120}" font-family="{SANS}" font-size="15" font-weight="700" fill="{INK}" text-anchor="middle">freshNfunky/benepar-corpus</text>')
parts.append(f'<text x="{x+cell/2:.1f}" y="{y+cell-28}" font-family="{SANS}" font-size="12" fill="#06493A" text-anchor="middle">MIT · Public Domain · CC-BY-4.0</text>')

parts.append("</svg>")
os.makedirs(ASSET,exist_ok=True)
open(OUT,"w",encoding="utf-8").write("\n".join(parts))
print("wrote",OUT, os.path.getsize(OUT),"bytes")

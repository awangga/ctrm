#!/usr/bin/env python3
"""Pass-2 resolver: NOT_FOUND/REVIEW anchors via arXiv API (authoritative).
Extract arXiv id + metadata from API response; DOI = 10.48550/arXiv.<id> (DataCite-registered).
Anti-fabrication: only write entries with title similarity >= 0.82."""
import urllib.parse, urllib.request, difflib, re, time, xml.etree.ElementTree as ET

OUT_BIB="/home/adb/awangga/trm/eksperimen/literatur/bibtex/scopusai_q1-q12_arxiv.bib"
OUT_REP="/home/adb/awangga/trm/eksperimen/literatur/doi_resolution_arxiv_report.md"
ATOM="{http://www.w3.org/2005/Atom}"; ARX="{http://arxiv.org/schemas/atom}"

# NOT_FOUND + REVIEW from pass-1 (scopus_id, surname, year, title, pillar)
REFS=[
 ("105010213243","Saunshi",2025,"Reasoning with Latent Thoughts: On the Power of Looped Transformers","P1"),
 ("85186371855","Yang",2024,"Looped Transformers are Better at Learning Learning Algorithms","P1"),
 ("105023640025","Xu",2025,"On Expressive Power of Looped Transformers: Theoretical Analysis and Enhancement via Timestep Encoding","P1"),
 ("85200579648","Jin",2024,"The Cost of Down-Scaling Language Models: Fact Recall Deteriorates Before In-Context Learning","P1"),
 ("105010225173","Snell",2025,"Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters","P2"),
 ("105027123539","Gema",2025,"Inverse Scaling in Test-Time Compute","P2"),
 ("105023640085","Chen",2025,"Do NOT Think That Much for 2+3=? On the Overthinking of o1-Like LLMs","P2"),
 ("85200604975","Lightman",2024,"Let's Verify Step by Step","P2"),
 ("85131887543","Schwarzschild",2021,"Can You Learn an Algorithm? Generalizing from Easy to Hard Problems with Recurrent Networks","P2"),
 ("85159351509","You",2023,"Zeus: Understanding and Optimizing GPU Energy Consumption of DNN Training","P3"),
 ("105010238045","Li",2025,"(Mis)Fitting: A Survey of Scaling Laws","P4"),
 ("105023635955","Choshen",2025,"A Hitchhiker's Guide to Scaling Law Estimation","P4"),
 ("85219577489","Brandfonbrener",2025,"Loss-to-Loss Prediction: Scaling Laws for All Datasets","P4"),
 ("85199876829","Caballero",2023,"Broken Neural Scaling Laws","P4b"),
 ("105023475637","Manhaeve",2026,"Benchmarking in Neuro-Symbolic AI","P5"),
 ("105023832374","Yang",2025,"The Hidden Joules: Evaluating the Energy Consumption of Vision Backbones","P6"),
 ("85029815608","Hallstedt",2017,"Sustainability integration in a technology readiness assessment framework","TKT"),
 ("105010219268","Mirzadeh",2025,"GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models","P2"),
 ("85124248843","Sharma",2022,"Scaling Laws from the Data Manifold Dimension","P4b"),
 ("105008281889","Li",2025,"Tackling the Abstraction and Reasoning Corpus with Vision Transformers","P5"),
 ("105003412401","Ates",2025,"Sudoku puzzle generation using mathematical programming and heuristics","P5"),
 ("105003623360","Wilhelm",2025,"Advocating Energy-per-Token in LLM Inference","P8"),
]

def norm(s): return re.sub(r"\s+"," ",re.sub(r"[^a-z0-9 ]"," ",(s or "").lower())).strip()

def query(title):
    # use simplified all: query of significant words to be robust
    words=[w for w in re.sub(r"[^A-Za-z0-9 ]"," ",title).split() if len(w)>2][:8]
    q='all:'+ " ".join(words)
    url="http://export.arxiv.org/api/query?"+urllib.parse.urlencode({"search_query":q,"max_results":"6"})
    req=urllib.request.Request(url,headers={"User-Agent":"trm-bib/1.0 (mailto:tisnasari.hafsah@unpad.ac.id)"})
    with urllib.request.urlopen(req,timeout=40) as r: return r.read()

def parse(xmlb,title,year):
    root=ET.fromstring(xmlb); best=None
    for e in root.findall(ATOM+"entry"):
        ct=(e.findtext(ATOM+"title") or "").strip()
        sim=difflib.SequenceMatcher(None,norm(title),norm(ct)).ratio()
        idu=e.findtext(ATOM+"id") or ""
        m=re.search(r"arxiv\.org/abs/([0-9]+\.[0-9]+)",idu)
        aid=m.group(1) if m else None
        pub=e.findtext(ATOM+"published") or ""
        yr=int(pub[:4]) if pub[:4].isdigit() else None
        auth=[a.findtext(ATOM+"name") for a in e.findall(ATOM+"author")]
        jref=e.findtext(ARX+"journal_ref"); pdoi=e.findtext(ARX+"doi")
        if not best or sim>best[0]: best=(sim,aid,yr,auth,ct,jref,pdoi)
    return best

entries=[]; report=[]
for sid,sur,yr,title,pil in REFS:
    st="NOT_ON_ARXIV"; note=""; doi=""
    try:
        b=parse(query(title),title,yr)
        if b and b[1]:
            sim,aid,ayr,auth,ct,jref,pdoi=b
            if sim>=0.82:
                st="VERIFIED_ARXIV"
                doi=pdoi or f"10.48550/arXiv.{aid}"
                key=re.sub(r"[^A-Za-z]","",sur).capitalize()+str(yr)
                au=" and ".join([f"{n.split()[-1]}, {' '.join(n.split()[:-1])}" for n in auth if n])
                f=[f"  author = {{{au}}}",f"  title = {{{ct}}}",f"  year = {{{ayr or yr}}}",
                   f"  journal = {{{jref if jref else 'arXiv preprint arXiv:'+aid}}}",
                   f"  eprint = {{{aid}}}",f"  archivePrefix = {{arXiv}}",f"  doi = {{{doi}}}",
                   f"  note = {{Scopus EID 2-s2.0-{sid}; pillar {pil}; arXiv sim={sim:.2f}{'; published: '+jref if jref else ''}}}"]
                entries.append("@ARTICLE{%s,\n%s\n}\n"%(key,",\n".join(f)))
            else:
                st="LOW_SIM"; note=f"best_sim={sim:.2f} matched={ct[:60]!r}"
        else: note="no arxiv id"
    except Exception as ex: st="ERROR"; note=str(ex)[:100]
    report.append((st,pil,sid,sur,yr,title,doi,note)); time.sleep(3.1)

with open(OUT_BIB,"w") as f:
    f.write("% Pass-2: anchors resolved via arXiv API (DOI = arXiv DataCite DOI or published DOI).\n\n")
    f.write("\n".join(entries))
from collections import Counter; c=Counter(r[0] for r in report)
with open(OUT_REP,"w") as f:
    f.write("# Laporan Resolusi Pass-2 (arXiv) — anchor NOT_FOUND/REVIEW\n\n")
    f.write(f"Total {len(REFS)}. Ringkasan: "+", ".join(f"{k}={v}" for k,v in sorted(c.items()))+"\n\n")
    f.write("VERIFIED_ARXIV ditulis ke `bibtex/scopusai_q1-q12_arxiv.bib`. Sisanya tetap perlu cek manual.\n\n")
    f.write("| Status | Pilar | Scopus ID | Penulis | Tahun | Judul | DOI | Catatan |\n|---|---|---|---|---|---|---|---|\n")
    for st,pil,sid,sur,yr,title,doi,note in sorted(report,key=lambda r:r[0]):
        t=title if len(title)<=55 else title[:52]+"..."
        f.write(f"| {st} | {pil} | {sid} | {sur} | {yr} | {t} | {doi} | {note} |\n")
print("DONE",dict(c),"entries:",len(entries))

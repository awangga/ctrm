#!/usr/bin/env python3
"""Resolve curated Scopus-AI references to DOI + verified metadata via CrossRef.
Anti-fabrication: only emit a .bib entry when title similarity is high AND year matches;
otherwise flag NEEDS_REVIEW. Never invent DOIs/metadata."""
import json, re, time, urllib.parse, urllib.request, difflib, sys

MAILTO = "tisnasari.hafsah@unpad.ac.id"
OUT_BIB = "/home/adb/awangga/trm/eksperimen/literatur/bibtex/scopusai_q1-q12.bib"
OUT_REP = "/home/adb/awangga/trm/eksperimen/literatur/doi_resolution_report.md"

# (scopus_id, first_author_surname, year, title, pillar)
REFS = [
 ("105010213243","Saunshi",2025,"Reasoning with Latent Thoughts: On the Power of Looped Transformers","P1"),
 ("85186371855","Yang",2024,"Looped Transformers are Better at Learning Learning Algorithms","P1"),
 ("105000536339","Csordas",2024,"MoEUT: Mixture-of-Experts Universal Transformers","P1"),
 ("105000500667","Sanford",2024,"Understanding Transformer Reasoning Capabilities via Graph Algorithms","P1"),
 ("105023640025","Xu",2025,"On Expressive Power of Looped Transformers: Theoretical Analysis and Enhancement via Timestep Encoding","P1"),
 ("85200579648","Jin",2024,"The Cost of Down-Scaling Language Models: Fact Recall Deteriorates Before In-Context Learning","P1"),
 ("85187456667","Veerabadran",2023,"Adaptive recurrent vision performs zero-shot computation scaling to unseen difficulty levels","P1"),
 ("85197008214","Bahri",2024,"Explaining neural scaling laws","P1"),
 ("105040135233","Yang",2025,"A Probabilistic Inference Scaling Theory for LLM Self-Correction","P2"),
 ("105010225173","Snell",2025,"Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters","P2"),
 ("105027123539","Gema",2025,"Inverse Scaling in Test-Time Compute","P2"),
 ("105023640085","Chen",2025,"Do NOT Think That Much for 2+3=? On the Overthinking of o1-Like LLMs","P2"),
 ("85200604975","Lightman",2024,"Let's Verify Step by Step","P2"),
 ("85131887543","Schwarzschild",2021,"Can You Learn an Algorithm? Generalizing from Easy to Hard Problems with Recurrent Networks","P2"),
 ("105010219268","Mirzadeh",2025,"GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models","P2"),
 ("85213910061","Yang",2024,"Accurate and Convenient Energy Measurements for GPUs: A Detailed Study of NVIDIA GPU's Built-In Power Sensor","P3"),
 ("85159351509","You",2023,"Zeus: Understanding and Optimizing GPU Energy Consumption of DNN Training","P3"),
 ("105003383331","Tschand",2025,"MLPerf Power: Benchmarking the Energy Efficiency of Machine Learning Systems from Microwatts to Megawatts for Sustainable AI","P3"),
 ("105029906534","van der Vlugt",2025,"PowerSensor3: A Fast and Accurate Open Source Power Measurement Tool","P3"),
 ("105032407647","Zhang",2025,"A Normalized Energy Efficiency Metric for AI Servers Under LLM Workloads","P3"),
 ("85143596581","Naser",2023,"Engineering Metrics to Enable Green Machine Learning from Tackling Accuracy-Energy Trade-offs","P3"),
 ("85165145257","Yarally",2023,"Uncovering Energy-Efficient Practices in Deep Learning Training: Preliminary Steps Towards Green AI","P3"),
 ("85163867329","Jay",2023,"An experimental comparison of software-based power meters: focus on CPU and GPU","P3"),
 ("105010238045","Li",2025,"(Mis)Fitting: A Survey of Scaling Laws","P4"),
 ("85159805124","Alabdulmohsin",2022,"Revisiting Neural Scaling Laws in Language and Vision","P4"),
 ("105028932436","Lourie",2025,"Scaling Laws Are Unreliable for Downstream Tasks: A Reality Check","P4"),
 ("105023635955","Choshen",2025,"A Hitchhiker's Guide to Scaling Law Estimation","P4"),
 ("85219577489","Brandfonbrener",2025,"Loss-to-Loss Prediction: Scaling Laws for All Datasets","P4"),
 ("65549085067","Clauset",2009,"Power-law distributions in empirical data","P4"),
 ("105040128618","Alnemari",2026,"Scaling Laws in the Tiny Regime: How Small Models Change Their Mistakes","P4b"),
 ("85124248843","Sharma",2022,"Scaling Laws from the Data Manifold Dimension","P4b"),
 ("85199876829","Caballero",2023,"Broken Neural Scaling Laws","P4b"),
 ("105023475637","Manhaeve",2026,"Benchmarking in Neuro-Symbolic AI","P5"),
 ("105032250570","Marinho Rocha",2025,"Program Synthesis Using Inductive Logic Programming for the Abstraction and Reasoning Corpus","P5"),
 ("105008281889","Li",2025,"Tackling the Abstraction and Reasoning Corpus with Vision Transformers","P5"),
 ("105003412401","Ates",2025,"Sudoku puzzle generation using mathematical programming and heuristics","P5"),
 ("84868276797","Ercsey-Ravasz",2012,"The chaos within Sudoku","P5"),
 ("85217768324","Aquino-Britez",2025,"Towards an Energy Consumption Index for Deep Learning Models: A Comparative Analysis of Architectures, GPUs, and Measurement Tools","Q4"),
 ("85124516601","Asperti",2022,"Dissecting FLOPs Along Input Dimensions for GreenAI Cost Estimations","Q4"),
 ("85191076521","Wan",2024,"Towards Cognitive AI Systems: a Workload and Characterization Study of Neuro-Symbolic AI","Q4"),
 ("105035350995","Peykani",2026,"Green Artificial Intelligence: A Comprehensive Review of Metrics, Tools, Challenges, Trends and Future Prospects","P6"),
 ("105023832374","Yang",2025,"The Hidden Joules: Evaluating the Energy Consumption of Vision Backbones","P6"),
 ("85208745118","Mao",2024,"Green Edge AI: A Contemporary Survey","P6"),
 ("105020466564","AlSideiri",2025,"Green Computing in Educational Settings: Leveraging Artificial Intelligence for Sustainable Teaching and Entertainment","P7a"),
 ("105030240446","Villegas-Ch",2026,"Integrating artificial intelligence and data envelopment analysis for sustainable efficiency assessment in higher education","P7a"),
 ("105001840746","Almatrafi",2025,"Leveraging generative AI for course learning outcome categorization using Bloom's taxonomy","P7a"),
 ("85197277473","Tariq",2024,"Complex artificial intelligence models for energy sustainability in educational buildings","P7a"),
 ("105040096470","Costa",2026,"Empowering Local Frugal Edge AI Innovation Based on Participatory Citizen Science in Developing Countries","P7a"),
 ("105041932528","Surjandari",2026,"Digital carbon footprint assessment of Universitas Indonesia data center based on the GHG protocol","P7b"),
 ("105013161244","Rahmandhika",2025,"Energy audit and optimization approach for university building energy efficiency improvement","P7b"),
 ("85213680502","Muzayyinah",2024,"Green Campus Implementation and Challenges in Education and Research in Indonesia","P7b"),
 ("105028170479","Rima Melati",2025,"The directions of limiting the use of artificial intelligence through regulation in Indonesia","P7b"),
 ("105021210910","Devita",2025,"The Green Gap: Lessons from the EU AI Act for Indonesia's Sustainable Digital Future","P7b"),
 ("105040074808","Yudatama",2026,"Dataset on Green IT adoption in Indonesian higher education institutions: A survey-based study","P7b"),
 ("105031056858","Babu",2026,"Towards Sustainable Lifecycle Maturity: Integrating Sustainable Development Goals into the eTRL Framework","TKT"),
 ("85029815608","Hallstedt",2017,"Sustainability integration in a technology readiness assessment framework","TKT"),
 ("105036827649","Marmouzi",2026,"A Systematic Review of Green and Sustainable AI: Taxonomy, Metrics, Challenges, and Open Research Directions","TKT"),
 ("85179003275","Sikand",2023,"Green AI Quotient: Assessing Greenness of AI-based software and the way forward","TKT"),
 ("105040767903","Rojahn",2026,"Green artificial intelligence and its lifecycle, hardware, and measurement dimensions","TKT"),
 ("105035959276","de Zarza",2026,"Energy-Aware Multilingual Evaluation of Large Language Models","P8"),
 ("105003623360","Wilhelm",2025,"Advocating Energy-per-Token in LLM Inference","P8"),
 ("105041062448","Tschand",2026,"MLPerf Power: Addressing the AI Power Wall","P8"),
 ("105037094110","Belkhiri",2026,"Recursive Weight Sharing for Parameter-Efficient Deep Convolutional Networks: Application to Skin Lesion Classification","P8"),
 ("105035967519","Zschache",2026,"Comparing energy consumption and accuracy in text classification inference","P8"),
]

def norm(s):
    s = re.sub(r"[^a-z0-9 ]"," ", (s or "").lower())
    return re.sub(r"\s+"," ", s).strip()

def fetch(title):
    q = urllib.parse.urlencode({"query.bibliographic": title, "rows": "5", "mailto": MAILTO})
    url = "https://api.crossref.org/works?" + q
    req = urllib.request.Request(url, headers={"User-Agent": f"trm-bib/1.0 (mailto:{MAILTO})"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["message"]["items"]

def best(title, year, items):
    nt = norm(title); cand=[]
    for it in items:
        ct = (it.get("title") or [""])[0]
        sim = difflib.SequenceMatcher(None, nt, norm(ct)).ratio()
        yr = None
        for k in ("published-print","published-online","published","issued"):
            if it.get(k,{}).get("date-parts",[[None]])[0][0]:
                yr = it[k]["date-parts"][0][0]; break
        cand.append((sim, yr, it, ct))
    cand.sort(key=lambda x:-x[0])
    return cand[0] if cand else None

def bibkey(sur, yr, used):
    base = re.sub(r"[^A-Za-z]","", sur).capitalize() + str(yr)
    k=base; i=0
    while k in used: i+=1; k=base+chr(96+i)
    used.add(k); return k

def fmt_authors(it):
    a=[]
    for au in it.get("author",[]):
        fam=au.get("family",""); giv=au.get("given","")
        a.append(f"{fam}, {giv}".strip(", ") if fam else au.get("name",""))
    return " and ".join([x for x in a if x])

used=set(); entries=[]; report=[]
for sid, sur, yr, title, pillar in REFS:
    status="ERROR"; doi=""; sim=0; note=""
    try:
        items = fetch(title)
        b = best(title, yr, items)
        if b:
            sim, cyr, it, ct = b
            doi = it.get("DOI","")
            ymatch = (cyr is None) or (abs((cyr or yr)-yr)<=1)
            if sim>=0.82 and doi and ymatch:
                status="VERIFIED"
                key=bibkey(sur,yr,used)
                authors=fmt_authors(it)
                cont=(it.get("container-title") or [""])[0]
                vol=it.get("volume",""); iss=it.get("issue","")
                pg=it.get("page",""); pub=it.get("publisher","")
                typ=it.get("type","")
                fields=[f"  author = {{{authors}}}", f"  title = {{{ct}}}", f"  year = {{{cyr or yr}}}"]
                if cont: fields.append(f"  journal = {{{cont}}}")
                if vol: fields.append(f"  volume = {{{vol}}}")
                if iss: fields.append(f"  number = {{{iss}}}")
                if pg: fields.append(f"  pages = {{{pg}}}")
                if pub: fields.append(f"  publisher = {{{pub}}}")
                fields.append(f"  doi = {{{doi}}}")
                fields.append(f"  note = {{Scopus EID 2-s2.0-{sid}; type {typ}; pillar {pillar}; verified CrossRef sim={sim:.2f}}}")
                entries.append("@ARTICLE{%s,\n%s\n}\n" % (key, ",\n".join(fields)))
            elif sim>=0.65 and doi:
                status="REVIEW"; note=f"sim={sim:.2f} doi={doi} cyr={cyr} matched_title={ct!r}"
            else:
                status="NOT_FOUND"; note=f"best_sim={sim:.2f} (likely arXiv/conf without CrossRef DOI)"
        else:
            status="NOT_FOUND"; note="no items"
    except Exception as e:
        status="ERROR"; note=str(e)[:120]
    report.append((status, pillar, sid, sur, yr, title, doi, note))
    time.sleep(0.35)

with open(OUT_BIB,"w") as f:
    f.write("% Auto-resolved from Scopus-AI curated refs via CrossRef (verified entries only).\n")
    f.write("% NEEDS_REVIEW / NOT_FOUND entries are NOT written here; see doi_resolution_report.md.\n\n")
    f.write("\n".join(entries))

from collections import Counter
c=Counter(r[0] for r in report)
with open(OUT_REP,"w") as f:
    f.write("# Laporan Resolusi DOI (CrossRef) — referensi Scopus AI q1-q12\n\n")
    f.write(f"Total {len(REFS)} referensi. Ringkasan: " + ", ".join(f"{k}={v}" for k,v in sorted(c.items())) + "\n\n")
    f.write("VERIFIED = ditulis ke `bibtex/scopusai_q1-q12.bib`. REVIEW/NOT_FOUND = perlu tindak lanjut manual (kemungkinan arXiv/ICLR/NeurIPS/PMLR/TMLR tanpa DOI CrossRef).\n\n")
    f.write("| Status | Pilar | Scopus ID | Penulis | Tahun | Judul | DOI | Catatan |\n|---|---|---|---|---|---|---|---|\n")
    order={"VERIFIED":0,"REVIEW":1,"NOT_FOUND":2,"ERROR":3}
    for st,pil,sid,sur,yr,title,doi,note in sorted(report,key=lambda r:(order.get(r[0],9),r[1])):
        t=title if len(title)<=60 else title[:57]+"..."
        f.write(f"| {st} | {pil} | {sid} | {sur} | {yr} | {t} | {doi} | {note} |\n")

print("DONE", dict(c), "entries:", len(entries))

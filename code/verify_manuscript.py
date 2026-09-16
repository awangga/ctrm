#!/usr/bin/env python3
"""Gerbang pra-submit: periksa 98 invarian naskah terhadap berkas yang sebenarnya.

Lahir dari fase BQ, setelah tiga suntingan naskah ternyata tidak pernah masuk ke berkas
karena tool call-nya gagal dan tidak diperiksa ulang, dan tiga judul usang lolos di
README/CLAUDE/lapakhir karena sapuannya tidak menyeluruh. Melaporkan hasil kerja tanpa
memeriksa berkasnya adalah kelas cacat tersendiri; skrip ini yang memeriksanya.

Yang diperiksa: judul di delapan kanal, koreksi family-wise, koreksi lantai idle,
kaveat NVML di empat tempat, angka statistik dan energi kunci, keberadaan dan isi
joules_to_target_faithful.csv, penamaan figur dan urutan kutipannya, sinkronnya TIGA
salinan highlights, posisi Acknowledgements dan nomor kontrak, kosakata yang sudah
dicabut (mengizinkan sebutan yang jelas bersifat pencabutan), dan kebersihan kompilasi.

Pakai:  python3 eksperimen/frontier/verify_manuscript.py   (dari akar repo)
Keluar: daftar OK/GAGAL per butir. GAGAL berarti naskah belum boleh disubmit.
"""
import re,os,subprocess,csv,statistics as st
M=open('manuscript/main.tex').read()
PDF=subprocess.run(['pdftotext','manuscript/main.pdf','-'],capture_output=True,text=True).stdout
ok=fail=0
def chk(label,cond,detail=""):
    global ok,fail
    if cond: ok+=1; print(f"  OK    {label}")
    else: fail+=1; print(f"  GAGAL {label}  {detail}")

print("== JUDUL ==")
T="The energy cost of recursion depth: half the joules for the same accuracy"
chk("judul di main.tex", f"\\title{{{T}}}" in M)
chk("judul tercetak di PDF", T.replace(": ",": ") .split(":")[0] in PDF and "half the joules" in PDF)
for f in ['manuscript/cover_letter.md','README.md','zenodo/README.md','zenodo/.zenodo.json',
          '/home/adb/awangga/ctrm/README.md','lapakhir/BAHAN_perubahan_setelah_lapkemajuan.md','CLAUDE.md']:
    s=open(f).read()
    flat=re.sub(r'\s+',' ',s)
    chk(f"judul baru ada di {os.path.basename(f)}", T in flat)
    hits=[]
    for old_t in ("more joules, no extra accuracy","recursion depth in tiny recursive models"):
        for m in re.finditer(re.escape(old_t), flat):
            ctx=flat[max(0,m.start()-220):m.start()+220].lower()
            if not any(e in ctx for e in ("tidak dipakai","dicabut","reproducibility package",
                                          "judul fase","sebelumnya","tiga judul","withdrawn")):
                hits.append(old_t)
    chk(f"judul lama mati di {os.path.basename(f)}", not hits, str(set(hits)))

print("== KOREKSI SUBSTANTIF ==")
chk("family-wise: tiga kontras disebut", "Three contrasts survive that threshold" in M)
for p in ["5.6{\\times}10^{-5}","0.0013","0.0010","0.0050","0.0047","0.0066","0.0083","0.0041"]:
    chk(f"  p={p} ada di paragraf family-wise", p in M)
chk("family-wise: klaim lama mati", "the two extreme baseline contrasts\nsurvive" not in M and "and the two extreme baseline contrasts" not in M)
chk("idle residu 4.3%", "at most $4.3\\%$ of net energy" in M)
chk("idle median 3.1%", "$3.1\\%$ at the median" in M)
chk("idle residu 2% mati", "at most $2\\%$ of it" not in M)
chk("sensitivitas idle 0/4.7/8.8", "$0$, $4.7$ and $8.8$" in M.replace("P_{\\text{idle}}{=}","") or "4.7$ and $8.8" in M)
chk("lonjakan setup: 18 dari 49", "$18$ of the $49$ runs" in M)
chk("lonjakan setup: at most 7 s", "for at most $7$\\,s" in M)
chk("lonjakan lama mati", "for between $7$ and $58$\\,s on about twenty runs" not in M)

print("== KAVEAT NVML ==")
chk("tidak ada 'two meters'", "two meters" not in M)
chk("tidak ada dual-instrument di naskah", "dual-instrument" not in M)
chk("Intro: bounds the bookkeeping", "bounds the bookkeeping, not the sensor" in M)
chk("Results: read the same sensor", "read the same sensor" in M)
chk("Threats: bias sensor diakui", "neither addresses bias in the sensor they share" in M)
z=open('zenodo/README.md').read()
chk("zenodo README ada kaveat NVML", "NVML" in z and ("same" in z.lower()))
chk("cover letter tanpa dual-instrument", "dual-instrument" not in open('manuscript/cover_letter.md').read())

print("== STATISTIK & ANGKA ==")
chk("Welch df rapuh diakui", "$2.1$--$2.8$" in M and "fragile" in M)
chk("lantai permutasi n=3", "\\binom{6}{3}{=}20" in M and "$0.10$" in M)
chk("kuantisasi dipropagasikan", "checkpoint quantisation of" in M and "about $16$\\,Wh on Sudoku-Extreme" in M)
chk("grid 11-13 Wh", "$11$--$13$\\,Wh on Maze-Hard" in M)
chk("grid 12-14 mati", "$12$--$14$\\,Wh" not in M)
chk("ARC 30.6-36.3", "($30.6$--$36.3\\%$ token)" in M)
chk("D18@24k 48.6", "the $48.6\\%$ of $D_{\\text{eff}}{=}18$" in M)
chk("crossover 130-210", "between roughly $130$ and\n$210$\\,Wh" in M or "roughly $130$ and $210$" in M)
chk("t baseline +10.22", "$t{=}10.22$" in M)
chk("t baseline -10.17", "$t{=}-10.17$" in M)
chk("plateau ARC D36 25.1-26.5", "$25.1$--$26.5\\%$" in M)

print("== BERKAS RILIS ==")
chk("kalimat menyebut CSV", "joules\\_to\\_target\\_faithful.csv" in M)
for pth in ['eksperimen/frontier/ablation/joules_to_target_faithful.csv',
            'eksperimen/frontier/joules_to_target_faithful.py',
            'zenodo/code/joules_to_target_faithful.py',
            '/home/adb/awangga/ctrm/code/joules_to_target_faithful.py']:
    chk(f"ada {os.path.basename(pth)} di {os.path.dirname(pth).split('/')[-1]}", os.path.exists(pth))
rows=list(csv.DictReader(open('eksperimen/frontier/ablation/joules_to_target_faithful.csv')))
def agg(t,tg,d,h=None):
    v=[float(r['net_Wh_at_target']) for r in rows if r['task']==t and r['target_pct']==tg
       and r['reached']=='yes' and r['D_eff']==d and (h is None or r['hidden']==h)]
    return (round(st.mean(v)),round(st.stdev(v)) if len(v)>1 else 0,len(v))
chk("CSV -> 268+-19 (D9 50%)", agg('Sudoku-Extreme','50','9','512')[:2]==(268,19))
chk("CSV -> 329+-17 (D18 50%)", agg('Sudoku-Extreme','50','18','512')[:2]==(329,17))
chk("CSV -> 353+-43 (baseline)", agg('Sudoku-Extreme','50','none')[:2]==(353,43))
chk("CSV -> 161+-1 (D9 ke 36.3)", agg('Sudoku-Extreme','36.26','9','512')[:2]==(161,1))
chk("CSV -> 155+-82 (ARC D9)", agg('ARC-AGI-1','30.59','9')[:2]==(155,82))

print("== FIGUR ==")
chk("fig_regime_map mati di main.tex", "fig_regime_map" not in M)
chk("fig_baseline_distance dipakai", "figures/fig_baseline_distance.pdf" in M)
chk("PDF figur baru ada", os.path.exists('manuscript/figures/fig_baseline_distance.pdf'))
chk("PDF figur lama hilang", not os.path.exists('manuscript/figures/fig_regime_map.pdf'))
for c in ['eksperimen/frontier/make_manuscript_figures.py','zenodo/code/make_manuscript_figures.py',
          '/home/adb/awangga/ctrm/code/make_manuscript_figures.py']:
    chk(f"tanpa 'regime' di {c.split('/')[-2]}/", 'regime' not in open(c).read().lower())
md5=[subprocess.run(['md5sum',c],capture_output=True,text=True).stdout.split()[0] for c in
     ['eksperimen/frontier/make_manuscript_figures.py','zenodo/code/make_manuscript_figures.py',
      '/home/adb/awangga/ctrm/code/make_manuscript_figures.py']]
chk("tiga salinan skrip figur identik", len(set(md5))==1, str(md5))
chk("caption pakai (a)/(b)", "(a) Best-checkpoint Sudoku-Extreme" in M and "(b) The width axis" in M)
chk("caption Left/Right mati", "differently. Left:" not in M)
spans=[(m.start(),m.end(),re.search(r'\\label\{(fig:[^}]*)\}',m.group(0)).group(1))
       for m in re.finditer(r'\\begin\{figure\}.*?\\end\{figure\}',M,re.S)]
order={l:i+1 for i,(a,b,l) in enumerate(spans)}
def inside(p): return any(a<=p<b for a,b,_ in spans)
first={}
for m in re.finditer(r'\\ref\{(fig:[^}]*)\}',M):
    if not inside(m.start()): first.setdefault(m.group(1),m.start())
seq=[order[l] for l,_ in sorted(first.items(),key=lambda kv:kv[1])]
chk("urutan kutipan figur monoton", seq==sorted(seq), str(seq))
chk("semua figur dikutip", len(first)==len(spans), f"{len(first)}/{len(spans)}")

print("== HIGHLIGHTS (TIGA SALINAN) ==")
tex_h=re.search(r'\\begin\{highlights\}(.*?)\\end\{highlights\}',M,re.S).group(1)
tex_items=[l.strip()[6:].strip() for l in tex_h.strip().split('\n') if l.strip().startswith('\\item')]
txt_items=[l[2:].strip() for l in open('manuscript/highlights.txt') if l.startswith('- ')]
import zipfile
zx=zipfile.ZipFile('manuscript/highlights.docx').read('word/document.xml').decode()
docx_items=re.findall(r'<w:t[^>]*>([^<]{20,})</w:t>',zx)
chk("main.tex 5 butir", len(tex_items)==5, str(len(tex_items)))
chk("txt == tex", txt_items==tex_items, f"{txt_items}\n     {tex_items}")
chk("docx == tex", docx_items==tex_items, f"{docx_items}")
chk("semua <=85 karakter", all(len(h)<=85 for h in tex_items), str([len(h) for h in tex_items]))
chk("tidak ada 'saturates' di highlights", not any('saturat' in h for h in tex_items+docx_items))
chk("tidak ada 'on three tasks'", not any('three tasks' in h for h in tex_items))

print("== BACK MATTER ==")
i_ack=M.find('\\section*{Acknowledgements}'); i_bib=M.find('\\bibliographystyle')
i_ai=M.find('\\section*{Declaration of generative AI')
chk("Acknowledgements sebelum bibliografi", 0<i_ack<i_bib)
chk("Acknowledgements setelah deklarasi lain", i_ai<i_ack)
chk("nomor kontrak di Acknowledgements", "283" in M[i_ack:i_bib] and "1650" in M[i_ack:i_bib] and "PKS.003" in M[i_ack:i_bib])
chk("nomor kontrak tercetak di PDF", "283/C3/DT.05.00/PL-BARU/2026" in PDF.replace("\n",""))

print("== KOSAKATA DICABUT ==")
files=['manuscript/main.tex','manuscript/highlights.txt','manuscript/cover_letter.md','README.md',
       'zenodo/README.md','zenodo/PROTOCOL.md','/home/adb/awangga/ctrm/README.md']
EXPL=('tidak ada klaim','withdrawn','retracted','dicabut','pencabutan','later withdrawn',
      'earlier manuscript','framing, an earlier','keeps framings','tidak dipakai')
for f in files:
    s=open(f).read().lower()
    bad=[]
    for w in ['regime map','saturated regime','sign flip']:
        for m in re.finditer(re.escape(w),s):
            ctx=s[max(0,m.start()-160):m.start()+160]
            if not any(e in ctx for e in EXPL): bad.append(w)
    chk(f"kosakata dicabut bersih di {os.path.basename(f)}", not bad, str(set(bad)))

print("== SUBBAGIAN & RUJUKAN ==")
chk("'Beyond the primary task'", "\\subsection{Beyond the primary task:" in M)
chk("'Cross-task generality' mati", "Cross-task generality" not in M)
chk("'Figure~\\ref{fig:costmodel}'", "Figure~\\ref{fig:costmodel} shows" in M)
chk("scope pelatihan dinyatakan", "training-budget allocation" in M)

print("== KOMPILASI ==")
log=open('manuscript/main.log').read()
chk("0 error LaTeX", len(re.findall(r'(?m)^!',log))==0)
chk("0 undefined", 'undefined' not in log.lower())
pg=subprocess.run(['pdfinfo','manuscript/main.pdf'],capture_output=True,text=True).stdout
chk("48 halaman", 'Pages:           48' in pg, pg.split('Pages:')[1].split('\n')[0] if 'Pages:' in pg else '?')
ab=re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',M,re.S).group(1)
ab=re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?','',ab); ab=re.sub(r'[{}$\\]',' ',ab)
chk("abstrak <=250 kata", len(ab.split())<=250, f"{len(ab.split())} kata")

print(f"\n== RINGKAS: {ok} OK, {fail} GAGAL ==")

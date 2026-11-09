#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
『난중일기』의 조사 운용과 장르 효과 — 분석 스크립트 (골격)
Particle Usage and Genre Effects in the Nanjungilgi — analysis skeleton

논문 2장 4)의 절차를 구현한다.
  ① 유효 글자 수 산출 (U+4E00–U+9FFF만)
  ② 7자(之·者·所·也·矣·焉·乎) 전수 추출 + 문맥(공기) 색인
  ③ 공기 규칙에 따른 기능 판별 (之: 助/代, 動은 수동 검토)
  ④ 100자당 밀도 (원계수·보정)
  ⑤ 쌍별 2×2 · 4×2 카이제곱, Cramér's V
  ⑥ 500자 창 부트스트랩 CI, Mann–Whitney

사용법:
  # 계산 재현(원문 불필요): 논문 <표 4> 계수로 표5·표6 재현
  python analysis.py stats

  # 색인 생성(원문 필요): corpora/<name>.txt 를 읽어 1차 판별 색인 생성
  python analysis.py index --corpora ./corpora --out ./out

주의:
  - 『무경칠서』 보정 원문 및 네 코퍼스 원문 전문은 저작권상 저장소에 포함하지 않는다.
    원문은 각 저본(README §3)에서 확보해 corpora/ 에 넣는다.
  - ③의 규칙 판별은 1차 자동 분류이며, 之의 動(동사 '가다')과 경계 사례는
    연구자 검토를 거쳐 확정한다(논문 <표 4>는 검토 반영 값).
"""
from __future__ import annotations
import re, os, sys, csv, argparse, glob
from collections import Counter
import numpy as np

try:
    from scipy.stats import chi2_contingency, mannwhitneyu
except ImportError:
    sys.exit("scipy가 필요합니다: pip install scipy")

# ───────────────────────── 상수 ─────────────────────────
HANZI = re.compile(r"[一-鿿]")          # 유효 글자(기본 한자)
PARTICLES = ["之", "者", "所", "也", "矣", "焉", "乎"]
STRUCT_OTHER = ["者", "所"]                      # 之 외 구조조사
MODAL = ["也", "矣", "焉", "乎"]                 # 문말어기조사(焉은 보정에서 제외)

# 之 목적어 대사(代) 판별용 후방 경계: 구두점 또는 절경계/종결 표지
ZHI_PRON_PUNCT = set("。，、；：？！.,·「」『』（）()【】〈〉《》○")
ZHI_PRON_FOLLOW = {"也", "矣", "而", "則", "以", "故", "乃", "遂", "焉", "乎"}

# 논문 <표 4> 계수(연구자 검토 반영) — 계산 재현의 입력값
#   name: (글자수, 之전체, 之助, 之動, 者, 所, 也, 矣, 焉, 乎)
PAPER_TABLE4 = {
    "난중일기":   (60287, 444, 355, 0, 33, 136,   93,  40,   4,  10),
    "이분 행록":  (10704, 314, 180, 1, 75,  44,   71,  26,  12,  13),
    "무경칠서":   (54355, 2207, 1502, 0, 1063, 403, 953, 114, 61, 118),
    "사서 경문":  (42557, 1954, 1359, 0, 699, 260, 1334, 325, 178, 312),
}
# 장계(보조, 저자 내 대조) — 之 助/代 분해는 색인 기준, 보조 검정 제외
PAPER_JANGGYE = dict(글자수=46457, 之전체=885, 者=105, 所=267,
                     也=26, 矣=97, 焉=2, 乎=7)


# ───────────────────── ① 유효 글자 수 ─────────────────────
def count_hanzi(text: str) -> int:
    return len(HANZI.findall(text))


# ───────────── ②③ 전수 추출 + 공기 규칙 1차 판별 ─────────────
def classify_zhi(text: str, i: int) -> str:
    """之[i]의 기능을 후방 공기로 1차 판별. 助(구조조사)/代(목적어 대사).
       動(동사 '가다')은 규칙만으로 확정 불가 → '助'로 두고 수동 검토 대상 표시."""
    j = i + 1
    while j < len(text) and text[j].isspace():
        j += 1
    nxt = text[j] if j < len(text) else ""
    if nxt == "" or nxt in ZHI_PRON_PUNCT or nxt in ZHI_PRON_FOLLOW:
        return "代"
    return "助"


def is_verb_candidate(text: str, i: int) -> bool:
    """之 動(가다) 후보: 앞이 접속/이동 맥락(而·遂·乃·往·如)이고 뒤가 명사.
       예: '浮海而之他'. 자동 확정하지 않고 수동 검토 플래그로만 사용."""
    prev = text[i - 1] if i > 0 else ""
    return prev in {"而", "遂", "乃", "往", "如", "復"}


def extract_index(name: str, text: str, win: int = 12) -> list[dict]:
    """7자 전수 추출 + ±win자 문맥 + 1차 판별 색인."""
    rows = []
    for i, ch in enumerate(text):
        if ch not in PARTICLES:
            continue
        pre = text[max(0, i - win):i]
        post = text[i + 1:i + 1 + win]
        flag = ""
        if ch == "之":
            pan = classify_zhi(text, i)
            if is_verb_candidate(text, i):
                flag = "動?(검토)"
            cat = "구조조사" if pan == "助" else "목적어대사"
        elif ch in STRUCT_OTHER:
            pan, cat = "助", "구조조사"
        else:  # 也矣焉乎
            pan, cat = "계수", "어기조사"
        rows.append(dict(텍스트=name, 표기자=ch, 앞문맥=pre, 뒤문맥=post,
                         판별=pan, 범주=cat, 비고=flag))
    return rows


# ───────────────────── ④ 계수 집계 ─────────────────────
def aggregate(name: str, text: str, index_rows: list[dict]) -> dict:
    """색인에서 <표 4> 계수를 집계."""
    chars = count_hanzi(text)
    zhi = [r for r in index_rows if r["표기자"] == "之"]
    zhi_zho = sum(r["판별"] == "助" for r in zhi)          # 之 助(구조조사)
    zhi_dae = sum(r["판별"] == "代" for r in zhi)          # 之 代(목적어 대사)
    cnt = Counter(r["표기자"] for r in index_rows)
    return dict(글자수=chars, 之전체=cnt["之"], 之助=zhi_zho, 之代=zhi_dae,
                者=cnt["者"], 所=cnt["所"],
                也=cnt["也"], 矣=cnt["矣"], 焉=cnt["焉"], 乎=cnt["乎"])


def from_paper_table4(name: str) -> dict:
    g, ztot, zzho, zdong, ja, so, ya, ui, eon, ho = PAPER_TABLE4[name]
    return dict(글자수=g, 之전체=ztot, 之助=zzho, 之代=ztot - zzho - zdong,
                者=ja, 所=so, 也=ya, 矣=ui, 焉=eon, 乎=ho)


# ── 구조/어기 묶음 (원계수·보정) ──
def struct_modal(c: dict, corrected: bool):
    """구조조사(之help+者) 대 문말어기조사. corrected=True면 보정(之助, 焉 제외)."""
    if corrected:
        struct = c["之助"] + c["者"]          # 보정 구조 (所 제외)
        modal = c["也"] + c["矣"] + c["乎"]    # 보정 어기 (焉 제외)
    else:
        struct = c["之전체"] + c["者"]         # 원계수 구조
        modal = c["也"] + c["矣"] + c["焉"] + c["乎"]
    return struct, modal


def density(c: dict):
    s0, m0 = struct_modal(c, corrected=False)
    s1, m1 = struct_modal(c, corrected=True)
    g = c["글자수"]
    return dict(원계수_구조밀도=s0 / g * 100, 원계수_어기밀도=m0 / g * 100,
                보정_구조밀도=s1 / g * 100, 보정_어기밀도=m1 / g * 100)


# ───────────────── ⑤ 카이제곱 · Cramér's V ─────────────────
def chi2_2x2(a, b):
    """a=(struct,modal), b=(struct,modal); 연속성 보정 없음."""
    tab = np.array([a, b], dtype=float)
    chi, p, _, _ = chi2_contingency(tab, correction=False)
    v = np.sqrt(chi / tab.sum())
    return chi, p, v


def chi2_rx2(rows):
    tab = np.array(rows, dtype=float)
    chi, p, _, _ = chi2_contingency(tab, correction=False)
    n, r, k = tab.sum(), tab.shape[0], tab.shape[1]
    v = np.sqrt(chi / (n * min(r - 1, k - 1)))
    return chi, p, v, int(n)


def table6(counts: dict[str, dict]):
    """보정 계수 기준 쌍별 2×2 + 4×2."""
    order = ["이분 행록", "난중일기", "무경칠서", "사서 경문"]
    sm = {k: struct_modal(counts[k], corrected=True) for k in order}
    labels = {"이분 행록": "행록", "난중일기": "난중일기",
              "무경칠서": "무경칠서", "사서 경문": "사서"}
    pairs = [("이분 행록", "난중일기"), ("이분 행록", "무경칠서"),
             ("이분 행록", "사서 경문"), ("난중일기", "무경칠서"),
             ("난중일기", "사서 경문"), ("무경칠서", "사서 경문")]
    print("\n<표 6> 쌍별 2×2 카이제곱 (보정 계수)")
    print(f"  {'비교 쌍':<16}{'χ²(1)':>9}{'p':>10}{'V':>8}")
    for a, b in pairs:
        chi, p, v = chi2_2x2(sm[a], sm[b])
        ps = "<.001" if p < .001 else f"{p:.3f}".lstrip("0")
        print(f"  {labels[a]+' 대 '+labels[b]:<16}{chi:>9.2f}{ps:>10}{('.'+f'{v:.3f}'.split('.')[1]):>8}")
    chi, p, v, n = chi2_rx2([sm[k] for k in order])
    print(f"  4×2 전체: χ²(3, N={n:,})={chi:.2f}, p={'<.001' if p<.001 else f'{p:.3g}'}, V={v:.3f}")


def table5(counts: dict[str, dict]):
    print("\n<표 5> 구조조사 : 문말어기조사 (원계수 / 보정)")
    print(f"  {'코퍼스':<10}{'구조:어기(원)':>14}{'구조:어기(보정)':>16}{'어기비중(보정)':>16}")
    for k in ["난중일기", "이분 행록", "무경칠서", "사서 경문"]:
        s0, m0 = struct_modal(counts[k], corrected=False)
        s1, m1 = struct_modal(counts[k], corrected=True)
        print(f"  {k:<10}{s0/m0:>13.2f}{s1/m1:>16.2f}{m1/(s1+m1)*100:>15.1f}%")


# ───────────────── ⑥ 500자 창 부트스트랩 · Mann–Whitney ─────────────────
def windows_struct_modal(text: str, index_rows: list[dict], size: int = 500):
    """텍스트를 유효글자 size자 창으로 나눠 각 창의 (보정 구조, 보정 어기)."""
    # 유효글자 위치만 추출
    hanzi_pos = [i for i, ch in enumerate(text) if HANZI.match(ch)]
    # 각 창에 속한 색인 행 집계(간이: 등장 순서 기반)
    # index_rows 는 text 등장 순서이므로 유효글자 번호로 창 배정
    wins = []
    # 빠른 집계를 위해 표기자별 보정 가담 여부
    def contrib(r):
        ch = r["표기자"]
        if ch == "之":
            return (1, 0) if r["판별"] == "助" else (0, 0)
        if ch in ("者",):
            return (1, 0)
        if ch in ("也", "矣", "乎"):
            return (0, 1)
        return (0, 0)  # 所(구조지만 밀도 제외), 焉(보정 제외)
    nwin = (len(hanzi_pos) + size - 1) // size
    acc = [[0, 0] for _ in range(max(nwin, 1))]
    # 각 색인 행의 텍스트 위치를 유효글자 순번으로 환산
    pos_rank = {p: r for r, p in enumerate(hanzi_pos)}
    # index_rows 에는 위치가 없으므로 재스캔
    k = 0
    occ = [(i, ch) for i, ch in enumerate(text) if ch in PARTICLES]
    for (i, ch), r in zip(occ, index_rows):
        # i 이하 유효글자 개수 = 창 번호
        rank = np.searchsorted(hanzi_pos, i)
        w = min(rank // size, len(acc) - 1)
        s, m = contrib(r)
        acc[w][0] += s
        acc[w][1] += m
    return np.array(acc, dtype=float)  # shape (nwin, 2): [struct, modal]


def bootstrap_modal_share_diff(winsA, winsB, n_boot=5000, seed=42):
    """두 코퍼스의 '어기 비중' 차이(A−B)에 대한 95% 부트스트랩 CI(창 리샘플)."""
    rng = np.random.default_rng(seed)
    def share(wins, idx):
        s = wins[idx].sum(axis=0)
        tot = s[0] + s[1]
        return s[1] / tot if tot else np.nan
    diffs = []
    for _ in range(n_boot):
        ia = rng.integers(0, len(winsA), len(winsA))
        ib = rng.integers(0, len(winsB), len(winsB))
        diffs.append(share(winsA, ia) - share(winsB, ib))
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    return float(np.nanmean(diffs)) * 100, lo * 100, hi * 100


def mann_whitney_density(winsA, winsB):
    """창별 구조조사 밀도(보정) 분포의 Mann–Whitney U (예: 장계 vs 난중일기)."""
    da = winsA[:, 0] / 500 * 100
    db = winsB[:, 0] / 500 * 100
    u, p = mannwhitneyu(da, db, alternative="two-sided")
    return u, p


# ───────────────────────── 실행 ─────────────────────────
def run_stats():
    """계산 재현: 논문 <표 4> 계수로 표5·표6 산출 (원문 불필요)."""
    counts = {k: from_paper_table4(k) for k in PAPER_TABLE4}
    print("== 밀도(100자당) ==")
    for k, c in counts.items():
        d = density(c)
        print(f"  {k:<10} 원계수 {d['원계수_구조밀도']:.2f}/{d['원계수_어기밀도']:.2f}"
              f"  보정 {d['보정_구조밀도']:.2f}/{d['보정_어기밀도']:.2f}")
    jg = PAPER_JANGGYE
    print(f"  {'장계(보조)':<10} 구조 {(jg['之전체']+jg['者'])/jg['글자수']*100:.2f}"
          f"  어기 {(jg['也']+jg['矣']+jg['焉']+jg['乎'])/jg['글자수']*100:.2f}")
    table5(counts)
    table6(counts)
    print("\n※ 각주 5·7(저본 일부 제외)과 각주 9(500자 창 부트스트랩)의 로버스트니스는"
          "\n  corpora/ 원문을 넣고 `python analysis.py index` 로 색인을 만든 뒤 재계산한다.")


def run_index(corpora_dir: str, out_dir: str):
    """원문 → 색인 생성 + 계수 집계 + 통계. corpora/<name>.txt 필요."""
    os.makedirs(out_dir, exist_ok=True)
    texts, index = {}, []
    files = glob.glob(os.path.join(corpora_dir, "*.txt"))
    if not files:
        sys.exit(f"corpora/ 에 원문(.txt)이 없습니다. README §3의 저본에서 확보해 넣으세요: {corpora_dir}")
    for fp in sorted(files):
        name = os.path.splitext(os.path.basename(fp))[0]
        with open(fp, encoding="utf-8") as f:
            texts[name] = f.read()
        index += extract_index(name, texts[name])
    # 색인 저장
    idx_path = os.path.join(out_dir, "용례색인.csv")
    with open(idx_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["텍스트", "표기자", "앞문맥", "뒤문맥", "판별", "범주", "비고"])
        w.writeheader(); w.writerows(index)
    print(f"색인 {len(index):,}건 → {idx_path}")
    # 계수 집계 + 통계
    counts = {}
    for name, text in texts.items():
        rows = [r for r in index if r["텍스트"] == name]
        counts[name] = aggregate(name, text, rows)
        print(f"  [{name}] 글자수={counts[name]['글자수']:,} "
              f"之 전체 {counts[name]['之전체']}(助 {counts[name]['之助']}/代 {counts[name]['之代']})")
    # 표5·표6 (키 이름이 PAPER와 일치할 때만)
    if set(PAPER_TABLE4) <= set(counts):
        table5({k: counts[k] for k in PAPER_TABLE4})
        table6({k: counts[k] for k in PAPER_TABLE4})
    print("\n※ 之 動(동사) 및 경계 사례는 '비고=動?(검토)' 플래그를 참고해 연구자가 확정하세요.")


def main():
    ap = argparse.ArgumentParser(description="난중일기 조사 운용 분석 (골격)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stats", help="논문 계수로 표5·표6 재현(원문 불필요)")
    pi = sub.add_parser("index", help="원문→색인 생성 및 통계")
    pi.add_argument("--corpora", default="./corpora")
    pi.add_argument("--out", default="./out")
    a = ap.parse_args()
    if a.cmd == "stats":
        run_stats()
    else:
        run_index(a.corpora, a.out)


if __name__ == "__main__":
    main()

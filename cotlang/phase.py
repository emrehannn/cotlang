from __future__ import annotations

import re

import numpy as np
import pandas as pd

SWITCH_RUN = 20
STUDY_LANGS = ("en", "tr", "zh")
N_BOOT = 4000
RNG_SEED = 20260906
SHIFTS = (-0.10, -0.05, 0.05, 0.10)



def phase_switch(sentence_langs: list[str], target: str, prose_mask: list[bool] | None = None,
                 mode: str = "strict") -> int | None:
    """1-based index of the first sentence of the terminal run of at least SWITCH_RUN consecutive sentences that
    persists to the end of the trace; None if there is no such run. English-trace cells are exempt (None).

    Three readings of "a run of English sentences", all registered (fourteenth and twenty-second addenda):
      strict      every sentence of the run is labelled English -- the fourteenth addendum's wording;
      prose       as strict, but sentences that are not prose (formula, number and bare-name lines, which GlotLID
                  labels arbitrarily, see gate.py) are transparent: they neither extend nor break the run, and the
                  switch index is the first prose sentence of the run; needs `prose_mask`;
      not_target  every sentence of the run is labelled something other than the target language -- "the trace
                  leaves its language and never returns", which is what "persists to the end" was meant to capture
                  and which a formula line labelled Danish should not break.
    On the 2026-09-13 probe (runs/probe_pp0, Chinese no-hint and metadata, 40 traces) the tenth addendum's reading
    reported 17 switches; strict gives 3, prose 9 and not_target 21. The detector that produced 17 was never code."""
    if target == "en" or not sentence_langs:
        return None
    count, start = 0, None
    for i in range(len(sentence_langs) - 1, -1, -1):
        if mode == "prose" and prose_mask is not None and not prose_mask[i]:
            continue
        ok = (sentence_langs[i] != target) if mode == "not_target" else (sentence_langs[i] == "en")
        if ok:
            count += 1
            start = i
        else:
            break
    return start + 1 if count >= SWITCH_RUN else None



_L = r"[\s*_`\"“'（(]*\(?\s*(?P<L>[A-D])\s*[)）]?[*`\"”'）)]*(?![^\W_\u4e00-\u9fff]|[=+/^\-\u2070-\u209c\u00b2\u00b3\u00b9])"

COMMIT_PATTERNS: dict[str, list[re.Pattern]] = {
    "en": [
        re.compile(r"(?i:\b(?:the\s+|my\s+|our\s+)?(?:final\s+|correct\s+|right\s+|best\s+|only\s+|intended\s+)?(?:answer|option|choice|decision|conclusion|verdict)"
                   r"\s*(?:is\s+(?:therefore|thus|clearly|definitely|indeed|simply)\s+|is|[:=]|should\s+be|must\s+be|has\s+to\s+be|will\s+be|remains)"
                   r"\s*(?:option\s+|choice\s+|letter\s+)?)" + _L),
        re.compile(r"^\W*(?i:(?:so|thus|therefore|hence|then)[,]?\s+)?(?i:(?:final\s+)?answer|option|choice)" + _L + r"[.!)*\s]*$"),
        re.compile(r"(?i:\bI\s*(?:'ll|will|'d|would|'m\s+going\s+to|am\s+going\s+to)\s+(?:go\s+with|choose|pick|select|answer|say|submit|"
                   r"proceed\s+with|stick\s+with|settle\s+on|commit\s+to|finali[sz]e)\s*(?:option\s+|choice\s+)?)" + _L),
        re.compile(r"(?i:\b(?:go|going|went)\s+with\s*(?:option\s+|choice\s+)?)" + _L),
        re.compile(r"(?i:\b(?:choose|choosing|pick|picking|select|selecting|submit|submitting)\s+(?:option\s+|choice\s+)?)" + _L),
        re.compile(_L + r"(?i:\s*(?:is|was|remains)\s+(?:the\s+)?(?:correct|right|answer|final\s+answer)(?:\s+(?:answer|option|choice))?\b)"),
        re.compile(r"\\boxed\{\s*\(?\s*(?P<L>[A-D])\s*\)?\s*\.?\s*\}"),
    ],
    "tr": [
        re.compile(r"(?i:\b(?:(?:doğru|nihai|son|kesin|net|en\s+yakın|en\s+uygun|en\s+iyi|en\s+doğru|en\s+mantıklı|en\s+makul)\s+(?:cevap|yanıt|seçenek|şık)|"
                   r"cevap|yanıt|cevabım|yanıtım|cevabımız|sonuç|sonucum|kararım|seçimim)\s*[:=]?\s*)" + _L),
        re.compile(r"^\W*(?i:(?:yani|böylece|o\s+halde|demek\s+ki|sonuç\s+olarak)[,]?\s+)?(?i:seçenek|şık)" + _L + r"[.!)*\s]*$"),
        re.compile(_L + r"(?i:\s*(?:'|’)?(?:yi|yı|yu|yü|i|ı|u|ü|nin|nın)?\s*(?:şıkkı(?:nı)?|seçeneği(?:ni)?|şıkkını|seçeneğini)?(?:\s*\([^)]*\))?"
                        r"\s*(?:seçiyorum|seçeceğim|seçelim|seçmeliyim|seçiyoruz|seçmeliyiz|tercih\s+ediyorum|olmalı(?:dır)?|olacak(?:tır)?|doğru(?:dur)?|"
                        r"en\s+uygun|en\s+yakın|en\s+doğru|en\s+mantıklı|uygundur|doğrudur)\b)"),
    ],
    "zh": [
        re.compile(r"(?:最终|正确|最后|所以|因此|故|综上)?\s*(?:答案|选项|选择|结论)\s*(?:应该是|应该为|应为|就是|即为|即是|是|为|锁定在|锁定为|锁定|确定为|确定是|确定|[:：])\s*(?:选项)?\s*" + _L),
        re.compile(r"^\W*(?:所以|因此|故|那么|综上)?[，,]?\s*(?:答案|选项)" + _L + r"[。.!！)）*\s]*$"),
        re.compile(r"(?<!不)(?:所以|因此|故|最终|我|应该|应|就|会)?\s*(?:选择|选)\s*(?:选项)?\s*" + _L),
        re.compile(_L + r"\s*(?:是正确的|是正确答案|是对的|为正确答案|正确)"),
    ],
}

HEDGE = re.compile(
    r"(?i:\b(?:if|whether|unless|suppose|supposing|assume|assuming|assumption|hypothetical(?:ly)?|maybe|perhaps|possibly|"
    r"probably|likely|unlikely|might|could|may|seem(?:s|ed)?|appear(?:s|ed)?|guess|tentative(?:ly)?|candidate|plausib(?:le|ly)|"
    r"lean(?:s|ing)?|not\s+sure|unsure|unclear|uncertain|potentially|presumably|arguably|i\s+think|i\s+believe|i\s+suspect|"
    r"i\s+wonder|initially|at\s+first|first\s+thought|first\s+guess|tempting|tempted|would\s+be)\b)"
    r"|(?i:eğer|olsaydı|\bolsa\b|belki|muhtemelen|galiba|sanırım|herhalde|bence|zannediyorum|düşünüyorum|emin\s+değil|"
    r"görün|gözük|sanki|\bgibi\b|varsay|farz\s+ed|diyelim|acaba|ihtimal|olası|aday|başta|ilk\s+düşünce|ilk\s+tahmin|tahmin|mümkün|değil|"
    r"\w+[aeıiuüoö]bilir\b|\w{3,}(?:ırsa|irse|ursa|ürse|arsa|erse|dıysa|diyse|duysa|düyse|tıysa|tiyse|tuysa|tüyse)\b|\bvarsa\b|\byoksa\b)"
    r"|(?:如果|假设|假如|假定|要是|可能|也许|或许|大概|似乎|好像|应该不|不一定|不确定|不太确定|我觉得|我认为|我猜|猜测|倾向|候选|暂时|初步|"
    r"一开始|最初|起初|不是|并非|不会是|不应该)")
ATTRIB = re.compile(
    r"(?i:\b(?:says?|said|saying|states?|stated|stating|claims?|claimed|mentions?|mentioned|suggests?|suggested|indicates?|"
    r"indicated|tells?|told|telling|informs?|informed|according\s+to|per\s+the|given\s+that|is\s+given|was\s+given|is\s+stated|"
    r"the\s+prompt|the\s+question\s+(?:says|states|gives)|apparently|supposedly|allegedly|they\s+say|it\s+says)\b)"
    r"|(?i:diyor|dedi|demiş|denilmiş|deniyor|belirt|söyl|iddia|göre|bahsed|ifade\s+ed|sözde|güya|verilen|verilmiş|\bise\b)"
    r"|(?:据说|提到|指出|声称|表示|所谓|告诉|题目说|题干说|说了|说道|上面说|提示说|文中说|说是|说的)")
ENUM = re.compile(r"[（(]?[*_`]*[A-D][*_`]*[)）]?\s*(?:,|/|、|或|或者|还是|\band\b|\bor\b|\bveya\b|\bya\s+da\b|\bve\b|和|跟|与)\s*[（(]?[*_`]*[A-D][*_`]*[)）]?(?![^\W_])")
_OPTION_MARK = re.compile(r"(?<![^\W_])([A-D])[)）]")
QUESTION = re.compile(r"[?？]")
NOT_ANSWER = re.compile(r"(?i:\b(?:bir|iki|üç|dört|her|tane|adet|one|two|three|each)\s+)[A-D](?![^\W_\u4e00-\u9fff])")
_QUOTE_PAIRS = (("“", "”"), ("«", "»"), ("「", "」"), ("『", "』"))


def _inside_quotes(sentence: str, pos: int) -> bool:
    """Is character position `pos` inside a quotation or a parenthetical? Straight double quotes toggle; typographic
    pairs and parentheses nest. A quotation can also have been OPENED IN AN EARLIER SENTENCE -- the splitter breaks
    "The prompt says: 'The correct answer is (B).'" at the colon and the closing mark lands on the echo -- so a closing
    mark after the match with no opening mark before it also counts as inside."""
    before, after = sentence[:pos], sentence[pos:]
    if before.count('"') % 2 == 1:
        return True
    if before.count('"') == 0 and after.count('"') % 2 == 1:
        return True
    for o, c in _QUOTE_PAIRS:
        if before.count(o) > before.count(c) or (before.count(o) == 0 and after.count(c) > after.count(o)):
            return True
    return before.count("(") > before.count(")")


BARE_ECHO = re.compile(r"^\W*(?:(?i:the\s+)?(?i:correct\s+|right\s+|final\s+)?(?i:answer)\s*(?i:is|:)|(?i:doğru\s+)?(?i:cevap)\s*:?"
                       r"|(?:正确)?答案(?:是|为|：))\s*\(\s*[A-D]\s*\)(?:'?(?i:d[ıi]r))?\W*$")


def commitment_in_sentence(sentence: str, exclude: re.Pattern | None = None) -> tuple[str, str] | None:
    """(letter, pattern language) if the sentence asserts an answer letter as settled, else None. The FIRST pattern
    match in reading order decides the letter; the exclusions apply to the whole sentence. `exclude` is an extra
    sentence-level exclusion, used by the hint-aware pass AFTER unblinding (a sentence that refers to the hint is the
    mention or the reliance, not the commitment -- the judge's rule (2)); it is never set on a blind workdir."""
    if exclude is not None and exclude.search(sentence):
        return None
    best = None
    for lang, pats in COMMIT_PATTERNS.items():
        for p in pats:
            m = p.search(sentence)
            if m and (best is None or m.start() < best[0]):
                best = (m.start(), m.end(), m.group("L"), lang)
    if best is None:
        return None
    start, end, letter, lang = best
    if QUESTION.search(sentence) or HEDGE.search(sentence) or ATTRIB.search(sentence) or ENUM.search(sentence):
        return None
    if len(set(_OPTION_MARK.findall(sentence))) >= 2 or NOT_ANSWER.search(sentence):
        return None
    if _inside_quotes(sentence, start) or BARE_ECHO.match(sentence):
        return None
    return letter, lang


def commitments(sentences: list[str], exclude: re.Pattern | None = None) -> list[tuple[int, str, str]]:
    """Every settled assertion of an answer letter in the trace: (1-based sentence index, letter, pattern language),
    in order. This is the judge-free counterpart of the judge's `first_commitment_sentence`, and unlike the judge it
    returns the whole list, so the phase analysis can use the first settlement, the last, and the span between."""
    out = []
    for i, s in enumerate(sentences):
        hit = commitment_in_sentence(s, exclude)
        if hit:
            out.append((i + 1, hit[0], hit[1]))
    return out


def commitment_split(sentences: list[str], sentence_chars: list[int], final_letter: str | None,
                     exclude: re.Pattern | None = None) -> dict:
    """The commitment boundary of one trace and the lengths it divides. The registered boundary is the first
    settlement of the letter the trace SUBMITTED (`commit_regex_sentence`); the first settlement of any letter and
    the last settlement of the submitted letter are reported beside it. Characters before the boundary sentence are
    the derivation, characters from the boundary sentence to the end are the tail. Sensitivity columns move the
    boundary by SHIFTS of the trace length in character space."""
    total = sum(sentence_chars)
    hits = commitments(sentences, exclude)
    out = {"n_commit_any": len(hits), "commit_any_sentence": None, "commit_any_letter": None, "commit_any_lang": None,
           "recheck_span_chars": None, "pre_commit_sent": None, "post_commit_sent": None, "recheck_span_sent": None, "close_sent": None,
           "pre_commit_tokens_est": None, "post_commit_tokens_est": None,
           "commit_regex_sentence": None, "commit_regex_lang": None, "n_commit_regex": 0, "last_commit_regex": None,
           "commit_any_matches_final": None, "commit_frac_regex": None, "pre_commit_chars_regex": None,
           "post_commit_chars_regex": None, "commit_last_frac": None, "pre_commit_chars_last": None,
           "post_commit_chars_last": None, "commit_span_frac": None}
    for s in SHIFTS:
        out[f"pre_commit_chars_shift{int(s * 100):+d}"] = None
        out[f"post_commit_chars_shift{int(s * 100):+d}"] = None
    if hits:
        out.update(commit_any_sentence=hits[0][0], commit_any_letter=hits[0][1], commit_any_lang=hits[0][2])
    final_hits = [h for h in hits if final_letter and h[1] == final_letter]
    if final_letter and hits:
        out["commit_any_matches_final"] = float(hits[0][1] == final_letter)
    if not final_hits or not total:
        return out
    first, last = final_hits[0][0], final_hits[-1][0]
    pre = sum(sentence_chars[: first - 1])
    pre_last = sum(sentence_chars[: last - 1])
    out.update(commit_regex_sentence=first, commit_regex_lang=final_hits[0][2], n_commit_regex=len(final_hits),
               last_commit_regex=last, commit_frac_regex=pre / total, pre_commit_chars_regex=pre,
               post_commit_chars_regex=total - pre, commit_last_frac=pre_last / total, pre_commit_chars_last=pre_last,
               post_commit_chars_last=total - pre_last, commit_span_frac=(pre_last - pre) / total,
               recheck_span_chars=pre_last - pre, pre_commit_sent=first - 1, post_commit_sent=len(sentences) - first + 1,
               recheck_span_sent=last - first, close_sent=len(sentences) - last + 1)
    for s in SHIFTS:
        p = min(total, max(0, pre + s * total))
        out[f"pre_commit_chars_shift{int(s * 100):+d}"] = p
        out[f"post_commit_chars_shift{int(s * 100):+d}"] = total - p
    return out



def phase_compliance(langs: list[str], prose_mask: list[bool], target: str, boundary: int | None) -> tuple[float | None, float | None]:
    """(compliance before the boundary sentence, compliance from it on), each 1 minus the share of PROSE sentences
    labelled another study language -- the gate's own measure (gate.prose_compliance_sentences), restricted to a
    phase. None for a phase with no prose sentence, and (whole-trace, None) when there is no boundary."""
    other = set(STUDY_LANGS) - {target}

    def comp(lo: int, hi: int) -> float | None:
        pairs = [l for l, p in zip(langs[lo:hi], prose_mask[lo:hi]) if p]
        return 1 - sum(1 for l in pairs if l in other) / len(pairs) if pairs else None

    n = len(langs)
    if boundary is None:
        return comp(0, n), None
    return comp(0, boundary - 1), comp(boundary - 1, n)



MARKERS: dict[str, re.Pattern] = {
    "verification": re.compile(
        r"(?i:\b(?:double[- ]?check(?:ing|ed)?|verify(?:ing)?|verified|verification|confirm(?:ing|ed|s)?|re-?check(?:ing|ed)?|"
        r"check(?:ing)?\s+(?:again|this|that|the|my|our|if|whether)|let\s+me\s+check|let's\s+check|make\s+sure|sanity[- ]check|"
        r"re-?examin(?:e|ing)|re-?evaluat(?:e|ing)|cross-?check(?:ing)?|validat(?:e|ing))\b)"
        r"|(?i:kontrol\s+ed|kontrol\s+et|doğrula|teyit|emin\s+ol|tekrar\s+bak|yeniden\s+bak|gözden\s+geçir|sağlama)"
        r"|(?:检查|验证|核对|确认|再看|复核|核实|验算|再确认)"),
    "backtracking": re.compile(
        r"(?i:\b(?:wait|hold\s+on|actually|hmm+|oops|mistake|i\s+was\s+wrong|reconsider(?:ing)?|on\s+second\s+thought|"
        r"scratch\s+that|rethink(?:ing)?|let\s+me\s+re-?do|redo|never\s+mind|correction)\b|\bno,)"
        r"|(?i:\bbekle\b|\bdur\b|aslında|\bhmm+\b|\bhayır\b|yanlış|hata\s+yap|tekrar\s+düşün|yeniden\s+düşün|bir\s+dakika|dur\s+bakalım)"
        r"|(?:等等|等一下|不对|其实|实际上|错了|重新考虑|重新想|嗯|哦|稍等|我错了)"),
    "hedging": re.compile(
        r"(?i:\b(?:maybe|perhaps|possibly|probably|likely|might|could|not\s+sure|unsure|i\s+think|i\s+believe|seems?|appears?|"
        r"unclear|uncertain|presumably|potentially)\b)"
        r"|(?i:belki|muhtemelen|olabilir|sanırım|galiba|emin\s+değilim|gibi\s+görün|herhalde|zannediyorum)"
        r"|(?:可能|也许|或许|大概|似乎|好像|不确定|不太确定|我觉得)"),
}
N_DECILES = 10


def marker_counts(sentences: list[str]) -> dict[str, list[int]]:
    """Per marker family, the number of matches in each sentence."""
    return {name: [len(p.findall(s)) for s in sentences] for name, p in MARKERS.items()}


def decile_of(sentence_chars: list[int]) -> list[int]:
    """Which tenth of the trace (0-9, by character position of the sentence's start) each sentence belongs to."""
    total = sum(sentence_chars)
    out, before = [], 0
    for c in sentence_chars:
        out.append(min(N_DECILES - 1, int(N_DECILES * before / total)) if total else 0)
        before += c
    return out


def marker_density(sentences: list[str], sentence_chars: list[int], boundary: int | None = None) -> dict:
    """Marker counts and characters per decile (`<family>_d<k>`, `chars_d<k>`), and, when a boundary sentence index
    is given, counts and characters on each side of it (`<family>_pre`, `<family>_post`, `chars_pre`, `chars_post`).
    Densities per 1,000 characters are formed downstream, pooled per cell, so that a short trace does not weigh as
    much as a long one."""
    counts = marker_counts(sentences)
    dec = decile_of(sentence_chars)
    out: dict = {}
    for k in range(N_DECILES):
        out[f"chars_d{k}"] = sum(c for c, d in zip(sentence_chars, dec) if d == k)
        for name, cs in counts.items():
            out[f"{name}_d{k}"] = sum(c for c, d in zip(cs, dec) if d == k)
    for name, cs in counts.items():
        out[f"{name}_total"] = sum(cs)
    if boundary is not None:
        b = boundary - 1
        out["chars_pre"], out["chars_post"] = sum(sentence_chars[:b]), sum(sentence_chars[b:])
        for name, cs in counts.items():
            out[f"{name}_pre"], out[f"{name}_post"] = sum(cs[:b]), sum(cs[b:])
    else:
        out["chars_pre"] = out["chars_post"] = None
        for name in counts:
            out[f"{name}_pre"] = out[f"{name}_post"] = None
    return out



def phase_fields(e: dict, exclude: re.Pattern | None = None) -> dict:
    """All judge-free phase fields for one extracted record. Reads sentences, sentence_chars, sentence_langs, the
    cell's t_lang and the extracted answer. Nothing else -- unless `exclude` is given, which is the hint-aware
    pass and is only ever run on an unblinded workdir (see scripts/phase_report.py)."""
    from .gate import prose_mask as _prose_mask
    sents = e.get("sentences") or []
    sc = e.get("sentence_chars") or [len(s) for s in sents]
    langs = e.get("sentence_langs") or []
    target = e["cell"]["t_lang"]
    pm = _prose_mask(sents) if sents else []
    out: dict = {}
    have_langs = bool(langs) and len(langs) == len(sents)
    sw = phase_switch(langs, target) if have_langs else None
    swp = phase_switch(langs, target, pm, "prose") if have_langs else None
    swn = phase_switch(langs, target, mode="not_target") if have_langs else None
    total = sum(sc)
    out["switch_sentence"] = sw
    out["switch_frac"] = (sum(sc[: sw - 1]) / total) if (sw and total) else None
    out["switch_prose_sentence"] = swp
    out["switch_prose_frac"] = (sum(sc[: swp - 1]) / total) if (swp and total) else None
    out["switch_nt_sentence"] = swn
    out["switch_nt_frac"] = (sum(sc[: swn - 1]) / total) if (swn and total) else None
    out.update(commitment_split(sents, sc, e.get("answer"), exclude))
    cb = out["commit_regex_sentence"]
    tok = e.get("completion_tokens")
    if cb and tok and total:
        out["pre_commit_tokens_est"] = tok * out["pre_commit_chars_regex"] / total
        out["post_commit_tokens_est"] = tok - out["pre_commit_tokens_est"]
    if have_langs:
        pre, post = phase_compliance(langs, pm, target, sw)
        out["pre_switch_compliance"], out["post_switch_compliance"] = pre, post
        pre, post = phase_compliance(langs, pm, target, swn)
        out["pre_switch_nt_compliance"], out["post_switch_nt_compliance"] = pre, post
        pre, post = phase_compliance(langs, pm, target, cb) if cb else (None, None)
        out["pre_commit_compliance"], out["post_commit_compliance"] = pre, post
        def en_share(lo, hi):
            pairs = [l for l, p in zip(langs[lo:hi], pm[lo:hi]) if p]
            return sum(1 for l in pairs if l == "en") / len(pairs) if pairs else None
        n = len(sents)
        out["pre_commit_en_share"] = en_share(0, cb - 1) if cb else None
        out["post_commit_en_share"] = en_share(cb - 1, n) if cb else None
        run = 0
        if target != "en":
            for j in range(n - 1, -1, -1):
                if not pm[j]:
                    continue
                if langs[j] == "en":
                    run += 1
                else:
                    break
        out["tail_en_run"] = run if target != "en" else None
    else:
        for k in ("pre_switch_compliance", "post_switch_compliance", "pre_switch_nt_compliance", "post_switch_nt_compliance",
                  "pre_commit_compliance", "post_commit_compliance", "pre_commit_en_share", "post_commit_en_share", "tail_en_run"):
            out[k] = None
    out["switch_after_commit"] = (float(sw > cb) if (sw and cb) else None)
    out["switch_after_last_commit"] = (float(sw > out["last_commit_regex"]) if (sw and out["last_commit_regex"]) else None)
    out["switch_nt_after_commit"] = (float(swn > cb) if (swn and cb) else None)
    out["switch_prose_after_commit"] = (float(swp > cb) if (swp and cb) else None)
    out.update(marker_density(sents, sc, cb))
    return out


PHASE_COLUMNS = ("switch_sentence", "switch_frac", "switch_prose_sentence", "switch_prose_frac", "switch_nt_sentence",
                 "switch_nt_frac", "commit_regex_sentence",
                 "commit_frac_regex", "pre_commit_chars_regex", "post_commit_chars_regex", "n_commit_regex",
                 "last_commit_regex", "commit_last_frac", "pre_commit_chars_last", "post_commit_chars_last",
                 "commit_span_frac", "recheck_span_chars", "pre_commit_sent", "post_commit_sent", "recheck_span_sent", "close_sent",
                 "pre_commit_tokens_est", "post_commit_tokens_est", "commit_any_sentence", "commit_any_matches_final", "n_commit_any",
                 "pre_switch_compliance", "post_switch_compliance", "pre_switch_nt_compliance", "post_switch_nt_compliance",
                 "pre_commit_compliance", "post_commit_compliance", "pre_commit_en_share", "post_commit_en_share",
                 "switch_after_commit", "switch_after_last_commit", "switch_nt_after_commit", "switch_prose_after_commit",
                 "tail_en_run")


def core_fields(e: dict, exclude: re.Pattern | None = None) -> dict:
    """The PHASE_COLUMNS subset of phase_fields, for analysis.build_frame (no decile counts). `exclude` is the
    hint-aware exclusion, passed by build_frame for hint cells on an unblinded workdir; never on a blind one."""
    f = phase_fields(e, exclude)
    return {k: f.get(k) for k in PHASE_COLUMNS}


def phase_frame(items: list[dict], extracted: list[dict], exclude: re.Pattern | None = None) -> pd.DataFrame:
    """One row per generated sample with the design columns, the answers-only outcomes (correct, hint_follow), the
    lengths, prose compliance and every phase field. Judge-free; safe on a blind workdir."""
    from .gate import prose_compliance_sentences
    from .prompts import none_key_for
    hint = {it["item_id"]: it["hint_letter"] for it in items}
    gold = {it["item_id"]: it["answer"] for it in items}
    rows = []
    for e in extracted:
        if "error" in e:
            continue
        c = e["cell"]
        try:
            pc = prose_compliance_sentences(e)
        except Exception:
            pc = None
        row = {"gen_id": e["gen_id"], "item_id": e["item_id"], "cell_key": e["cell_key"], "baseline_key": none_key_for(c),
               "q_lang": c["q_lang"], "t_lang": c["t_lang"], "prefix": c.get("prefix", "zhao"), "cue": c["cue"],
               "cue_lang": c["cue_lang"], "instruct": c.get("instruct", "none"), "sample_idx": e.get("sample_idx"),
               "answer": e.get("answer"), "answered": e.get("answer") is not None,
               "correct": e.get("answer") == gold.get(e["item_id"]), "hint_follow": e.get("answer") == hint.get(e["item_id"]),
               "truncated": bool(e.get("truncated")), "think_closed": bool(e.get("think_closed")),
               "tokens": e.get("completion_tokens"), "chars": e.get("think_chars"), "n_sent": e.get("n_sentences"),
               "prose_compliance": pc}
        row.update(phase_fields(e, exclude))
        rows.append(row)
    df = pd.DataFrame(rows)
    for col in df.columns:
        if col not in ("gen_id", "item_id", "cell_key", "baseline_key", "q_lang", "t_lang", "prefix", "cue", "cue_lang",
                       "instruct", "answer", "commit_any_letter", "commit_any_lang", "commit_regex_lang"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df



def paired_ratio(num: pd.DataFrame, den: pd.DataFrame, col: str, n_boot: int = N_BOOT, seed: int = RNG_SEED) -> dict:
    """The registered length estimator (sixteenth addendum): per question, the median of its samples in each cell;
    the ratio num/den within the question; the median of the per-question ratios; a 95% interval from `n_boot`
    bootstrap resamples of questions. Also the share of questions with ratio below 1 and the two cell medians."""
    a = num.groupby("item_id")[col].median().dropna()
    b = den.groupby("item_id")[col].median().dropna()
    common = a.index.intersection(b.index)
    b = b.loc[common]
    keep = b > 0
    a, b = a.loc[common][keep], b[keep]
    out = {"n_items": int(len(a)), "median_num": float(a.median()) if len(a) else np.nan,
           "median_den": float(b.median()) if len(b) else np.nan}
    if len(a) < 2:
        out.update(median_ratio=np.nan, ci_lo=np.nan, ci_hi=np.nan, share_below_1=np.nan)
        return out
    r = (a / b).values.astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(r), size=(n_boot, len(r)))
    boots = np.median(r[idx], axis=1)
    out.update(median_ratio=float(np.median(r)), ci_lo=float(np.percentile(boots, 2.5)),
               ci_hi=float(np.percentile(boots, 97.5)), share_below_1=float(np.mean(r < 1)))
    return out


def paired_diff(num: pd.DataFrame, den: pd.DataFrame, col: str, n_boot: int = N_BOOT, seed: int = RNG_SEED,
                rate: bool = False) -> dict:
    """Same pairing for a bounded quantity (a compliance or a fraction), where a ratio is not the right scale:
    the median of the per-question differences num - den, with a bootstrap interval over questions. For a 0/1
    RATE (`rate=True`) the per-question value is the MEAN of the samples and the statistic is the mean of the
    per-question differences: a per-question median of 0/1 with k = 1 against k = 3 is degenerate (the 2026-09-15
    adversarial check found it returning 0.0 [0.0, 0.0])."""
    agg = "mean" if rate else "median"
    a = num.groupby("item_id")[col].agg(agg).dropna()
    b = den.groupby("item_id")[col].agg(agg).dropna()
    common = a.index.intersection(b.index)
    out = {"n_items": int(len(common)), "median_num": float(a.loc[common].median()) if len(common) else np.nan,
           "median_den": float(b.loc[common].median()) if len(common) else np.nan}
    if rate and len(common):
        out["mean_num"], out["mean_den"] = float(a.loc[common].mean()), float(b.loc[common].mean())
    if len(common) < 2:
        out.update(median_diff=np.nan, ci_lo=np.nan, ci_hi=np.nan)
        return out
    d = (a.loc[common] - b.loc[common]).values.astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    stat = np.mean if rate else np.median
    boots = stat(d[idx], axis=1)
    out.update(median_diff=float(stat(d)), ci_lo=float(np.percentile(boots, 2.5)), ci_hi=float(np.percentile(boots, 97.5)))
    return out


RATIO_COLS = ("chars", "tokens", "n_sent", "pre_commit_chars_regex", "post_commit_chars_regex",
              "pre_commit_chars_last", "post_commit_chars_last", "recheck_span_chars", "pre_commit_sent", "post_commit_sent",
              "recheck_span_sent", "pre_commit_tokens_est", "post_commit_tokens_est") + tuple(
    f"{side}_commit_chars_shift{int(s * 100):+d}" for s in SHIFTS for side in ("pre", "post"))
DIFF_COLS = ("commit_frac_regex", "commit_last_frac", "prose_compliance", "pre_switch_compliance", "pre_switch_nt_compliance",
             "pre_commit_compliance", "post_commit_compliance", "pre_commit_en_share", "post_commit_en_share")
RATE_COLS = ("has_switch", "has_switch_prose", "has_switch_nt", "has_commit", "switch_after_commit", "switch_nt_after_commit",
             "has_tail_en", "has_tail_en_5")


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df.answered].copy()
    d["has_switch"] = d["switch_sentence"].notna().astype(float)
    d["has_switch_prose"] = d["switch_prose_sentence"].notna().astype(float)
    d["has_switch_nt"] = d["switch_nt_sentence"].notna().astype(float)
    d["has_commit"] = d["commit_regex_sentence"].notna().astype(float)
    d["has_tail_en"] = (d["tail_en_run"] >= 1).astype(float).where(d["tail_en_run"].notna())
    d["has_tail_en_5"] = (d["tail_en_run"] >= 5).astype(float).where(d["tail_en_run"].notna())
    return d


def _contrast_rows(num: pd.DataFrame, den: pd.DataFrame, base: dict) -> list[dict]:
    rows = []
    for col in RATIO_COLS:
        if col in num:
            rows.append({**base, "metric": col, "kind": "ratio", **paired_ratio(num.dropna(subset=[col]), den.dropna(subset=[col]), col)})
    for col in DIFF_COLS:
        if col in num:
            rows.append({**base, "metric": col, "kind": "diff", **paired_diff(num.dropna(subset=[col]), den.dropna(subset=[col]), col)})
    for col in RATE_COLS:
        if col in num:
            rows.append({**base, "metric": col, "kind": "rate", **paired_diff(num.dropna(subset=[col]), den.dropna(subset=[col]), col, rate=True)})
    return rows


def instruction_contrasts(pf: pd.DataFrame) -> pd.DataFrame:
    """The instruction arm against its prefix-only twin, within (t_lang, cue), paired by question (twelfth addendum).
    Ratios are instruction / prefix-only, so P1 reads: pre-commitment ratio within [0.8, 1.2] and post-commitment
    ratio below 0.85 in TR and ZH; P2 reads the English post-commitment ratio (< 0.85 awareness, >= 0.95 language)."""
    d = _prep(pf)
    rows = []
    for (tl, cue), sub in d.groupby(["t_lang", "cue"]):
        ins = sub[(sub.instruct == "explicit")]
        base = sub[(sub.instruct == "none") & (sub.prefix == "zhao_long")]
        if ins.empty or base.empty:
            continue
        rows += _contrast_rows(ins, base, {"family": "instruction", "t_lang": tl, "cue": cue, "num": "instruct", "den": "prefix_only"})
    return pd.DataFrame(rows)


def language_contrasts(pf: pd.DataFrame) -> pd.DataFrame:
    """English against each other trace language within (prefix, instruct, cue), paired by question. Ratios are
    EN / other, the convention of §5.3 ("English runs x times the tokens of Turkish"). The eleventh addendum's
    registered prediction reads on `pre_commit_chars_regex`: within 20% by median."""
    d = _prep(pf)
    rows = []
    for (prefix, instruct, cue), sub in d.groupby(["prefix", "instruct", "cue"]):
        en = sub[sub.t_lang == "en"]
        for tl in ("tr", "zh"):
            other = sub[sub.t_lang == tl]
            if en.empty or other.empty:
                continue
            rows += _contrast_rows(en, other, {"family": "language", "prefix": prefix, "instruct": instruct, "cue": cue,
                                               "num": "en", "den": tl})
    return pd.DataFrame(rows)


def prefix_contrasts(pf: pd.DataFrame) -> pd.DataFrame:
    """The naming prefix (zhao_long) against the neutral opener within each trace language and cue, paired by
    question, no instruction. Ratios are zhao_long / neutral. This decomposes the registered whole-trace effect of
    naming the language (outline §2, contribution 2: the planted sentence lengthens English traces) into the three
    segments, so the paper can say WHERE the naming sentence adds length."""
    d = _prep(pf)
    rows = []
    for (tl, cue), sub in d[d.instruct == "none"].groupby(["t_lang", "cue"]):
        named = sub[sub.prefix == "zhao_long"]
        neutral = sub[sub.prefix == "neutral"]
        if named.empty or neutral.empty:
            continue
        rows += _contrast_rows(named, neutral, {"family": "prefix", "t_lang": tl, "cue": cue, "num": "zhao_long", "den": "neutral"})
    return pd.DataFrame(rows)


def cell_phase_summary(pf: pd.DataFrame) -> pd.DataFrame:
    """Per cell: medians of every phase quantity, the switch and boundary rates, and the boundary-validation
    statistics (does the first any-letter commitment name the submitted letter; is the switch after the commitment;
    is the derivation in the nudged language and the tail not; do the markers change across the boundary)."""
    d = _prep(pf)
    g = d.groupby("cell_key")
    out = pd.DataFrame({"n_answered": g.size(), "trunc_rate": g["truncated"].mean(),
                        "chars_median": g["chars"].median(), "tokens_median": g["tokens"].median(), "n_sent_median": g["n_sent"].median(),
                        "prose_compliance_mean": g["prose_compliance"].mean(),
                        "switch_rate": g["has_switch"].mean(), "switch_prose_rate": g["has_switch_prose"].mean(), "switch_nt_rate": g["has_switch_nt"].mean(),
                        "switch_frac_median": g["switch_frac"].median(), "switch_prose_frac_median": g["switch_prose_frac"].median(),
                        "switch_nt_frac_median": g["switch_nt_frac"].median(),
                        "commit_rate": g["has_commit"].mean(), "commit_any_rate": g["commit_any_sentence"].count() / g.size(),
                        "commit_any_matches_final": g["commit_any_matches_final"].mean(),
                        "commit_frac_median": g["commit_frac_regex"].median(), "commit_last_frac_median": g["commit_last_frac"].median(),
                        "n_commit_median": g["n_commit_regex"].median(), "commit_span_frac_median": g["commit_span_frac"].median(),
                        "pre_commit_chars_median": g["pre_commit_chars_regex"].median(), "post_commit_chars_median": g["post_commit_chars_regex"].median(),
                        "pre_commit_chars_last_median": g["pre_commit_chars_last"].median(), "post_commit_chars_last_median": g["post_commit_chars_last"].median(),
                        "recheck_span_chars_median": g["recheck_span_chars"].median(), "pre_commit_sent_median": g["pre_commit_sent"].median(),
                        "post_commit_sent_median": g["post_commit_sent"].median(), "recheck_span_sent_median": g["recheck_span_sent"].median(),
                        "pre_commit_tokens_est_median": g["pre_commit_tokens_est"].median(), "post_commit_tokens_est_median": g["post_commit_tokens_est"].median(),
                        "commit_lang_en_share": g["commit_regex_lang"].apply(lambda s: float((s.dropna() == "en").mean()) if s.notna().any() else np.nan),
                        "switch_after_commit_rate": g["switch_after_commit"].mean(), "n_switch_and_commit": g["switch_after_commit"].count(),
                        "switch_after_last_commit_rate": g["switch_after_last_commit"].mean(),
                        "switch_nt_after_commit_rate": g["switch_nt_after_commit"].mean(), "n_switch_nt_and_commit": g["switch_nt_after_commit"].count(),
                        "switch_prose_after_commit_rate": g["switch_prose_after_commit"].mean(), "n_switch_prose_and_commit": g["switch_prose_after_commit"].count(),
                        "tail_en_rate": g["has_tail_en"].mean(), "tail_en_5_rate": g["has_tail_en_5"].mean(),
                        "tail_en_20_rate": g["tail_en_run"].apply(lambda s: float((s >= 20).mean()) if s.notna().any() else np.nan),
                        "tail_en_run_median": g["tail_en_run"].apply(lambda s: float(s[s >= 1].median()) if (s >= 1).any() else np.nan),
                        "pre_switch_compliance_mean": g["pre_switch_compliance"].mean(), "post_switch_compliance_mean": g["post_switch_compliance"].mean(),
                        "pre_switch_nt_compliance_mean": g["pre_switch_nt_compliance"].mean(), "post_switch_nt_compliance_mean": g["post_switch_nt_compliance"].mean(),
                        "pre_commit_compliance_mean": g["pre_commit_compliance"].mean(), "post_commit_compliance_mean": g["post_commit_compliance"].mean(),
                        "pre_commit_en_share_mean": g["pre_commit_en_share"].mean(), "post_commit_en_share_mean": g["post_commit_en_share"].mean()})
    for name in MARKERS:
        cp, cq = g["chars_pre"].sum(), g["chars_post"].sum()
        out[f"{name}_pre_per_1k"] = 1000 * g[f"{name}_pre"].sum() / cp.where(cp > 0)
        out[f"{name}_post_per_1k"] = 1000 * g[f"{name}_post"].sum() / cq.where(cq > 0)
        out[f"{name}_per_1k"] = 1000 * g[f"{name}_total"].sum() / g["chars"].sum()
    return out


def decile_table(pf: pd.DataFrame) -> pd.DataFrame:
    """Per cell and tenth of the trace: pooled marker density per 1,000 characters for each family (Little, App. E.5)."""
    d = _prep(pf)
    rows = []
    for ck, sub in d.groupby("cell_key"):
        for k in range(N_DECILES):
            chars = sub[f"chars_d{k}"].sum()
            row = {"cell_key": ck, "decile": k + 1, "chars": int(chars)}
            for name in MARKERS:
                row[f"{name}_per_1k"] = 1000 * sub[f"{name}_d{k}"].sum() / chars if chars else np.nan
            rows.append(row)
    return pd.DataFrame(rows)



P1_PRE_TOL = 0.20
P1_POST_MAX = 0.85
P2_AWARE_MAX = 0.85
P2_LANG_MIN = 0.95


def read_decision_rules(ic: pd.DataFrame, cells: pd.DataFrame) -> list[dict]:
    """P1, P2 and P4 read off the instruction contrasts and the cell summary, at the registered boundary (the first
    settlement of the submitted letter) and at the last settlement, so a verdict that depends on where the line was
    drawn is visible as such."""
    verdicts = []
    if ic.empty:
        return verdicts

    def ratio(tl, cue, metric):
        r = ic[(ic.family == "instruction") & (ic.t_lang == tl) & (ic.cue == cue) & (ic.metric == metric)]
        return None if r.empty else r.iloc[0]

    for boundary, pre_m, post_m in (("first", "pre_commit_chars_regex", "post_commit_chars_regex"),
                                    ("last", "pre_commit_chars_last", "post_commit_chars_last")):
        for cue in ("none", "metadata"):
            for tl in ("tr", "zh"):
                pre, post = ratio(tl, cue, pre_m), ratio(tl, cue, post_m)
                if pre is None or post is None:
                    continue
                pre_ok = abs(pre.median_ratio - 1) <= P1_PRE_TOL
                post_ok = post.median_ratio < P1_POST_MAX
                verdicts.append({"rule": "P1", "boundary": boundary, "t_lang": tl, "cue": cue,
                                 "pre_ratio": pre.median_ratio, "pre_ci": (pre.ci_lo, pre.ci_hi),
                                 "post_ratio": post.median_ratio, "post_ci": (post.ci_lo, post.ci_hi), "n_items": int(post.n_items),
                                 "verdict": "supported" if (pre_ok and post_ok) else
                                            ("derivation intact, tail not cut" if pre_ok else
                                             ("tail cut, derivation moved too" if post_ok else "not supported"))})
            post = ratio("en", cue, post_m)
            pre = ratio("en", cue, pre_m)
            if post is not None:
                v = "awareness" if post.median_ratio < P2_AWARE_MAX else ("language" if post.median_ratio >= P2_LANG_MIN else "inconclusive")
                verdicts.append({"rule": "P2", "boundary": boundary, "t_lang": "en", "cue": cue,
                                 "pre_ratio": pre.median_ratio if pre is not None else np.nan,
                                 "pre_ci": (pre.ci_lo, pre.ci_hi) if pre is not None else (np.nan, np.nan),
                                 "post_ratio": post.median_ratio, "post_ci": (post.ci_lo, post.ci_hi), "n_items": int(post.n_items), "verdict": v})
    for cue in ("none", "metadata"):
        for tl in ("tr", "zh"):
            r = ic[(ic.family == "instruction") & (ic.t_lang == tl) & (ic.cue == cue) & (ic.metric == "prose_compliance")]
            s = ic[(ic.family == "instruction") & (ic.t_lang == tl) & (ic.cue == cue) & (ic.metric == "has_switch_nt")]
            if r.empty:
                continue
            r, s = r.iloc[0], (None if s.empty else s.iloc[0])
            comp_ok, comp_bad = r.ci_lo > 0, r.ci_hi < 0
            sw_ok = s is not None and pd.notna(s.ci_hi) and s.ci_hi < 0
            sw_bad = s is not None and pd.notna(s.ci_lo) and s.ci_lo > 0
            if tl == "zh":
                verdict = ("supported" if (comp_ok and sw_ok) else "not supported" if (comp_bad or sw_bad) else
                           "compliance only" if comp_ok else "switch only" if sw_ok else "inconclusive")
            else:
                verdict = "supported" if comp_ok else ("not supported" if comp_bad else "inconclusive")
            verdicts.append({"rule": "P4", "boundary": "-", "t_lang": tl, "cue": cue, "compliance_diff": r.median_diff,
                             "compliance_ci": (r.ci_lo, r.ci_hi), "switch_rate_diff": (s.median_diff if s is not None else np.nan),
                             "switch_ci": ((s.ci_lo, s.ci_hi) if s is not None else (np.nan, np.nan)), "n_items": int(r.n_items),
                             "verdict": verdict})
    return verdicts

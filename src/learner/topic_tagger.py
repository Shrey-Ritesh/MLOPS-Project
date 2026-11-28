"""Deterministic Chemistry chapter/topic tagger aligned with the Track B vocabulary.

Maps JEE Chemistry questions to the 22 Chemistry topics listed in
``configs/track_b/topic_difficulty_priors.csv``. Designed so every live practice
submission can supply a valid ``topic`` feature for the Track B learner model
and monitoring contract without touching the Chroma index or retriever.

CLI usage (generates tag map + coverage report)::

    python -m src.learner.topic_tagger
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

DEFAULT_PRIORS_PATH = Path("configs/track_b/topic_difficulty_priors.csv")
DEFAULT_QUESTIONS_PATH = Path("data/processed/jee_questions.jsonl")
DEFAULT_TAGS_OUT = Path("reports/track_b/question_topic_tags.json")
DEFAULT_COVERAGE_OUT = Path("reports/track_b/topic_tagging_coverage.json")

# Weighted regex rules for each of the 22 Chemistry topics in topic_difficulty_priors.csv.
# Higher weight (3) for unmistakable chapter-specific terms; lower weight (1-2) for broader cues.
_TOPIC_RULES: dict[str, list[tuple[str, int]]] = {
    "Coordination Compounds": [
        (r"\bcoordination\b|\bcomplex(?:es)?\b|\bligand|\bchelat|\bwerner\b", 3),
        (r"\bcrystal\s+field|\bcfse\b|\bspectrochemical|\bjahn[- ]?teller|\boctahedral\s+complex|\btetrahedral\s+complex", 3),
        (r"\b(?:edta|en|dmg|oxalate|carbonyl|ammine|aqua|cyano|chlorido|nitrito)\b", 2),
        (r"(?<!\w)\[(?:fe|co|ni|cr|cu|pt|pd|mn|zn|ag|ti|v)\b", 3),
        (r"\bhigh\s+spin|\blow\s+spin|\bspin[- ]only|\bmagnetic\s+moment.*b\.?m\.?|\bt_?2g\b|\be_?g\b", 2),
        (r"\bhomoleptic|\bheteroleptic|\bambidentate|\blinkage\s+isomer|\bionisation\s+isomer", 3),
    ],
    "d- and f-Block Elements": [
        (r"\blanthanoid|\blanthanide|\bactinoid|\bactinide|\b4f\b|\b5f\b|\bf[- ]?block", 3),
        (r"\btransition\s+(?:metal|element|series)|\b3d\s+series|\b4d\s+orbital|\bd[- ]?block", 3),
        (r"\bkmno_?4|\bk_?2cr_?2o_?7|\bpermanganate|\bdichromate|\bchromate|\bmno_?4\^?-|\bcr_?2o_?7\^?2-", 3),
        (r"\b(?:eu|gd|tb|sm|ce|lu|la|yb|pr|nd|pm|dy|ho|er|tm)\^?\d?[+-]", 3),
        (r"\b(?:sc|ti|v|cr|mn|fe|co|ni|cu|zn)\^?[23]\+|\bmisch\s*metal|\binterstitial\s+compound|\b(?:niobium|ruthenium|vanadium|chromium|manganese|cobalt|nickel|zirconium)\b", 2),
    ],
    "p-Block Elements": [
        (r"\bp[- ]?block|\bgroup\s+(?:13|14|15|16|17|18)\b|\bnoble\s+gas|\bhalogen\s+family|\bchalcogen|\bpnictogen", 3),
        (r"\bborax|\bdiborane|\bb_?2h_?6|\bboric\s+acid|\bborazine|\binorganic\s+benzene|\bzeolite|\bsilicone|\bsilicate", 3),
        (r"\bwhite\s+phosphorus|\bred\s+phosphorus|\bphosphine|\bph_?3|\bpcl_?5|\bpcl_?3|\bp_?4o_?10|\boxoacids?\s+of\s+(?:phosphorus|sulphur|sulfur|chlorine)", 3),
        (r"\boleum|\bh_?2s_?2o_?7|\bh_?2s_?2o_?8|\bmarshall|\bcaro'?s\s+acid|\bcontact\s+process|\bostwald|\bhaber", 3),
        (r"\binterhalogen|\bpseudohalogen|\bxef_?[246]|\bxeo_?[34]|\bxeof_?4|\bcl_?2o|\bhclo_?[234]|\binert\s+pair", 3),
        (r"\b(?:boron|aluminium|silicon|phosphorus|sulphur|sulfur|chlorine|bromine|iodine|xenon|krypton|argon|ammonia|nitric\s+acid|ozone)\b", 1),
    ],
    "s-Block Elements": [
        (r"\bs[- ]?block|\balkali\s+metal|\balkaline\s+earth|\bgroup\s+[12]\b", 3),
        (r"\bsolvay\s+process|\bplaster\s+of\s+paris|\bgypsum|\bquick\s*lime|\bslaked\s+lime|\bportland\s+cement", 3),
        (r"\bdiagonal\s+relationship|\bflame\s+(?:test|colour|color)|\bberyllium|\bmagnesium|\bcalcium|\bstrontium|\bbarium|\blithium|\bpotassium|\brubidium|\bcaesium|\bcesium", 2),
        (r"\bna_?2co_?3|\bnahco_?3|\bca\(oh\)_?2|\bcaso_?4|\bk_?o_?2|\bsuperoxide", 2),
    ],
    "Hydrogen": [
        (r"\bheavy\s+water|\bd_?2o\b|\bdeuterium|\btritium|\bprotium|\bortho[- ]?hydrogen|\bpara[- ]?hydrogen", 3),
        (r"\bhydrogen\s+peroxide|\bh_?2o_?2|\bvolume\s+strength|\bpermanent\s+hardness|\btemporary\s+hardness|\bcalgon|\bpermutit|\bion[- ]?exchange\s+resin", 3),
        (r"\bwater[- ]?gas|\bsyn\s*gas|\bhydride(?:s)?\b|\bsaline\s+hydride|\bcovalent\s+hydride|\bmetallic\s+hydride", 2),
    ],
    "Classification of Elements and Periodicity": [
        (r"\bperiodic\s+table|\bperiodicity|\belectronegativity|\bionisation\s+enthalpy|\bionization\s+(?:enthalpy|energy|potential)", 3),
        (r"\belectron\s+gain\s+enthalpy|\belectron\s+affinity|\batomic\s+radi(?:us|i)|\bionic\s+radi(?:us|i)|\bisoelectronic", 3),
        (r"\bmendeleev|\bmoseley|\bslater|\beffective\s+nuclear\s+charge|\bshielding\s+effect|\bscreening\s+effect", 3),
        (r"\biupac\s+name.*atomic\s+number|\bunnil|\bunbi|\bunun|\bmetallic\s+character|\bamphoteric\s+oxide", 2),
    ],
    "Metallurgy": [
        (r"\bmetallurgy|\bellingham|\bfroth\s+floatation|\bfroth\s+flotation|\bleaching|\bcalcination|\broasting|\bsmelting", 3),
        (r"\bzone\s+refining|\bmond(?:'s)?\s+process|\bvan\s+arkel|\bhall[- ]?heroult|\bbayer(?:'s)?\s+process|\bblast\s+furnace|\bblister\s+copper", 3),
        (r"\b(?:haematite|hematite|magnetite|bauxite|galena|cinnabar|calamine|siderite|malachite|copper\s+pyrites|cassiterite|dolomite|carnallite)\b", 3),
        (r"\bgangue|\bslag|\bflux|\bcollector|\bdepressant|\bmatte|\bpig\s+iron|\bcast\s+iron|\bwrought\s+iron|\b(?:bronze|brass|german\s+silver|alloy)\b", 2),
    ],
    "Chemical Bonding": [
        (r"\bhybridi[sz]ation|\bvsepr|\bmolecular\s+orbital|\bmo\s+theory|\bbond\s+order|\bdipole\s+moment", 3),
        (r"\bfajan(?:'s)?\s+rule|\blattice\s+enthalpy|\bborn[- ]?haber|\bhydrogen\s+bond|\bformal\s+charge|\bresonance\s+structure", 3),
        (r"\bsigma\s+bond|\bpi\s+bond|\blone\s+pair|\bbond\s+angle|\bbond\s+length|\bbond\s+enthalpy|\bparamagnetic|\bdiamagnetic", 2),
        (r"\b(?:sp3d2|sp3d|sp3|sp2|sp)\b|\btrigonal\s+bipyramidal|\bsquare\s+planar|\bsee[- ]?saw|\bt[- ]?shaped|\blinear\s+shape|\bbent\s+shape", 3),
    ],
    "Atomic Structure": [
        (r"\bbohr(?:'s)?\s+(?:model|orbit|radius|atom)|\bde\s*broglie|\bheisenberg|\buncertainty\s+principle|\bschr[oö]dinger", 3),
        (r"\bquantum\s+number|\bprincipal\s+quantum|\bazimuthal|\bmagnetic\s+quantum|\bspin\s+quantum|\baufbau|\bpauli|\bhund(?:'s)?\s+rule", 3),
        (r"\bphotoelectric\s+effect|\bwork\s+function|\blyman|\bbalmer|\bpaschen|\bbrackett|\bpfund|\brydberg|\bwave\s*number|\bspectral\s+line|\bx[- ]?ray\s+spectra", 3),
        (r"\bradial\s+node|\bangular\s+node|\bnodal\s+plane|\borbital\s+angular\s+momentum|\bplanck(?:'s)?\s+constant|\bwavelength.*(?:electron|photon)|\ballowed\s+orbitals", 3),
        (r"\bradioactive\s+decay|\bhalf[- ]?life.*(?:decay|nuclei|isotope)", 2),
    ],
    "Chemical Thermodynamics": [
        (r"\bthermodynamic|\benthalpy|\bentropy|\bgibbs|\bfree\s+energy|\bhess(?:'s)?\s+law|\bcarnot|\badiabatic|\bisothermal|\bisochoric|\bisobaric", 3),
        (r"\binternal\s+energy|\bheat\s+capacity|\bc_?p\s*-\s*c_?v|\bthermally\s+insulated|\bcalorimeter|\bbomb\s+calorimeter", 3),
        (r"\b(?:delta|Δ)\s*[hgsue]\b|\bΔ[hgsue]\b|\bspontaneous|\bspontaneity|\bnon[- ]?spontaneous|\bwork\s+done.*(?:expansion|compression|gas)", 2),
        (r"\bborn[- ]?haber|\benthalpy\s+of\s+(?:combustion|formation|neutralisation|neutralization|atomisation|vaporisation|fusion)", 3),
    ],
    "Chemical Equilibrium": [
        (r"\bequilibrium\s+constant|\bk_?p\b|\bk_?c\b|\bk_?sp\b|\bk_?a\b|\bk_?b\b|\bk_?w\b|\ble\s+chatelier", 3),
        (r"\bsolubility\s+product|\bcommon\s+ion\s+effect|\bbuffer\s+solution|\bhenderson|\bhydrolysis\s+of\s+salt", 3),
        (r"\b(?:ph|poh|pka|pkb)\s*(?:=|of|is|value|will)|\bacid[- ]?base\s+(?:titration|indicator|conjugate)|\bconjugate\s+(?:acid|base)|\blewis\s+(?:acid|base)|\bbronsted", 3),
        (r"\bdegree\s+of\s+dissociation|\bostwald\s+dilution|\bionic\s+equilibrium|\breaction\s+quotient|\bpartial\s+pressure.*equilibrium", 3),
    ],
    "Chemical Kinetics": [
        (r"\brate\s+(?:constant|law|equation|of\s+reaction)|\border\s+of\s+reaction|\bfirst[- ]?order|\bsecond[- ]?order|\bzero[- ]?order|\bpseudo[- ]?first", 3),
        (r"\barrhenius|\bactivation\s+energy|\be_?a\b|\bhalf[- ]?life|\bt_?1/2|\bcollision\s+theory|\bmolecularity|\brate\s+determining\s+step", 3),
        (r"\bpre[- ]?exponential|\bfrequency\s+factor|\btemperature\s+coefficient|\bintegrated\s+rate", 3),
    ],
    "Solutions": [
        (r"\bcolligative|\braoult(?:'s)?\s+law|\bhenry(?:'s)?\s+law|\bosmotic\s+pressure|\bvan'?t\s+hoff\s+factor", 3),
        (r"\belevation\s+(?:in|of)\s+boiling\s+point|\bdepression\s+(?:in|of)\s+freezing\s+point|\bebullioscopic|\bcryoscopic|\bk_?b\b.*boiling|\bk_?f\b.*freezing|\bboiling\s+points?\b", 3),
        (r"\bazeotrop|\bideal\s+solution|\bnon[- ]?ideal\s+solution|\bvapour\s+pressure|\bvapor\s+pressure|\brelative\s+lowering", 3),
        (r"\bmolality|\bmolarity|\bmole\s+fraction|\bparts\s+per\s+million|\bppm\b|\bisotonic|\bhypertonic|\bhypotonic|\bfinal\s+concentration\s+of\s+the\s+solution", 2),
    ],
    "Electrochemistry": [
        (r"\belectrolysis|\belectrochemical|\bgalvanic\s+cell|\bvoltaic\s+cell|\bdaniell\s+cell|\bfuel\s+cell|\blead\s+storage", 3),
        (r"\bnernst\s+equation|\bfaraday(?:'s)?\s+law|\bkohlrausch|\bmolar\s+conduct(?:ivity|ance)|\bspecific\s+conduct(?:ivity|ance)|\bcell\s+constant", 3),
        (r"\belectrode\s+potential|\bstandard\s+reduction\s+potential|\be\^?[°o0]\s*_?cell|\bemf\s+of|\bcathode|\banode|\bsalt\s+bridge", 3),
        (r"\bdeposited\s+at|\bcurrent\s+of\s+\d+.*(?:ampere|\ba\b|coulomb)|\bcorrosion|\boverpotential", 2),
    ],
    "Basic Concepts of Chemistry": [
        (r"\bstoichiometr|\blimiting\s+reagent|\bempirical\s+formula|\bmolecular\s+formula|\bavogadro|\bmolar\s+mass", 3),
        (r"\bequivalent\s+(?:weight|mass)|\bnormality|\btitration.*(?:oxalic|kmno|mohr|naoh|hcl)|\bsignificant\s+figures|\blaw\s+of\s+(?:multiple|definite|conservation)", 3),
        (r"\bpercentage\s+composition|\bcombustion\s+analysis|\bdumas\s+method|\bkjeldahl|\bcarius\s+method|\blassaigne|\bdisproportionation|\boxidation\s+(?:state|number)", 3),
        (r"\bnumber\s+of\s+moles|\bmoles?\s+of|\bmol\s+of|\bmole\s+ratio|\bgram\s+molecular|\batomic\s+mass\s+unit|\bamu\b|\bdalton|\bmolecules\s+from", 2),
        (r"\bchromatography|\bdistillation|\bsublimation|\bfractional\s+crystallisation|\bqualitative\s+analysis|\bquantitative\s+analysis", 2),
    ],
    "Basic Principles of Organic Chemistry": [
        (r"\biupac\s+name|\bgeometrical\s+isomer|\bstereoisomer|\boptical\s+isomer|\benantiomer|\bdiastereomer|\bmeso\s+compound|\bchiral|\btautomer", 3),
        (r"\bcarbocation|\bcarbanion|\bfree\s+radical|\binductive\s+effect|\bmesomeric|\bhyperconjugation|\bresonance\s+effect|\belectromeric", 3),
        (r"\baromaticity|\bh[uü]ckel(?:'s)?\s+rule|\banti[- ]?aromatic|\bnon[- ]?aromatic|\bnucleophile|\belectrophile|\bconform(?:er|ation)|\bnewman|\bsawhorse", 3),
        (r"\b(?:cis|trans|e/z|r/s)\s+isomer|\bstructural\s+isomer|\bfunctional\s+isomer|\bmetamer|\bchair\s+conformation", 2),
    ],
    "Hydrocarbons": [
        (r"\balkane|\balkene|\balkyne|\bpropyne|\bpropene|\bbutene|\bbutyne|\bethane|\bethene|\bethyne|\bacetylene|\bmethane|\bpropane", 3),
        (r"\bmarkovnikov|\banti[- ]?markovnikov|\bperoxide\s+effect|\bkharasch|\bozonolysis|\bwurtz|\bkolbe(?:'s)?\s+electrolysis|\bdecarboxylation", 3),
        (r"\bfrid?el[- ]?crafts|\bbenzene|\btoluene|\bxylene|\bnaphthalene|\bl\s*indlar|\bbirch\s+reduction|\bhydroboration|\boxymercuration", 3),
        (r"\bbaum?eyer(?:'s)?\s+reagent|\bterminal\s+alkyne|\bdehydrohalogenation|\bsaytzeff|\bzaitsev|\bhofmann\s+elimination", 2),
    ],
    "Organic Compounds Containing Halogens": [
        (r"\bs_?n\s*1\b|\bs_?n\s*2\b|\be1\b|\be2\b|\balkyl\s+halide|\baryl\s+halide|\bhaloalkane|\bhaloarene|\bvinyl\s+halide|\ballylic\s+halide|\bsolvolysis", 3),
        (r"\bgrignard\s+reagent|\brmgx|\bfinkelstein|\bswarts|\bsandmeyer|\bgattermann|\bchloroform|\biodoform|\bfreon|\bddt\b|\bbhc\b", 3),
        (r"\bchlorobenzene|\bbromobenzene|\biodobenzene|\bbenzyl\s+chloride|\bprimary\s+halide|\bsecondary\s+halide|\btertiary\s+halide|\bwalden\s+inversion", 3),
        (r"\b(?:ch_?3|c_?2h_?5|r)[- ]?(?:cl|br|i)\b|\bwilliamson\s+synthesis.*halide", 2),
    ],
    "Organic Compounds Containing Oxygen": [
        (r"\bfehling(?:'s)?\s+test|\btollens(?:'s)?\s+(?:reagent|test)|\bschiff(?:'s)?\s+test|\blucas\s+(?:reagent|test)|\bvictor\s+meyer", 3),
        (r"\baldol|\bcannizzaro|\bperkin|\bclemmensen|\bwolff[- ]?kishner|\breimer[- ]?tiemann|\bkolbe(?:'s)?\s+reaction|\bhell[- ]?volhard|\bhvz\b", 3),
        (r"\b(?:alcohol|phenol|ether|aldehyde|ketone|carboxylic\s+acid|ester|anhydride|acyl\s+chloride|epoxide)\b", 2),
        (r"\b(?:acetaldehyde|formaldehyde|acetone|benzaldehyde|acetophenone|benzoic\s+acid|formic\s+acid|acetic\s+acid|salicylic\s+acid|picric\s+acid|anisole)\b", 3),
        (r"\bwilliamson\s+(?:ether\s+)?synthesis|\beste?rification|\bsaponification|\bpcc\b|\bjones\s+reagent|\blialh_?4|\bnabh_?4|\brosenmund|\bstephen\s+reduction|\betard", 3),
    ],
    "Organic Compounds Containing Nitrogen": [
        (r"\bamine(?:s)?\b|\baniline|\bdiazonium|\bazo\s+dye|\bnitrobenzene|\bcyanide|\bisocyanide|\bnitrile|\bamide|\bimide", 3),
        (r"\bhinsberg|\bcarbylamine|\bgabriel\s+phthalimide|\bhofmann\s+bromamide|\bmendius|\bcoupling\s+reaction|\bdiazotisation|\bdiazotization", 3),
        (r"\b(?:primary|secondary|tertiary)\s+amine|\bbasic(?:ity)?\s+of\s+amine|\bpyridine|\bpyrrole|\bzwitterion.*amine", 3),
    ],
    "Biomolecules": [
        (r"\bvitamin|\bcarbohydrate|\bglucose|\bfructose|\bsucrose|\bmaltose|\blactose|\bstarch|\bcellulose|\bglycogen|\bmonosaccharide|\bdisaccharide|\bpolysaccharide|\bglyceraldehyde", 3),
        (r"\bamino\s+acid|\bpeptide|\bprotein|\benzyme|\bnucleic\s+acid|\bdna\b|\brna\b|\bnucleotide|\bnucleoside|\bpurine|\bpyrimidine|\bdouble\s+helix", 3),
        (r"\b(?:ascorbic\s+acid|thiamine|riboflavin|pyridoxine|cobalamin|retinol|tocopherol|calciferol|niacin|biotin|folic\s+acid)\b", 3),
        (r"\b(?:glycine|alanine|valine|leucine|isoleucine|proline|serine|threonine|cysteine|methionine|lysine|arginine|histidine|tyrosine|tryptophan|phenylalanine|glutamic|aspartic)\b", 3),
        (r"\banomer|\bepimer|\bmutarotation|\bglycosidic|\bdenaturation|\bzwitter\s*ion|\bisoelectric\s+point|\breducing\s+sugar|\bnon[- ]?reducing\s+sugar", 3),
    ],
    "Polymers": [
        (r"\bpolymer|\bmonomer|\bcopolymer|\bhomopolymer|\bvulcani[sz]ation|\bthermoplastic|\bthermosetting|\belastomer", 3),
        (r"\b(?:nylon|bakelite|teflon|neoprene|buna[- ]?[ns]|dacron|terylene|glyptal|melamine|polythene|polyethylene|pvc|polystyrene|orlon|polyacrylonitrile|phbv)\b", 3),
        (r"\bcaprolactam|\badipic\s+acid|\bhexamethylenediamine|\bisoprene|\bchloroprene|\bbutadiene|\bstyrene|\bziegler[- ]?natta|\bcondensation\s+polymer|\baddition\s+polymer", 3),
    ],
}


@dataclass(frozen=True)
class TopicTagResult:
    topic: str
    score: int
    matched_rules: tuple[str, ...]
    is_unknown: bool = False


def load_chemistry_topics(priors_path: str | Path = DEFAULT_PRIORS_PATH) -> tuple[str, ...]:
    """Load the canonical Chemistry topic vocabulary from topic_difficulty_priors.csv."""
    topics: list[str] = []
    with open(priors_path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if (row.get("subject") or "").strip().lower() == "chemistry":
                topic = (row.get("topic") or "").strip()
                if topic:
                    topics.append(topic)
    return tuple(topics)


def _compiled_rules() -> dict[str, list[tuple[re.Pattern[str], int, str]]]:
    compiled: dict[str, list[tuple[re.Pattern[str], int, str]]] = {}
    for topic, rules in _TOPIC_RULES.items():
        compiled[topic] = [(re.compile(pat, re.IGNORECASE), weight, pat) for pat, weight in rules]
    return compiled


_COMPILED_RULES = _compiled_rules()


def tag_text(text: str, *, default_topic: str = "unknown") -> TopicTagResult:
    """Tag a Chemistry question's combined text to a Track B Chemistry topic."""
    cleaned = (text or "").strip()
    if not cleaned:
        return TopicTagResult(topic=default_topic, score=0, matched_rules=(), is_unknown=(default_topic == "unknown"))

    best_topic = default_topic
    best_score = 0
    best_matches: list[str] = []

    for topic, rules in _COMPILED_RULES.items():
        score = 0
        matched: list[str] = []
        for pattern, weight, raw_pat in rules:
            if pattern.search(cleaned):
                score += weight
                matched.append(raw_pat)
        if score > best_score:
            best_score = score
            best_topic = topic
            best_matches = matched

    if best_score == 0:
        return TopicTagResult(topic=default_topic, score=0, matched_rules=(), is_unknown=(default_topic == "unknown"))
    return TopicTagResult(topic=best_topic, score=best_score, matched_rules=tuple(best_matches), is_unknown=False)


def question_to_text(question: Any) -> str:
    """Extract searchable text from a JEEQuestionRecord or raw question dict."""
    if isinstance(question, dict):
        parts = [str(question.get("question_text") or "")]
        for opt in question.get("options") or []:
            if isinstance(opt, dict):
                parts.append(str(opt.get("text") or ""))
        if question.get("solution_text"):
            parts.append(str(question["solution_text"]))
        return "\n".join(parts)

    parts = [str(getattr(question, "question_text", "") or "")]
    for opt in getattr(question, "options", None) or []:
        parts.append(str(getattr(opt, "text", "") or ""))
    sol = getattr(question, "solution_text", None)
    if sol:
        parts.append(str(sol))
    return "\n".join(parts)


def tag_question(question: Any, *, default_topic: str = "unknown") -> TopicTagResult:
    """Tag a JEEQuestionRecord or dict to a Track B Chemistry topic."""
    return tag_text(question_to_text(question), default_topic=default_topic)


class QuestionTopicTagger:
    """Fast lookup + on-the-fly topic tagger for Chemistry questions."""

    def __init__(
        self,
        precomputed_tags: dict[str, str] | None = None,
        valid_topics: Iterable[str] | None = None,
    ) -> None:
        self._tags = dict(precomputed_tags or {})
        self._valid_topics = frozenset(valid_topics or load_chemistry_topics())

    @classmethod
    def from_file(cls, tags_path: str | Path = DEFAULT_TAGS_OUT, priors_path: str | Path = DEFAULT_PRIORS_PATH) -> "QuestionTopicTagger":
        valid = load_chemistry_topics(priors_path)
        path = Path(tags_path)
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            tags = data.get("question_topics", data) if isinstance(data, dict) else {}
            return cls(precomputed_tags=tags, valid_topics=valid)
        return cls(precomputed_tags={}, valid_topics=valid)

    def get_topic(self, question: Any, *, fallback_topic: str = "unknown") -> str:
        """Return a Track B topic for a question (uses precomputed map, else tags on the fly).

        If no keyword matches, returns ``fallback_topic`` (default ``"unknown"`` so untagged
        questions are reported honestly as unseen topics rather than inflating a real chapter).
        """
        qid = question.get("question_id") if isinstance(question, dict) else getattr(question, "question_id", None)
        if qid and qid in self._tags and self._tags[qid] in self._valid_topics:
            return self._tags[qid]
        result = tag_question(question, default_topic="unknown")
        if not result.is_unknown and result.topic in self._valid_topics:
            return result.topic
        return fallback_topic


def build_coverage_report(
    questions_path: str | Path = DEFAULT_QUESTIONS_PATH,
    priors_path: str | Path = DEFAULT_PRIORS_PATH,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Tag all Chemistry questions in a JSONL file and return (tag_map, coverage_report)."""
    valid_topics = set(load_chemistry_topics(priors_path))
    tag_map: dict[str, str] = {}
    counts: Counter[str] = Counter()
    untagged: list[str] = []
    total_chemistry = 0

    with open(questions_path, encoding="utf-8") as fh:
        for line in fh:
            raw = line.strip()
            if not raw:
                continue
            record = json.loads(raw)
            if str(record.get("subject", "")).strip().lower() != "chemistry":
                continue
            total_chemistry += 1
            qid = str(record["question_id"])
            res = tag_question(record, default_topic="unknown")
            if not res.is_unknown and res.topic in valid_topics:
                tag_map[qid] = res.topic
                counts[res.topic] += 1
            else:
                tag_map[qid] = "unknown"
                untagged.append(qid)

    tagged_count = total_chemistry - len(untagged)
    coverage_pct = round((tagged_count / total_chemistry) * 100.0, 2) if total_chemistry else 0.0
    report = {
        "questions_path": str(questions_path),
        "priors_path": str(priors_path),
        "total_chemistry_questions": total_chemistry,
        "tagged_count": tagged_count,
        "untagged_count": len(untagged),
        "coverage_pct": coverage_pct,
        "topics_covered": len(counts),
        "total_vocabulary_topics": len(valid_topics),
        "counts_by_topic": dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "untagged_question_ids": untagged,
    }
    return tag_map, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--priors", type=Path, default=DEFAULT_PRIORS_PATH)
    parser.add_argument("--tags-out", type=Path, default=DEFAULT_TAGS_OUT)
    parser.add_argument("--coverage-out", type=Path, default=DEFAULT_COVERAGE_OUT)
    args = parser.parse_args(argv)

    tag_map, report = build_coverage_report(args.questions, args.priors)
    args.tags_out.parent.mkdir(parents=True, exist_ok=True)
    args.coverage_out.parent.mkdir(parents=True, exist_ok=True)
    args.tags_out.write_text(
        json.dumps({"vocabulary_source": str(args.priors), "question_topics": tag_map}, indent=2),
        encoding="utf-8",
    )
    args.coverage_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"Tagged {report['tagged_count']}/{report['total_chemistry_questions']} "
        f"Chemistry questions ({report['coverage_pct']}%) across {report['topics_covered']}/{report['total_vocabulary_topics']} topics."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

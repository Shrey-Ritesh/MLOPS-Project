# Public difficulty evidence — Physics and Chemistry (Track B MLOps demo)

Retrieved 2026-10-14. Scope: Physics (18 chapters) and Chemistry (22 chapters); Mathematics is deferred.

## How to read this
Public sources can suggest *which kinds* of difficulty to represent in a simulation. They **cannot** establish how often these difficulties occur across all JEE aspirants. Every number below is a simulation prior with the stated confidence, not a measured fact. Weightage (how often a chapter appears) is **not** difficulty, and several sources only give weightage.

## Sources
| Key | URL | What it gives | Basis it states | Confidence |
|---|---|---|---|---|
| SATHEE | https://sathee.iitk.ac.in/pyqs/difficulty-analysis/ | Per-chapter score (1–5), success rate, avg time, Easy/Medium/Hard/Very-Hard split for 40 Physics + Chemistry chapters | Weighted formula over five parameters; cites "15 years of data" but gives no sample sizes or validation; some accuracy claims are unsupported | Low–medium |
| PW-TOUGH | https://www.pw.live/iit-jee/exams/toughest-chapters-in-jee-mains | A short list of "hardest" Physics topics (no reasons); chapter lists by weightage | "Expert analysis of previous years' question papers"; no data shown; dated 10 Feb 2026 | Low |
| VED-CHEM | https://www.vedantu.com/jee-main/chemistry | A list of "important" Chemistry chapters, weightage by marks; no per-chapter difficulty labels | Previous five years of papers (years not specified) | Low |
| Shift-level analyses (Careers360, Aakash, Vedantu; found through search, pages not opened) | e.g. https://news.careers360.com/jee-main-2025-january-28-shift-1-analysis-exam-challenging-tougher-physics-difficulty-level-student-reactions | Whole-section difficulty for individual shifts (Physics hard in one shift, easy in another; Chemistry usually easiest) | Student feedback and coaching-institute review | Low; varies shift to shift |

## SATHEE priors used for the simulation
Hardest Physics chapters:
  - Rotational Motion (score 3.7, success 42.1%, ~5.6 min)
  - Electromagnetic Induction (score 3.6, success 44.7%, ~5.4 min)
  - Thermodynamics (score 3.4, success 48.9%, ~5.2 min)
  - Atoms and Nuclei (score 3.4, success 49.2%, ~5.1 min)
  - Magnetic Effects of Current (score 3.3, success 51.8%, ~4.9 min)

Easiest Physics chapters:
  - Units and Dimensions (score 1.2, success 89.4%, ~1.8 min)
  - Kinematics (score 2.1, success 76.8%, ~2.9 min)
  - Ray Optics (score 2.5, success 70.3%, ~3.6 min)
  - Work, Energy and Power (score 2.6, success 68.7%, ~3.8 min)
  - Electromagnetic Waves (score 2.7, success 67.1%, ~3.9 min)

Hardest Chemistry chapters:
  - Chemical Thermodynamics (score 3.5, success 46.8%, ~5.3 min)
  - Coordination Compounds (score 3.4, success 49.5%, ~5.0 min)
  - Chemical Equilibrium (score 3.3, success 51.9%, ~4.9 min)
  - Electrochemistry (score 3.2, success 54.1%, ~4.8 min)
  - Organic Compounds Containing Nitrogen (score 3.2, success 52.8%, ~4.8 min)

Easiest Chemistry chapters:
  - Basic Concepts of Chemistry (score 1.3, success 87.6%, ~1.9 min)
  - Classification of Elements and Periodicity (score 1.8, success 79.4%, ~2.6 min)
  - Hydrogen (score 2.1, success 75.2%, ~2.9 min)
  - s-Block Elements (score 2.3, success 72.8%, ~3.2 min)
  - Basic Principles of Organic Chemistry (score 2.4, success 71.5%, ~3.4 min)

The full table, with the Very-Hard share folded into Hard for the Easy/Medium/Hard schema, is `configs/track_b/topic_difficulty_priors.csv`.

## Recurring themes (qualitative, low confidence)
1. **Multi-concept and multi-step problems are harder than single-concept ones.** SATHEE's highest scores sit in chapters that combine ideas (Rotational Motion, Electromagnetic Induction, Chemical Thermodynamics, Coordination Compounds). PW-TOUGH also names "Mechanics" and "Heat and Thermodynamics" as hard. Simulation use: more mistakes, hints and time on these topics.
2. **Numerical and calculation-heavy chapters.** SATHEE's study-priority notes link the hardest chemistry chapters to mathematical or problem-solving skills; several shift analyses describe lengthy, calculative Physics papers. Simulation use: longer response times and more mistakes on numerical topics.
3. **Foundation and memory-based chapters are easier.** SATHEE rates Units and Dimensions, Basic Concepts of Chemistry and Classification of Elements lowest; shift analyses call Chemistry the most scoring section. Simulation use: fewer mistakes and hints, more switching to harder levels.
4. **Section difficulty varies by exam shift.** Reports for different shifts disagree about whether Physics or Chemistry was harder, so per-topic difficulty is best treated as an average with large noise. Simulation use: substantial random variation, not fixed per-topic outcomes.

## Where the sources agree or disagree with SATHEE
- **Agree:** Thermodynamics, Electromagnetic Induction, and Electrostatics/Magnetism are hard in both SATHEE and PW-TOUGH.
- **Disagree:** PW-TOUGH names Current Electricity and Optics among the hardest topics; SATHEE rates them Medium (2.8, 2.5/3.0). Sources also disagree about Modern Physics (hard in PW-TOUGH; high-scoring in another analysis).
- **Weightage vs difficulty:** VED-CHEM calls Chemical Thermodynamics and Equilibrium important, while PW-TOUGH lists both as low weightage. Neither speaks to difficulty.
- For most chapters (see `cross_check_note` in the CSV) no independent chapter-level difficulty evidence was found.

## Limits
- SATHEE's methodology is undocumented; its figures cannot be checked here.
- Coaching-site rankings may reflect their own marketing and are expert opinion, not student outcome data.
- Sources found through search summaries were not opened and are used only as weak context.
- No personal data or individual posts are used. Repeated claims from different sites are not treated as independent evidence.
- Nothing here validates a model for real students; the simulation is a controlled experiment about MLOps process.

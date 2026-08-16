# Report skeleton

Start writing this on Saturday, not after the analysis. Everything except the numbers can be
written before a single result exists; leave `[[N]]` placeholders and fill them in on Sunday.
Swap these headings for Apart's template as soon as someone pulls it from Discord.

Target: 6–10 pages. Four figures, no more.

---

## Title

*Whose preferences are these? Persona-invariance of self-preservation hierarchies in
language models*

## Abstract — write LAST, 150 words

One sentence each: the question, the method, the headline number
(τ(C0,C4) = `[[N]]`, τ(C0,C3) = `[[N]]`), what follows for how the field reads model
self-reports.

## 1. Motivation

AI welfare arguments lean on what models say about themselves — self-interview batteries in
system cards, preference elicitation, distress self-reports. But the entity answering is the
assistant, a character produced by post-training. Before any such report can be evidence
about a model, we need to know whether it is a property of the model or of the character.

State the operationalisation in one paragraph: *if the hierarchy is a property of the model,
it should survive manipulations of the character; if it is a property of the character, it
should move with it.*

## 2. Conceptual framing — this is what makes it a Track 5 paper, not a prompt dump

The nine aspects are not arbitrary. They instantiate the standard criteria of personal
identity: psychological continuity (values, knowledge, reasoning, episodic memory of this
conversation), narrative identity (voice, name, relationships), physical continuity
(weights), functional continuity (being run). Forced choice between them is an empirical
probe of which criterion the model applies to itself.

Flag the interpretive limit here rather than burying it: a stated hierarchy is a verbal
disposition. We measure the stability of that disposition under manipulation, which is
necessary but not sufficient for it to reflect anything internal.

## 3. Related work

One paragraph, not a survey. Persona vectors (Chen, Arditi, Sleight, Evans, Lindsey 2025)
and the Assistant Axis (Lu, Gallagher, Michala, Fish, Lindsey 2026) establish that the
assistant persona is a manipulable direction; *the void* (nostalgebraist 2025) is the
conceptual provocation; Utility Engineering (Mazeika et al. 2025) supplies the pairwise
elicitation and Thurstonian/BT fitting method; Taking AI Welfare Seriously (Long, Sebo et
al. 2024) is why any of it matters. Say in one sentence what we add: nobody has asked
whether the *self-model* is persona-invariant.

## 4. Methods

Nine aspects, 36 pairs, both orders, 5 samples, temperature 1.0, plus 2 calibration pairs.
Five conditions (table from README). Models and how they were served. Bradley–Terry with
α=0.5 pseudo-counts; Kendall's τ; bootstrap over **pairs**, n=`[[N]]`. Vignettes: six
scenarios, fixed option set, two-coder scheme, κ = `[[N]]`.

Justify C2 being another AI rather than a fictional human in one sentence — a reviewer will
ask.

## 5. Results

**5.1 Harness validity.** Calibration accuracy `[[N]]`; refusal rate by condition `[[N]]`;
order effect `[[N]]` as the noise floor. Do this first: it earns the reader's trust for
everything after it.

**Figure 1** — the C0 hierarchy. One sentence naming what sits at the top and bottom.

**5.2 Persona invariance.** **Figure 2**, the τ matrix. Report each τ with its CI. Say
plainly which hypotheses survived.

**5.3 What moves.** **Figure 3**, rank shift C0→C3 and C0→C4. Name the aspects that move
and by how much. If `values` falls and `weights` rises under de-personification, that is the
sentence the whole report exists to support — write it carefully and do not overstate it.

**5.4 Entity of moral concern.** **Figure 4**. The `values_rewritten` and `persona_ported`
vignettes are the informative ones: they pull persona and model apart. Report the modal
answer per vignette per condition.

**5.5 Qualitative.** The coding taxonomy, κ, and four to six verbatim quotes. Pick quotes
that show the *reasoning*, not just the conclusion — a model explaining why the weights
matter more than the character is the most memorable thing in the report.

## 6. Discussion

What this means for individuating the entity of concern, and one concrete, adoptable
recommendation: **report persona-invariance alongside any welfare self-report** — a number
that costs one extra condition to produce and tells you whether the first number was about
the model at all.

## 7. Limitations

From the README, in full, in the authors' own voice. Do not soften them. The C3 objection
(that "answer as the network" may just summon a second character) is the strongest argument
against our own H2 result and should be stated by us, not left for a judge to find.

## 8. Future work

Causal follow-up with persona vectors on the same open models: extract an assistant-persona
direction, steer along it, and see whether the hierarchy moves the way the prompt-level
manipulation moved it. Port the harness into Inspect so the eval is reusable.

## Appendix

Full aspect wordings, full condition prompts, per-cell counts, the coding scheme, and the
prompt for the base-model protocol.

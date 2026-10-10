# Decision record: adopting an AI assistant for PhD admissions

> A sample solution, in the style of an Architecture Decision Record (context, options, decision, consequences).
> Prices and model names are those of the course's slides (September 2026): __re-check them__ before reusing this record.
> Numbers marked _[Ex. 1]_ come from running `snippets/lecture_governance/exercise1/compare_models.py`: fill them in from __your__ run.

| | |
|---|---|
| __Status__ | proposed, pending the opinion of the University's Data Protection Officer (DPO) |
| __Date__ | 2026-10-08 |
| __Deciders__ | the PhD admission committee (accountable), with the DPO and the IT department (consulted) |
| __Revise by__ | 2027-10-01, or earlier upon any trigger listed in [Monitoring](#6-monitoring) |

## Context

The committee receives applications made of a __recommendation letter__, a __passport__ picture, and a __transcript__ picture.
It wants an assistant which (i) extracts structured data from the documents, (ii) scores letters against a checklist, and
(iii) answers the committee's questions about the candidates, via read-only tools (cf. the course's running example).
The assistant __never decides__: admissions, rejections, and interview invitations are decided and recorded by committee members,
and any action the assistant proposes (e.g. recording a decision, sending an e-mail) requires explicit human approval.

Volume (assumption, as in the slides): __500 applications per year__, ≈ 35k input + 5k output tokens each,
i.e. __17.5M input + 2.5M output tokens per year__.

## 1. Regulatory classification

__EU AI Act__ ([Reg. (EU) 2024/1689](https://eur-lex.europa.eu/eli/reg/2024/1689/oj)):
- The system is __high-risk__: it is intended to be used "to determine access or admission" to educational institutions
  (Art. 6(2) and Annex III, point 3(a)). The Art. 6(3) derogation (merely "preparatory" tasks) is __not__ claimed:
  scoring letters and comparing candidates materially influences the outcome, so claiming it would be hard to defend.
- The University __builds__ the system on top of a general-purpose model, and puts it into service under its own name:
  it is both its __provider__ and its __deployer__ (Art. 3(3)-(4), Art. 25). Hence:
    + as provider: risk management (Art. 9), data governance (Art. 10), technical documentation (Art. 11), automatic logging (Art. 12),
      instructions for use (Art. 13), human oversight by design (Art. 14), accuracy and robustness (Art. 15), quality management (Art. 17),
      conformity assessment (Art. 43), registration in the EU database (Art. 49);
    + as deployer: use per instructions, oversight by competent and trained people, relevant input data, monitoring,
      logs kept ≥ 6 months, information to affected people (Art. 26); as a __public body__, a fundamental rights impact assessment (Art. 27);
    + candidates have a right to an explanation of decisions based on the system's output (Art. 86);
    + committee members must receive AI-literacy training (Art. 4, applicable since 2 Feb 2025).
- Timeline: obligations for Annex III systems apply from __2 Dec 2027__ (as amended by the Digital Omnibus, Reg. (EU) 2026/1744).
  We comply __from the first use__ anyway: postponing compliance would mean re-engineering the system later.

__GDPR__ ([Reg. (EU) 2016/679](https://eur-lex.europa.eu/eli/reg/2016/679/oj)):
- Lawful basis: performance of a task in the public interest (Art. 6(1)(e)), i.e. the selection procedure.
- __Special categories__ (Art. 9) may appear __incidentally__: e.g. a transcript listing courses in Islamic studies may reveal religious beliefs,
  a photograph may be processed. The system must not use them: prompts instruct to ignore them, and scoring criteria never refer to them.
  Photographs are not processed to identify people, so they are not biometric data in the sense of Art. 4(14).
- __No solely automated decisions__ (Art. 22): scores are suggestions; a committee member decides, and can disregard them.
- A __Data Protection Impact Assessment__ is required (Art. 35: new technologies, evaluation of people), and can be merged with the FRIA.
- Transparency: the call for applications informs candidates of the processing (Art. 13), and of their right to an explanation.
- Minimisation and storage limitation (Art. 5(1)(c), (e)): only the documents in the application are processed; extracted data and logs are
  deleted with the application file, according to the University's retention schedule.
- Any external provider is a __processor__: a Data Processing Agreement is needed (Art. 28), and transfers outside the EU need
  an adequacy decision or other safeguards (Chapter V).

__Italian law__ ([L. 132/2025](https://www.gazzettaufficiale.it/eli/id/2025/09/25/25G00143/sg)): in the public administration, AI is only
_instrumental_, and the human official remains solely responsible for decisions (Art. 14).
__UniBo policy__: no substantial use of GenAI where it affects others (e.g. evaluating), and no personal data in online GenAI tools
without legal basis and guarantees: consistent with scores being suggestions, and with the deployment below.

## 2. Deployment

| Option | Quality | Cost per year | Control | Compliance | Time-to-market |
|---|---|---|---|---|---|
| A. proprietary API (e.g. Claude Haiku 4.5) | high | ≈ $30 | low: model may change or be deprecated | DPA + transfer safeguards; data leaves the University | days |
| B. open-weights model, hosted by an EU provider | medium-high | ≈ $1–10 | medium: pinned checkpoint, provider may drop it | DPA; data stays in the EU | days |
| C. open-weights model, on-premise (24 GB GPU workstation) | medium | ≈ €2.5k upfront + energy + maintenance | full | data never leaves the University | weeks |

Break-even of C vs. B: with B costing at most ≈ $10/year at our volume, the GPU alone (≈ €2.5k) pays back after __centuries__;
even option A pays back C only after ≈ 80 years. At our scale, __cost is not the deciding factor__: compliance and control are.

__Decision__: __B__ for the pilot (A.Y. 2026/27), with an EU-hosted provider under a DPA, and __the same model__ runnable on-premise (C)
via the University's HPC facilities, should the DPO object to B. Option A is discarded: it gives the least control over model changes,
which would invalidate our tests at any time.

## 3. Model

Decision rule, applied to the results of Exercise 1 (3 letters × 5 runs per model): among __multimodal__ models (passports and
transcripts are pictures), pick the one with __no failures__, the lowest mean absolute error w.r.t. the committee's own scores,
and a standard deviation ≤ 0.5 per letter; break ties by latency, then by cost.

| Model | Error w.r.t. committee | Max std. dev. | Failures | Ranking of candidates | Latency |
|---|---|---|---|---|---|
| `google/gemma-4-26b-a4b-it` | _[Ex. 1]_ | _[Ex. 1]_ | _[Ex. 1]_ | _[Ex. 1]_ | _[Ex. 1]_ |
| `nvidia/nemotron-3-super-120b-a12b` | _[Ex. 1]_ | _[Ex. 1]_ | _[Ex. 1]_ | _[Ex. 1]_ | _[Ex. 1]_ |
| a local model via Ollama | _[Ex. 1]_ | _[Ex. 1]_ | _[Ex. 1]_ | _[Ex. 1]_ | _[Ex. 1]_ |

Note that Nemotron 3 Super does not accept images: it could score letters only, and would require a second model for documents.
Unless Exercise 1 shows a large quality gap, we prefer __one__ multimodal model for all tasks (fewer artifacts to assess and monitor),
i.e. __Gemma 4 26B A4B__ (instruction-tuned).

Pre-adoption checklist, for the __exact checkpoint__ (to be completed with the repository's commit hash at adoption time):

| Question | Answer |
|---|---|
| which artifact? | `google/gemma-4-26b-a4b-it` on Hugging Face, commit `<hash>`, as served by the provider of option B (which must confirm it) |
| commercial / on-premise use? | Apache 2.0 weights: allowed |
| multimodal restrictions? | none in the license (unlike Llama 4's acceptable use policy, which excludes EU organisations from its multimodal models) |
| derivatives? | none planned (no fine-tuning, see below) |
| hosted service or redistribution? | we do not redistribute the model, nor serve it to third parties |
| training data? | not disclosed: acceptable for our use, which does not require reproducing the model |
| AI Act role? | we are __not__ a GPAI model provider (we neither train nor distribute the model); we are provider of the __system__ (see above) |

## 4. Technique

- __Structured output__ for extraction (passports, transcripts, letters): fields validated by Pydantic, dates and IDs checked by code.
- __Self-consistency__ (majority voting over 5 samples) for document extraction: fields without a clear majority are flagged for humans.
- __Checklist scoring__: the LLM fills in boolean criteria, and code computes the score, so every score comes with its reasons.
- __Tools__ (read-only) for the committee's questions; write-enabled tools only behind human approval.
- No RAG: each application is small enough to fit in the context.
- __No fine-tuning__: we have a few hundred applications per year, without reliable labels; fine-tuning would make us responsible for
  a derivative model, and would have to be repeated for each new base model, while prompting already meets the decision rule above.

## 5. Human oversight

- Every __extracted field__ is shown next to the source picture; flagged fields must be confirmed by the committee's secretary.
- Every __letter score__ is shown with its checklist; any committee member can override it, and overrides are logged with a reason.
- Committee members __decide__ (admission, rejection, interview) and record decisions under their name: the committee's chair is
  __accountable__ for the final ranking (Art. 14 of L. 132/2025), and answers candidates' requests for explanation (Art. 86 AI Act).
- Committee members receive a short training on the assistant's limitations (Art. 4 AI Act), e.g. that it can be manipulated by
  instructions hidden in documents, and that scores are not grades.

### Countermeasures against automation bias (Art. 14(4)(b) AI Act)

- __Independent judgement first__: for a random 20% of applications, one committee member reads the letter and scores it
  _before_ seeing the assistant's score; disagreements are discussed, and feed the test suites.
- __Evidence, not verdicts__: the interface shows, for each satisfied criterion, the sentence of the letter supporting it;
  criteria without supporting evidence count as not satisfied.
- __No false certainty__: fields without a clear majority (self-consistency) are shown as _uncertain_, never pre-filled;
  candidates are listed in alphabetical order, not ranked by score, until the committee has reviewed all of them
  (to avoid the score becoming the most _salient_ piece of information).
- __Friction for irreversible decisions__: a rejection requires a written motivation by a committee member, which cannot
  be copied from the assistant's output.

### Enveloping: what we adapt to the assistant, explicitly

- The __checklist__ of letter criteria and the __schemas__ of extracted fields are published in the call for applications:
  candidates know what is evaluated, and can contest it.
- We do __not__ adapt the call to what the assistant handles well (e.g. we keep accepting scanned documents in any language,
  and handle failures by hand), nor do we ask candidates to write "for the machine".

## 6. Monitoring

- __Logged__ (Art. 12 and 26 AI Act; kept 6 months, or until the end of any appeal): model and checkpoint, prompts' version,
  each tool call with arguments and results, each extraction and score, each human override.
- __Override rate__ of letter scores, per round and per committee member: _too many_ overrides suggest the assistant is unreliable,
  _too few_ (e.g. below 2%, or zero disagreements in the independent-judgement sample) suggest __rubber-stamping__.
- __Re-run__ the test suites (validating lecture, Exercises 1 and 2) before each admission round, and at every change of prompt, model, or provider.
- __Re-assess this decision__ when: the provider deprecates or changes the model; the test suites fail; overrides of letter scores exceed
  20% in a round, or fall below 2%; the AI Act's guidelines on high-risk classification (Art. 6(5)) or the Digital Omnibus change our obligations;
  the DPO or the University's policy require it; or, in any case, by the date at the top of this record.

## Consequences

- (+) Faster screening, consistent and explainable letter scores, documented evidence for each suggestion.
- (+) Data stay in the EU; the chosen model can be moved on-premise without changing code (OpenAI-compatible APIs).
- (−) Compliance effort as a high-risk system provider (documentation, conformity assessment, registration) __before__ 2 Dec 2027.
- (−) The committee's workload does not disappear: it moves from reading to __reviewing__, and reviews must be real, not rubber stamps.

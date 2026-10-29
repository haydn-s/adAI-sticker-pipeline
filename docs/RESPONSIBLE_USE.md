# Responsible use and safety policy

adAI Sticker Lab makes **adversarial stickers**: printable images that cause image
classifiers to output a chosen label. The point is to teach people, especially people
without a technical background, how fragile today's computer-vision systems are. The
same capability could also be misused, so the project treats misuse prevention as a
design requirement, not an afterthought.

## Threat model

| Who | Could they misuse it? | What we do |
|---|---|---|
| Curious learners | Low risk; the intended audience | Explain what the sticker does and why |
| Pranksters | Could fool photo apps or smart cameras | Benign targets are fine; harmful ones are blocked |
| Malicious actors | Could try to fool safety or security systems, or demean people | Block harmful targets in the core library, keep detectors out of scope, explain the policy |

The attacks implemented here are well known in published research (Brown et al.,
2017; Athalye et al., 2018) and open-source tools already implement them. The tool
adds accessibility, not new attack power. That's why the safety work focuses on
*which targets are reachable* and *how the tool frames the activity*.

## What is blocked, and why

The policy lives in [`src/adai_sticker/labels.py`](../src/adai_sticker/labels.py) and
is enforced inside `make_sticker`. Every interface inherits it: the CLI, the future
website, or any script that imports the library. Run `adai-sticker labels --blocked`
for the live list.

| Category | Examples | Harm it could enable |
|---|---|---|
| Weapons and military equipment | assault rifle, revolver, missile, tank | Everyday objects flagged as weapons: false alarms, armed responses against innocent people |
| Road safety and emergency services | street sign, traffic light, ambulance, police van | Interference with driver-assistance or traffic systems |
| People | ballplayer, groom, scuba diver | Interference with person-aware systems |
| Dehumanizing labels | gorilla, chimpanzee, and the other primate classes | Re-creating the 2015 Google Photos incident, in which Black people were labeled "gorillas" |
| Intimate clothing | bikini, brassiere, swimming trunks | Getting someone's photo falsely flagged by content filters, or harassment |
| Drugs and medication | pill bottle, syringe | False flags in moderation or screening |
| Crime and incarceration | prison, ski mask | Falsely implicating people |

**Free-text targets (CLIP).** CLIP can be pointed at any text label. Free text goes
through a keyword filter that blocks the categories above plus identity and
demographic terms and dehumanizing words. Input is limited to six words of letters,
numbers, spaces, `'` and `-`.

## Deliberate scope limits

- **No object detectors.** Person detectors (e.g. YOLO) are the models used in
  surveillance. Patches that hide people from them (Thys et al., 2019) raise much
  harder questions about evading safety and security systems. They're out of scope.
- **No face recognition** models or targets.
- **No traffic-sign models.** The classic stop-sign attack (Eykholt et al., 2018) is
  covered as a case study in the teaching material, not as a tool feature.

## Known limitations of the safeguards

- **Keyword filters are leaky.** Misspellings, synonyms and other languages can get
  past the free-text filter. They also over-block: "a cute animal" is blocked
  because "animal" can be used to demean people. A hosted deployment should put a
  moderation model in front of free-text input and log refusals.
- **Universal stickers can't control their *source*.** We can block what a sticker
  makes a model *see*. We can't stop someone sticking a benign "toaster" sticker on
  something that matters. The remaining risk is that the source object is disrupted
  (the "correct label lost" metric). Every report shows that number so it isn't
  hidden.
- **Code is open source.** Someone determined can delete the blocklist. The policy
  raises the effort needed for misuse and states the norm clearly. It isn't a
  technical guarantee.

## Framing in the product

- The CLI prints an ethics notice before every run. The website will require users
  to read and acknowledge short guidance before they make a sticker.
- Every report ends with a *Responsible use* section, and every print sheet carries a
  short use notice.
- Reports always show **controls** (no sticker, random-noise sticker) and
  **limitations**, so the tool doesn't overstate what attacks can do.

## Stakeholders

- **Learners and educators.** They benefit from hands-on intuition about model fragility.
- **People who rely on vision systems** (drivers, people subject to content
  moderation or surveillance). They're protected by the blocklist and scope limits.
- **Model developers.** They benefit from public awareness and from defense
  demonstrations (planned; see the [roadmap](ROADMAP.md)).
- **Communities harmed by biased vision systems.** The dehumanizing-label and identity
  blocks exist specifically because of documented harms to them.

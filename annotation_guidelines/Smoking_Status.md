# Smoking Status \- Current

- Involves NER and Assertion
- Tips and good practice: <https://www.johnsnowlabs.com/tips-and-tricks-on-how-to-annotate-assertion-in-clinical-texts/>
- Article on how to achieve higher accuracy in assertion classification: <https://arxiv.org/abs/2503.17425>

# Alignment on annotations

## NER (General)

- NER Labels: **Smoking\_Status, Substance\_Quantity, Substance\_Duration, Substance\_Frequency, Smoking\_Type, Section\_Header**
- When Smoking / Tobacco is part of section header, this should be annotated appropriately as “section header” and does need any assertion
- Is “Passive Exposure” a type of smoking that we should annotate?
    -  we do not need to worry about it for this set of annotations.  This would be more for specific studying regarding second-hand smoke
- In many ambulatory notes, *patient education* is listed, these often mention recommendations/help on quitting smoking, alcoholism in the family. 
    - Do **NOT** need to annotate patient education info or patient counseling info
- “Smokeless tobacco use: “ and “Electronic Cigarette/Vaping: “
    - We will ignore these phrases for now, as they are default “fields” in the form - Do **NOT** need to annotate as NER
- Do not annotate Nicotine when listed as drug
- Do not annotate Marijuana related information
- Do not annotate the word 'history'
- DO not ignore Smoking Type words like 'cigaratte' or 'cigarattes' word in the note though it is part of another lebel like subtance frequency
- Do not annotate questionnaires
- Mind section headers
- Do not ignore smoking status word like 'smoking','tobacco' excapt words in treatment plans/discussions will be ignored (e.g. discussed with patient on tobacco cessation, patient visited for tobacco cessation).
- Substance_Frequency, Substance_Quantity and Substance_Duration must describe an explicit patient value. If the note states that the value is unknown, unavailable, undocumented, or not recorded, do not annotate that span.
- Do **NOT** annotate Substance\_Frequency or Substance\_Quantity orSubstance\_Duration when the frequency/quantity/duration described belongs to someone else (e.g. a parent, family member, or other person), not the patient — even though the associated Smoking\_Status word can still be tagged with the **Someone Else** assertion


## Assertion (General)

THE CONTEXT WINDOW FOR ASSERTION IS THE **SENTENCE**

ASSERTION LABELS ARE ASSIGNED ONLY TO **SMOKING\_STATUS **NER

- **Current smoker**: active smoker, current every day smoker, patient smokes, smokes daily, smokes occasionally
- **Former smoker**: ex-smoker, past smoker, remote smoker, distant smoker, reformed smoker, in remission, remote smoker
- **Never smoker**: lifelong non-smoker, never smoked, no history of smoking
- **Smoker current status unknown**: if unclear if still smoking, "patient has history of smoking", "History of tobacco use", “Tobacco use HX”, please note Current and Former smokers will be subgroups of this group
    - This is used when we know the historical status but not current status
- **Unknown if ever smoked**: if unclear if former or never smoker, often mentioned as "non smoker" without any other detail, please note Never smokers will be a subgroup of this group. If in one sentence patient is described as nonsmoker without any other detail, please use this label. Even if in a different sentence patient is described as a former smoker. The context for assertion is within that one sentence.
    - This is used when we know the current status but not historical status
- **Someone Else**: relates to family, household (for example "parents smoke outdoors", "family history of tobacco use")


 NER and Assertion combinations for smoking

- “Tobacco use (date): never (less than 100 cigarettes in lifetime)”
    - NER: **Smoking\_Status**:** “**Tobacco” with Assertion: **Never smoker**
    - NER: **Smoking\_Status**:** “**cigarettes with assertion  **Never smoker**
    - NER: **Substance\_Quantity: **“less than 100 cigarettes in lifetime“ without Assertion
- “Tobacco: (date); Use: Former smoker, quit more; Smokeless tobacco use: Never; Type: Cigarettes;
    - Both “Tobacco” and “smoker” annotate as **Smoking\_Status **with **Assertion Former smoker**
    - “Cigarettes” in this case annotate as **Smoking\_Type**
    - “Smokeless tobacco” ignore altogether
- “PATIENT WITH 50 PACK YEAR HX OF SMOKING”
    - NER: **Substance\_Quantity: **“50 PACK YEAR” - Assertion skipped
    - NER: **Smoking\_Status **with **Assertion Smoker current status unknown**
- “PATIENT WITH 50 YEAR PACK HX OF SMOKING”
    - NER: **Substance\_Quantity: **“50 YEAR PACK” - Assertion skipped
    - NER: **Smoking\_Status **with **Assertion Smoker current status unknown**
- "2 packs per day for 25 years" 
    - NER: **Substance\_Frequency: **“2 packs per day”
    - NER: **Substance\_Duration: **“25 years”
- “He smokes 5 to 7 cigarettes per day”
    - NER: **Smoking\_Status**: “smokes” with **Assertion Current smoker**
    - NER: **Substance\_Frequency: **“5 to 7 cigarettes per day”
- “He quit smoking over 50 years ago”
    - NER: **Smoking\_Status**: “smoking” with **Assertion Former smoker**
    - “50 years ago” is **NOT** Substance\_Duration — it describes how long ago the patient quit, not how long they smoked, so it should be left unannotated
- “NONSMOKER”
- NER: **~~Smoking\_Status~~**~~:~~**~~~~**~~“NONSMOKER”~~ updated tokenization rule: exceptionally “smoker” in the full word “nonsmoker” should be tagged
- Assertion: **Unknown if ever smoked**
- “non-smoker” or “ex-smoker”
    - NER: **Smoking\_Status**:“smoker”
    - **Unknown if ever smoked**: for “non” and  **Former smoker** for “ex”
- “Social smoker in the past”
    - NER: **Smoking\_Status**:“smoker”
    - NER: **Substance\_Frequency**: “social”
- “female patient who is heavy smoker presented with a burning sensation”
    - NER: **Smoking\_Status**: “smoker” with **Assertion Current smoker**
    - NER: **Substance\_Frequency**: “heavy”
    - “burning sensation” is a symptom, unrelated to smoking - do not annotate
- “Nicotine Polacrilex 2mg Chew 2 mg, Oral, Q2H, PRN: Nicotine Withdrawal”
    - DO NOT LABEL NICOTINE WHEN LISTED AS DRUG
- “He was a chronic smoker of 80 packet years and a social alcoholic.”
    - NER: **Smoking\_Status**: “smoker” with **Assertion Current smoker**
    - NER: **Substance\_Quantity**: “80 packet years” without Assertion
    - “alcoholic” is not smoking-related — ignore
    - **Why Current smoker and not Former smoker**: “chronic” describes an ongoing, habitual smoking behavior, not a discontinued one. The past-tense verb “was” here reflects the tense of the note/sentence (often how a clinical note narrates a patient's history), **not** that the patient quit. Do not rely on verb tense alone to decide current vs. former — look for explicit cues of cessation (“quit”, “ex-”, “former”, “in the past”) before assigning **Former smoker**. Absent such cues, “chronic smoker” should be read as **Current smoker** even when phrased in past tense
- “History of the mother was uneventful except for smoking 1-2 cigarettes per day”
    - NER: **Smoking\_Status**: “smoking” with **Assertion Someone Else** — the smoking described relates to the mother, not the patient
    - **Substance\_Frequency**: “1-2 cigarettes per day” should **NOT** be annotated — this frequency belongs to the mother's smoking, not the patient's own use. Do not label Substance\_Frequency or Substance\_Quantity when the amount/frequency described belongs to someone else

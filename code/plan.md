Your goal is to create an LLM driven filter which scans a large repository of forum posts and identifies very specific kinds of harassment and adverse behaviour. There are two kinds of behaviour to be specifically looking for "plans for harassment" and "doxing".

## 1. Definitions

Plans for harassment are defined as when an individual attempts to mobilise others online to collaborate to conduct online harassment. This definition includes discussions of possible harassment (eg. "It would be so funny to dox the target"), planning of specific acts, reporting carrying out those acts (eg "I have set up mass reports of this person's social media"), committing to specific acts (eg "I will make sure to send them hate mail when I see them online on Discord") and encouraging those acts ("eg You should go harass them!")

Doxing refers to sharing, broadcasting, or publishing personal information about an individual without their consent and with the intention to do harm. As above, this refers to both sharing that personal information (eg "Fuck it, here's their phone number 0434...), requesting that someone share their personal information (eg "Hey, can you give me their address?") or calling for their personal information to be divulged.

In both of the above scenarios, there is a specific actor either harassing, doxing, or organising and mobilising that harassment and doxing. Simply sharing negative content about the person, harmful rumours, or using slurs or toxic language does not qualify. It is essential that you note that this task is NOT the same as creating a moderation system for harmful or toxic content detection, these should be treated as false positives, we are only interested in calls to harassment and doxing.

## 2. Data

Data will come from a large directory of jsonl files. Each jsonl file represents a distinct thread from the forum that has been collected, and contains a large number of posts. See /home/ubuntu/anonymisation/sample_data/samplejsonl for an example set. The filter must make a decision for each post as to whether to flag it for inclusion of the kind of harmful content we are targeting.

## 3. Deliverables

Build a two-stage filter: the first should be a fast heuristic check to ensure that the LLM does not need to read every single post in the dataset, as this would take too long. See section 4 for details. The second must be a more nuanced LLM based filter which acts as the final classifier. It should provide structured json output, and must classify, for each post, whether it includes the kinds of harmful content defined in section 1.

## 4. Requirements

#### 4.1 Heuristic Check
Forum-scale data is mostly irrelevant; running a 27B model on every post is wasteful. Add a
cheap first pass tuned for **high recall, low precision** to discard obvious negatives:
- Lightweight signals: first-person verbs of action + person reference, presence of
  threat/stalking/contact lexicon (as a *gate*, never as the classifier), posts where the PII
  audit log flagged the author publishing another party's identifying info, etc.
- **This gate must err toward keeping** — anything it passes goes to the LLM; anything it drops
  is logged and a sample is periodically audited to measure false-negative leakage. Tune the
  gate so its recall on the synthetic positive set is ~0.95; precision doesn't matter here.

#### 4.2 The LLM 
This must be able to be run at large scale on a multi-GPU instance using an LLM for filtering. Qwen3 30B A3B is suggested due to its fast MoE architecture, and good capabilities for structured output and instruction-following. vLLM is suggested as a serving model due to its effectiveness in HPC environments. The prompt must be structured so that plain hate speech is excluded - a post can be vile, but still be classified as a "none". Err towards precision here; it may be the case that only one in a hundred posts is in fact a specific, actionable call to harassment from the poster towards an individual victim.

#### 4.3 Outputs

Outputs should be written into another jsonl file. It is imperative that if the script fails part way through, already-run outputs can easily be recovered, and it can be resumed from where it left off. You may wish to create other outputs along the way - this is permissible, but do not create excessively large files as a side effect where possible.

## 5. Directory Structure

Any and all dependencies must be installed into the environment anon_env (accessible at /home/ubuntu/anon_env). When you add dependencies, ensure that these are logged suitably in a requirements.txt file. All code should be generated within /home/ubuntu/anonymisation/vllm-harassment-filter

## 6. Ethics Notes

This pipeline surfaces content describing real-world harm and possibly criminal acts. It is a
  **triage aid for human moderators, not an autonomous decision-maker**; all data will see human annotation
- Outputs (flagged posts, evidence spans) are sensitive; these must never be logged directly to raw observability systems such as git

## 7. Claude Code Build Order 

1. Scaffold the repository and outline project plan
2. Write system prompt in detail
3. Write  `classify_llm.py`  to run prompts against a local VLLM endpoint, ensure strict JSON output and schema validation. Tests must be passed.
4. Write `ingest.py` and ensure that data can be drawn safely from sample data
5. Write `prefilter.py` + `route.py` to reduce amount of data handed to llm + `pipeline.py` for resuability

Good hunting!
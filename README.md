# CreaEval

> Decoupled Analysis-Judging: An Automated Creativity Evaluator Using LLMs in Complex Multi-step Creativity Tasks

[insert the link of the paper here]


# Motivation

Automated creativity evaluation remains highly challenging for LLM-as-a-Judge, especially in complex multi-step creativity tasks.

Existing methods mainly focus on relatively simple creativity benchmarks such as AUT and TTCT, where responses are short and structurally simple (only single-step). However, **Contextually-Grounded and Procedurally-Structured Tasks (CGPST)** introduce substantially greater challenges:

- Multiple interdependent steps
- Highly subjective evaluation dimensions
- Wide scoring ranges
- Strong contextual grounding

Directly applying typical LLM-as-a-Judge methods to such tasks often leads to unstable and biased evaluations, including verbosity bias and leniency bias, inconsistent scoring across judges, poor discrimination in subjective dimensions.

To address these limitations, we propose **CreaEval**, a decoupled analysis-judging framework for automated creativity evaluation in complex multi-step creativity tasks, such as **CGPST**.


# Method

[insert CreaEval framework figure here]

## Overview

CreaEval decomposes typical LLM-as-a-Judge into two phases: **(1) Memory-augmented Analysis** and **(2)Evidence-based Judging**

Instead of directly scoring raw responses, CreaEval first converts responses into **structured intermediate evaluation evidence** in the form of Structure-of-Thought (SoT), and then performs judging solely based on this extracted evidence without accessing raw responses. This decoupled design constrains the judging process with structured evidence, improving scoring consistency and mitigating common judging biases.

## Phase 1: Memory-augmented Analysis

In the first phase, SoT-LLM incrementally extracts structured evaluation evidence from each CGPST step. 

For each step, SoT-LLM jointly considers the future scenario, current step response, dimension descriptions within this step and cross-step memory.

### Cross-step Memory Mechanism

The steps of CGPST exhibit strong temporal dependencies, as responses in later steps are conditioned on decisions made in earlier ones. For instance, the solution proposed in Step-3 is designed to address the problem identified in Step-2. To model these dependencies, CreaEval introduces **a memory mechanism** that preserves task-relevant state across steps. Specifically, in addition to generating evidence, SoT-LLM produces
a corresponding *task_status_memory* , which summarizes the key information of the current step and is passed as contextual input to subsequent steps. This mechanism retains cross-step contextual dependencies, thereby enhancing the accuracy and consistency of evidence extraction.

---

## Phase 2: Evidence-based Judging

In the second phase, Judge-LLM performs scoring solely based on the extracted evidence by SoT-LLM, scoring rubrics and scenario information. Importantly, Judge-LLM **does not access raw responses**. This evidence-grounded judging significantly improves scoring stability and reduces subjective interference caused by raw textual responses.

Compared with typical LLM-as-a-Judge methods, the decoupled design can improves human-LLM agreement, mitigate verbosity bias and leniency bias, stabilizes scoring across different Judge-LLMs. More analysis are shown in Discussion Section of the paper.

---

** All prompts are provided in the paper appendix.  
** All experimental results, figures, flowcharts, and related materials will be released upon paper acceptance.


# Citation

```bibtex
[insert citation here after acceptance]
```

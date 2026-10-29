# Bangla Sentiment Classifier

A small, end-to-end NLP project that classifies Bangla text as **positive**, **negative**, or **neutral**.

Built as a compact, reproducible baseline — trains in seconds on CPU, runs offline, no GPU, no internet, no external Bangla NLP libraries required.

---

## Table of Contents

- [Overview](#overview)
- [Why this project](#why-this-project)
- [Features](#features)
- [Project structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Train](#1-train-the-model)
  - [Predict](#2-predict-on-new-text)
- [Example output](#example-output)
- [How it works](#how-it-works)
- [Dataset](#dataset)
- [Evaluation](#evaluation)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Tech stack](#tech-stack)
- [Resume / portfolio notes](#resume--portfolio-notes)
- [License](#license)

---

## Overview

Bangla (Bengali) is one of the most spoken languages in the world, yet it remains under-served in practical NLP tooling. This project is a deliberately small but complete sentiment classifier for Bangla, designed to demonstrate a full NLP pipeline from raw text to deployed inference — without pulling in a large pretrained model or a GPU.

It is intentionally **small enough to read in 10 minutes** and **reproducible on any laptop**.

---

## Why this project

- **Real gap.** Most accessible sentiment-classifier tutorials are English-only. Bangla needs its own text handling (Unicode normalisation, Bangla digits, agglutinative morphology, no reliable stemmer).
- **Full pipeline, not just a notebook.** Preprocessing, training, evaluation, model persistence, and a CLI are all included.
- **Fast to reproduce.** Trains in under 2 seconds. No downloads, no API keys, no GPU.
- **A baseline to extend.** If you later want to try BanglaBERT or fastText, this gives you the scaffolding and an honest reference number to beat.

---

## Features

- Bangla-aware text normalisation (Unicode NFC, Bangla → ASCII digit mapping, URL/punctuation stripping).
- Word (1–2 gram) **and** character (2–4 gram) TF-IDF features — character n-grams matter a lot for Bangla because of its rich inflectional morphology.
- Logistic Regression with balanced class weights for a strong, interpretable baseline.
- **Stratified hold-out evaluation** + **5-fold cross-validation** in one command.
- Per-class precision / recall / F1, confusion matrix, and top discriminative n-grams per class.
- Saved model (`joblib`) with a small CLI for **single-sentence** and **interactive** inference.
- Zero non-standard dependencies beyond `scikit-learn`, `pandas`, `joblib`.

---

## Project structure

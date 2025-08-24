# Play Store Preprocessing

**Project:** Play Store Apps — Data Cleaning & Preprocessing

## الوصف (Arabic)
مشروع لتنظيف ومعالجة بيانات تطبيقات متجر Google Play. يهدف المشروع إلى إصلاح القيم المفقودة، تحويل صيغ الأعمدة، التعامل مع القيم الشاذة (outliers)، وتجهيز مجموعة بيانات نظيفة للتحليل أو للنمذجة. يتضمن المشروع خطوات مفصلة لكل عمود رئيسي (Rating, Size, Price, Installs, Android version, وغيرها) ويجيب على أسئلة تحليلية مهمة مع رسوم توضيحية.

## Description (English)
This repository contains a Colab-ready Jupyter Notebook and scripts to clean and preprocess the Play Store apps dataset. The pipeline standardizes columns (Rating, Size, Price, Installs, Android version, etc.), handles missing values and outliers, and produces analysis answering multiple domain questions.

---

## What’s included / ماذا يحتوي المشروع
- `playstore_preprocessing.ipynb` — Jupyter Notebook with step-by-step cleaning, explanations, visualizations, and answers to analytical questions.  
- `playstore_preprocessing.py` — Standalone script to run the preprocessing pipeline (Colab or local).  
- `outputs/` — Contains processed CSV and saved charts after running the notebook/script.  
- `googleplaystore.csv` — Original dataset (if included).

---

## Quick Start (Colab)
1. افتح Google Colab.  
2. حمّل الملفات أو استنخِب الريبو في Colab.  
3. لتشغيل السكربت:  
```bash
!python playstore_preprocessing.py

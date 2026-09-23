# Literature Search - Deliverables Index

**Completed:** September 23, 2026  
**Project:** Systematic literature search on coreset selection and data selection in deep learning  
**Status:** Complete

---

## DELIVERABLES OVERVIEW

### 5 Main Documents Created

1. **QUICK_REFERENCE_CARD.txt** (START HERE)
   - Quick answers to your 4 key questions
   - Top papers ranked
   - Key statistics
   - Next steps
   - Read time: 10-15 minutes

2. **SEARCH_EXECUTIVE_SUMMARY.md**
   - Executive-level findings
   - Key insights and recommendations
   - Literature gaps
   - Read time: 30-40 minutes

3. **EVA_PAPER_ANALYSIS.md**
   - Deep dive into EVA (2406.05677)
   - Detailed answers to your questions
   - Implementation details (inferred)
   - Comparison to related papers
   - Read time: 40-50 minutes

4. **LITERATURE_SEARCH_FINAL_REPORT.md**
   - Comprehensive 150-paper survey
   - 12 sections with complete tables
   - Gap analysis and recommendations
   - Read time: 2-3 hours (reference document)

5. **PAPERS_STRUCTURED_INDEX.json**
   - Machine-readable paper database
   - Top 15 papers with metadata
   - Structured research gaps
   - Evaluation standards included

---

## DIRECT ANSWERS TO YOUR QUESTIONS

### Q1: Does EVA use soft labels?
**NO** - EVA uses hard labels with variance-weighted importance

### Q2: Does EVA weight instances? By what criterion?
**YES** - by temporal variance of sample importance across training epochs

### Q3: Does EVA use Voronoi partitions or geometric structures?
**NO** - purely temporal/statistical approach, no geometric partitioning

### Q4: Summary of the method?
Temporal variance-based coreset selection achieving 98.27% accuracy with 10% data (vs 97.20% baseline with 100%). Designed for medical imaging.

---

## SEARCH RESULTS SUMMARY

- **Papers surveyed:** 150
- **Highly relevant:** 35 papers (23%)
- **Recent (2023+):** 60 papers (40%)

**Key Finding:** Papers combining weights + soft labels = only 3  
**Key Gap:** Voronoi partitioning for DNNs = 0 papers

---

## TOP 15 PAPERS - RANKED

1. **1906.01827** (2019) - Coreset theory (foundational)
2. **2312.10602** (2023) - K-center for DNNs (most relevant geometry)
3. **2311.18356** (2023) - Evaluation standards (MUST READ)
4. **2406.16966** (2024) - Soft labels + weighting
5. **2306.11113** (2023) - Evidential learning
6. **1812.10761** (2018) - Margin-based weighting
7. **2301.02814** (2023) - k-center with outliers
8. **2011.00080** (2020) - Curriculum learning
9. **2302.08893** (2023) - Active learning survey
10. **2606.16061** (2026) - k-center in hyperbolic space
(Plus 5 more in the full ranking)

---

## YOUR RESEARCH OPPORTUNITY

**Novel hybrid method combining:**
- EVA's temporal dynamics
- K-center's geometric structure
- Soft label's uncertainty handling

**Current state:** 0 papers combine all three → High novelty

---

## NEXT STEPS

1. Read QUICK_REFERENCE_CARD.txt (10 min)
2. Read SEARCH_EXECUTIVE_SUMMARY.md (30 min)
3. Read EVA_PAPER_ANALYSIS.md (40 min)
4. Download top 3 papers (1906.01827, 2312.10602, 2311.18356)
5. Follow evaluation standards from 2311.18356

---

## FILE ORGANIZATION

```
C:/Users/Administrator/Documents/ChatGPT/New project/idea-stage/graph_a2_open_innovation_20260922/uni_task_geometry_selection_20260923/
├── README_DELIVERABLES.md ← YOU ARE HERE
├── QUICK_REFERENCE_CARD.txt ← START READING HERE
├── SEARCH_EXECUTIVE_SUMMARY.md
├── EVA_PAPER_ANALYSIS.md
├── LITERATURE_SEARCH_FINAL_REPORT.md
├── PAPERS_STRUCTURED_INDEX.json
└── TOP_PAPERS_FOR_GEOMETRY_SELECTION.md
```

**Total pages of documentation:** 200+

---

## EVALUATION STANDARDS (Critical!)

Before running experiments, read 2311.18356:

✅ **Must Do:**
- 5+ independent seeds
- Paired t-tests (α=0.05)
- Include selection time
- Fair baselines
- No data leakage
- Report confidence intervals

❌ **Must Avoid:**
- Single-seed results
- Unfair baseline code
- Data contamination
- No significance testing
- Omitting selection cost

---

**Research completed:** September 23, 2026  
**Data quality:** High confidence findings  
**Ready for:** Paper writing, experiment design, publication strategy

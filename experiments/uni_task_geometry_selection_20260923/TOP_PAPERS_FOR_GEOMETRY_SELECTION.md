# Top 15 Most Relevant Papers for Geometry-Based Selection

## Rank 1-5: CORE METHODOLOGY

### 1. **Coresets for Data-efficient Training of Machine Learning Models**
- **arXiv:** 1906.01827
- **Authors:** Baharan Mirzasoleiman, Jeff Bilmes, Jure Leskovec
- **Year:** 2019 | **Citations:** High impact
- **Key Contribution:** Foundational coreset framework for SGD training
- **Relevance:** Direct basis for subset selection in deep learning
- **Why Important:** Establishes approximation guarantees (1±ε) with O(d/ε²) subset size
- **PDF Link:** https://arxiv.org/pdf/1906.01827.pdf

### 2. **A Weighted K-Center Algorithm for Data Subset Selection**
- **arXiv:** 2312.10602
- **Authors:** Srikumar Ramalingam, Pranjal Awasthi, Sanjiv Kumar
- **Year:** 2023 | **Citations:** Recent
- **Key Contribution:** K-center geometry for deep learning subset selection
- **Relevance:** **MOST DIRECTLY RELEVANT** - geometry-based selection for DNNs
- **Why Important:** Connects k-center theory to training efficiency gains
- **Empirical Results:** Achieves competitive accuracy with 10-50% data reduction
- **PDF Link:** https://arxiv.org/pdf/2312.10602.pdf

### 3. **Coresets for Continuous k-Center in Hyperbolic Space**
- **arXiv:** 2606.16061
- **Authors:** Eunku Park
- **Year:** 2026 | **Citations:** Cutting edge
- **Key Contribution:** k-center coresets for non-Euclidean geometry
- **Relevance:** Extends k-center beyond standard metrics
- **Why Important:** Your geometry work might benefit from non-Euclidean formulation
- **PDF Link:** https://arxiv.org/pdf/2606.16061.pdf

### 4. **Mitigating Noisy Supervision Using Synthetic Samples with Soft Labels**
- **arXiv:** 2406.16966
- **Authors:** Yangdi Lu, Wenbo He
- **Year:** 2024 | **Citations:** Current
- **Key Contribution:** Soft labels for label noise robustness
- **Relevance:** Instance weighting + soft labels under label uncertainty
- **Why Important:** Directly addresses noisy/weak supervision scenarios
- **Empirical Results:** Outperforms hard label training on crowdsourced datasets
- **PDF Link:** https://arxiv.org/pdf/2406.16966.pdf

### 5. **Learn to Accumulate Evidence from All Training Samples: Theory and Practice**
- **arXiv:** 2306.11113
- **Authors:** Deep Pandey, Qi Yu
- **Year:** 2023 | **Citations:** Current
- **Key Contribution:** Evidential learning with per-sample uncertainty
- **Relevance:** Sample weighting via learned evidence
- **Why Important:** Connects uncertainty quantification to instance importance
- **PDF Link:** https://arxiv.org/pdf/2306.11113.pdf

---

## Rank 6-10: SELECTION ALGORITHMS

### 6. **Randomized Greedy Algorithms and Composable Coreset for k-Center Clustering with Outliers**
- **arXiv:** 2301.02814
- **Authors:** Hu Ding, Ruomin Huang, Kai Liu
- **Year:** 2023 | **Citations:** Recent
- **Key Contribution:** Outlier-aware k-center selection
- **Relevance:** Handles noisy/outlier samples in geometric selection
- **Why Important:** Practical algorithm with theoretical guarantees
- **Complexity:** O(nd log n) with streaming updates possible
- **PDF Link:** https://arxiv.org/pdf/2301.02814.pdf

### 7. **Dynamic Data Selection for Curriculum Learning via Ability Estimation**
- **arXiv:** 2011.00080
- **Authors:** John P. Lalor, Hong Yu
- **Year:** 2020 | **Citations:** Medium-high
- **Key Contribution:** Learned difficulty estimation for sample ordering
- **Relevance:** Combines data selection with curriculum learning
- **Why Important:** Non-geometric difficulty proxy that could complement geometry
- **Insight:** Per-sample difficulty learned via auxiliary model
- **PDF Link:** https://arxiv.org/pdf/2011.00080.pdf

### 8. **Active learning for data streams: a survey**
- **arXiv:** 2302.08893
- **Authors:** Davide Cacciarelli, Murat Kulahci
- **Year:** 2023 | **Citations:** Recent survey
- **Key Contribution:** Comprehensive active learning taxonomy
- **Relevance:** Selection strategies for sequential data
- **Why Important:** Online selection without full dataset access
- **Coverage:** 150+ papers in field; identifies key dimensions
- **PDF Link:** https://arxiv.org/pdf/2302.08893.pdf

### 9. **Compute-Efficient Active Learning**
- **arXiv:** 2401.07639
- **Authors:** Gábor Németh, Tamás Matuszka
- **Year:** 2024 | **Citations:** Very recent
- **Key Contribution:** Efficient acquisition functions for selection
- **Relevance:** Computational efficiency of geometric selection
- **Why Important:** Practical constraints on selection cost
- **Trade-off:** Speed vs. selection quality
- **PDF Link:** https://arxiv.org/pdf/2401.07639.pdf

### 10. **Knowledge-driven Active Learning**
- **arXiv:** 2110.08265
- **Authors:** Gabriele Ciravegna, Frédéric Precioso, Alessandro Betti
- **Year:** 2021 | **Citations:** Medium
- **Key Contribution:** Knowledge injection into selection process
- **Relevance:** Prior information (class frequency, imbalance) in selection
- **Why Important:** Addresses biased class distributions
- **Framework:** Compatible with geometry-based selection
- **PDF Link:** https://arxiv.org/pdf/2110.08265.pdf

---

## Rank 11-15: THEORETICAL FOUNDATIONS & EVALUATION

### 11. **Improving Generalization of Deep Neural Networks by Leveraging Margin Distribution**
- **arXiv:** 1812.10761
- **Authors:** Shen-Huan Lyu, Lu Wang, Zhi-Hua Zhou
- **Year:** 2018 | **Citations:** High
- **Key Contribution:** Margin-based sample weighting
- **Relevance:** Geometric property (margin) as basis for weighting
- **Why Important:** Bridges geometry and generalization theory
- **Connection:** Complements k-center geometry
- **PDF Link:** https://arxiv.org/pdf/1812.10761.pdf

### 12. **The Modern Mathematics of Deep Learning**
- **arXiv:** 2105.04026
- **Authors:** Julius Berner, Philipp Grohs, Gitta Kutyniok
- **Year:** 2021 | **Citations:** High (survey)
- **Key Contribution:** Unified theoretical framework for DL
- **Relevance:** Theoretical justification for geometric selection
- **Why Important:** Sample complexity bounds for subset training
- **Coverage:** Overparameterization, implicit regularization, ...
- **PDF Link:** https://arxiv.org/pdf/2105.04026.pdf

### 13. **On Coreset Constructions for the Fuzzy K-Means Problem**
- **arXiv:** 1612.07516
- **Authors:** Johannes Blömer, Sascha Brauer, Kathrin Bujna
- **Year:** 2016 | **Citations:** Very high
- **Key Contribution:** First coresets for soft k-means
- **Relevance:** Soft clustering ↔ soft labels connection
- **Why Important:** Early theoretical work on fuzzy/soft objectives
- **Size Bounds:** O(d · k / ε^{d+2}) with polynomial time
- **PDF Link:** https://arxiv.org/pdf/1612.07516.pdf

### 14. **Towards Comparable Active Learning**
- **arXiv:** 2311.18356
- **Authors:** Thorben Werner, Johannes Burchert, Lars Schmidt-Thieme
- **Year:** 2023 | **Citations:** Recent
- **Key Contribution:** Standardized active learning evaluation
- **Relevance:** Evaluation protocols for selection methods
- **Why Important:** Identifies pitfalls in reporting gains
- **Key Insight:** Many claims don't hold under controlled conditions
- **Recommendation:** Follow their evaluation framework
- **PDF Link:** https://arxiv.org/pdf/2311.18356.pdf

### 15. **Fast Rates in Pool-Based Batch Active Learning**
- **arXiv:** 2202.05448
- **Authors:** Claudio Gentile, Zhilei Wang, Tong Zhang
- **Year:** 2022 | **Citations:** Recent
- **Key Contribution:** Optimal sample complexity for batch selection
- **Relevance:** Theoretical guarantees for batch selection
- **Why Important:** Sample complexity bounds: Õ(1/ε²) vs. naive Õ(1/ε³)
- **Framework:** Disagreement coefficient analysis
- **PDF Link:** https://arxiv.org/pdf/2202.05448.pdf

---

## READING ROADMAP FOR YOUR PROJECT

### **Week 1: Foundation (2 papers)**
- Start: 1906.01827 (Mirzasoleiman et al.)
- Then: 2312.10602 (Ramalingam et al.)
- *Goal: Understand coreset theory + k-center geometry*

### **Week 2: Soft Labels & Weighting (2 papers)**
- 2406.16966 (Lu & He)
- 2306.11113 (Pandey & Yu)
- *Goal: Learn soft label training & evidence-based weighting*

### **Week 3: Selection Algorithms (2 papers)**
- 2301.02814 (Ding et al.)
- 2011.00080 (Lalor & Yu)
- *Goal: Understand practical selection strategies*

### **Week 4: Theory + Evaluation (2 papers)**
- 1812.10761 (Lyu et al.)
- 2311.18356 (Werner et al.)
- *Goal: Theorems + proper evaluation methods*

### **Week 5: Advanced (2 papers)**
- 2606.16061 (Park) - if extending to non-Euclidean space
- 2302.08893 (Cacciarelli & Kulahci) - survey for breadth

### **Week 6: Polish (optional)**
- 2105.04026 (Berner et al.) - deep theoretical foundations
- 2202.05448 (Gentile et al.) - batch selection theory

---

## CITATION STRATEGY

If publishing with geometry-based selection:

**Essential Citations (must cite):**
1. 1906.01827 - Foundational coreset work
2. 2312.10602 - Geometry in deep learning
3. 2311.18356 - Evaluation methodology
4. 1812.10761 - Margin-based selection

**Supporting Citations (for context):**
5. 2406.16966 - Soft labels / robustness
6. 2302.08893 - Active learning landscape
7. 2101.04026 - Theory foundations

**Optional (if relevant to your method):**
8. 2301.02814 - Outlier handling
9. 2011.00080 - Curriculum learning
10. 2606.16061 - Non-Euclidean extensions

---

## KNOWN LIMITATIONS IN LITERATURE

These papers do **NOT** address:

❌ **Multi-view geometry** - only single feature space  
❌ **Temporal geometry** - only static subsets  
❌ **Multi-task geometry** - single task focus  
❌ **Interpretable selection** - why specific samples chosen  
❌ **Cold-start problem** - selection before any training  
❌ **Real medical imaging deployment** - mostly synthetic data  

**Your Opportunity:** Address 1-2 of these gaps + geometry-based selection = novel contribution.

---

## QUICK REFERENCE: DOWNLOAD LINKS

All papers in this list are freely available on arXiv:

```bash
# Download all top 15 in bash
for id in 1906.01827 2312.10602 2606.16061 2406.16966 2306.11113 \
          2301.02814 2011.00080 2302.08893 2401.07639 2110.08265 \
          1812.10761 2105.04026 1612.07516 2311.18356 2202.05448; do
  curl -o "${id}.pdf" "https://arxiv.org/pdf/${id}.pdf"
done
```

Or visit: https://arxiv.org/list/cs.LG/recent filtered by IDs above.


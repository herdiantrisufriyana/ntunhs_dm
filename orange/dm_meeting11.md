# Hands-on with Orange: Hypothesis Generation

**Herdiantri Sufriyana**
Graduate Institute of Artificial Intelligence and Big Data in Healthcare
National Taiwan University of Nursing and Health Sciences

---

## Table of Contents

1. [Subtopics](#subtopics)
2. [What is Orange?](#what-is-orange)
3. [Step 1: Load and Explore Data (Meeting 02)](#step-1-load-and-explore-data-meeting-02)
4. [Step 2: Split Data (Meeting 01)](#step-2-split-data-meeting-01)
5. [Step 3: Hypothesis Generation by Data Visualization (Meeting 03)](#step-3-hypothesis-generation-by-data-visualization-meeting-03)
6. [Step 4: Hypothesis Generation by PCA and Clustering (Meeting 04)](#step-4-hypothesis-generation-by-pca-and-clustering-meeting-04)
7. [Step 5: Hypothesis Generation by Correlation Matrix (Meeting 05)](#step-5-hypothesis-generation-by-correlation-matrix-meeting-05) — 5a. Correlation Matrix, 5b. Correlation Network
8. [Step 6: Ontology Inference (Meeting 09)](#step-6-ontology-inference-meeting-09)

---

## Subtopics

- Loading and exploring data
- Splitting data for hypothesis generation and testing
- Hypothesis generation by data visualization
- Hypothesis generation by PCA and clustering
- Hypothesis generation by correlation matrix and network
- Ontology inference via variable clustering

[Back to Table of Contents](#table-of-contents)

---

## What is Orange?

- A **visual programming** tool for data mining and machine learning
- Build workflows by connecting **widgets** (drag-and-drop blocks)
- Each widget performs one task (load data, visualize, cluster, etc.)
- Connections between widgets define the data flow

**Install:** https://orangedatamining.com/download/

[Back to Table of Contents](#table-of-contents)

---

## Step 1: Load and Explore Data (Meeting 02)

1. Drag **File** widget onto the canvas → load your dataset (.csv)
2. In the **File** widget:
   - Rename your outcome column to **outcome**
   - Set its role to **Target**
   - Set predictor columns to **Feature**
   - Set ID or irrelevant columns to **Meta**
   - Click **Apply**
3. Connect **File** → **Data Table** (label it "Raw data") → inspect rows, columns, values
4. Connect **Data Table** → **Column Statistics** (label it "Raw data summary") via **Selected Data → Data** connection

**Check these:**
- How many samples (rows) and features (columns)?
- Which column is the outcome?
- Are there missing values?
- What are the variable types (numeric, categorical)?

> **Recall Meeting 02:** Variable types matter — categorical variables should be coded as numbers (e.g., 0/1). Setting id as meta keeps it visible but excluded from analysis.

[Back to Table of Contents](#table-of-contents)

---

## Step 2: Split Data (Meeting 01)

**Widgets:** File (x2), Data Sampler (x2), Data Table (x4)

**2a. Stratified partition (for EDA in Steps 3–4)**
1. Rename the **File** widget from Step 1 to **"File (stratified)"**
2. Connect **File (stratified)** → **Data Sampler** (label it "Data partition (stratified)") — set 20%, check **Stratify**
3. Connect **Data Sample → Data** output → **Data Table** (label it "Hypothesis generation (stratified)")

**2b. Non-stratified partition (for correlation analysis in Steps 5–6)**
1. Drag a second **File** widget (label it "File") → load the same dataset
2. Connect **File** → **Data Sampler** (label it "Data partition") — set 20%, check **Stratify**
3. Connect **Data Sample → Data** output → **Data Table** (label it "Hypothesis generation") — used for Steps 5–6 (outcome included as feature), and **Data Table** (label it "Hypothesis testing (stratified)") — set aside
4. Connect **Remaining Data → Data** output → **Data Table** (label it "Hypothesis testing") — set aside

| Partition | Source | % of total | Use |
|-----------|--------|-----------|-----|
| Hypothesis generation (stratified) | File (stratified) | 20% | EDA + pattern discovery (Steps 3–4) |
| Hypothesis generation | File | 20% | Correlation analysis (Steps 5–6), outcome included |
| Hypothesis testing (stratified) | File (stratified) | 20% | Hypothesis testing only |
| Hypothesis testing | File | 80% | Hypothesis testing only |

> **Recall Meeting 01:** Hypothesis generation and hypothesis testing should use different partitions of the data. The hypothesis generation set is used for EDA and pattern discovery. The hypothesis testing set is set aside and only used later for regression analysis.

[Back to Table of Contents](#table-of-contents)

---

## Step 3: Hypothesis Generation by Data Visualization (Meeting 03)

**Widgets:** Distributions, Scatter Plot, Box Plot, Violin Plot

All visualizations are connected from **Hypothesis generation (stratified)** (Data Table) via **Selected Data → Data**, and all are split by **outcome**.

**3a. Histogram**
1. Connect **Hypothesis generation (stratified)** → **Distributions** via **Selected Data → Data**
2. Select a numeric variable from the dropdown
3. Histograms are automatically split by **outcome** (target variable)

**3b. Scatter Plot**
1. Connect **Hypothesis generation (stratified)** → **Scatter Plot** via **Selected Data → Data**
2. Set x-axis and y-axis to two numeric predictors
3. Set **Color** to **outcome**
4. Check **Show regression line** to see the trend
5. Look for separation between outcome groups and outliers

**3c. Box Plot**
1. Connect **Hypothesis generation (stratified)** → **Box Plot** via **Selected Data → Data**
2. Set **Variable** to a numeric predictor
3. Set **Group by** to **outcome**
4. Compare medians, spread, and outliers between groups

**3d. Violin Plot**
1. Connect **Hypothesis generation (stratified)** → **Violin Plot** via **Selected Data → Data**
2. Set **Variable** to a numeric predictor
3. Set **Group by** to **outcome**
4. Compare distribution shapes between groups — skew is more visible than in box plots

**3e. Stacked Bar Chart**
1. Connect **Hypothesis generation (stratified)** → **Distributions** (a second one) via **Selected Data → Data**
2. Select a **categorical** variable from the dropdown
3. Check **Stack columns**
4. The bars are automatically split by **outcome** — disproportionate stacking suggests association

> **Recall Meeting 03:** Each visualization type reveals different patterns. Histograms show distribution shape, scatter plots show correlations and outliers, box plots compare medians across groups, violin plots show distribution shape differences, and stacked bar charts show categorical-to-categorical relationships. All are tools for generating hypotheses.

[Back to Table of Contents](#table-of-contents)

---

## Step 4: Hypothesis Generation by PCA and Clustering (Meeting 04)

**Widgets:** PCA, Distances, Hierarchical Clustering, k-Means, Scatter Plot (x3)

**4a. PCA**
1. Connect **Hypothesis generation (stratified)** → **PCA** via **Selected Data → Data**
2. Set **Components** to **2**
3. Connect **PCA** → **Scatter Plot** (label it "Score plot by outcome") via **Data**
   - Set x-axis: **PC1**, y-axis: **PC2**
   - Color by: **outcome**
   - Do outcome groups separate in the reduced space?

**4b. Hierarchical Clustering**
1. Connect **PCA** → **Distances** (label it "Sample distances") via **Data**
2. Connect **Distances** → **Hierarchical Clustering**
3. Set height ratio to **50%** to cut the dendrogram
4. Connect **Hierarchical Clustering** → **Scatter Plot** (label it "Score plot by hierarchical cluster") via **Selected Data → Data**
   - Set x-axis: **PC1**, y-axis: **PC2**
   - Color by: **Cluster**
   - Compare: do hierarchical clusters match outcome groups?

**4c. k-Means**
1. Connect **PCA** → **k-Means** via **Data**
2. Set **From** to **2**, **To** to **10**
3. Orange will evaluate silhouette scores for each k — pick the best
4. Connect **k-Means** → **Scatter Plot** (label it "Score plot by spatial cluster") via **Data**
   - Set x-axis: **PC1**, y-axis: **PC2**
   - Color by: **Cluster**
   - Compare: do k-means clusters match outcome groups?

> **Recall Meeting 04:** PCA projects high-dimensional data into fewer dimensions that preserve the most variation. Clustering is unsupervised — it finds groups without knowing the outcome. If clusters align with outcome groups in PC1-PC2 space, the predictors collectively contain information about the outcome.

[Back to Table of Contents](#table-of-contents)

---

## Step 5: Hypothesis Generation by Correlation Matrix (Meeting 05)

**Widgets:** Preprocess, Correlations, Select Rows, Data Table, Transpose, Distances, Network From Distances, Network Explorer

This step uses the non-stratified **Hypothesis generation** (from Step 2b), so the outcome variable is included as a feature.

1. Connect **Hypothesis generation** → **Preprocess** (label it "Continuize & standardize") via **Selected Data → Data**
   - Add **Continuize** — set most frequent as base
   - Add **Standardize**

**5a. Correlation Matrix**

2. Connect **Continuize & standardize** → **Correlations** via **Preprocessed Data → Data**
3. View pairwise correlation coefficients between all features (including outcome)
4. Connect **Correlations** → **Select Rows** (label it "FDR <= 0.05") via **Correlations → Data**
   - Add condition: **FDR** ≤ **0.05**
5. Connect **Select Rows** → **Data Table** (label it "Significant correlations") via **Matching Data → Data**
6. Inspect the significant correlations — look for:
   - **High correlations** between predictors (collinearity)
   - **Moderate correlations** with the outcome — potential predictors
   - **Unexpected correlations** — might suggest confounding
7. Note the number of significant correlations — use this for Step 5b

**5b. Correlation Network**

8. Connect **Continuize & standardize** → **Transpose** via **Preprocessed Data → Data**
9. Connect **Transpose** → **Distances** (label it "Feature distances") via **Data**
   - Set distance metric to **Pearson**
10. Connect **Feature distances** → **Network From Distances** via **Distances**
    - Set the number of edges to match the number of significant correlations from Step 5a
11. Connect **Network From Distances** → **Network Explorer** via **Network**
    - Label nodes by **feature name**
    - Visually inspect which variables cluster together

> **Recall Meeting 05:** Correlation does not imply causation — a significant correlation between X and Y may be due to a confounder C. This is why we need causal graphs (Meeting 06) to decide which variables to adjust for. Also recall the Simpson's paradox: an association can reverse after adjustment.

[Back to Table of Contents](#table-of-contents)

---

## Step 6: Ontology Inference (Meeting 09)

**Widget:** Hierarchical Clustering

This step clusters **variables** (not samples) to discover which predictors belong together. Uses the same **Feature distances** from Step 5b.

1. Connect **Feature distances** → **Hierarchical Clustering** (label it "Ontology inference") via **Distances**
2. Set height ratio to **20%**
3. View the dendrogram — which variables cluster together?

> **Recall Meeting 09:** Ontology inference groups correlated variables using hierarchical clustering on transposed data. Variables within the same cluster can be combined via PCA (PC1) to reduce the number of predictors — critical when sample size is limited (Meeting 10).

[Back to Table of Contents](#table-of-contents)

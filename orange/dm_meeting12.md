# Hands-on with Orange: Hypothesis Testing

**Herdiantri Sufriyana**
Graduate Institute of Artificial Intelligence and Big Data in Healthcare
National Taiwan University of Nursing and Health Sciences

---

## Table of Contents

1. [Subtopics](#subtopics)
2. [Prerequisites](#prerequisites)
3. [Step 7: Identify Statistical Confounders (Meeting 06)](#step-7-identify-statistical-confounders-meeting-06) — 7a. Correlations on hypothesis testing data, 7b. Variables correlated with VOI, 7c. Variables correlated with VOI2, 7d. Find matching confounders, 7e. Draw causal graph
4. [Step 8: Determine Causal Directions and Covariates (Meeting 08)](#step-8-determine-causal-directions-and-covariates-meeting-08)
5. [Step 9: Hypothesis Testing by Regression Analysis (Meeting 10)](#step-9-hypothesis-testing-by-regression-analysis-meeting-10) — 9a. Univariate, 9b. Multivariate, 9c. Compare

---

## Subtopics

- Identifying statistical confounders from correlation evidence
- Determining causal directions and covariates using domain knowledge
- Hypothesis testing by univariate and multivariate regression
- Comparing unadjusted and adjusted odds ratios

[Back to Table of Contents](#table-of-contents)

---

## Prerequisites

This hands-on continues from **Week 11** (Hands-on with Orange: Hypothesis Generation). Open your Week 11 workflow — you will need:

- **Hypothesis testing** (Data Table from Step 2b) — 80% partition, set aside for this session

Choose your two **variables of interest (VOI)**:
- **VOI** — the exposure variable (predictor of interest)
- **VOI2** — the outcome variable

[Back to Table of Contents](#table-of-contents)

---

## Step 7: Identify Statistical Confounders (Meeting 06)

**Widgets:** Preprocess, Correlations, Select Rows (x5), Data Table (x2), Select Columns (x4), Edit Domain (x4), Concatenate (x2), Merge Data

A statistical confounder is a variable significantly correlated with **both** VOI and VOI2. This step finds those variables by running correlations on the hypothesis testing data and finding matching pairs.

**7a. Correlations on hypothesis testing data**

1. Copy and paste the **Continuize & standardize**, **Correlations**, and **FDR filter** (Select Rows) widgets from the hypothesis generation pipeline
2. Connect **Hypothesis testing** → **Continuize & standardize (1)** via **Selected Data → Data**
   - The rest of the copied chain reconnects automatically
3. Change the filter condition to **p-value** ≤ **0.05**
4. Connect the filter → **Data Table** (label it "Significant correlations (1)") via **Matching Data → Data**

**7b. Variables correlated with VOI**

Copy and paste the Select Rows → Select Columns → Edit Domain chain twice, once for each Feature column:

5. Connect **Significant correlations (1)** → **Select Rows** (label it "Select VOI in Feature 1") via **Selected Data → Data**
   - Add condition: **Feature 1** is equal to **VOI**
6. Connect → **Select Columns** (label it "Select Feature 2") via **Matching Data → Data**
   - Keep only the **Feature 2** column
7. Connect → **Edit Domain** (label it "Rename Feature 2 to Feature") via **Data**
   - Rename **Feature 2** → **Feature**
8. Connect **Significant correlations (1)** → **Select Rows** (label it "Select VOI in Feature 2") via **Selected Data → Data**
   - Add condition: **Feature 2** is equal to **VOI**
9. Connect → **Select Columns** (label it "Select Feature 1") via **Matching Data → Data**
   - Keep only the **Feature 1** column
10. Connect → **Edit Domain** (label it "Rename Feature 1 to Feature") via **Data**
    - Rename **Feature 1** → **Feature**
11. Connect **Rename Feature 2 to Feature** → **Concatenate** via **Data → Additional Data**
12. Connect **Rename Feature 1 to Feature** → **Concatenate** via **Data → Additional Data**
    - Check **"Treat variables with the same name as the same variable"**

**7c. Variables correlated with VOI2**

Copy and paste the same chain from 7b, replacing VOI with VOI2:

13. Connect **Significant correlations (1)** → **Select Rows** (label it "Select VOI2 in Feature 1") — condition: **Feature 1** is equal to **VOI2**
14. → **Select Columns** (label it "Select Feature 2 (1)") — keep only **Feature 2**
15. → **Edit Domain** (label it "Rename Feature 2 to Feature (1)") — rename **Feature 2** → **Feature**
16. Connect **Significant correlations (1)** → **Select Rows** (label it "Select VOI2 in Feature 2") — condition: **Feature 2** is equal to **VOI2**
17. → **Select Columns** (label it "Select Feature 1 (1)") — keep only **Feature 1**
18. → **Edit Domain** (label it "Rename Feature 1 to Feature (1)") — rename **Feature 1** → **Feature**
19. Connect both Edit Domain outputs → **Concatenate (1)** via **Data → Additional Data**
    - Check **"Treat variables with the same name as the same variable"**

**7d. Find matching confounders**

20. Connect **Concatenate** → **Merge Data** (label it "Identify statistical confounders") via **Data**
21. Connect **Concatenate (1)** → **Merge Data** via **Data → Extra Data**
    - Select **Find matching pairs of rows**
    - Row matching: **Feature** matches **Feature**
22. Connect **Merge Data** → **Data Table** (label it "Statistical confounders") via **Data**

**7e. Classify relationships and draw causal graph**

23. Open the **Statistical confounders** table — these are variables significantly correlated with both VOI and VOI2
24. For each confounder, classify using **domain knowledge**:
    - **Confounder**: C → VOI and C → VOI2 (common cause — must adjust)
    - **Mediator**: VOI → M → VOI2 (on the causal path — do NOT adjust for total effect)
    - **Collider**: VOI → C ← VOI2 (common effect — do NOT adjust)
25. On paper or whiteboard, draw the DAG:
    - **VOI** (exposure) and **VOI2** (outcome) connected by an arrow
    - Add all identified confounders, mediators, and colliders with their arrows

> **Recall Meeting 06:** A causal graph (DAG) encodes assumptions about which variables cause which. The d-separation rules determine what to adjust for: adjusting for a confounder removes bias, adjusting for a collider introduces bias, and adjusting for a mediator blocks the indirect causal path. Always construct the causal graph BEFORE deciding covariates — never use p-value-based variable selection. Remember the Berkeley admission example: an association can reverse after adjustment (Simpson's paradox).

[Back to Table of Contents](#table-of-contents)

---

## Step 8: Determine Causal Directions and Covariates (Meeting 08)

**No new widgets** — this step is done on paper using the statistical confounders from Step 7d and domain knowledge.

1. For each variable in the **Statistical confounders** table, determine the causal direction relative to VOI and VOI2 using domain knowledge
2. Update your DAG from Step 7e with the determined directions
3. Apply d-separation rules to decide covariates:
   - **Include** all confounders (block backdoor paths)
   - **Exclude** all colliders (adjusting opens spurious paths)
   - **Decide** on mediators: exclude for total effect, include for direct effect
4. Write your regression equations:
   - Total effect: VOI2 ~ VOI + C₁ + C₂ + ...
   - Direct effect: VOI2 ~ VOI + C₁ + C₂ + ... + M

> **Recall Meeting 08:** Determining causal directions requires domain knowledge or LLM assistance. For each variable pair, ask: which is more likely to cause the other? The causal graph must be finalized before selecting covariates for regression.

[Back to Table of Contents](#table-of-contents)

---

## Step 9: Hypothesis Testing by Regression Analysis (Meeting 10)

**Widgets:** Select Columns (x4), Logistic Regression (x2), Data Table (x5), Formula (x2), Merge Data

This step uses the **Hypothesis testing** partition (80%, from Step 2b).

**9a. Univariate regression**

1. Connect **Hypothesis testing** → **Select Columns** (label it "Select VOI2 with VOI as target") via **Selected Data → Data**
   - Move **VOI** to **Features**
   - Move **VOI2** to **Target**
   - Move all other variables to **Meta**
2. Drag **Logistic Regression** (label it "Univariate regression analysis") onto the canvas
   - Set **Regularization** to **No regularization**
3. Connect **Select Columns** → **Logistic Regression** via **Data**
4. Connect **Logistic Regression** → **Data Table** (label it "Unadjusted estimate") via **Coefficients → Data**
5. Connect **Unadjusted estimate** → **Formula** (label it "Compute unadjusted OR") via **Selected Data → Data**
   - Add new feature: **OR = exp(coefficient)**
6. Connect **Formula** → **Data Table** (label it "Unadjusted OR") via **Data**
7. Connect **Unadjusted OR** → **Select Columns** (label it "Select unadjusted OR") via **Selected Data → Data**

**9b. Multivariate regression (guided by causal graph)**

8. Connect **Hypothesis testing** → **Select Columns** (label it "Select VOI2 and covariates with VOI as target") via **Selected Data → Data**
   - Move **VOI** and all covariates from Step 8 to **Features**
   - Move **VOI2** to **Target**
   - Move remaining variables to **Meta**
9. Drag **Logistic Regression** (label it "Multivariate regression analysis") onto the canvas
   - Set **Regularization** to **No regularization**
10. Connect **Select Columns** → **Logistic Regression** via **Data**
11. Connect **Logistic Regression** → **Data Table** (label it "Adjusted estimate") via **Coefficients → Data**
12. Connect **Adjusted estimate** → **Formula** (label it "Compute adjusted OR") via **Selected Data → Data**
    - Add new feature: **OR = exp(coefficient)**
13. Connect **Formula** → **Data Table** (label it "Adjusted OR") via **Data**
14. Connect **Adjusted OR** → **Select Columns** (label it "Select adjusted OR") via **Selected Data → Data**

**9c. Compare unadjusted and adjusted OR**

15. Connect **Select unadjusted OR** → **Merge Data** (label it "Inner join unadjusted and adjusted OR") via **Data**
16. Connect **Select adjusted OR** → **Merge Data** via **Data → Extra Data**
    - Select **Find matching pairs of rows**
17. Connect **Merge Data** → **Data Table** (label it "Unadjusted and adjusted OR") via **Data**
18. Compare the OR for VOI across models:
    - Did OR change direction (cross 1.0)? → Simpson's paradox
    - Did it move closer to 1.0? → Confounders explained the association
    - Did it move further from 1.0? → Confounders were masking the effect

> **Recall Meeting 10:** Regression analysis tests hypotheses, but the causal graph decides what to adjust for — never use stepwise or p-value-based variable selection. The coefficient from logistic regression is log-odds; exponentiate to get the odds ratio (OR). Comparing unadjusted and adjusted OR reveals whether confounders inflate, mask, or reverse the association.

[Back to Table of Contents](#table-of-contents)

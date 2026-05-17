# Hands-on with Orange: Text Mining & Image Analytics

**Herdiantri Sufriyana**
Graduate Institute of Artificial Intelligence and Big Data in Healthcare
National Taiwan University of Nursing and Health Sciences

---

## Table of Contents

1. [Subtopics](#subtopics)
2. [Prerequisites](#prerequisites)
3. [Step 10: Text Mining](#step-10-text-mining) — 10a. Load and preprocess text, 10b. Sentiment analysis, 10c. Topic modeling
4. [Step 11: Image Analytics](#step-11-image-analytics) — 11a. Load and embed images, 11b. Cluster and classify images

---

## Subtopics

- Text preprocessing and visualization
- Sentiment analysis and topic modeling
- Image embedding and similarity
- Image clustering and classification

[Back to Table of Contents](#table-of-contents)

---

## Prerequisites

Install the required add-ons before this session:
1. Open Orange → **Options** → **Add-ons**
2. Check **Orange3-Text** and **Orange3-ImageAnalytics**
3. Click **OK** and restart Orange

Prepare data:
- A **.csv** file with at least one text column (e.g., clinical notes, survey responses, article abstracts)
- A folder of images organized into subfolders by class label (e.g., `images/cats/`, `images/dogs/`)

[Back to Table of Contents](#table-of-contents)

---

## Step 10: Text Mining

**10a. Load and preprocess text**

**Widgets:** Corpus, Preprocess Text, Corpus Viewer, Word Cloud

1. Drag **Corpus** onto the canvas → load your .csv file with a text column
   - Select the column containing text as the **Text** field
   - Set any label/category column as **Class**
2. Connect **Corpus** → **Preprocess Text** via **Corpus**
   - Add transformations in order:
     - **Lowercase**
     - **Tokenize** — select **Word & Punctuation**
     - **Filter** — remove **Stopwords** (select language)
     - **Filter** — by **Document Frequency** (min: 2)
3. Connect **Preprocess Text** → **Corpus Viewer** via **Corpus** — inspect individual documents after preprocessing
4. Connect **Preprocess Text** → **Word Cloud** via **Corpus** — visualize the most frequent terms

**Check these:**
- Which terms dominate the word cloud?
- Did stopword removal clean up meaningless terms?
- Are there domain-specific stopwords you should add?

**10b. Sentiment analysis**

**Widgets:** Sentiment Analysis, Data Table, Distributions

5. Connect **Preprocess Text** → **Sentiment Analysis** via **Corpus**
   - Select method: **Vader** (works well for English text)
6. Connect **Sentiment Analysis** → **Data Table** via **Corpus** — inspect sentiment scores per document
   - Vader outputs: **neg**, **neu**, **pos**, and **compound** scores
7. Connect **Sentiment Analysis** → **Distributions** via **Corpus**
   - Select the **compound** variable
   - If a class variable is set, distributions are split by class automatically

**Check these:**
- What is the overall sentiment distribution?
- Do different categories (classes) differ in sentiment?
- Are there outlier documents with extreme sentiment?

**10c. Topic modeling**

**Widgets:** Bag of Words, Topic Modelling, Data Table, Word Cloud

8. Connect **Preprocess Text** → **Bag of Words** via **Corpus**
   - Leave defaults (term frequency)
9. Connect **Bag of Words** → **Topic Modelling** via **Corpus**
   - Set method: **Latent Dirichlet Allocation (LDA)**
   - Set **Number of topics**: start with **3**
10. Connect **Topic Modelling** → **Data Table** via **Corpus** — each document is assigned topic probabilities
11. Connect **Topic Modelling** → **Word Cloud** via **All Topics** — view keywords per topic
    - Use the dropdown to switch between topics
12. Adjust the number of topics (e.g., 3 → 5) and compare — do the topics become more specific or more fragmented?

**Check these:**
- Can you interpret each topic based on its top keywords?
- Do the topic assignments match the class labels?
- What happens when you increase or decrease the number of topics?

[Back to Table of Contents](#table-of-contents)

---

## Step 11: Image Analytics

**11a. Load and embed images**

**Widgets:** Import Images, Image Viewer, Image Embedding, Image Grid

1. Drag **Import Images** onto the canvas → point to your image folder
   - Subfolders become class labels automatically
2. Connect **Import Images** → **Image Viewer** via **Data** — browse the loaded images
3. Connect **Import Images** → **Image Embedding** via **Data**
   - Select embedder: **SqueezeNet** (fast, good for most tasks)
   - Wait for embedding to complete (progress bar in the widget)
4. Connect **Image Embedding** → **Image Grid** via **Data**
   - Images are arranged by similarity — similar images cluster together

**Check these:**
- How many images per class were loaded?
- In the Image Grid, do images of the same class appear near each other?
- Are there any obvious misplacements?

**11b. Cluster and classify images**

**Widgets:** Distances, Hierarchical Clustering, Image Viewer, Test & Score, Logistic Regression, Confusion Matrix

*Clustering:*

5. Connect **Image Embedding** → **Distances** via **Data**
6. Connect **Distances** → **Hierarchical Clustering** via **Distances**
   - Cut the dendrogram to form clusters
7. Connect **Hierarchical Clustering** → **Image Viewer** via **Selected Data → Data**
   - Click different clusters in the dendrogram to see which images belong to each

*Classification:*

8. Connect **Image Embedding** → **Test & Score** via **Data**
9. Drag **Logistic Regression** onto the canvas
   - Set **Regularization** to **No regularization**
10. Connect **Logistic Regression** → **Test & Score** via **Learner**
    - Set evaluation to **5-Fold Cross Validation**
    - Note the **AUC** and **CA** (classification accuracy)
11. Connect **Test & Score** → **Confusion Matrix** via **Evaluation Results**
    - Which classes are most often confused?
12. Try other classifiers: drag **kNN** and **Tree** onto the canvas, connect each to **Test & Score** via **Learner** — compare which performs best

**Check these:**
- Do hierarchical clusters correspond to the class labels?
- Which classifier achieves the highest AUC?
- Which class pairs are hardest to distinguish?

[Back to Table of Contents](#table-of-contents)

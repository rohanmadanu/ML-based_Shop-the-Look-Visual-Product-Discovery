# Hybrid Cascade Pipeline for Visual Product Discovery

## Overview

The objective of this project is to solve the **Shop-the-Look** problem:

> Given an unstructured lifestyle scene image containing a fashion item, automatically locate the item, extract its visual characteristics, and retrieve identical or highly relevant products from a structured e-commerce catalog.

Instead of training a large end-to-end object detection and retrieval model from scratch, this system employs a **Two-Stage Hybrid Cascade Architecture** that combines:

* Deep Learning Foundation Models
* Deterministic Heuristic Filtering

This design enables:

* Zero-shot product retrieval
* High color consistency
* Reduced training requirements
* Faster deployment
* Improved interpretability

---

# System Architecture

```text
Input Scene Image
        │
        ▼
┌─────────────────┐
│     YOLOv8      │
│ Object Detection│
└─────────────────┘
        │
        ▼
┌─────────────────┐
│  Color Filter   │
│ RGB Analysis    │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│   OpenAI CLIP   │
│ Feature Encoder │
└─────────────────┘
        │
        ▼
Recommended Product
```

---

# Model Architecture & Methodology

The retrieval pipeline consists of three major stages:

## 1. Object Localization Layer (YOLOv8)

Lifestyle images often contain:

* Background clutter
* Furniture
* Multiple people
* Environmental distractions

Using global image embeddings directly can significantly reduce retrieval accuracy.

To isolate the target fashion item, the system uses:

**Model:** `yolov8n.pt`

### Process

1. Run YOLOv8 inference on the scene image.
2. Extract bounding box coordinates.
3. Crop the detected apparel region.
4. Forward the cropped image to downstream modules.

### Benefits

* Removes irrelevant background information.
* Improves feature quality.
* Reduces embedding noise.

---

## 2. Attribute Alignment Layer (Deterministic Color Cascade)

Foundation vision models often prioritize:

* Item category
* Shape
* Style

over exact color matching.

To enforce color consistency, a deterministic filtering stage is introduced.

### Color Extraction

The cropped image is reduced to a **1×1 thumbnail**:

```python
avg_rgb = image.resize((1,1))
```

This produces an average RGB representation of the garment.

### Color Classification

A predefined color palette dictionary is used:

```python
{
    "red": (255,0,0),
    "green": (0,255,0),
    "blue": (0,0,255),
    ...
}
```

The nearest palette color is determined using Euclidean distance:

```text
distance = √((R1-R2)² + (G1-G2)² + (B1-B2)²)
```

The resulting color label is used to partition the catalog search space.

### Benefits

* Enforces strict color consistency.
* Prevents mismatches such as:

  * Green shirt → Blue shirt recommendation
  * Black hoodie → Gray hoodie recommendation

---

## 3. Feature Extraction Layer (OpenAI CLIP)

After color filtering, the system extracts fine-grained visual semantics using:

**Model:** `openai/clip-vit-base-patch32`

### Captured Features

* Sleeve length
* Neckline style
* Garment silhouette
* Fabric texture
* Overall fashion structure

### Output

Each image is encoded into a:

```text
512-Dimensional Embedding Vector
```

which represents the item's visual identity within CLIP's latent space.

---

# Product Representation Strategy

Efficient retrieval requires an optimized catalog index.

The catalog ingestion pipeline processes:

```text
product_catalog.jsonl
```

and converts it into a searchable embedding database.

---

## Catalog Ingestion Pipeline

### 1. Asset Processing

Catalog entries are loaded sequentially and image URLs are generated.

### 2. Color Pre-Computation

Each catalog image passes through the same color extraction module.

The resulting labels are stored in:

```python
catalog_colors
```

### 3. Embedding Generation

CLIP generates a 512-dimensional embedding for every catalog item.

### 4. Feature Normalization

All embeddings are L2-normalized:

```text
feat = feat / ||feat||₂
```

This transforms cosine similarity into a simple dot product operation.

### 5. Tensor Aggregation

Embeddings are stacked into a single matrix:

```text
Catalog Matrix
[N × 512]
```

allowing efficient batch computations on CPU or GPU.

---

# Retrieval & Ranking Methodology

```text
Query Crop
     │
     ▼
Extract Color Class
     │
     ▼
Filter Catalog By Color
     │
     ▼
Compute Cosine Similarity
     │
     ▼
Top-1 Selection
     │
     ▼
Return Best Match
```

---

## Stage 1: Color Filtering

Example:

```text
Query Color = Green
```

The system only evaluates catalog entries satisfying:

```python
catalog_color == "green"
```

### Fallback Mechanism

If no matching color subset exists:

```text
Filtered Set = Empty
```

the search automatically expands to the full catalog.

This guarantees a recommendation is always produced.

---

## Stage 2: Similarity Search

Cosine similarity is computed between:

* Query embedding
* Filtered catalog embeddings

Formula:

```text
Similarity(A,B) =
(A · B) / (||A|| ||B||)
```

Since vectors are pre-normalized:

```text
Similarity(A,B) = A · B
```

which greatly improves computational efficiency.

---

## Stage 3: Ranking

The highest similarity score is selected:

```python
best_match = torch.argmax(scores)
```

### Decision Threshold

| Similarity Score | Interpretation             |
| ---------------- | -------------------------- |
| ≥ 0.94           | Exact Catalog Match        |
| < 0.94           | Most Relevant Catalog Item |

This confidence flag can be consumed by downstream business logic.

---

# Evaluation Methodology

The system is evaluated using three primary metrics.

## 1. Color Class Precision

Measures:

```text
Percentage of recommendations
sharing the same color category
as the query item.
```

The deterministic color cascade keeps this metric close to **100%** under standard lighting conditions.

---

## 2. Semantic Latent Proximity

Measures:

```text
Cosine Similarity Distribution
```

between retrieved products and query items.

High-density distributions above:

```text
0.85
```

indicate strong semantic understanding of:

* Collar styles
* Sleeve variations
* Fabric structures
* Garment shapes

without task-specific fine-tuning.

---

## 3. End-to-End Latency

Measures total execution time from:

```text
Image Upload
        ↓
Recommendation Output
```

Latency tracking includes:

* YOLO inference
* HTTP image handling
* CLIP encoding
* Similarity search
* UI rendering

---

# Limitations

## Shadow & Illumination Sensitivity

The 1×1 RGB pooling heuristic is sensitive to:

* Shadows
* Bright reflections
* Background leakage

Examples:

* White shirt → Gray
* Green shirt → Black

under extreme lighting conditions.

---

## Memory Scalability

Current implementation stores:

```text
Entire Catalog Embedding Matrix
```

in memory.

While efficient for small and medium catalogs, memory consumption increases linearly with SKU count.

---

# Future Improvements

## 1. Instance Segmentation

Replace bounding-box crops with:

* Segment Anything Model (SAM)
* YOLO Segmentation

Benefits:

* Background removal
* Better color estimation
* Reduced shadow interference

---

## 2. Approximate Nearest Neighbor Search

Transition from flat tensor search to vector databases such as:

* Pinecone
* Milvus
* FAISS

using:

```text
HNSW Indexing
```

Benefits:

* Sub-millisecond retrieval
* Million-scale catalog support

---

## 3. Supervised Metric Learning

Fine-tune the retrieval model using:

```text
Triplet Loss
```

on fashion-specific datasets.

Benefits:

* Improved texture awareness
* Better silhouette discrimination
* Higher retrieval precision

---

# Tech Stack

| Component          | Technology                |
| ------------------ | ------------------------- |
| Object Detection   | YOLOv8                    |
| Color Filtering    | RGB Heuristic Cascade     |
| Feature Extraction | OpenAI CLIP ViT-B/32      |
| Similarity Search  | Cosine Similarity         |
| Tensor Operations  | PyTorch                   |
| UI                 | Gradio                    |
| Future Vector DB   | Pinecone / Milvus / FAISS |

---

# Key Contributions

✅ Hybrid Cascade Architecture

✅ Zero-Shot Product Retrieval

✅ Deterministic Color-Constrained Search

✅ CLIP-Based Semantic Matching

✅ Efficient Cosine Similarity Retrieval

✅ Extensible Vector Database Integration

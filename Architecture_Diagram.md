# Overall Architecture Diagram — Eigenvalue-Driven GCN-TCN Pipeline

Use this as a reference to redraw in any diagram tool (draw.io, Lucidchart, PowerPoint, etc.)

---

## Dimension Flow (Step-by-Step from Raw Video to Prediction)

```
STEP 1: Raw Video
─────────────────
  Input:  A video with F total RGB frames
  Output: Sampled frames (every 2nd frame)
  
  F frames  ──(skip factor s=2)──►  T = F/2 sampled frames
  
  Example: 120 frame video → T = 60 sampled frames


STEP 2: MediaPipe Holistic (per frame)
──────────────────────────────────────
  Input:  Each sampled frame (single RGB image)
  Output: 75 keypoints, each with (x, y, z) coordinates
  
  1 frame  ──►  75 × 3 matrix
                 ↑       ↑
                 │       └── 3 coordinates: x, y, z (normalized 0-1)
                 └────────── 75 keypoints: 33 pose + 21 left hand + 21 right hand

  For all T frames:
  
  T frames  ──►  T × 75 × 3  tensor
                 ↑    ↑    ↑
                 │    │    └── 3 coords per keypoint
                 │    └─────── 75 keypoints per frame
                 └──────────── T time steps


STEP 3: Eigenvalue-Based Keypoint Selection
───────────────────────────────────────────
  Input:  T × 75 × 3
  Output: T × 35 × 3
  
  Removes 40 low-variance (mostly static) keypoints, keeps top 35:
  
  T × 75 × 3  ──(keep only 35 high-variance keypoints)──►  T × 35 × 3
       │                                                         │
  75 keypoints                                             35 keypoints
  (includes static                                         (hands + upper body,
   legs, ankles,                                            the parts that
   feet, etc.)                                              actually move)


STEP 4: Graph Adjacency Reconstruction
──────────────────────────────────────
  Input:  Original adjacency matrix A ∈ ℝ^{75×75}
  Output: Reduced adjacency matrix  ∈ ℝ^{35×35}
  
  75 × 75  ──(extract rows & cols of selected 35 keypoints)──►  35 × 35
  
  (This matrix tells the GCN which joints are connected to which)


STEP 5: Coordinate Normalisation
────────────────────────────────
  Input:  T × 35 × 3  (values in range [0, 1])
  Output: T × 35 × 3  (values in range [-1, 1])
  
  x' = 2 × (x - 0.5)     maps 0→-1, 0.5→0, 1→+1
  
  Shape doesn't change, only the value range.


STEP 6: Batching (DataLoader)
────────────────────────────
  Multiple videos are batched together, padded to same length:
  
  T × 35 × 3  ──(batch B videos, pad to max length T_max)──►  B × T_max × 35 × 3
  
  Example: Batch of 2 videos, longest has 60 frames:
  
  Video 1: 60 × 35 × 3  ─┐
                           ├──►  2 × 60 × 35 × 3
  Video 2: 45 × 35 × 3  ─┘     (Video 2 padded with zeros from frame 46-60)
                                 Actual lengths stored: [60, 45]


STEP 7: Spatial GCN (processes each frame independently)
───────────────────────────────────────────────────────
  Input:  B × T × 35 × 3
  
  GC Layer 1:                                    GC Layer 2:
  B × T × 35 × 3  ──►  B × T × 35 × 128        B × T × 35 × 128  ──►  B × T × 35 × 256
              ↑                     ↑                          ↑                      ↑
          3 features            128 features               128 features           256 features
          per joint             per joint                  per joint               per joint
           (x,y,z)            (learned spatial)
  
  Global Average Pool over 35 joints:
  B × T × 35 × 256  ──(average across 35 joints)──►  B × T × 256
                ↑                                            ↑
           35 joints                                  joints collapsed
           still present                              into 1 vector
  
  Summary:
  B × T × 35 × 3  ──►  B × T × 35 × 128  ──►  B × T × 35 × 256  ──►  B × T × 256
       input            after GC layer 1         after GC layer 2       after avg pool


STEP 8: Temporal TCN (processes the time sequence)
─────────────────────────────────────────────────
  Input:  B × T × 256
  
  TB1 (dilation=1):   B × T × 256  ──►  B × T × 256    (sees 3 consecutive frames)
  TB2 (dilation=2):   B × T × 256  ──►  B × T × 256    (sees frames spaced 2 apart)
  TB3 (dilation=4):   B × T × 256  ──►  B × T × 512    (sees frames spaced 4 apart)  ← channels increase!
  TB4 (dilation=8):   B × T × 512  ──►  B × T × 512    (sees frames spaced 8 apart)
  
  Summary:
  B × T × 256  ──►  B × T × 256  ──►  B × T × 256  ──►  B × T × 512  ──►  B × T × 512
     input           after TB1          after TB2          after TB3          after TB4


STEP 9: Masked Temporal Average Pooling
──────────────────────────────────────
  Input:  B × T × 512
  Output: B × 512
  
  Averages across time (only real frames, ignores padding):
  
  B × T × 512  ──(average across T time steps, masked)──►  B × 512
       ↑                                                       ↑
  still has T                                            time collapsed
  time steps                                             into 1 vector
  
  Example for Video 2 (actual length 45, padded to 60):
  Only frames 1-45 are averaged, frames 46-60 (zeros) are ignored.


STEP 10: Fully Connected Classifier
────────────────────────────────────
  Input:  B × 512
  
  FC Layer 1:   B × 512  ──►  B × 512   (+ ReLU + Dropout)
  FC Layer 2:   B × 512  ──►  B × 6     (6 = number of sentence classes)
  Softmax:      B × 6    ──►  B × 6     (probabilities, sums to 1)
  
  Summary:
  B × 512  ──►  B × 512  ──►  B × 6  ──►  B × 6
    input       after FC1     after FC2    probabilities
```

### Complete Dimension Flow in One Line

```
Video (F frames)
  → T × 75 × 3          (MediaPipe extraction)
  → T × 35 × 3          (Eigenvalue selection)
  → T × 35 × 3          (Normalisation)
  → B × T × 35 × 3      (Batching + Padding)
  → B × T × 35 × 128    (GCN Layer 1)
  → B × T × 35 × 256    (GCN Layer 2)
  → B × T × 256          (Global Avg Pool over joints)
  → B × T × 256          (TCN Block 1, d=1)
  → B × T × 256          (TCN Block 2, d=2)
  → B × T × 512          (TCN Block 3, d=4)
  → B × T × 512          (TCN Block 4, d=8)
  → B × 512              (Masked Temporal Avg Pool)
  → B × 512              (FC Layer 1 + ReLU)
  → B × 6                (FC Layer 2)
  → B × 6                (Softmax → predicted class)
```

---

## Main Pipeline (Left → Right Flow)

```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────────┐      ┌─────────────────────┐      ┌─────────────────┐
│   VIDEO INPUT   │ ───► │   MediaPipe     │ ───► │   Eigenvalue-Based  │ ───► │   Graph Adjacency   │ ───► │   Coordinate    │
│                 │      │   Holistic      │      │   Keypoint Selection│      │   Reconstruction    │      │   Normalisation │
│  {I₁, I₂,…,Iₓ} │      │                 │      │                     │      │                     │      │   [-1, 1]       │
│  (RGB Frames)   │      │  Extracts 75    │      │  Ranks keypoints by │      │  Reduces A from     │      │                 │
│  skip factor=2  │      │  keypoints/frame│      │  variance, keeps    │      │  75×75 → 35×35      │      │  x' = 2(x-0.5) │
│                 │      │  (33 pose +     │      │  top K=35           │      │  Fixes isolated     │      │                 │
│                 │      │   21 LH + 21 RH)│      │  (~95% variance)    │      │  nodes              │      │                 │
└─────────────────┘      └─────────────────┘      └─────────────────────┘      └─────────────────────┘      └─────────────────┘
                          T × 75 × 3               T × 75 × 3 → T × 35 × 3    Â ∈ ℝ^{35×35}
```

```
     ┌───────────────────────┐          ┌───────────────────────┐          ┌──────────────┐     ┌──────────────┐     ┌──────────┐
───► │     SPATIAL GCN       │ ──────► │     TEMPORAL TCN       │ ──────► │   Masked     │ ──► │ FC Classifier│ ──► │ Softmax  │
     │                       │         │                        │         │   Temporal   │     │              │     │ Output   │
     │  2 Graph Conv Layers  │         │  4 Temporal Blocks     │         │   Avg Pool   │     │ 512 → 512   │     │          │
     │  + Global Avg Pool    │         │  with Dilated Conv     │         │              │     │ → |C| classes│     │ |C| probs│
     └───────────────────────┘         └───────────────────────┘         └──────────────┘     └──────────────┘     └──────────┘
     B × T × 35 × 3                    B × T × 256                       B × T × 512          B × 512              B × |C|
              → B × T × 256                     → B × T × 512                  → B × 512            → B × |C|
```

---

## Detailed Sub-Components

### 1. Spatial GCN (Internal)

```
Input: B × T × 35 × 3
   │
   ▼
┌──────────────────────┐
│  GC Layer 1           │
│  In: 3  →  Out: 128   │
│  + Batch Norm          │
│  + ReLU                │
│  + Dropout (p=0.3)     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  GC Layer 2           │
│  In: 128 →  Out: 256  │
│  + Batch Norm          │
│  + ReLU                │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Global Average Pool   │
│  (over K=35 nodes)     │
│  Collapses node dim    │
└──────────┬───────────┘
           │
           ▼
Output: B × T × 256
```

**What each GC layer does:**
```
For each node (joint):
   new_feature[node] = ReLU( BN( Σ (normalized_weight × neighbor_feature) × W ))
                                   ↑                                       ↑
                           aggregation from                         learnable
                           graph neighbours                         transformation
```

---

### 2. Temporal TCN (Internal)

```
Input: B × T × 256
   │
   ▼
┌─────────────────────────────┐
│  Temporal Block 1            │
│  Dilation = 1, Channels: 256 │
│  ┌────────────────────────┐  │
│  │ DilConv → BN → ReLU    │  │
│  │ DilConv → BN → ReLU    │  │
│  │ + Dropout               │  │
│  └────────────┬───────────┘  │
│  + Residual Connection ──────│
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Temporal Block 2            │
│  Dilation = 2, Channels: 256 │
│  (same internal structure)   │
│  + Residual Connection       │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Temporal Block 3            │
│  Dilation = 4, Channels: 512 │
│  (same internal structure)   │
│  + 1×1 Conv on residual path │ ← dimension mismatch (256→512)
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Temporal Block 4            │
│  Dilation = 8, Channels: 512 │
│  (same internal structure)   │
│  + Residual Connection       │
└──────────────┬──────────────┘
               │
               ▼
Output: B × T × 512
```

**Why increasing dilation?**
```
Dilation = 1:  looks at frames  [t, t-1, t-2]           → local motion
Dilation = 2:  looks at frames  [t, t-2, t-4]           → short-range
Dilation = 4:  looks at frames  [t, t-4, t-8]           → medium-range
Dilation = 8:  looks at frames  [t, t-8, t-16]          → long-range

Combined receptive field = 31 frames (covers the full gesture)
```

---

### 3. Temporal Block (Single Block Detail)

```
         Input
           │
     ┌─────┴─────┐
     │            │
     ▼            │ (Residual Path)
┌─────────┐       │
│DilConv 1│       │ If channels change:
│  + BN   │       │   apply 1×1 Conv
│  + ReLU │       │ Otherwise:
│+Dropout │       │   identity
└────┬────┘       │
     │            │
     ▼            │
┌─────────┐       │
│DilConv 2│       │
│  + BN   │       │
│  + ReLU │       │
│+Dropout │       │
└────┬────┘       │
     │            │
     └─────┬──────┘
           │ ADD
           ▼
        ReLU
           │
         Output
```

---

### 4. Classification Head

```
B × T × 512
     │
     ▼
┌────────────────────────┐
│  Masked Temporal        │
│  Average Pooling        │
│  (only valid frames)    │
└──────────┬─────────────┘
           │
           ▼
        B × 512
           │
           ▼
┌────────────────────────┐
│  FC Layer 1             │
│  512 → 512              │
│  + ReLU + Dropout       │
└──────────┬─────────────┘
           │
           ▼
┌────────────────────────┐
│  FC Layer 2             │
│  512 → |C| (6 classes)  │
└──────────┬─────────────┘
           │
           ▼
┌────────────────────────┐
│  Softmax                │
│  → Class Probabilities  │
└────────────────────────┘
```

---

## Complete Pipeline Summary (Single Row)

| Stage | Component | Input | Output | Key Detail |
|-------|-----------|-------|--------|------------|
| 1 | Video Input | RGB video | Sampled frames | Skip factor s=2 |
| 2 | MediaPipe Holistic | Frames | T × 75 × 3 | 33 pose + 21 LH + 21 RH |
| 3 | Eigenvalue Selection | T × 75 × 3 | T × 35 × 3 | 95% variance retained |
| 4 | Graph Reconstruction | A (75×75) | Â (35×35) | Connectivity preserved |
| 5 | Normalisation | T × 35 × 3 | T × 35 × 3 | Maps to [-1, 1] |
| 6 | Spatial GCN | B×T×35×3 | B×T×256 | 2 GC layers + avg pool |
| 7 | Temporal TCN | B×T×256 | B×T×512 | 4 blocks, dil=[1,2,4,8] |
| 8 | Temporal Pooling | B×T×512 | B×512 | Masked average |
| 9 | FC Classifier | B×512 | B×\|C\| | 2 FC layers + softmax |

---

## Color Coding Suggestion

| Component | Suggested Color |
|-----------|----------------|
| Video Input / MediaPipe | Orange |
| Eigenvalue Selection / Graph Recon | Yellow / Gold |
| Normalisation | Light Cyan |
| Spatial GCN | Blue |
| Temporal TCN | Green |
| Pooling | Purple |
| Classifier / Output | Red / Pink |

---

## Arrows & Annotations

- **Thick arrows** between major stages (GCN → TCN → Pool → Classifier)
- **Thin arrows** for internal sub-component flows
- **Dashed arrows** from main blocks down to their expanded sub-diagrams
- **Dimension labels** on every arrow (e.g., "B × T × 256")
- **Stage numbers** above each major block (Stage 1, Stage 2, etc.)

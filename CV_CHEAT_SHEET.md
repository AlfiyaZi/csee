# Computer Vision - Exam Cheat Sheet
## Based on ALL Past Exams (2022-2025) + Lecture Slides

---

# LECTURE 2: Capturing Digital Images

## f-number (ASKED IN EVERY EXAM)
```
f-number (N) = focal_length / aperture_diameter
```
| Exam | Question | Answer |
|------|----------|--------|
| 2022 | Diameter of 80mm f/4 lens? | 80/4 = **20mm** |
| 2023 | Diameter of 60mm f/4 lens? | 60/4 = **15mm** |
| 2024 | Diameter of 80mm f/4 lens? | 80/4 = **20mm** |
| 2025 Jan | What does f/2.8 mean? | Focal length is 2.8x aperture diameter |
| 2025 Jan | f/2.8 vs f/4 light? | f/2.8 captures **2x more light** than f/4 |
| 2025 Retry | f/8, aperture 16mm → focal length? | 8 × 16 = **128mm** |

**Rule:** Each f-stop change = 2x light difference. Lower f-number = more light, shallower depth of field.

## Bayer Filter & Color Sensors
- Bayer filter: **50% green, 25% red, 25% blue** (NOT 50% red!)
- Each pixel measures ONLY ONE color channel
- Missing colors obtained by **demosaicing** (interpolation from neighbors)
- Human vision is more sensitive to green → more green pixels

## Sensor Noise & Signal-to-Noise Ratio (SNR)
- **SNR is more important** than absolute noise amount
- Dark regions → low signal → worse SNR → noise has stronger relative impact
- Noise suppression: averaging multiple images, filtering
- Noise causes image gradients (2023 exam)

## Sensor Response
- Sensor response is normally **linear**
- Human visual system is **nonlinear**
- Consumer cameras apply nonlinear response to produce pleasing images for humans
- **Linearization** needed for computer vision (invert the nonlinear response function)

## Radial Distortion
- **Barrel distortion**: magnification DECREASES with distance from optical axis (wide-angle/fisheye)
  - First polynomial coefficient is **positive**
- **Pincushion distortion**: magnification INCREASES with distance (telephoto lenses)
- Makes camera model **non-linear** (polynomial function)

## Camera Model & Intrinsic/Extrinsic Parameters
```
p = K[R|T] * P
    ↑  ↑ ↑
    K  R T
```
**Intrinsic parameters (K):** focal length, principal point, skew angle, lens parameters
**Extrinsic parameters:** rotation (R), translation (T) = camera pose

- Multiplying by K maps from **normalized image plane → physical image plane**
- Inverse of K maps from physical → normalized
- Linear model for thin lenses, non-linear for thick lenses (distortion)

## Projection Types
- **Perspective projection**: parallel lines converge at vanishing points, x = fX/Z
- **Orthographic projection**: NO vanishing points, parallel lines stay parallel
- **Weak perspective**: f/z = constant

## Homogeneous Coordinates
- Heterogeneous → Homogeneous: (x, y) → (x, y, 1)
- Homogeneous → Heterogeneous: (x, y, w) → (x/w, y/w)
  - **Divide all coefficients by the last one**

## Projection Matrix M (3x4)
- 12 unknowns (3×4 coefficients)
- Need **6 corresponding (P,p) pairs** minimum to solve for M
  - Each pair gives 2 equations: u = (m1·P)/(m3·P), v = (m2·P)/(m3·P)
  - 6 pairs × 2 equations = 12 equations for 12 unknowns

---

# LECTURE 3: Digital Image Processing

## Convolution Properties (ASKED REPEATEDLY)
```
Commutative:    f * g = g * f               →  FG = GF
Associative:    f * (g * h) = (f * g) * h   →  (FG)H = F(GH)
Distributive:   f*h + g*h = (f+g) * h       →  FH + GH = (F+G)H
Linear:         (af + bg)*h = af*h + bg*h   →  (aF+bG)H = aFH+bGH
Shift-Invariant: f(x+t)*h = (f*h)(x+t)
```
- Convolution is a **shift-invariant LINEAR** operation
- I+(H1*H2)=I*(H1+H2) is **WRONG** (nonsensical)

## Convolution Theorem (ASKED REPEATEDLY)
```
Convolution in spatial domain = Multiplication in frequency domain
  f*g ↔ F·G

Multiplication in spatial domain = Convolution in frequency domain
  f·g ↔ (1/NM) F*G

BOTH DIRECTIONS ARE TRUE!
```
- If f*g = g*f, then FG = GF (NOT FG = G/F!)

## How to Compute Convolution
1. **Flip the kernel 180°** (negate both indices: h[k,l] → h[-k,-l])
2. Slide flipped kernel over image
3. Element-wise multiply and sum

## Filter Types
| Filter | Type | Effect | Sum of kernel |
|--------|------|--------|---------------|
| Box filter (1/9 * ones) | Low-pass | Blurring/smoothing | 1 |
| Gaussian filter | Low-pass | Smooth blur | 1 |
| Laplacian filter | High-pass | Edge/detail detection | 0 |
| Impulse filter (1 in center) | Identity | No change | 1 |
| Sharpening filter | High-pass | Accentuates differences | 1 |

**Sharpening kernel example:**
```
(-1, -1, -1,
 -1, 17, -1,
 -1, -1, -1) / 9
```
Sum = 1, accentuates differences → **sharpening filter**

**Laplacian approximation:** Unit (impulse) MINUS Gaussian = Laplacian
(NOT Gaussian minus impulse!)

## Gaussian Convolution Property
- Convolving once with σ = convolving twice with **σ/√2**
- Convolution of two Gaussians is a Gaussian
- Fourier transform of Gaussian = another Gaussian
- Gaussians are **separable** (2×1D instead of 1×2D)

## Image Pyramids
| Pyramid | Filter type | Represents |
|---------|-------------|------------|
| **Gaussian** pyramid | Low-pass | Progressively blurred & downsampled |
| **Laplacian** pyramid | Band-pass | Different frequency sub-bands (details at each scale) |

## Frequency Domain
- Spectrum with more high-frequency content = image with more spatial details
- Low frequencies = smooth regions, overall shapes
- High frequencies = fine details, sharp transitions

## Gradient Domain
- Image gradient field IS a **conservative vector field**
- Image gradient field HAS **zero curl**
- **Divergence of gradient field = Laplacian** of the image
- After PROCESSING gradient field → no longer conservative (has non-zero curl)
- Cannot reconstruct by simple integration → must solve **2D Poisson equation**

## Deconvolution
- Division in frequency domain is naive inverse filtering
- Problems: sensitive to noise, ringing artifacts
- Better: Wiener filter, regularized filter, Lucy-Richardson, blind deconvolution

## Transposed Convolution
- Used for **upsampling** in CNNs
- Output size formula: output = (input - 1) × stride - 2 × padding + kernel_size
- Example: input 4×4, kernel 3×3, stride 2, padding 1 → output **7×7**

---

# LECTURE 4: Machine Learning

## PCA / Eigenfaces

### PCA Fundamentals
- Goal: reduce d dimensions to k dimensions (k < d) with minimal information loss
- Finds directions of **maximum variance** in data
- Eigenvectors of covariance matrix C = X^T·X
- Eigenvalues = amount of variance in each direction
- Select eigenvectors with **largest eigenvalues**
- Eigenvectors represent directions of **maximum** variance (NOT minimum!)

### Eigenfaces
- Each face = point in high-dimensional space (100×100 = 10,000 dimensions)
- PCA finds lower-dimensional "face space"
- Eigenfaces = eigenvectors of covariance matrix of face images
- Any face = mean face + weighted linear combination of eigenfaces
- Recognition: project onto eigenfaces, find nearest neighbor
- **Limitation: Fails quickly when variation between training and test becomes too large**
  (NOT robust to extreme lighting/pose/expression changes)
- Choose K eigenfaces based on eigenvalue decay (ignore low-variance ones)

## ML Classes
| Type | Description |
|------|-------------|
| Unsupervised | Find structure in unlabeled data (clustering) |
| Supervised | Labeled data for prediction (classification, regression) |
| Semi-supervised | Labeled + unlabeled data |
| Self-supervised | Properties of data itself as supervisory signals (e.g., predict rotation) |

## Dataset Splits
| Split | Purpose |
|-------|---------|
| **Training** | Train model parameters |
| **Validation** | Tune hyperparameters |
| **Testing** | Report final accuracy (don't touch before!) |

## CNNs

### ReLU: f(x) = max(0, x)
- Introduces **non-linearity**
- Helps mitigate **vanishing gradient problem**
- ReLU(-1234) = **0**
- ReLU(5) = **5**

### Pooling (Max Pooling)
- **Reduces spatial dimensions** (downsampling)
- Provides large response **regardless of exact feature position**
- Achieves scale invariance in CNNs (replaces image pyramids in model-based approaches)
- Usually 2×2 or 3×3

### Receptive Field
- Region of input image a neuron "looks at"
- Depends on kernel size, stride, pooling

### CNN Parameters Calculation
```
Parameters per filter = kernel_h × kernel_w × input_channels
Total parameters = num_filters × params_per_filter (+ num_filters if bias)
```
Example: input 32×32×3, 16 filters of 3×3×3, no bias:
- Per filter: 3 × 3 × 3 = 27
- Total: 16 × 27 = **432**

### CNN Feature Levels
| Level | What it learns | Example |
|-------|---------------|---------|
| Low-level (conv1) | Edges, textures, **Gaussian-like** filters | Gabor-like patterns |
| Mid-level | Object parts, textures | Eyes, wheels |
| High-level | Object-level features | Faces, cars |

### Key Architectures
| Architecture | Year | Depth | Notable |
|-------------|------|-------|---------|
| AlexNet | 2012 | 5 conv | Breakthrough, started DL era |
| VGG-Net | 2014 | 16 conv | Deeper, uniform 3×3 kernels |
| GoogLeNet | 2015 | 22 conv | Inception modules |
| ResNet | 2016 | <100 conv | Skip connections, **better top-5 than humans** |

---

# LECTURE 5: Feature Extraction

## Model-Based vs Learning-Based
| Aspect | Model-Based | Learning-Based |
|--------|-------------|----------------|
| Kernels | Manually designed, fixed | Learned from training data |
| Examples | Gaussian, Laplacian, SIFT | CNN filters |
| Invariance | Explicit handling needed | Learned via pooling/augmentation |
| Pipeline | Features → Clustering → Classification | End-to-end learned |

## SIFT (Scale-Invariant Feature Transform)
1. **Scale-space**: Build Gaussian pyramid, compute DoG (Difference of Gaussians = Laplacian pyramid)
2. **Detect local extrema**: Find peaks in DoG pyramid
3. **Filter**: Remove low-contrast regions (noise)
4. **Orientation**: Assign dominant gradient direction from histogram
5. **Descriptor**: 4×4 tiles × 8 orientation bins = **128-dimensional** descriptor

- SIFT IS scale invariant (exam question: "SIFT is not scale invariant" → **FALSE**)

## Model-Based Classification Pipeline
Features (SIFT) → **Clustering** → Feature Representation → Pooling → Classification
- Feature detectors are manually designed and remain fixed
- Mid-level representations formed by **clustering low-level features**
- Feature kernels are NOT learned (that's learning-based)
- DOES require invariance handling (scale, rotation)

---

# LECTURE 6: Segmentation

## Types of Segmentation
| Type | Description |
|------|-------------|
| **Semantic** | Pixels → class labels (all cars = same label, no instance distinction) |
| **Instance** | Detect individual objects + segment pixels per instance |
| **Panoptic** | Semantic + Instance combined (both object instances AND background classes) |

- Instance segmentation CAN differentiate between instances (exam trap: "cannot" → **FALSE**)

## K-Means Clustering
```
1. Choose k random cluster centers
2. Assign each point to nearest cluster
3. Replace centers with mean of cluster points
4. Repeat until convergence
```
- K-means is **NOT deterministic** (random initialization → different results)
- Requires predefining k

## Graph-Based Segmentation (Clustering by Graph Eigenvectors)
```
1. Construct affinity matrix A (NxN for N pixels)
2. Compute eigenvalues and eigenvectors of A
3. Until stopping criterion:
   - Take eigenvector for next largest eigenvalue
   - Assign elements to cluster
   - Zero out clustered elements in A
4. Stop when eigenvalue drops below threshold
```
- Number of clusters = number of successful rounds before stopping

## Hough Transform (ASKED REPEATEDLY)

### Key: Hough space dimensionality = number of parameters defining the shape

| Shape | Parameters | Hough Space Dimension |
|-------|-----------|----------------------|
| Line in 2D | θ, r (angle + distance) | **2D** |
| Circle in 2D | x, y, r (center + radius) | **3D** |
| Axis-aligned ellipse in 2D | x, y, a, b (center + semi-axes) | **4D** |
| General ellipse in 2D | x, y, a, b, θ (center + semi-axes + rotation) | **5D** |
| Plane in 3D | θ₁, θ₂, d (two angles + distance) | **3D** |
| Ellipsoid in 3D | 9 params (center + semi-axes + 3 rotations) | But exam says **5D** for 3D ellipsoids |

**Important from 2024 exam:** "Hough space for finding ellipsoids (3D version of ellipses) in a 3D volume is five-dimensional" was the correct answer.

## U-Nets (Connected Autoencoders)
- **Skip connections** between encoder and decoder layers
- Purpose: **Preserve high-resolution spatial information** lost during downsampling
- Concatenation of encoder output with decoder input at corresponding levels
- Better for segmentation than unconnected autoencoders (which lose spatial detail)
- 1×1 convolution used for **dimensionality reduction** (linear projection of feature stacks)

## CNN Architecture for Segmentation
- Full resolution convolutions only → too expensive
- Standard CNN (downsampling only) → wrong output size
- **Correct: Encoder-Decoder with down- AND up-sampling** inside the network

---

# LECTURE 7: Optical Flow

## Optical Flow Equation
```
Ix·u + Iy·v + It = 0
```
Where Ix, Iy = spatial gradients, It = temporal gradient, (u,v) = flow vector

### Key Assumptions
| Assumption | Description |
|-----------|-------------|
| **Brightness constancy** | I(x+u, y+v, t+1) = I(x, y, t) |
| **Small motion** | Displacement (u,v) is small |

## Lucas-Kanade vs Horn-Schunck (ASKED REPEATEDLY)
| Aspect | Lucas-Kanade | Horn-Schunck |
|--------|-------------|-------------|
| Type | **Local** least-squares | **Global** energy minimization |
| Output | **Sparse** flow | **Dense** flow |
| Extra assumption | **Spatial coherence** (neighbors have same flow) | **Smoothness** constraint |
| Equations | 5×5 window → 25 equations for 2 unknowns | E = Σ[Ed + λEs] for all pixels |
| λ parameter | N/A | Larger λ = smoother flow |

### Lucas-Kanade assumptions: Brightness constancy + Spatial coherence
### Horn-Schunck assumptions: Brightness constancy + Smoothness
(Smooth flow fields is Horn-Schunck ONLY, not Lucas-Kanade!)

## Handling Large Motion
- Basic optical flow works for ~1 pixel displacement
- Solution: **Gaussian image pyramids** (compute flow at lower resolution, warp & upsample)

## Deep Learning for Optical Flow
- **Encoder-decoder** (U-Net-like) architectures: FlowNetS, FlowNetCorr
- Input: pair of RGB images → Output: dense flow field
- Learns **direct mapping** from image pairs to flow fields (end-to-end)
- Does NOT require explicit computation of image gradients
- Training data often generated with computer graphics

---

# LECTURE 8: Object Detection

## Detection Architectures Evolution
```
R-CNN → Fast R-CNN → Faster R-CNN → Mask R-CNN (instance segmentation)
         ↓
    YOLO/SSD/RetinaNet (single-shot)
```

### R-CNN: ~2k region proposals → CNN per region → SVM classification (SLOW)
### Fast R-CNN: Single CNN for whole image → crop features per region
### Faster R-CNN: **Region Proposal Network (RPN)** replaces external proposals
### Mask R-CNN: Adds 28×28 binary mask per RoI for instance segmentation

## Region Proposal Network (RPN) in Faster R-CNN
- Primary function: **Propose regions that likely contain objects**
- Uses anchor boxes of different sizes/scales at each feature map position
- Binary classification: object vs. not-object (objectness score)
- Predicts bounding box corrections (4 numbers: dx, dy, dw, dh)
- Sorts by objectness score, takes top ~300 proposals
- Does NOT classify into specific categories (that's the per-region network)

## YOLO (You Only Look Once)
1. Split image into S×S grid
2. Each cell predicts: Y = [pc, bx, by, bh, bw, c1, c2, ...]
   - pc = confidence score (probability of object)
   - bx,by,bh,bw = bounding box coordinates
   - c1,c2,... = class scores
3. **IoU** (Intersection over Union): discard low-IoU predictions
4. **NMS** (Non-Maximum Suppression): remove **too similar** bounding boxes
   - Select highest confidence → suppress all with high IoU → repeat

## Evaluation Metrics
```
Precision = TP / (TP + FP)    → "Of all positive predictions, how many are correct?"
Recall    = TP / (TP + FN)    → "Of all actual positives, how many did we find?"
Accuracy  = (TP+TN) / (TP+TN+FP+FN)
FPR       = FP / (FP + TN)
```

**Example:** TP=80, FP=20, FN=40 → Precision = 80/100 = **0.8**

### Precision-Recall Curve & mAP
- Plot precision vs recall for many confidence thresholds
- **AP** = area under P-R curve (0..1) per class
- **mAP** = mean of AP across all classes
- AP = 0.5 with smooth curve → **linear** relationship between precision and recall
- IoU is always between **0 and 1** (>0.5 and <1 for overlapping boxes)

## Vision Transformers (ViTs)
- Sequence of words = sequence of **image patches**
- Only the **encoder** part is used (NOT decoder, NOT both)
- Flattened patches → **linear projection** (lower dimension, learned) + **positional embedding**
- Add **class token** for classification
- **Attention** learned between all tokens
- Key idea: not only features (like CNNs) but their **relationships/context**
- Output → MLP/softmax for classification

---

# LECTURE 9: Multi-View Geometry

## Epipolar Geometry (ASKED IN EVERY EXAM)

### Essential Matrix vs Fundamental Matrix
| Matrix | Applied to | Plane | K required? |
|--------|-----------|-------|-------------|
| **Essential (ε)** | **Normalized** image plane | Known K | Yes (calibrated) |
| **Fundamental (F)** | **Physical** image plane | Unknown K | No (uncalibrated) |

```
l = ε·p'  (calibrated, normalized plane)
l = F·p'  (uncalibrated, physical plane)
```

- Need **>8 corresponding point-pairs** to compute F or ε
- For one pixel in camera A, there is exactly **1 epipolar line** in camera B

### Epipolar lines are NOT the same as rectified scanlines!
- "Epipolar lines are rectified scanlines" → **FALSE** (they become scanlines AFTER rectification)

## Image Rectification
- Transforms images so epipolar lines become **horizontal scanlines**
- Simplifies correspondence search from **2D → 1D** (along horizontal lines)
- Process called **rectification** (not relation, correlation, or normalization)

## Correspondence Search Enhancements
- Image rectification
- Applying epipolar constraints
- Using image pyramids
- Normalizing correlating image patches
- Order constraint: CANNOT be applied (order may differ due to occlusion) → **TRUE**

## Stereoscopic Depth Reconstruction
```
Z = f·T / (XL - XR)
         ↑ disparity
```
- Depth Z is **inversely proportional** to disparity
- Disparity = pixel distance of same scene point in rectified images (on x-axis)
- Dense optical flow of rectified stereo pair correlates to disparity map → **TRUE**

## Structure from Motion (ASKED REPEATEDLY)
```
Equations:  2 × n × m       (2 per point per view)
Unknowns:   12m + 3n        (12 per camera matrix + 3 per 3D point)
Solvable:   2nm > 12m + 3n
```

| What | Formula | Example |
|------|---------|---------|
| Equations (m=3, n=20) | 2×20×3 | **120** |
| Unknowns (m=3, n=25) | 12×3 + 3×25 | **111** |
| Solvable (m=2, n=25)? | 100 > 99? | **Yes** |
| Solvable (m=3, n=3)? | 18 > 45? | **No!** |

### Affine SfM (from 2022 exam):
- 2nm equations, 8m+3n unknowns, ambiguity matrix Q: 2nm > 8m+3n-12
- Min for 2 views: **4 points**

### Projective SfM:
- Projection matrix 3×4 (11 unknowns, lower-right = 1)
- Q is 4×4 (15 unknowns): 2nm > 11m+3n-15
- Min for 2 views: **7 points** → answer format "72"

## Calibration vs SfM
- Calibration pattern: can reconstruct intrinsic params, defines coordinate system origin
- SfM: reconstructs in **relative** coordinate system (not absolute)

---

# LECTURE 10: 3D Vision

## Depth Representations
| Type | Description | Pros | Cons |
|------|-------------|------|------|
| **Depth Map** | 2D matrix, each pixel = distance | Simple, direct | 2.5D only |
| **Voxel Grid** | V×V×V occupancy grid | Simple 3D | High resolution needed |
| **Point Cloud** | Set of 3D points | Fine structures, efficient | No explicit surface |
| **Mesh** | Set of triangles (vertices + faces) | Standard for graphics, adaptive | Complex |
| **Implicit Surface** | Function: inside/outside classification | Flexible | Needs sampling |

## Depth Map Prediction
- RGB image → CNN → predicted depth map
- Loss: per-pixel L2 distance (in log space)
- RGB + Depth = **RGB-D** image (2.5D)
- Recordable with 3D sensors (e.g., Microsoft Kinect)

## Scale-Depth Ambiguity
- Small close object looks same as large far object in single image
- Solution: **Scale-invariant loss function** (considers global scene scale)
- NOT: segmentation first, NOT: smoothness assumptions

## Surface Normals
- Unit vectors perpendicular to surface at each pixel
- Can compute normals from depth (gradients) and depth from normals (integration)
- Loss function: solid angle difference (dot product based)

## Neural Radiance Fields (NeRFs)
- Architecture: **Fully connected network** (NOT CNN, NOT transformer, NOT U-Net)
- Input: 3D position (x,y,z) + viewing direction (θ,φ)
- Output: **Volume density (σ)** + **RGB color**
- Uses **volume rendering equation**: C(r) = ∫ T(t)·σ(t)·c(t,d) dt
- Transmittance T = exp(-∫σ ds) (accumulated opacity along ray)
- Retrained for **every scene**
- Solves an **inverse problem** → TRUE
- If opacity is binary → can only represent **diffuse + isotropic** reflections

## Depth from X
- X = Stereo: use perspective difference
- X = Shading: use shading differences (inverse shading)
- X = Defocus: use focus differences
- ResNet features CAN be used for depth from defocus (saliency proportional to focus, but only for non-uniform regions)

---

# LECTURE 11: Trends in Computer Vision

## Large Multi-Modal Models (LMMs)
- Handle multiple data types: text, images, audio, video
- Examples: GPT, Gemini, Claude
- CAN do high-level reasoning but MAY struggle with precise spatial tasks
- **Hallucinate** if they can't generalize
- Don't truly "reason" — compile statistical essence of data
- Still mainly based on natural language
- Don't know rules of how the world works

## CLIP
- Model that learns correlations of **images and text descriptions**
- Not YOLO, not ResNet, not Grad-CAM

## Image Aggregation (ASKED REPEATEDLY)
Examples include:
- Panorama imaging ✓
- HDR imaging ✓
- Multi-spectral imaging ✓
- Multi-focal imaging ✓
- Single-shot object detection ✗

## Event Cameras
- Asynchronously report **changes** in pixel intensity (NOT full frames)
- Higher temporal resolution, lower latency than frame-based cameras
- Each event encodes: pixel location + timestamp + brightness change (polarity)
- Do NOT suffer from motion blur
- Require LESS bandwidth (only transmit changes)

## Self-Supervised Learning
- Converts unsupervised into supervised by creating labels from data itself
- Example: rotate images, predict the rotation angle

## Generative Models (not covered in detail)
- VAEs, GANs, Diffusion Models
- Uses: training data augmentation, solving inverse problems

## Limitations of ML
- Cannot detect objects never seen before (can't count novel objects)
- More parameters ≠ better
- Human knowledge helps (e.g., regularization in loss functions)

---

# MISCELLANEOUS EXAM TOPICS

## ICP (Iterative Closest Point)
- Unoptimized: **O(N²)** complexity
- With k-d trees: **O(N log N)** complexity

## Gray Code (Structured Light)
- Hamming distance = **1**
- Reason: to **detect errors** in transmission
- Bit difference between subsequent code words is 1

## Light Transport
- Forward: c = T·p (camera image from projector image)
- Inverse: p = T⁻¹·c (inverse light transport, inverted colors)
- Transpose: p = T'·c (dual photography: scene from projector's perspective under camera illumination)

## Linear vs Non-Linear Equation Systems
- Linear: Householder, Jacobi iterations, Gauss-Seidel
- Non-linear: Newton, Gauss-Newton, Levenberg-Marquardt

## Pixel Value Calculations
- Average of pixels: set up linear equation system
- Example: avg(x,y,z)=180, x=2y, x=z → x=216, y=108, z=216

## Rotation Representations
- Rotation matrix R: rotate vector v → **Rv**
- Quaternion q (conjugate q'): rotate vector v → **qvq'**
- NOT q'vq, NOT qv alone, NOT vR

---

# QUICK FORMULA REFERENCE

```
f-number:           N = f / d
Projection:         x = fX/Z,  y = fY/Z
Camera model:       p = K[R|T] · P
Disparity→Depth:    Z = fT / (XL - XR)
SfM equations:      2nm
SfM unknowns:       12m + 3n
SfM solvable:       2nm > 12m + 3n
Precision:          TP / (TP + FP)
Recall:             TP / (TP + FN)
Accuracy:           (TP+TN) / (TP+TN+FP+FN)
Optical flow:       Ix·u + Iy·v + It = 0
ReLU:               f(x) = max(0, x)
CNN params:         num_filters × (kH × kW × channels) [+ bias]
Gaussian conv:      σ_once = σ_twice × √2
Hough dim:          = number of shape parameters
```

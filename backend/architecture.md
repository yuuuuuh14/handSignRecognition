Technical System Architecture Specification: Multi-Branch Transformer-CNN Hybrid for Korean Sign Language Recognition

1. Executive System Overview and Design Philosophy

Korean Sign Language (KSL) recognition presents a unique set of computer vision challenges, primarily stemming from environmental variables such as non-uniform light illumination and high background complexity. While Convolutional Neural Networks (CNNs) are the standard for local spatial feature extraction, they lack the global context necessary for complex gestural syntax. Conversely, Vision Transformers (ViTs) excel at capturing long-range dependencies but suffer from an "efficiency gap" due to the exponential computational complexity of their self-attention mechanisms—particularly in sequential CNN-Meet-Transformer (CMT) models that repeat these stages multiple times.

The strategic objective of this architecture is to resolve this bottleneck through a parallel multi-branch design. By synthesizing long-range dependency capture with high-fidelity local feature extraction, the system achieves state-of-the-art accuracy with a significantly lower computational footprint. Unlike sequential hybrids that incur compounding latency, this parallel approach extracts disparate feature sets simultaneously, ensuring the model preserves fine-grained hand orientations while maintaining a global understanding of the signer’s posture.

Architectural Design Goals vs. Technical Solutions

Design Goal	Technical Solution
Long-range dependency capture	Lightweight Multi-head Self-Attention (LMHSA) with K/V dimensionality reduction.
Dense local feature preservation	Parallel 4-block 3 \times 3 Conv branch utilizing GELU activation and Batch Normalization.
Mitigation of patch projection loss	Grain Architecture Module replacing linear projection with convolutional down-sampling.
Translation and scale invariance	Local Perception Unit (LPU) utilizing element-wise convolutions for structural patch information.
High-dimensional representation	Modified Inverted Residual Feed-Forward Network (IRFFN) with 4\times dimension expansion.
Computational cost optimization	Parallel execution reducing the depth-induced complexity of stage-wise hybrid designs.

This multi-branch synergy begins with the Grain Module, which serves as a specialized convolutional front-end to refine visual input for downstream processing.

2. Front-End Processing: The Grain Module Mechanism

Traditional Vision Transformers rely on a "linear projection of patches" which often fails to capture multi-scale feature hierarchies and ignores low-resolution details critical for finger-level gesture clarity. The Grain Module is engineered as a superior alternative, acting as a convolutional initializer that provides a structured, fine-grained representation of the input image.

The Grain Module is structured in two distinct stages to optimize the input for the parallel branches. The primary objective is a 2 \times down-sampling of spatial resolution coupled with a 2 \times enlargement of the feature dimension. This transition from raw pixels to "grain features" preserves local information density while reducing the input size to a 32-output channel configuration, effectively solving the "patch processing problem" by maintaining spatial connectivity that linear splitting destroys.

Technical Composition of the Grain Module

* Block 1 (Spatial Reduction): Initial 3 \times 3 convolution with stride 2 for aggressive resolution reduction, followed by two 3 \times 3 convolutions with stride 1 to stabilize the feature maps.
* Block 2 (Feature Refinement): A final 3 \times 3 convolution with stride 2 to reach the target resolution, followed by Layer Normalization (LN).
* Channel Configuration: Projects the input to 32 output channels.
* Activation: Utilizes GELU for non-linear feature mapping throughout the module.

By refining these grain features, the architecture ensures that both the global and local branches receive a high-entropy data stream for parallel processing.

3. Parallel Feature Extraction: Sub-Path A (Convolutional Transformer)

Sub-Path A is a Convolutional Layer-Based Transformer designed to capture global context without the quadratic memory overhead of standard self-attention. This branch allows the model to understand the relationship between distant gestural components—such as the distance between the hands and the face—efficiently.

Core Architectural Components

1. Local Perception Unit (LPU): To mitigate image translation dependency (where slight signer movement affects accuracy), the LPU implements the logic IM(X) = EWConv(X) + X. This element-wise convolution allows the model to leverage token order while preserving the structural information inside the patches, effectively replacing absolute positional encoding with a more robust local context mechanism.
2. Lightweight Multi-head Self-Attention (LMHSA): This unit achieves "lightweight" status by applying an element-wise convolution with a stride k to the Key (K) and Value (V) matrices. This reduces the dimensionality of the attention operation, significantly lowering computational complexity per layer. It further incorporates a relative position bias (B) to maintain spatial awareness.
3. MLP Convolution: Utilizing 1 \times 1 convolutions, this module converts the global attention features back into local pixel information, ensuring the global context is spatially mapped to the original image dimensions.

Unlike brute-force MHSA, which scales poorly with input size, the LMHSA mechanism provides a pragmatic balance of long-range perception and memory efficiency, critical for high-resolution sign language datasets.

4. Parallel Feature Extraction: Sub-Path B (CNN Branch)

While Sub-Path A abstracts data for global context, the dedicated CNN Branch serves as a high-fidelity pass-through for dense local features. In KSL, subtle changes in finger orientation or palm direction can radically alter semantic meaning; this branch ensures these fine-grained spatial details are not smoothed over by the self-attention mechanism.

Structural Logic and Impact

The CNN branch consists of a four-block sequence of 3 \times 3 convolution layers. Each block is followed by Batch Normalization (BN) and GELU activation. This path acts as a direct counterpart to the Transformer branch, specifically processing the Grain Features to maintain high spatial resolution. By isolating local feature extraction in a parallel path, the system prevents the "feature smoothing" often observed in pure Transformer architectures, ensuring that the orientation of small gestural tokens (like fingers) remains distinct in the final feature map.

5. Integration and Classification Module Logic

The synthesis of global and local features occurs at the point of concatenation, where the two disparate feature sets are unified into a high-dimensional representation. This unification is processed by the Classification Module, which determines the final KSL label.

Classification Architecture and IRFFN Logic

The module utilizes a modified Inverted Residual Feed-Forward Network (IRFFN) structure. A critical design choice here is the "expansion-reduction" cycle: weight matrices W_1 and W_2 expand the feature dimension to a factor of 4d before reducing it back to d. This allows the system to learn complex interdependencies between the local CNN features and global Transformer features.

Final Classification Sequence:

* Global Average Pooling (GAP): Aggregates the high-dimensional feature maps into a consolidated vector.
* Fully Connected (FC) Layer: Projects the pooled vector into the hidden feature space.
* n-way Softmax: Produces the final probability distribution across KSL labels.

Normalization and Activation Standards:

* GELU Activation: Employed for all non-linear transformations in the classification module.
* Hybrid Normalization: Strategic application of Batch Normalization (BN) for the CNN and MLP components, with Layer Normalization (LN) reserved for Transformer-specific elements to stabilize training across different feature distributions.

6. System Performance and Resource Optimization Analysis

The proposed architecture prioritizes "mobile-size" efficiency, maintaining a low parameter count of approximately 1.5M. By adopting a parallel multi-branch approach, the system avoids the exponential growth in computational operations required by sequential stage-wise hybrids.

Performance Benchmarks and Computational Complexity

Dataset	Accuracy	Parameters	Computational Complexity
Lab-Controlled Dataset (20 Labels)	98.30%	~1.52M	1.17 GMac
KSL Benchmark Dataset (77 Labels)	89.00%	~1.50M	245.5 MMac

Hardware Impact and Efficiency Analysis

System validation was conducted on an NVIDIA GPU with 32GB RAM utilizing CUDA 11.7. The results demonstrate that the model outperforms existing state-of-the-art architectures like TSN (which achieved 79.80% on the KSL dataset) while requiring fewer FLOPs. This efficiency is a direct result of the parallel design; the model minimizes GMacs/MMacs by avoiding redundant feature processing stages, making it highly suitable for real-time, signer-independent applications where hardware resources are constrained.

7. Technical Implementation Requirements

To replicate the system performance and ensure architectural stability, the following implementation guidelines are mandatory:

1. Hardware & Environmental Configuration:
  * GPU: NVIDIA GPU with 32GB RAM minimum (NVIDIA-SMI with Persistence Mode enabled).
  * Software Stack: CUDA version 11.7, Adam Optimizer.
2. Optimizer & Regularization Settings:
  * Learning Rate: 0.001 (initial).
  * Weight Decay: 0.0001.
  * Dropout Rate: 0.1 (applied to mitigate overfitting in the classification module).
3. Training Protocol:
  * Batch Size: 32.
  * Epoch Limit: 300 (to ensure convergence across the multi-branch features).
  * Data Split: 70% Training / 30% Testing (adhering to a signer-independent evaluation where possible).

By integrating a grain-based convolutional front-end with parallel global and local feature branches, this architecture represents a significant contribution to the field of sign language recognition, delivering a robust and computationally efficient solution for real-world KSL translation.

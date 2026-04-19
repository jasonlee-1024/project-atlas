# COMS 6998 - High Performance Machine Learning
## Homework Assignment 4

**Instructor:** Dr. Kaoutar El Maghraoui
**Due Date:** April 20, Spring 2026 | **Max Points:** 100

---

## Instructions

This lab is to be completed individually. Submissions will be checked to verify that each student is the sole author of their work.

In this lab, you will explore two model compression techniques that allow machine learning models to run efficiently on resource-constrained hardware:

- **Post-training quantization** — reducing the precision of a trained model's weights and activations from 32-bit floating-point to 8-bit fixed-point integers.
- **Pruning** — inducing sparsity by setting individual weights to zero. You will examine how pruning affects model accuracy, how fine-tuning can recover lost accuracy, and how the timing of pruning (before vs. after training) impacts the result.

You will complete this assignment in a Google Colaboratory notebook that has been prepared for you.

---

## Background

Machine learning models are typically trained using 32-bit floating-point data. However, floating-point arithmetic is expensive in terms of area, performance, and energy to implement in hardware. While 32 bits of precision may be needed during training to capture very small gradient steps, that level of precision is usually unnecessary during inference. Lowering arithmetic precision saves both circuit area and memory bandwidth, helping hardware designers reduce costs on several fronts.

ML accelerators therefore typically implement fixed-point arithmetic at much lower precision. 8-bit unsigned integers are a common target, but some accelerators go as low as single-bit binary arithmetic. There are two broad approaches to quantization: post-training quantization and quantization-aware training. Quantization-aware training [1] can preserve more accuracy, but we focus only on post-training quantization in this assignment. Post-training quantization comes in two variants: 1) static and 2) dynamic, which you can read about here and here.

Pruning reduces model size by removing redundant weights. In unstructured magnitude pruning, individual weights with the smallest absolute values are set to zero, producing a sparse weight matrix. The pruned model can then be fine-tuned, essentially retrained briefly with a small learning rate, to recover most of the lost accuracy. A key design choice is when to prune: pruning a trained model removes weights that the network has learned are least important (as measured by magnitude), whereas pruning at initialization forces the network to learn with a fixed sparse structure from the start. You will explore both strategies in Q6 and Q7 using PyTorch. For a detailed walkthrough, refer here.

---

## Getting Started

1. Navigate to the following Google Drive link here
2. Select **File → Save a copy in Drive**
3. Select **Runtime → Change runtime type**, then select **GPU** under the Hardware accelerator drop-down menu as shown in Figure 1.

> **Figure 1:** How to switch the Google Colaboratory Runtime

The Colab notebook loads the CIFAR-10 dataset and trains a simple convolutional neural network (CNN) to classify it. Complete the notebook by filling in the code blocks and answering questions to quantize and prune the CNN step by step.

> **Note on training time:** The setup cell, Q5, and Q6–Q7 each trigger training runs on GPU that take a few minutes. Make sure your runtime is set to GPU before starting the notebook to avoid long waits.

> **Scope note:** For quantization, this assignment covers static post-training quantization only. Dynamic PTQ (where scale factors are computed at inference time) is not covered.

---

## Submission

Submit the completed notebook (`.ipynb`) along with a PDF writeup containing all your written responses. Make sure to include your name and UNI in the writeup.

| What | Where to submit |
|------|----------------|
| Completed notebook with all cells run (Q1–Q7) | `.ipynb` file |
| Code written by the student (Q1–Q7) | `.ipynb` file |
| Plots, if applicable (Q1–Q7) | `.ipynb` file |
| Written reflections for Q6 and Q7 | PDF writeup |
| Written answers for Q8 | PDF writeup |

Make sure your code is nicely formatted and has comments.

---

## Grading Structure

| Question | Topic | Points |
|----------|-------|--------|
| Q1 | Visualize Weights | 15 |
| Q2 | Quantize Weights | 15 |
| Q3 | Visualize Activations | 15 |
| Q4 | Quantize Activations | 15 |
| Q5 | Quantize Biases | 10 |
| Q6 | Pruning and Fine-Tuning | 10 |
| Q7 | Pruning Before vs. After Training | 10 |
| Q8 | GPTQ — Accurate Post-Training Quantization for Generative Pre-trained Transformers | 10 |
| | **Total** | **100** |

---

## Q8: GPTQ (10 points)

Read the GPTQ paper [2], which introduces a method for compressing large language models like OPT-175B and BLOOM-176B by reducing their bit-widths to 3 or 4 bits with minimal accuracy loss. Answer each sub-question in 2–3 sentences.

1. What are the main innovations of GPTQ that enable efficient quantization of models with hundreds of billions of parameters?

2. What makes GPTQ's approach more scalable than prior methods, and how does its use of approximate second-order information address challenges in layer-wise quantization?

3. Identify one limitation of GPTQ discussed in the paper. Suggest a possible way to address this in future work.

---

## References

[1] Benoit Jacob et al. *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference.* https://arxiv.org/abs/1712.05877

[2] Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* ICLR 2023. https://arxiv.org/abs/2210.17323

# Table of Contents (ongoing)
* [Large Vision Language Models](#large-vision-language-models)
	* [Architecture Modification](##architecture-modification)
* [Video Comprehension MLLMs](#video-comprehension-mllms)
* [Vision-Audio-Text](#vision-audio-text)
* [Understanding and Generation](#understanding-and-generation)
* [Vision Encoder](#vision-encoder)
* [Efficient](#efficient)
* [Benchmarks](#benchmarks)

# Large Vision Language Models

| Paper                                                                                                                                                        | Date  | Description                                                                               |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----- | ----------------------------------------------------------------------------------------- |
| [Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs](http://arxiv.org/abs/2406.16860)                                                   | 24.06 | Integrate multiple vision encoders.<br>Open source dataset.                               |
| [Unveiling Encoder-Free Vision-Language Models](http://arxiv.org/abs/2406.11832)                                                                             | 24.06 | Without vision encoder;<br>from scratch;<br>Add new supervision signal into vision tokens |
| [PaliGemma: A versatile 3B VLM for transfer](http://arxiv.org/abs/2407.07726)                                                                                | 24.07 |                                                                                           |
| [xGen-MM (BLIP-3): A Family of Open Large Multimodal Models](http://arxiv.org/abs/2408.08872)                                                                | 24.08 |                                                                                           |
| [EAGLE: Exploring The Design Space for Multimodal LLMs with Mixture of Encoders](http://arxiv.org/abs/2408.15998)                                            | 24.08 | Integrate multiple vision encoders.                                                       |
| [LLaVA-OneVision: Easy Visual Task Transfer](http://arxiv.org/abs/2408.03326)                                                                                | 24.08 | Image, video                                                                              |
| [Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Multimodal Models](http://arxiv.org/abs/2409.17146)                                        | 24.09 |                                                                                           |
| [NVLM: Open Frontier-Class Multimodal LLMs](http://arxiv.org/abs/2409.11402)                                                                                 | 24.09 |                                                                                           |
| [Mono-InternVL: Pushing the Boundaries of Monolithic Multimodal Large Language Models with Endogenous Visual Pre-training](https://arxiv.org/abs/2410.08202) | 24.10 | Without vision encoder; finetune                                                          |
| [ARIA : An Open Multimodal Native Mixture-of-Experts Model](http://arxiv.org/abs/2410.05993)                                                                 | 24.10 | MoE                                                                                       |
## Architecture Modification

| Paper                                                                                                                               | Date  | Description                                                                      |
| ----------------------------------------------------------------------------------------------------------------------------------- | ----- | -------------------------------------------------------------------------------- |
| [Ovis: Structural Embedding Alignment for Multimodal Large Language Model](https://arxiv.org/abs/2405.20797)                        | 24.05 | Replace projector with MLP which place a softmax in middle other than GeLU       |
| [WINGS: Learning Multimodal LLMs without Text-only Forgetting](http://arxiv.org/abs/2406.03496)                                     | 24.06 | Address text-only forgetting by adding lora in multi-head self-attention modules |
| [TroL: Traversal of Layers for Large Language and Vision Models](http://arxiv.org/abs/2406.12246)                                   | 24.06 | Reuse LLM's layers                                                               |
| [CLIP-MoE: Towards Building Mixture of Experts for CLIP with Diversified Multiplet Upcycling](https://arxiv.org/abs/2409.19291)     | 24.09 | Integrate MoE into CLIP                                                          |
| [Phantom of Latent for Large Language and Vision Models](http://arxiv.org/abs/2409.14713)                                           | 24.09 | Add fake dimension into multi-head self-attention modules                        |
| [TG-LLaVA: Text Guided LLaVA via Learnable Latent Embeddings](http://arxiv.org/abs/2409.09564)                                      | 24.09 | Enhance visual features with text tokens                                         |
| [Deciphering Cross-Modal Alignment in Large Vision-Language Models with Modality Integration Rate](http://arxiv.org/abs/2410.07167) | 24.10 | Add scaler and bias after nomalization layer                                     |
## 3D

| Paper                                                                                                            | Date  | Description |
| ---------------------------------------------------------------------------------------------------------------- | ----- | ----------- |
| [LLaVA-3D: A Simple yet Effective Pathway to Empowering LMMs with 3D-awareness](http://arxiv.org/abs/2409.18125) | 24.09 |             |


# Video Comprehension MLLMs

| Paper                                                                                                                     | Date  |
| ------------------------------------------------------------------------------------------------------------------------- | ----- |
| [LLaMA-VID: An Image is Worth 2 Tokens in Large Language Models](http://arxiv.org/abs/2311.17043)                         | 23.11 |
| [Flash-VStream: Memory-Based Real-Time Understanding for Long Video Streams](http://arxiv.org/abs/2406.08085)             | 24.06 |
| [Exploring the Design Space of Visual Context Representation in Video MLLMs](http://arxiv.org/abs/2410.13694)             | 24.10 |
| [xGen-MM-Vid (BLIP-3-Video): You Only Need 32 Tokens to Represent a Video Even in VLMs](https://arxiv.org/abs/2410.16267) | 24.10 |



# Vision-Audio-Text

| Paper                                                                                                                       | Date  |
| --------------------------------------------------------------------------------------------------------------------------- | ----- |
| [Mirasol3B: A Multimodal Autoregressive Model for Time-Aligned and  Contextual Modalities](http://arxiv.org/abs/2311.05698) | 23.11 |



# Understanding and Generation

| Paper                                                                                                                    | Date  | Description                                                |
| ------------------------------------------------------------------------------------------------------------------------ | ----- | ---------------------------------------------------------- |
| [Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model](http://arxiv.org/abs/2408.11039)     | 24.08 | AR+Diffusion                                               |
| [Emu3: Next-Token Prediction is All You Need](http://arxiv.org/abs/2409.18869)                                           | 24.09 | AR; Generation and understaning are different models       |
| [Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation](http://arxiv.org/abs/2410.13848) | 24.10 | Generation and understanding use different vision encoders |

# Visual Encoder

| Paper                                                                                                                                                                                                                          | Date  | Description                                                |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----- | ---------------------------------------------------------- |
| [LiT: Zero-Shot Transfer with Locked-image text Tuning](http://arxiv.org/abs/2111.07991)                                                                                                                                       | 21.11 |                                                            |
| [MoVQ: Modulating Quantized Vectors for High-Fidelity Image Generation](https://arxiv.org/abs/2209.09002)                                                                                                                      | 22.09 | Improve VQ-based autoencoder                               |
| [When and why vision-language models behave like bags-of-words, and what to do about it?](http://arxiv.org/abs/2210.01936)                                                                                                     | 22.10 | Contrastive learning is hard to learn compostion and order |
| [Rethinking Video ViTs: Sparse Video Tubes for Joint Image and Video Learning](http://arxiv.org/abs/2212.03229)                                                                                                                | 22.12 | Extend ViT to video                                        |
| [ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders](https://openaccess.thecvf.com/content/CVPR2023/html/Woo_ConvNeXt_V2_Co-Designing_and_Scaling_ConvNets_With_Masked_Autoencoders_CVPR_2023_paper.html) | 23    | ConvNeXt——CNN<br>ConvNeXt——CNN+MAE                         |
| [When are Lemons Purple? The Concept Association Bias of Vision-Language Models](https://aclanthology.org/2023.emnlp-main.886)                                                                                                 | 23    | concept association bias in CLIP                           |
| [Image Captioners Are Scalable Vision Learners Too](http://arxiv.org/abs/2306.07915)                                                                                                                                           | 23.06 | Image Captioners Are Scalable Vision Learners Too          |
| [LanguageBind: Extending Video-Language Pretraining to N-modality by Language-based Semantic Alignment](http://arxiv.org/abs/2310.01852)                                                                                       | 23.10 | Contrastive learning+token masking+LoRA                    |
| [Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs](http://arxiv.org/abs/2401.06209)                                                                                                                        | 24.01 | Integrate VLM with DINOv2                                  |


# Efficient

| Paper                                                                                                                                | Date  | Description         |
| ------------------------------------------------------------------------------------------------------------------------------------ | ----- | ------------------- |
| [Fit and Prune: Fast and Training-free Visual Token Pruning for Multi-modal Large Language Models](https://arxiv.org/abs/2409.10197) | 24.09 | Prune vision tokens |
| [Less is More: A Simple yet Effective Token Reduction Method for Efficient Multi-modal LLMs](http://arxiv.org/abs/2409.10994)        | 24.09 | Prune vision tokens |

# Benchmarks

| Paper                                                                                                                                                   | Date  | Description               |
| ------------------------------------------------------------------------------------------------------------------------------------------------------- | ----- | ------------------------- |
| [Do Pre-trained Vision-Language Models Encode Object States?](https://arxiv.org/abs/2409.10488)                                                         | 24.09 | Object states recognition |
| [HumanEval-V: Evaluating Visual Understanding and Reasoning Abilities of Large Multimodal Models Through Coding Tasks](http://arxiv.org/abs/2410.12381) | 24.10 | Multimodal coding task    |

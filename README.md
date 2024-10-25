# Table of Contents (ongoing)
* [Large Vision Language Models](#large-vision-language-models)
	* [Architecture Modification](##architecture-modification)
* [Video Comprehension MLLMs](#video-comprehension-mllms)
* [Vision-Audio-Text](#vision-audio-text)
* [Understanding and Generation](#understanding-and-generation)
* [Vision Encoder](#vision-encoder)
* [Benchmarks](#benchmarks)

# Large Vision Language Models

| Paper                                                                                                                                                        | Date  | Description                                                 |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----- | ----------------------------------------------------------- |
| [Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs](http://arxiv.org/abs/2406.16860)                                                   | 24.06 | Integrate multiple vision encoders.<br>Open source dataset. |
| [PaliGemma: A versatile 3B VLM for transfer](http://arxiv.org/abs/2407.07726)                                                                                | 24.07 |                                                             |
| [xGen-MM (BLIP-3): A Family of Open Large Multimodal Models](http://arxiv.org/abs/2408.08872)                                                                | 24.08 |                                                             |
| [EAGLE: Exploring The Design Space for Multimodal LLMs with Mixture of Encoders](http://arxiv.org/abs/2408.15998)                                            | 24.08 | Integrate multiple vision encoders.                         |
| [Mono-InternVL: Pushing the Boundaries of Monolithic Multimodal Large Language Models with Endogenous Visual Pre-training](https://arxiv.org/abs/2410.08202) | 24.10 | Without vision encoder; finetune                            |
| [ARIA : An Open Multimodal Native Mixture-of-Experts Model](http://arxiv.org/abs/2410.05993)                                                                 | 24.10 | MoE                                                         |
## Architecture Modification

| Paper                                                                                                                               | Date  | Description                                  |
| ----------------------------------------------------------------------------------------------------------------------------------- | ----- | -------------------------------------------- |
| [CLIP-MoE: Towards Building Mixture of Experts for CLIP with Diversified Multiplet Upcycling](https://arxiv.org/abs/2409.19291)     | 24.09 | Integrate MoE into CLIP                      |
| [Deciphering Cross-Modal Alignment in Large Vision-Language Models with Modality Integration Rate](http://arxiv.org/abs/2410.07167) | 24.10 | Add scaler and bias after nomalization layer |


# Video Comprehension MLLMs

| Paper                                                                                                         | Date  |
| ------------------------------------------------------------------------------------------------------------- | ----- |
| [LLaMA-VID: An Image is Worth 2 Tokens in Large Language Models](http://arxiv.org/abs/2311.17043)             | 23.11 |
| [Flash-VStream: Memory-Based Real-Time Understanding for Long Video Streams](http://arxiv.org/abs/2406.08085) | 24.06 |
| [Exploring the Design Space of Visual Context Representation in Video MLLMs](http://arxiv.org/abs/2410.13694) | 24.10 |



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

| Paper                                                                                                                                    | Date  | Description                                                |
| ---------------------------------------------------------------------------------------------------------------------------------------- | ----- | ---------------------------------------------------------- |
| [LiT: Zero-Shot Transfer with Locked-image text Tuning](http://arxiv.org/abs/2111.07991)                                                 | 21.11 |                                                            |
| [When and why vision-language models behave like bags-of-words, and what to do about it?](http://arxiv.org/abs/2210.01936)               | 22.10 | Contrastive learning is hard to learn compostion and order |
| [When are Lemons Purple? The Concept Association Bias of Vision-Language Models](https://aclanthology.org/2023.emnlp-main.886)           | 23    | concept association bias in CLIP                           |
| [Image Captioners Are Scalable Vision Learners Too](http://arxiv.org/abs/2306.07915)                                                     | 23.06 | Image Captioners Are Scalable Vision Learners Too          |
| [LanguageBind: Extending Video-Language Pretraining to N-modality by Language-based Semantic Alignment](http://arxiv.org/abs/2310.01852) | 23.10 | Contrastive learning+token masking+LoRA                    |
| [Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs](http://arxiv.org/abs/2401.06209)                                  | 24.01 | Integrate VLM with DINOv2                                  |

# Benchmarks

| Paper                                                                                                                                                   | Date  | Description            |
| ------------------------------------------------------------------------------------------------------------------------------------------------------- | ----- | ---------------------- |
| [HumanEval-V: Evaluating Visual Understanding and Reasoning Abilities of Large Multimodal Models Through Coding Tasks](http://arxiv.org/abs/2410.12381) | 24.10 | Multimodal coding task |

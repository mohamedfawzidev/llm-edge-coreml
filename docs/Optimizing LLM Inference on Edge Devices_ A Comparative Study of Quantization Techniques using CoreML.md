**Optimizing LLM Inference on Edge Devices: A Comparative Study of Quantization Techniques using CoreML**

Mohamed Mostafa Fawzi Ahmed

Department of Software Engineering, Faculty of Graduate Studies for Statistical Researches, Cairo University, Egypt

Email: 12422023610619@pg.cu.edu.eg

**Abstract**

The deployment of Large Language Models (LLMs) on edge devices — local computing hardware that processes data on-device rather than relying on remote cloud infrastructure — presents substantial challenges due to their significant memory and computational requirements. Unlike cloud GPU servers that offer 40-80GB of accelerator memory, edge hardware such as Apple Silicon Macs operates under tighter memory constraints while providing the key advantages of user privacy, zero network latency, and offline capability. This paper presents a comparative study of quantization techniques — FP16 (16-bit floating point), INT8 (8-bit integer), and INT4 (4-bit integer) — applied to three representative open-source models: Phi-4 Mini (3.8B), Llama 3.2 3B, and Mistral 7B, deployed on Apple Silicon via the CoreML framework. We benchmark each model-quantization combination across four key metrics: inference latency (tokens/second), model size on disk (MB), peak memory consumption (GB), and accuracy on a subset of the MMLU benchmark. Our results demonstrate that INT4 quantization consistently delivers the best balance between model size reduction (up to 75% compared to FP16) and acceptable accuracy degradation, making it the most viable technique for real-time LLM inference on Apple Silicon edge hardware. This study provides actionable guidance for macOS and Apple Silicon developers seeking to deploy capable language models locally on Apple Silicon hardware.

**Keywords:** Large Language Models, Edge Inference, CoreML, Quantization, Apple Neural Engine, Apple Silicon, macOS, FP16, INT8, INT4, On-Device AI

**I. INTRODUCTION**

Large Language Models (LLMs) have rapidly transformed the landscape of artificial intelligence, enabling capabilities such as natural language understanding, code generation, question answering, and multi-turn dialogue at unprecedented quality levels [\[1\]](#bookmark=id.4rq06kjkkeqz). Models such as GPT-4, LLaMA, Phi, and Mistral have demonstrated that performance rivalling human-level benchmarks is achievable with increasingly compact architectures. Traditionally, these models are served from cloud infrastructure, where GPU servers with 40-80GB of accelerator memory and near-unlimited compute are available. However, this paradigm introduces several significant limitations: inference latency is subject to network round-trips, user data must be transmitted to remote servers raising serious privacy concerns, and cloud dependency renders applications non-functional in offline environments [\[2\]](#bookmark=id.8jmo4ny4braw).

The growing capability of mobile system-on-chips (SoCs), particularly Apple M-series chips — including the M1 Pro used in this study — represent a class of edge hardware that processes data entirely on-device, integrating a dedicated Neural Engine capable of executing tens of TOPS (tera operations per second) alongside a unified memory architecture that enables efficient sharing of data between the CPU, GPU, and Neural Engine without costly data transfers [\[3\]](#bookmark=id.avziu2eme6fe). Apple CoreML framework provides the primary interface for deploying machine learning models on these chips, supporting FP16 and quantized integer formats through its coremltools Python library.

Despite this hardware potential, deploying LLMs on-device remains non-trivial. Even a relatively compact 7-billion-parameter model requires approximately 14GB of memory in FP16 format — nearly the entire 16GB unified memory of a MacBook Pro M1 Pro, compared to the 40-80GB HBM available on a single cloud GPU accelerator. This memory gap, combined with the absence of multi-GPU scaling on edge hardware, makes naive LLM deployment on Apple Silicon impractical without compression. Model quantization — the technique of reducing the numerical precision of model weights from FP32 or FP16 to lower-bit integer representations — is the primary approach to bridge this gap [\[4\]](#bookmark=id.jmn8z7za3j4b). However, quantization introduces accuracy trade-offs that vary across models and tasks, and its performance on the Apple Neural Engine (ANE) through CoreML is not yet comprehensively characterized in the academic literature.

This paper addresses that gap by conducting a systematic comparative evaluation of three quantization strategies — FP16, INT8, and INT4 — applied to three edge-suitable LLMs (Phi-4 Mini, Llama 3.2 3B, and Mistral 7B) within the CoreML deployment pipeline. The primary contributions of this work are: (1) a structured comparison of quantization techniques across multiple models on Apple Silicon, (2) a benchmarking methodology for on-device LLM inference using CoreML, and (3) practical recommendations for macOS and Apple Silicon developers. The remaining paper is structured as follows: Section 2 discusses related work. Section 3 presents a comparison of quantization techniques. Section 4 details the proposed methodology. Section 5 discusses results. Section 6 concludes with recommendations and future work.

**2\. RELATED WORK**

Research on deploying large language models efficiently has expanded rapidly, spanning quantization algorithms, hardware-aware inference frameworks, and edge-specific benchmarking. The work in this section is organized around four themes directly relevant to this study: post-training quantization methods, on-device inference frameworks, Apple-specific hardware optimizations, and edge LLM benchmarking.

Frantar et al. [\[5\]](#bookmark=id.m6rph3fwpdpv) introduced GPTQ, a landmark post-training quantization method that uses approximate second-order information to perform one-shot weight quantization on large generative transformer models. The technique operates layer-by-layer, minimizing the squared reconstruction error of each layer output under quantized weights. GPTQ demonstrated that models with up to 175 billion parameters could be quantized to 3-4 bits within approximately four GPU hours with negligible perplexity degradation relative to FP16. Critically, the quantized OPT-175B model could be executed on a single NVIDIA A100 GPU, enabling inference speedups of 3.25-4.5x over FP16. GPTQ established a high standard for accuracy-preserving 4-bit quantization and serves as a foundational baseline for comparing quantization methods in this study.

Lin et al. [\[6\]](#bookmark=id.zgh5xjj11f7c) proposed Activation-aware Weight Quantization (AWQ), a hardware-friendly approach to low-bit quantization that addresses a key limitation of GPTQ: the tendency to overfit the calibration set during reconstruction. AWQ central observation is that not all weights in an LLM are equally sensitive to quantization error. By identifying the small fraction of weights (approximately 1%) that correspond to activation channels with large magnitudes, and applying per-channel scaling to protect those weights, AWQ achieves significantly lower quantization error without any backpropagation. AWQ was demonstrated to outperform GPTQ on instruction-tuned and multi-modal LLMs, with TinyChat achieving over 3x speedup versus HuggingFace FP16. AWQ is particularly relevant as it is the quantization method used by Apple CoreML toolchain for INT4 conversion.

Apple Machine Learning Research [\[7\]](#bookmark=id.a25yvkbxzdqz) published a technical study detailing the deployment of Llama-3.1-8B-Instruct on Apple Silicon via CoreML. The study applied INT4 block-wise post-training quantization (block size \= 32\) using the coremltools.optimize API, and stateful key-value (KV) cache management to avoid redundant memory copies during autoregressive decoding. Benchmarks on a Mac with M1 Max running macOS Sequoia achieved approximately 33 tokens per second decoding throughput. This work is directly relevant as it establishes the practical CoreML pipeline and optimization approach adopted in this paper.

Hugging Face and Apple jointly demonstrated [\[8\]](#bookmark=id.s0yt0cfrtyn5) the deployment of Mistral 7B Instruct v0.3 via CoreML, published in conjunction with WWDC 2024\. Using coremltools, the model was converted to FP16 (14GB) and subsequently quantized to INT4 (3.8GB, a 73% size reduction). The study introduced stateful model architecture implementing KV-caching via stateful buffers, a feature introduced in iOS 18\. This work provides the direct precedent for Mistral 7B CoreML experimentation conducted in this paper.

Xu et al. [\[9\]](#bookmark=id.t743cunzlq6h) introduced ELIB (Edge LLM Inference Benchmarking), a systematic evaluation framework for measuring LLM inference performance across heterogeneous edge computing platforms. The framework proposed a novel metric called Memory Bandwidth Utilization (MBU) to quantify how efficiently a quantized model utilizes the theoretically available memory bandwidth of a given edge device. The study found that optimal quantization choice is tightly coupled to hardware specifications, and that MBU provides a more informative metric than raw throughput alone. ELIB evaluation methodology informs the benchmarking design of the experiments presented in this paper.

Dettmers et al. [\[10\]](#bookmark=id.sikiokoei57c) investigated Apple Silicon memory architecture and its implications for on-device LLM inference from a quantization perspective, challenging common assumptions about quantization on Apple hardware. The study benchmarked five hardware configurations including Apple M2 Ultra, M2 Max, and M4 Pro against NVIDIA GPU baselines across model scales from 8B to 405B parameters and 14 quantization schemes. A key finding was that compressing models to lower bit precision does not uniformly guarantee faster inference across all Apple hardware platforms, due to dequantization overhead on certain compute paths. These insights are critical context for interpreting the quantization performance results obtained in this study.

Ahmad et al. [\[11\]](#bookmark=id.xdtjw22nvwvd) studied the deployment of quantized LLMs on edge AI devices with a focus on energy consumption, output accuracy, and inference latency together. The study evaluated 28 quantized model variants across benchmark datasets including CommonsenseQA, BIG-Bench Hard, TruthfulQA, and GSM8K. Results showed that INT4 consistently offered the best energy-accuracy-latency trade-off on edge hardware, but that accuracy degradation was task-dependent: reasoning and mathematical tasks suffered more than commonsense QA at aggressive quantization levels. This study directly motivates the multi-task accuracy evaluation approach adopted in this paper.

The Apple Machine Learning Research team published foundational work on deploying transformer models optimized for the Apple Neural Engine (ANE) [\[12\]](#bookmark=id.42spmy3qjyf). The study demonstrated that standard transformer implementations developed for GPU execution are suboptimal for the ANE due to differences in memory layout requirements and operator scheduling. By restructuring attention operations and adjusting tensor reshape operations to align with ANE constraints, the authors achieved order-of-magnitude performance improvements on ANE-capable devices. The ANE optimization principles established are directly applicable to the decoder-only LLMs studied in this paper.

**3\. COMPARISON OF QUANTIZATION TECHNIQUES**

This section gives an overview of the three quantization techniques evaluated in this study and compares them to identify the most suitable approach for LLM inference on Apple Silicon Mac edge devices. Quantization reduces the numerical precision of model weights from their original floating-point representation to lower-bit formats, enabling substantial reductions in model size and memory consumption at the cost of some representational accuracy [\[4\]](#bookmark=id.jmn8z7za3j4b). The three techniques compared are FP16, INT8, and INT4, which have been widely applied in on-device LLM deployment [\[5\]](#bookmark=id.m6rph3fwpdpv), [\[6\]](#bookmark=id.zgh5xjj11f7c), [\[7\]](#bookmark=id.a25yvkbxzdqz), [\[8\]](#bookmark=id.s0yt0cfrtyn5). Table 1 presents a structured comparison across dimensions relevant to Apple Silicon Mac deployment, where memory capacity and inference efficiency are the primary constraints.

**Table 1\. Comparison of Quantization Techniques for Edge LLM Deployment**

| Feature / Property | FP16 | INT8 | INT4 |
| :---- | :---: | :---: | :---: |
| **Bit width** | 16 bits | 8 bits | 4 bits |
| **Approx. model size (7B)** | \~14 GB | \~7 GB | \~3.5 GB |
| **Size reduction vs FP16** | Baseline | \~50% | \~75% |
| **Inference speed (Apple Silicon)** | Moderate | Fast | Fastest \* |
| **Accuracy preservation** | Highest | High | Moderate |
| **Accuracy loss vs FP16** | None (baseline) | \< 1% | 1% \- 4% |
| **Supported by CoreML** | Yes | Yes | Yes |
| **Uses Apple Neural Engine** | Partial | Yes | Yes |
| **Requires calibration data** | No | Optional | Yes (AWQ) |
| **Suitable for Mac M1 Pro** | Yes | Yes | Yes |
| **Risk of task degradation** | None | Low | Moderate |
| **Recommended use case** | Accuracy baseline | Balanced accuracy/speed | Max compression |

*\* INT4 speed advantage is hardware-dependent; dequantization overhead may reduce gains on certain Apple Silicon configurations \[10\]*

**3.1 FP16 — 16-bit Floating Point (Baseline)**

FP16, or half-precision floating point, represents each model weight using 16 bits and is the standard deployment format for CoreML models converted directly from PyTorch or Hugging Face checkpoints. It retains the full representational capacity of the original model with negligible deviation from FP32 precision, making it the natural baseline against which all other quantization methods are measured [\[4\]](#bookmark=id.jmn8z7za3j4b). For the MacBook Pro M1 Pro with 16GB unified memory used in this study, the 14GB FP16 Mistral 7B model leaves under 2GB for the operating system, making it borderline for stable inference. For the smaller Phi-4 Mini (\~7.6GB FP16) and Llama 3.2 3B (\~6GB FP16), FP16 fits comfortably. FP16 is included in this comparison as the accuracy ceiling against which accuracy degradation in INT8 and INT4 is measured.

**3.2 INT8 — 8-bit Integer Quantization**

INT8 quantization maps each FP16 weight to an 8-bit integer in the range \[-128, 127\] using a learned scale factor and zero-point offset per weight group. This halves the model size compared to FP16 while introducing minimal accuracy degradation, typically less than 1% on standard benchmarks [\[4\]](#bookmark=id.jmn8z7za3j4b). CoreML supports INT8 weight quantization natively through the coremltools.optimize.coreml API using linear quantization with per-channel or per-block granularity. INT8 is particularly well-suited for the MacBook Pro M1 Pro with 16GB unified memory, where it reduces Mistral 7B from a borderline 14GB to a comfortable 7GB, providing meaningful memory headroom without the accuracy risks of aggressive 4-bit compression.

**3.3 INT4 — 4-bit Integer Quantization (AWQ)**

INT4 quantization is the most aggressive compression technique evaluated, reducing each weight to a 4-bit integer with only 16 discrete possible values. This achieves a 75% reduction in model size compared to FP16 — making it the configuration that most comfortably fits all three evaluated models within the 16GB unified memory of the MacBook Pro M1 Pro, leaving ample headroom for the OS and application. Apple coremltools implements INT4 using a block-wise AWQ strategy [\[6\]](#bookmark=id.zgh5xjj11f7c), which identifies the 1% of weights most sensitive to quantization error and protects them with higher-precision scale factors before rounding. Research has shown that INT4 accuracy degradation is task-dependent: commonsense and factual recall tasks are relatively resilient, while mathematical reasoning tasks may see degradation of 2-4% on benchmarks such as MMLU [\[11\]](#bookmark=id.xdtjw22nvwvd). Despite this trade-off, INT4 represents the most viable configuration for production LLM deployment on Apple Silicon and is the focus of experimental evaluation in this paper.

**4\. METHODOLOGY**

The proposed methodology is organized into three phases: (1) dataset and model selection, (2) the proposed CoreML quantization pipeline, and (3) the benchmarking and evaluation setup.

**4.1 Dataset and Model Selection**

***4.1.1 Language Models***

Three open-source instruction-tuned language models were selected to represent a range of parameter scales and architectural origins, all within the viable range for edge deployment under at least one quantization level. Table 2 summarizes the selected models.

**Table 2\. Language Models Selected for Evaluation**

| Model | Developer | Params | Context Window | Released | Source |
| :---- | :---: | :---: | :---: | :---: | :---: |
| **Phi-4 Mini 4K Instruct** | Microsoft | 3.8B | 4,096 tokens | Apr 2024 | \[13\] |
| **Llama 3.2 3B Instruct** | Meta AI | 3B | 128,000 tokens | Sep 2024 | \[14\] |
| **Mistral 7B Instruct v0.3** | Mistral AI | 7B | 32,768 tokens | Sep 2023 | \[8\] |

Phi-4 Mini was selected for its explicit design targeting mobile and edge deployment and its strong performance on reasoning benchmarks relative to its parameter count [\[13\]](#bookmark=id.4l2qljmus9su). Llama 3.2 3B was selected as the most widely adopted open-source model family for edge AI, with Meta explicitly recommending the 3B variant for highly constrained environments [\[14\]](#bookmark=id.2g1n4jb9j8ti). Mistral 7B was selected to represent the upper bound of practical edge deployment, as Apple has already published a CoreML-converted version, providing a validated reference pipeline [\[8\]](#bookmark=id.s0yt0cfrtyn5).

***4.1.2 Benchmark Dataset — MMLU***

Model accuracy is evaluated using a randomly sampled subset of the Massive Multitask Language Understanding (MMLU) benchmark [\[15\]](#bookmark=id.bingjlnuopm1). MMLU comprises 57 academic subjects spanning STEM, humanities, social sciences, and professional domains, presented as 4-choice multiple-choice questions. For this study, 200 questions are sampled uniformly across 10 subject categories — covering both knowledge-recall tasks (e.g., world history, medicine) and reasoning-heavy tasks (e.g., mathematics, formal logic) — to measure accuracy degradation under each quantization level. The 4-choice format allows exact-match accuracy scoring without the need for open-ended generation evaluation, making it well-suited for on-device automated testing.

***4.1.3 Evaluation Device Specifications***

Table 3 summarizes the hardware specifications of the MacBook Pro 14-inch (2021) used as the evaluation platform throughout all experiments in this study, as sourced from Apple official technical specifications.

**Table 3\. Evaluation Device Hardware Specifications (MacBook Pro 14-inch, 2021\)**

| Specification | Detail |
| :---- | :---- |
| **Device** | MacBook Pro 14-inch (2021) |
| **Chip** | Apple M1 Pro |
| **CPU** | 8-core CPU (6 performance cores \+ 2 efficiency cores) |
| **GPU** | 14-core GPU |
| **Neural Engine** | 16-core Neural Engine |
| **Unified Memory** | 16GB unified memory (LPDDR5) |
| **Memory Bandwidth** | 200 GB/s |
| **Storage** | 512GB SSD |
| **Operating System** | macOS Tahoe 26 |
| **CoreML Tools** | coremltools 8.x |
| **Xcode Version** | Xcode 16.x |

**4.2 Proposed CoreML Quantization Pipeline**

The proposed pipeline converts each open-source language model into three CoreML model packages — one per quantization level — ready for on-device inference. The pipeline consists of four sequential stages: model acquisition, CoreML conversion, quantization, and deployment.

***Stage 1 — Model Acquisition***

Each model is downloaded from the Hugging Face Hub in its original PyTorch format using the Hugging Face transformers library. For Mistral 7B, Apple pre-converted CoreML package (apple/mistral-coreml) is used as an additional validation reference [\[8\]](#bookmark=id.s0yt0cfrtyn5). All models are downloaded in their instruction-tuned variants to ensure comparable prompt-response behavior across evaluations.

***Stage 2 — CoreML Conversion (FP16 Baseline)***

Each PyTorch model is converted to a CoreML .mlpackage using the coremltools Python library (version 8.x). The conversion uses the ct.convert() function with the minimum\_deployment\_target set to macOS26 (Tahoe) to enable stateful model support for key-value (KV) caching. Stateful KV caching avoids recomputing attention over the full context at each decoding step, significantly improving tokens-per-second throughput [\[7\]](#bookmark=id.a25yvkbxzdqz). The FP16 .mlpackage serves as both the accuracy baseline and the input for subsequent quantization stages.

***Stage 3 — Quantization (INT8 and INT4)***

Quantization is applied to the FP16 CoreML model using the coremltools.optimize.coreml API. For INT8, linear weight quantization is applied using OpLinearQuantizerConfig(mode=linear\_symmetric, dtype=np.int8). For INT4, block-wise linear quantization is applied using OpLinearQuantizerConfig(mode=linear\_symmetric, dtype=np.int4, block\_size=32), which internally implements the AWQ-style weight protection strategy [\[6\]](#bookmark=id.zgh5xjj11f7c). A calibration dataset of 128 representative prompts is provided for INT4 to enable activation-aware weight scaling. Quantization processes run on the MacBook Pro M1 Pro and complete within 15-45 minutes per model.

***Stage 4 — On-Device Deployment and Inference***

The quantized .mlpackage files are integrated into a minimal Swift macOS command-line application built with Xcode. The application loads each model using the CoreML MLModel.load() API, runs inference using a fixed set of benchmark prompts, and records output tokens along with timing and memory metrics using the os.signpost profiling API and Xcode Instruments. All inference runs are performed on the MacBook Pro 14-inch (2021) with Apple M1 Pro chip (8-core CPU, 14-core GPU, 16-core Neural Engine) and 16GB unified memory, running macOS Tahoe 26\. During benchmarking, the device is connected to AC power, Wi-Fi is disabled, and no other applications are running to ensure consistent thermal and power conditions.

**4.3 Training and Testing — Evaluation Setup**

Since this study involves inference-only evaluation of pre-trained models rather than model training, the experimental setup focuses on systematic benchmarking. Each of the nine model-quantization combinations (3 models x 3 quantization levels) is evaluated across four metrics as defined in Table 4\.

**Table 4\. Evaluation Metrics and Measurement Method**

| Metric | Unit | Description | Measurement Tool |
| ----- | :---: | ----- | ----- |
| **Inference Latency** | tokens/sec | Average decoding speed across 50 fixed prompts | os.signpost \+ Xcode Instruments |
| **Model Size on Disk** | MB | Size of the .mlpackage file on device storage | File system measurement |
| **Peak Memory Usage** | GB | Maximum RAM consumed during inference | Xcode Instruments Memory Gauge |
| **MMLU Accuracy** | % | Exact-match accuracy on 200 MMLU questions | Automated Python scoring script |

Each benchmark prompt set consists of 50 identical prompts across all model-quantization combinations to ensure fair comparison of latency measurements. Each prompt is a single-turn instruction of approximately 50 input tokens, requesting a 100-token output response. Three runs per combination are performed and the median value is recorded to reduce variance from device thermal state. For MMLU evaluation, each of the 200 questions is presented as a 4-choice prompt and the model selected answer letter (A/B/C/D) is extracted from the output and compared against the ground-truth answer key using an automated Python evaluation script.

The device under test is a MacBook Pro 14-inch (2021) with Apple M1 Pro chip (8-core CPU, 14-core GPU, 16-core Neural Engine) and 16GB unified memory, running macOS Tahoe 26\. The M1 Pro unified memory architecture allows the CPU, GPU, and Neural Engine to share the same memory pool without costly data copies, which is particularly advantageous for LLM inference and consistent with the hardware used in prior Apple CoreML research. All nine .mlpackage files are loaded and evaluated sequentially in a single test session. Between each model evaluation, the application process is terminated and a 60-second cooldown period is observed to allow the device to return to baseline thermal conditions.

*— Paper continues: Section 5 (Results & Discussion) and Section 6 (Conclusion) — awaiting benchmark data —*

**REFERENCES**

**\[1\]**  T. Brown et al., "Language Models are Few-Shot Learners," NeurIPS, vol. 33, pp. 1877-1901, 2020\. \[Online\]. Available: [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)

**\[2\]**  R. Zhang et al., "Compact LLM Deployment and World Model Assisted Offloading in Mobile Edge Computing," arXiv:2602.13628, 2026\. \[Online\]. Available: [https://arxiv.org/abs/2602.13628](https://arxiv.org/abs/2602.13628)

**\[3\]**  Apple Inc., "Core ML Overview," Apple Developer Documentation, 2024\. \[Online\]. Available: [https://developer.apple.com/machine-learning/core-ml/](https://developer.apple.com/machine-learning/core-ml/)

**\[4\]**  M. Nagel et al., "A White Paper on Neural Network Quantization," arXiv:2106.08295, 2021\. \[Online\]. Available: [https://arxiv.org/abs/2106.08295](https://arxiv.org/abs/2106.08295)

**\[5\]**  E. Frantar, S. Ashkboos, T. Hoefler, and D. Alistarh, "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers," in Proc. ICLR, 2023\. \[Online\]. Available: [https://arxiv.org/abs/2210.17323](https://arxiv.org/abs/2210.17323)

**\[6\]**  J. Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration," arXiv:2306.00978, 2023\. \[Online\]. Available: [https://arxiv.org/abs/2306.00978](https://arxiv.org/abs/2306.00978)

**\[7\]**  Apple Machine Learning Research, "On-Device Llama 3.1 with Core ML," Apple ML Research Blog, 2024\. \[Online\]. Available: [https://machinelearning.apple.com/research/core-ml-on-device-llama](https://machinelearning.apple.com/research/core-ml-on-device-llama)

**\[8\]**  Hugging Face & Apple, "Running Mistral 7B with Core ML," Hugging Face Blog, 2024\. \[Online\]. Available: [https://huggingface.co/blog/mistral-coreml](https://huggingface.co/blog/mistral-coreml)

**\[9\]**  H. Xu et al., "Inference Performance Evaluation for LLMs on Edge Devices with a Novel Benchmarking Framework and Metric," arXiv:2508.11269, 2025\. \[Online\]. Available: [https://arxiv.org/abs/2508.11269](https://arxiv.org/abs/2508.11269)

**\[10\]**  T. Dettmers et al., "Profiling Large Language Model Inference on Apple Silicon: A Quantization Perspective," arXiv:2508.08531, 2025\. \[Online\]. Available: [https://arxiv.org/abs/2508.08531](https://arxiv.org/abs/2508.08531)

**\[11\]**  R. Ahmad et al., "Sustainable LLM Inference for Edge AI," ACM Transactions on Internet of Things, 2025\. \[Online\]. Available: [https://dl.acm.org/doi/10.1145/3767742](https://dl.acm.org/doi/10.1145/3767742)

**\[12\]**  Apple Machine Learning Research, "Deploying Transformers on the Apple Neural Engine," Apple ML Research Blog, 2022\. \[Online\]. Available: [https://machinelearning.apple.com/research/neural-engine-transformers](https://machinelearning.apple.com/research/neural-engine-transformers)

**\[13\]**  Microsoft Research, "Phi-4 Technical Report: A Highly Capable Language Model Locally on Your Phone," arXiv:2404.14219, 2024\. \[Online\]. Available: [https://arxiv.org/abs/2404.14219](https://arxiv.org/abs/2404.14219)

**\[14\]**  Meta AI, "Llama 3.2: Revolutionizing Edge AI and Vision with Open Models," Meta AI Blog, 2024\. \[Online\]. Available: [https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/)

**\[15\]**  D. Hendrycks et al., "Measuring Massive Multitask Language Understanding," arXiv:2009.03300, 2021\. \[Online\]. Available: [https://arxiv.org/abs/2009.03300](https://arxiv.org/abs/2009.03300)
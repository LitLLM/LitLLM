# LitLLM: an AI-powered Literature Review Assistant

<p align="center">
  <img src="https://github.com/user-attachments/assets/8901d158-366d-49b7-bde9-7e955c8cd42b" alt="LitLLM Logo" width="200"/>
</p>

> LitLLM is a powerful AI toolkit that transforms how researchers write literature reviews using advanced Retrieval-Augmented Generation (RAG) to create accurate, well-structured related work sections in minutes rather than hours or days.

[![TMLR 2025](https://img.shields.io/badge/TMLR-2025-blue)](https://arxiv.org/abs/2412.15249)
[![arXiv](https://img.shields.io/badge/arXiv-2412.15249-b31b1b.svg)](https://arxiv.org/abs/2412.15249)
[![Website](https://img.shields.io/badge/Demo-Live-green)](https://litllm.onrender.com)

## Try It Now!
**🔗 [LitLLM Web App](https://litllm.onrender.com)**

## LitLLM Demo

<p align="center">
  <a href="https://www.youtube.com/watch?v=u6nCvpH0o30">
    <img src="https://img.youtube.com/vi/u6nCvpH0o30/maxresdefault.jpg" alt="LitLLM Demo" width="800"/>
  </a>
</p>

## What is LitLLM?

LitLLM is a tool that helps researchers write literature reviews with the assistance of Large Language Models (LLMs). Writing literature reviews is one of the most time-consuming aspects of academic research, particularly in rapidly evolving fields like machine learning. LitLLM addresses this challenge by decomposing the task into two key components:

1. **Retrieval**: Finding the most relevant papers for your research
2. **Generation**: Creating a coherent, well-structured literature review based on the retrieved papers

## How It Works

LitLLM follows principles of Retrieval-Augmented Generation (RAG):

1. **Keyword Extraction**: LLMs identify meaningful keywords from your research abstract
2. **Multi-Strategy Search**: Combines keyword-based and embedding-based search to query academic databases (Google Scholar, OpenAlex)
3. **Re-ranking with Attribution**: An LLM re-ranks search results to prioritize the most relevant papers 
4. **Structured Generation**: Creates a literature review following a plan-based approach that organizes the content meaningfully

![LitLLM Framework](https://litllm.github.io/static/images/litllm.png)

## Key Features

- **Hybrid Retrieval**: Combines keyword and embedding-based search for optimal coverage
- **Attribution-based Re-ranking**: Prioritizes papers by relevance and importance
- **Plan-based Generation**: Creates structured, coherent literature reviews with fewer hallucinations
- **Interactive Interface**: User-friendly web interface for generating literature reviews

## Interface
> Note: the current interface has some new features not shown in the screenshot below (exporting citation, adding papers directly, high-level plan generation)

![LitLLM Interface](https://litllm.github.io/static/images/react_demo.png)

To use LitLLM:
1. Provide your research idea or abstract in the textbox
2. Select papers from the search results to base your literature review on
3. (Optional) Provide a writing plan to guide the generation of literature review
4. Generate literature review!

## Papers

LitLLM is based on the following two papers:

```
@article{agarwal2024llms,
  title={LitLLMs, LLMs for Literature Review: Are we there yet?},
  author={Agarwal*, Shubham and Sahu*, Gaurav and Puri*, Abhay and Laradji, Issam H and Dvijotham, Krishnamurthy DJ and Stanley, Jason and Charlin, Laurent and Pal, Christopher},
  journal={arXiv preprint arXiv:2412.15249},
  year={2024}
}
```

```
@article{agarwal2024litllm,
  title={Litllm: A toolkit for scientific literature review},
  author={Agarwal*, Shubham and Sahu*, Gaurav and Puri*, Abhay and Laradji, Issam H and Dvijotham, Krishnamurthy DJ and Stanley, Jason and Charlin, Laurent and Pal, Christopher},
  journal={arXiv preprint arXiv:2402.01788},
  year={2024}
}
```

## Feedback and Contributions
For feedback:
- Open an [issue](../../issues) in this repository
- For private feedback, please fill out our [feedback form](https://docs.google.com/forms/d/e/1FAIpQLSfZxWZWjQELIQA-Iz61Vp3rQhkuUuag166REYJXD-EPOQzHPA/viewform?usp=send_form)
- Alternatively, you can also email us at litllm [at] duck [dot] com

We welcome contributions from the community to make LitLLM even better! Please reach out to us at litllm [at] duck [dot] com with any queries. 

---

<p align="center">Made with ❤️ by researchers at Mila and ServiceNow Research</p> 

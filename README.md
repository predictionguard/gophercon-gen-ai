# GopherCon 2023 Generative AI Workshop Materials

This repo includes the materials for the 2023 GopherCon workshop: "Generative AI - Go Applications that Integrate GPT-Like Models." The materials were prepared by Daniel Whitenack for live attendees at the conference. However, others might benefit from them as they build generative AI applications with Go. 

## Description

Learn how to build state-of-the-art, practical AI applications in Go with a focus on generative AI models like those used in ChatGPT.

Are you wondering how models like these can create practical value for you? Are you hesitant to integrate these models because of new/weird tooling, unclear reliability, or privacy concerns? Well, the goal of this half-day workshop is to share how you can leverage the latest AI-driven functionalities to build domain-specific chat assistants, fast neural search over knowledge bases, and automated agents. From the beginning, you will pair program with the instructor, walking through the fundamentals of the current wave of AI models along with key concepts like prompt engineering, augmentation, chaining, and agents.

What a student is expected to learn:
- AI Fundamentals
- Building a Chat Assistant
- Integrating External Knowledge
- Chaining and Agents

Prerequisites: 
- Basic understanding of the Go programming language is recommended. Take the Go tour here: https://tour.golang.org/welcome/
- Basic understanding of and comfort in working on the command line. Learn the command line here: https://www.codecademy.com/learn/learn-the-command-line
- Basic understanding of and comfort with REST API interactions.

Recommended Preparation:
- Install and configure an editor for Go.
- Have a functioning Go environment installed.
- Sign up for a Github account, if you do not already have one.
- Computers should be capable of modern software development, such as access to install and run binaries, install a code editor, etc.

## Agenda

0. Introductions, logistics
1. Accessing LLMS
    1. [Example 1](accessing-llms/example1/) - Prompting a text completion model
    2. [Example 2](accessing-llms/example2/) - Generating Go code with WizardCoder
2. Basic Prompting
    1. [Example 1](basic-prompting/example1/) - Autocomplete
    2. [Example 2](basic-prmopting/example2/) - Zero shot, prompt structure
    3. [Example 2](basic-prmopting/example3/) - Few shot
3. Prompt Engineering
    1. [Example 1](prompt-engineering/example1/) - Prompt templates
    2. [Example 2](prompt-engineering/example2/) - Prompt templates continued
    3. [Example 3](prompt-engineering/example3/) - Parameters, temperature
    4. [Example 4](prompt-engineering/example4/) - Parameters, max tokens
    5. [Example 5](prompt-engineering/example5/) - Contraints, types
    6. [Example 6](prompt-engineering/example6/) - Guards on factuality, toxicity
4. Retrieval Augmented Generation
    1. [Example 1](retrieval-augmentation/example1/) - Preparing data for injection into prompts
    2. [Example 2](retrieval-augmentation/example2/) - Embedding chunks
    3. [Example 3](retrieval-augmentation/example3/) - Similarity search
    4. [Example 4](retrieval-augmentation/example4/) - Similarity search, RAG, continued
    5. [Example 5](retrieval-augmentation/example5/) - Chaining
    6. [Example 6](retrieval-augmentation/example6/) - Generalizing retrieval
5. Conclusions
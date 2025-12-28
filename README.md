# Kemet AI Task

## Overview
Kemet AI Task is a modular and extensible framework designed for building and orchestrating AI-driven applications. It provides capabilities for language generation, retrieval-augmented generation (RAG), and seamless integration with multiple large language model (LLM) providers. The project is structured to ensure scalability, maintainability, and ease of use.

## Features
- **Language Generation**: Generate text using various LLM providers.
- **Retrieval-Augmented Generation (RAG)**: Combine retrieval techniques with language generation for enhanced results.
- **Multi-Provider Support**: Easily switch between LLM providers like OpenAI, Cohere, and Gemini.
- **Customizable Pipelines**: Build and orchestrate custom pipelines for ingestion, retrieval, and generation.
- **UI Components**: Interactive user interface for chat-based applications.

## Project Structure
```
Kemet AI Task
├── app.py                 # Entry point for the application
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
├── data/                  # Data storage (e.g., vector store, metadata)
├── logs/                  # Log files
├── Planning/              # Documentation and planning resources
├── src/                   # Source code
│   ├── config/            # Configuration files
│   ├── core/              # Core utilities (e.g., logging, language utilities)
│   ├── generation/        # Text generation modules
│   ├── ingestion/         # Data ingestion and processing
│   ├── llm/               # LLM provider interfaces and templates
│   ├── llmproviders/      # Legacy LLM provider implementations
│   ├── orchestrator/      # Orchestration logic for pipelines
│   ├── retrieval/         # Retrieval and ranking modules
├── ui/                    # User interface components
```

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd kemet-ai-task
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Start the application:
   ```bash
   python app.py
   ```

2. Access the user interface or interact with the system via the command line.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- Inspired by modern AI frameworks and best practices.
- Special thanks to the contributors and the open-source community.

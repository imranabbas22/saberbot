# Changelog

## v2.3.0 - 2026-05-04
### Added
- **User Guidelines**: Authored a comprehensive user manual (`docs/USER_GUIDE.md`) to instruct users on how to query the chatbot, use the compliance checker, and understand offline AI limitations.

## v2.2.1 - 2026-05-04
### Fixed
- Fixed an issue where the chatbot responded with an `LLM Offline` error due to a failing Hugging Face download (`gemma-4-E4B`). Implemented a graceful fallback mechanism to the local `qwen.gguf` file to ensure continuous operation.
- Deleted the legacy Streamlit `app.py` codebase to prevent confusion and enforce the new architecture.

## v2.2.0 - 2026-05-04
### Changed
- Standardized the default local LLM to `Gemma-4-E4b`.
- Removed all residual `Ollama` fallback logic. The application is now strictly self-contained.
### Added
- Integrated `huggingface_hub` into the backend startup sequence to automatically pull the required GGUF model directly from Hugging Face if it isn't present locally.

## v2.1.0 - 2026-05-04
### Added
- **Compliance Check Engine**: New functionality allowing users to upload documents to check against local laws.
- **Multi-Format Processing**: Full offline extraction for `.pdf`, `.docx`, `.xlsx`, and `.txt` files.
- **OCR Support**: Integrated Tesseract OCR (`pytesseract`) to dynamically extract text from images (`.png`, `.jpg`) and feed it to the local LLM.
- **Aesthetic UI**: Added a beautiful visual dashboard for the compliance results featuring glassmorphism cards and an overall compliance score gauge.

## v2.0.0 - 2026-05-04
### Changed
- Massively pivoted architecture from a Streamlit prototype to a production-ready **Vite (React) + FastAPI** stack.
- Completely removed the dependency on external Ollama. Integrated **`llama-cpp-python`** to allow the application to run 100% offline internally.
- Redesigned the UI with a modern, glassmorphism aesthetic.

### Added
- Integrated **BM25 Lexical Search** to complement existing Vector and PageTree indices.
- Implemented **Reciprocal Rank Fusion (RRF)** to combine three retrieval streams.
- Added RAGAS faithfulness testing integrated with the local LLM.
- Scaffolding for local document ingestion endpoints to allow the client to update legislation without an internet connection.

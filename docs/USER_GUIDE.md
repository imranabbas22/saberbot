# UAE Law RAG Chatbot - User Guidelines

Welcome to the UAE Law RAG Chatbot! This tool provides an offline, secure, and privacy-focused environment to query UAE federal laws and assess documents for legal compliance. Because the entire system runs locally on your machine, your data and documents never leave your computer.

---

## 1. Getting Started

1. **Access the Interface:** Open your web browser and navigate to `http://localhost:8000/` (or the URL provided by your administrator).
2. **Select a Module:** Use the sidebar to toggle between the **Chat Interface** and the **Compliance Check** module.
3. **Data Privacy:** 100% of the document processing and AI generation happens locally.

---

## 2. Using the Chat Interface

The Chat Interface allows you to ask free-form questions about UAE Laws. 

### How to Ask Good Questions
To get the most accurate answers from the AI, follow these best practices:
- **Be Specific:** Instead of "What are labor rights?", ask "What are the rules regarding annual leave calculation under the UAE Labor Law?".
- **Ask for Citations:** Include "Please cite the specific law and article number" in your prompt to ensure the AI provides traceable references.
- **Provide Context:** If your question relates to a specific industry (e.g., healthcare, construction), mention it so the retrieval engine can prioritize relevant statutes.

### Understanding Retrieval Modes
Behind the scenes, the system searches through a database of laws. You may see references to the retrieval mode used:
- **Vector Search:** Good for semantic meaning and conceptual questions.
- **PageTree:** Good for finding structured clauses and exact text matches.
- **Auto (Hybrid):** Combines both methods for maximum accuracy. 

*Note: If the system returns "I don't know," try rephrasing your question to use standard legal terminology.*

---

## 3. Using the Compliance Checker

The Compliance Checker allows you to upload a company policy, SOP, or contract to assess its alignment with UAE laws.

### Supported File Formats
You can upload the following file types:
- **PDF** (`.pdf`)
- **Word Documents** (`.docx`)
- **Excel Spreadsheets** (`.xlsx`)
- **Text Files** (`.txt`)
- **Images/Scans** (`.jpg`, `.png`, `.jpeg`) - *Text will be extracted automatically via OCR.*

### Understanding the Results
After processing, the tool will provide an Overall Compliance Score and a breakdown of findings:
- 🟢 **Compliant:** The uploaded clause aligns with known UAE law.
- 🟡 **Gray Area:** The clause might be missing context, or the law is ambiguous. Human review recommended.
- 🔴 **Non-Compliant:** The clause directly contradicts a retrieved UAE law.

*Disclaimer: The Compliance Score is an AI-generated metric for internal assessment and does not constitute a legally binding audit.*

---

## 4. Limitations & Disclaimers

⚠️ **Important Notice:**
- **No Legal Advice:** This AI assistant is a compliance support tool, not a certified lawyer. The responses generated do not constitute formal legal advice.
- **AI Hallucinations:** While the system uses RAG (Retrieval-Augmented Generation) to ground its answers in real documents, language models can occasionally misinterpret text or "hallucinate" incorrect information. Always cross-reference the cited Article numbers.
- **Offline Limitations:** The local AI model (`Gemma-4-E4b` / `Qwen`) has been optimized to run on standard hardware. Responses may take a few seconds to generate depending on your computer's processing power.

Always consult a qualified legal professional in the UAE for final decisions and formal legal compliance.

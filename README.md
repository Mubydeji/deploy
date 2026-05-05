---

## Key Design Decisions

**Why RAG over fine-tuning?**
Fine-tuning is expensive, slow to update, and still hallucinates. RAG is modular — update the knowledge base without retraining anything.

**Why all-MiniLM-L6-v2?**
Fast, lightweight, and strong on semantic similarity for short to medium text chunks. Runs without a GPU.

**Why ChromaDB?**
Simple, in-memory, no infrastructure required. Right tool for a focused single-domain knowledge base at this scale.

**Why chunk overlap?**
Answers often span chunk boundaries. 150-character overlap ensures no key sentence is split across two non-adjacent chunks.

---

## Tools and Libraries

| Tool | Purpose |
|---|---|
| Python | Core language |
| Streamlit | UI and deployment |
| Gemini 2.5 Flash | Answer generation |
| Sentence Transformers | Query and document embeddings |
| ChromaDB | Vector storage and retrieval |
| LangChain Text Splitters | Recursive chunking |
| Trafilatura | Web scraping and content extraction |

---

## Author

**Mubarak Adesola Adedeji**
Data Analyst · AI Developer | Python · SQL · R · Power BI
[LinkedIn](https://linkedin.com/in/mubarak-adedeji-776804273) · [GitHub](https://github.com/Mubydeji)

---

## License

MIT License — free to use, adapt, and build on with attribution.

# DIY RAG Solution with Vertex AI, LangChain, and Gemini

This project implements a DIY Retrieval-Augmented Generation (RAG) solution that leverages Vertex AI embeddings and vector search to enhance AI-generated responses. The implementation uses the LangChain framework along with the Gemini model for natural language generation. Flipkart product data from Kaggle is used as the dataset, which is stored in BigQuery. The system is built using various GCP services, with Cloud Run hosting the backend and the Streamlit UI for user interaction.

---

## 1. Summary

This DIY RAG solution is designed to integrate robust data retrieval with advanced language generation. By generating embeddings with Vertex AI and employing vector search via Chroma DB, the system fetches relevant information from a BigQuery-hosted dataset containing Flipkart product data. LangChain orchestrates the process, while Gemini generates context-aware responses. Users interact with the solution through a Streamlit-based UI, which is deployed on Cloud Run to provide easy access via endpoints.

---

## 2. Features & Services

### Features

- **Enhanced AI Responses:** Combines vector search and generative models to deliver context-aware answers.
- **Memory Management:** Maintains conversation history to improve the continuity of interactions.
- **User-Friendly Interface:** A Streamlit UI allows users to submit questions and receive immediate responses.
- **Scalable Deployment:** Cloud Run ensures the backend is scalable and accessible via API endpoints.

### Services & Technologies

- **Google Cloud Platform (GCP):**
  - **BigQuery:** Serves as the dataset source for storing Flipkart product data.
  - **Cloud Run:** Deploys the backend and Streamlit UI for scalable and accessible service.
- **Vertex AI:** Generates high-quality embeddings for queries and dataset entries.
- **LangChain Framework:** Orchestrates the retrieval-augmented generation pipeline.
- **Chroma DB:** Acts as the vector database for performing similarity searches.
- **Gemini Model:** Powers the natural language generation to create coherent, context-rich responses.
- **Streamlit:** Provides an interactive UI for users to engage with the system.

---

## 3. Architecture: Big Picture Overview

The system is built to seamlessly blend retrieval and generation for an optimal user experience:

- **User Interaction:**  
  Users access the solution via a Streamlit UI deployed on Cloud Run. The UI is designed for ease of use, enabling users to input questions and receive answers in real time.

- **Backend Processing:**  
  When a query is submitted, it is sent to an API endpoint hosted on Cloud Run. LangChain orchestrates the following processes:
  - **Embedding Generation:** Vertex AI converts the user query and dataset entries into vector embeddings.
  - **Vector Search:** Chroma DB performs a vector similarity search against the BigQuery-hosted Flipkart product data to find the most relevant information.
  - **Context Management:** The system manages conversation history to ensure that responses are context-aware.

- **Response Generation:**  
  The Gemini model integrates the retrieved context with the user's query to generate a coherent and informed response.

- **Result Delivery:**  
  The final AI-generated response is returned to the user through the Streamlit UI, completing the interaction loop.

---

## 4. How to Run

`pip install -r requirements. txt`

`streamlit run demo_1.py`

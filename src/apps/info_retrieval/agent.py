import streamlit as st
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

DEVELOPER_MODE = False  # set to False before demo

# Global Settings
# Set Ollama globally so nothing falls back to OpenAI
Settings.llm = OllamaLLM(model="llama3", base_url="http://localhost:11434")
Settings.chunk_size = 500  # długość chunku (np. 300–700 tokenów)
Settings.chunk_overlap = 50  # ile znaków nakłada się między chunkami np 50-100
# Chunking Config
Settings.node_parser = SentenceSplitter(chunk_size=500, chunk_overlap=50)

# LLM Init
llm = OllamaLLM(model="llama3", base_url="http://localhost:11434", temperature=0)
# Choose an embedding model available in Ollama (e.g. "nomic-embed-text" or "mxbai-embed-large")
embed_model = OllamaEmbedding(model_name="mxbai-embed-large")
# reranker wybiera najlepsze chunki top_n
reranker = FlagEmbeddingReranker(top_n=3, model="BAAI/bge-reranker-large")


def is_grounded(answer_text: str, nodes):
    answer_text = answer_text.lower()
    # sprawdzamy czy fragment odpowiedzi (np. nazwisko) występuje w retrieved
    for n in nodes:
        content = n.node.get_content(metadata_mode="none").lower()
        # heurystyka: czy jakieś słowo >4 litery z odpowiedzi jest w treści chunku
        for word in answer_text.split():
            if len(word) > 4 and word in content:
                return True
    return False


# ---- Przygotowanie lokalnej bazy dokumentów (RAG) ----
# Load your data
documents = SimpleDirectoryReader("docs").load_data()
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

# Agent 1: Analyzer – wyciąga odpowiedź + cytat
analyzer_prompt = ChatPromptTemplate.from_template(
    """Masz dostęp WYŁĄCZNIE do poniższego KONTEKSTU.
    Kontekst:
    {context}

    Pytanie użytkownika: {user_input}

    Zasady:
    - Odpowiadaj TYLKO na podstawie kontekstu (bez wiedzy zewnętrznej).
    - Jeśli brak odpowiedzi w kontekście, zwróć dokładnie: "Brak informacji".
    - Zwróć:
    Odpowiedź: <dokładna wartość wycięta z kontekstu>
    Fragment źródłowy: "<dokładny cytat z kontekstu>"
    """
)
analyzer_chain = analyzer_prompt | llm

# Agent 2: Responder – odpowiada użytkownikowi
if DEVELOPER_MODE:
    responder_prompt = ChatPromptTemplate.from_template(
        """Pytanie: {user_input}
        Analiza:
        {analysis}

        Na podstawie analizy zwróć:
        - Odpowiedź: <krótko, dokładnie>
        - Uzasadnienie: <jedno zdanie, dlaczego ta odpowiedź wynika z cytatu>
        - Pewność (niska/średnia/wysoka): <ocena>
        Jeśli analiza zawiera "Brak informacji", zwróć dokładnie:
        - Odpowiedź: Brak informacji
        - Uzasadnienie: Brak cytatu potwierdzającego w dostarczonym kontekście.
        - Pewność: niska
        """
    )
else:
    responder_prompt = ChatPromptTemplate.from_template(
        """Pytanie: {user_input}
        Analiza:
        {analysis}

        Na podstawie analizy zwróć:
        - Odpowiedź: <dokładnie>
        Jeśli analiza zawiera "Brak informacji", zwróć dokładnie:
        - Odpowiedź: Brak informacji
        """
    )
responder_chain = responder_prompt | llm

# ---- 3. Streamlit UI ----
st.title("🚀 Information Retrieval")
st.write("Zadaj pytanie, a agenci odpowiedzą korzystając z dokumentów.")

user_input = st.text_input("Twoje pytanie:")

if user_input:
    # 1. Pobierz dokumenty powiązane z pytaniem
    # Increase top_k (number of chunks retrieved):
    query_engine = index.as_query_engine(
        similarity_top_k=8,  # podnieś top_k 8-12
        # reranker wybiera najlepsze 3 - Settings
        node_postprocessors=[reranker],
    )
    docs = query_engine.query(user_input)

    # 1a. Zbuduj kontekst z rzeczywiście znalezionych fragmentów
    context = "\n\n---\n\n".join(
        [n.node.get_content(metadata_mode="none") for n in docs.source_nodes]
    )

    # 2. Analyzer
    analysis = analyzer_chain.invoke({"user_input": user_input, "context": context})

    # 3. Responder
    response = responder_chain.invoke({"user_input": user_input, "analysis": analysis})

    if DEVELOPER_MODE:
        # 👉 Inspect what was retrieved
        st.write("### 🔎 Retrieved source chunks:")
        # Why here?
        # docs (the query result from index.as_query_engine()) still contains response.source_nodes.
        # After you pass through your chains, you lose the retrieval metadata — only text flows forward.
        for n in docs.source_nodes:
            st.write("---")
            st.write(n.node.get_content(metadata_mode="none")[:500])  # first 500 chars
            # optionally also show retriever score
            st.write(f"(retriever score: {getattr(n, 'score', None)})")
            # retriever score z source_nodes – to jest relevance score z wyszukiwarki wektorowej
            # (retrievera). On potrafi być dodatni lub ujemny,
            # zależnie od backendu (czasem cosine similarity, czasem dot product).
            # Wynik nie oznacza błędu w grounded check, tylko po prostu taki wynik podobieństwa
            # (np. im mniejszy, tym gorzej dopasowany).
            # To po prostu metryka podobieństwa z retrievera,
            # nie dowód na poprawność.
            # Możesz go logować dla debugowania, ale nie mieszać z grounding.
        # Sprawdzenie grounding
        grounded = is_grounded(analysis, docs.source_nodes)
        if not grounded:
            st.warning(
                "⚠️ Uwaga: odpowiedź mogła nie być dostatecznie uziemiona w źródłach."
            )

    st.write(response)

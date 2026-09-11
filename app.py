import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np


@st.cache_resource
def carregar_modelo():
    return SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


st.title("RAG com PDF por página")

arquivo = st.file_uploader(
    "Envie um arquivo PDF",
    type=["pdf"]
)

if arquivo is not None:
    leitor = PdfReader(arquivo)

    st.success(
        f"PDF carregado com sucesso: {len(leitor.pages)} páginas."
    )

    chunks = []

    for numero, pagina in enumerate(leitor.pages, start=1):
        texto = pagina.extract_text() or ""
        texto = texto.strip()

        if texto:
            chunks.append({
                "pagina": numero,
                "texto": texto
            })

    st.write(f"Foram criados {len(chunks)} chunks.")

    if chunks:
        modelo = carregar_modelo()

        textos = [
            chunk["texto"]
            for chunk in chunks
        ]

        embeddings = modelo.encode(
            textos,
            normalize_embeddings=True
        )

        embeddings = np.array(embeddings)

        st.success("Embeddings criados com sucesso.")

        st.write(
            f"Formato dos embeddings: {embeddings.shape}"
        )

        pergunta = st.text_input(
            "Faça uma pergunta sobre o PDF"
        )

        if pergunta:
            embedding_pergunta = modelo.encode(
                [pergunta],
                normalize_embeddings=True
            )[0]

            similaridades = embeddings @ embedding_pergunta

            indices = np.argsort(similaridades)[::-1][:3]

            st.subheader("Páginas mais relacionadas")

            for indice in indices:
                st.write(
                    f"Página {chunks[indice]['pagina']} "
                    f"- Similaridade: {similaridades[indice]:.3f}"
                )

    with st.expander("Ver páginas separadas"):
        for chunk in chunks:
            st.subheader(f"Página {chunk['pagina']}")
            st.write(chunk["texto"])

import streamlit as st
from pypdf import PdfReader

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

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

    chunks = []

    for numero, pagina in enumerate(leitor.pages, start=1):
        texto = pagina.extract_text()

        if texto:
            chunks.append({
                "pagina": numero,
                "texto": texto
            })

    st.write(f"Foram criados {len(chunks)} chunks.")

    with st.expander("Ver páginas separadas"):
        for chunk in chunks:
            st.subheader(f"Página {chunk['pagina']}")
            st.write(chunk["texto"])

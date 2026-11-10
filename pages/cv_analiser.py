import streamlit as st
from database import db
from services import analiser_service
import os

def save_uploaded_file(uploaded_file):
    cv_folder = "cv"
    os.makedirs(cv_folder, exist_ok=True)
    file_path = os.path.join(cv_folder, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path  # Retornar o caminho do arquivo salvo

# Inicializa variáveis de sessão se não existirem
if "vaga_selecionada" not in st.session_state:
    st.session_state.vaga_selecionada = None
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

st.title("🔍 Analisar Currículo")
st.write("Aqui você pode fazer a análise automática de um CV.")

vagas = db.all() or []
vaga_selecionada = st.selectbox(
    "Selecione a vaga para análise:",
    [vaga['cargo'] for vaga in vagas if 'cargo' in vaga],
    key="vaga_selecionada"
)

uploaded_file = st.file_uploader(
    "Faça upload do seu currículo (PDF)", 
    type="pdf"
)

if uploaded_file is not None:
    st.session_state.uploaded_file = uploaded_file
    st.success(f"Arquivo '{uploaded_file.name}' carregado com sucesso!")

if st.button("Analisar Currículo"):
    if st.session_state.uploaded_file is not None:
        # Salva o arquivo primeiro e obtém o caminho
        file_path = save_uploaded_file(st.session_state.uploaded_file)

        with st.spinner("Analisando o currículo..."):
            vaga_descricao = next(
            (vaga['descricao'] for vaga in vagas if vaga['cargo'] == vaga_selecionada), 
            "Descrição não encontrada."
        )
            # Passa o caminho do arquivo salvo para o analiser_service
            res=analiser_service.analise(file_path, vaga_descricao)
            if res:
                st.write("Resultado da análise:")
                st.json(res)
            else:
                st.warning("Nenhum resultado foi retornado pela análise.")

        st.info(f"Arquivo salvo em: {file_path}")
        st.success("Análise concluída com sucesso!")

        
        # Exibir descrição da vaga se quiser
        # st.write(f"Descrição da vaga: {vaga_descricao}")

    else:
        st.error("Por favor, faça upload de um currículo antes de analisar.")

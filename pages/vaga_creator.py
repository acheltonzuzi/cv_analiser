import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from database import db
from datetime import datetime
load_dotenv()

st.title("📄 Criador de Anúncios de Vaga")

# Inicializa o modelo LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=1)

# Prompt do sistema
prompt_template = """
Você é um especialista em recursos humanos com experiência em criar descrições de vagas de emprego atrativas, claras e objetivas. Seu objetivo é redigir uma descrição de vaga profissional com base nas informações fornecidas abaixo.

## Informações da Vaga:
Cargo: {cargo}
Senioridade: {senioridade}
Tipo de contratação: {tipo_contratacao}
Local de trabalho: {local_trabalho}
Modelo de trabalho: {modelo_trabalho}
Área: {area}
Responsabilidades:
{lista_de_responsabilidades}

Requisitos obrigatórios:
{lista_de_requisitos}

Diferenciais:
{lista_de_diferenciais}

Benefícios:
{lista_de_beneficios}

Sobre a empresa:
{descricao_empresa}

## Instruções:
1. Comece com um título atrativo,direto e profissional.
2. Faça um breve resumo da oportunidade e do impacto do cargo.
3. Estruture a descrição com seções claras: “Responsabilidades”, “Requisitos”, “Diferenciais”, “Benefícios”, “Sobre a empresa”.
4. Use uma linguagem empática e acolhedora, que reflita a cultura da empresa.
5. Evite jargões técnicos excessivos ou linguagem muito formal.

## Resultado esperado:
Uma descrição de vaga pronta para ser publicada em plataformas como LinkedIn, Gupy ou Vagas.com.

Aqui esta um exemplo:
**Título:** Desenvolvedor Backend Pleno - Oportunidade em São Paulo
**Resumo:** Estamos em busca de um Desenvolvedor Backend Pleno para se juntar à nossa equipe em São Paulo. Se você é apaixonado por tecnologia e deseja fazer parte de um time inovador, essa é a sua chance!
**Responsabilidades:**
- Desenvolver e manter aplicações backend em Python.
- Colaborar com a equipe de frontend para integrar APIs.
- Participar de reuniões diárias e contribuir para a melhoria contínua dos processos.
**Requisitos:**
- Experiência em Python e Django.
- Conhecimento em bancos de dados relacionais (PostgreSQL).
- Familiaridade com metodologias ágeis (Scrum/Kanban).
**Diferenciais:**
- Experiência com AWS ou Google Cloud.
- Conhecimento em Docker e Kubernetes.
- Participação em projetos open source.
**Benefícios:**
- Salário competitivo.
- Vale refeição e vale transporte.
- Plano de saúde e odontológico.
- Oportunidade de crescimento e desenvolvimento profissional.
**Sobre a empresa:** Somos uma startup inovadora focada em soluções de inteligência artificial. Valorizamos a diversidade e buscamos criar um ambiente inclusivo e colaborativo.

"""

template = ChatPromptTemplate.from_messages([
    ("system", prompt_template),
    ("user", "Aqui estão as informações da vaga: {curriculo_texto}")
])

tab1,tab2= st.tabs(["📄 Criar Anúncio", "💾 Todas vagas"])

with tab1:
    # Formulário de entrada
    with st.form("vaga_form"):
        st.header("Preencha os detalhes da vaga")
        col1, col2 = st.columns(2)
        with col1:
            cargo = st.text_input("**Cargo**", placeholder="Ex.: Desenvolvedor Backend")
            senioridade = st.selectbox("**Senioridade**", ["Júnior", "Pleno", "Sênior"])
            tipo_contratacao = st.selectbox("**Tipo de contratação**", ["CLT", "PJ", "Freelancer", "Estágio"])
        with col2:
            local_trabalho = st.text_input("**Local de trabalho**", placeholder="Ex.: São Paulo, SP")
            modelo_trabalho = st.selectbox("**Modelo de trabalho**", ["Presencial", "Híbrido", "Remoto"])
            area = st.text_input("**Área**", placeholder="Ex.: Tecnologia da Informação")

        st.markdown("### Descrição da Vaga")
        lista_de_responsabilidades = st.text_area("**Responsabilidades** (uma por linha)")
        lista_de_requisitos = st.text_area("**Requisitos obrigatórios** (uma por linha)")
        col3, col4 = st.columns(2)
        with col3:
            lista_de_diferenciais = st.text_area("**Diferenciais** (opcional, uma por linha)")
        with col4:
            lista_de_beneficios = st.text_area("**Benefícios** (uma por linha)")

        descricao_empresa = st.text_area("**Sobre a empresa**", placeholder="Ex.: Somos uma startup inovadora focada em IA")

        submitted = st.form_submit_button("✨ Criar Anúncio")

    # Geração do anúncio
    # Geração do anúncio
    if submitted:
        campos_obrigatorios = [cargo, senioridade, tipo_contratacao, local_trabalho, modelo_trabalho, area, lista_de_responsabilidades, lista_de_requisitos, lista_de_beneficios, descricao_empresa]

        if all(campos_obrigatorios):
            vaga_contexto = {
                "cargo": cargo,
                "senioridade": senioridade,
                "tipo_contratacao": tipo_contratacao,
                "local_trabalho": local_trabalho,
                "modelo_trabalho": modelo_trabalho,
                "area": area,
                "lista_de_responsabilidades": "\n- ".join(lista_de_responsabilidades.splitlines()),
                "lista_de_requisitos": "\n- ".join(lista_de_requisitos.splitlines()),
                "lista_de_diferenciais": "\n- ".join(lista_de_diferenciais.splitlines()) if lista_de_diferenciais else "Nenhum",
                "lista_de_beneficios": "\n- ".join(lista_de_beneficios.splitlines()),
                "descricao_empresa": descricao_empresa,
                "curriculo_texto": ""  # compatibilidade
            }

            with st.spinner("🔄 Gerando anúncio de vaga, por favor aguarde..."):
                messages = template.invoke(vaga_contexto)
                response = llm.invoke(messages)

                # 💾 Salva temporariamente no estado da sessão
                st.session_state['anuncio_gerado'] = response.content
                st.session_state['cargo'] = cargo  # opcional para salvar depois

            st.success("✅ Anúncio de vaga gerado com sucesso!")
        else:
            st.error("❗ Por favor, preencha todos os campos obrigatórios.")


    # Mostrar o anúncio se já estiver no estado da sessão
    if 'anuncio_gerado' in st.session_state:
        st.markdown("---")
        st.subheader("📢 Anúncio Gerado:")
        st.write(st.session_state['anuncio_gerado'])

        if st.button("💾 Salvar Anúncio"):
            data_criacao = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            db.insert({
                "cargo": st.session_state.get('cargo', 'Sem cargo'),
                "descricao": st.session_state['anuncio_gerado'],
                "data_criacao": data_criacao
            })
            st.success("✅ Anúncio salvo com sucesso!")



with tab2:
    st.header("Todas Vagas")
    # Exibe todas as vagas salvas
    vagas = db.all()
    if vagas:
        for vaga in vagas:
            formatted_date = datetime.strptime(vaga['data_criacao'], "%Y-%m-%d %H:%M:%S").strftime("%d/%m/%Y")
            with st.expander(f"{vaga['cargo']} - {formatted_date}"):
                st.write(vaga['descricao'])
    else:
        st.write("Nenhuma vaga salva ainda.")   
import streamlit as st
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Configuração da Página Streamlit ---
st.set_page_config(page_title="Cérebro Digital de IA - Demonstração", layout="wide")
st.title("🧠 Cérebro Digital de IA: Seu Assistente de Conteúdo Imediato")
st.subheader("Demonstração de Prova de Conceito (POC) para Clientes")

# --- Variáveis de Configuração ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("🔑 ERRO: A chave da API da OpenAI não está configurada.")
    st.info("""
    **Para configurar no Streamlit Cloud:**
    1. Vá em Settings → Secrets
    2. Adicione: `OPENAI_API_KEY = "sua-chave-aqui"`
    
    **Para rodar localmente:**
    Execute: `export OPENAI_API_KEY='sua-chave-aqui'`
    """)
    st.stop()

# --- Funções do RAG ---

@st.cache_resource
def setup_rag_system(file_path):
    """
    Configura o sistema RAG (Retrieval-Augmented Generation) moderno.
    """
    try:
        # 1. Carregar o documento
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()

        # 2. Dividir o texto
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)

        # 3. Criar Embeddings e Vector Store
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vectorstore = Chroma.from_documents(texts, embeddings)
        retriever = vectorstore.as_retriever()
        
        # 4. Criar o prompt template
        template = """Você é um assistente especializado em responder perguntas sobre políticas da empresa.
Use APENAS as informações do contexto abaixo para responder. Se a informação não estiver no contexto, diga que não sabe.

Contexto:
{context}

Pergunta: {question}

Resposta:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # 5. Configurar o LLM
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1, openai_api_key=OPENAI_API_KEY)
        
        # 6. Criar a chain moderna
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return rag_chain, retriever

    except Exception as e:
        st.error(f"Erro ao configurar o sistema RAG: {e}")
        return None, None

# --- Inicialização ---
FILE_PATH = "politicas_empresa.txt"

if not os.path.exists(FILE_PATH):
    st.error(f"❌ Arquivo '{FILE_PATH}' não encontrado!")
    st.info("Certifique-se de que o arquivo está no repositório Git.")
    st.stop()

rag_chain, retriever = setup_rag_system(FILE_PATH)

if rag_chain and retriever:
    st.success(f"✅ Sistema treinado com sucesso! (Base: {os.path.basename(FILE_PATH)})")
    st.markdown("---")

    # --- Interface de Chat ---
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Exibe o histórico de chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Captura a entrada do usuário
    if prompt := st.chat_input("Pergunte algo sobre as políticas da empresa..."):
        # Adiciona a mensagem do usuário ao histórico
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Processa a pergunta
        with st.spinner("🤔 A IA está consultando a base de conhecimento..."):
            try:
                # Buscar documentos relevantes
                source_docs = retriever.get_relevant_documents(prompt)
                
                # Gerar resposta
                response = rag_chain.invoke(prompt)
                
                # Adiciona a resposta da IA ao histórico
                st.session_state.messages.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
                    
                    # Mostra a fonte para o cliente
                    with st.expander("📄 Ver Fontes Consultadas"):
                        for i, doc in enumerate(source_docs, 1):
                            st.code(f"Fonte {i}:\n{doc.page_content[:200]}...", language="text")

            except Exception as e:
                st.error(f"❌ Ocorreu um erro durante a consulta: {e}")
                st.session_state.messages.append({"role": "assistant", "content": "Desculpe, ocorreu um erro ao processar sua solicitação."})

# --- Instruções para o Cliente ---
st.sidebar.title("📋 Instruções para a Demonstração")
st.sidebar.markdown("""
Este protótipo demonstra como sua IA pode responder **apenas** com base nos seus documentos internos.

**Perguntas de Teste:**
1. Qual é a política de home office da TechCorp?
2. Qual o prazo para submeter despesas de viagem?
3. Qual é a missão da empresa?
4. Quantos dias de férias tenho direito?

**O que o cliente vê:** 
- ✅ Respostas precisas
- ✅ Fonte da informação
- ✅ Baseado 100% nos documentos

**O que o cliente compra:** 
A certeza de que a IA não "alucina" e usa apenas o conhecimento da empresa.
""")

st.sidebar.markdown("---")
st.sidebar.info(f"🔑 API Key: {'✅ Configurada' if OPENAI_API_KEY else '❌ Ausente'}")
st.sidebar.info(f"📁 Arquivo: {FILE_PATH}")
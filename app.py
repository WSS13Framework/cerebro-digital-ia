import streamlit as st
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# --- Configuração da Página Streamlit ---
st.set_page_config(page_title="Cérebro Digital de IA - Demonstração", layout="wide")
st.title("🧠 Cérebro Digital de IA: Seu Assistente de Conteúdo Imediato")
st.subheader("Demonstração de Prova de Conceito (POC) para Clientes")

# --- Variáveis de Configuração ---
# O cliente precisará da chave da OpenAI.
# Para a demo, usaremos uma variável de ambiente.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("ERRO: A chave da API da OpenAI não está configurada. Por favor, defina a variável de ambiente 'OPENAI_API_KEY'.")
    st.stop()

# --- Funções do RAG ---

@st.cache_resource
def setup_rag_system(file_path):
    """
    Configura o sistema RAG (Retrieval-Augmented Generation).
    1. Carrega o documento.
    2. Divide o texto em pedaços (chunks).
    3. Cria embeddings e armazena no Vector Store (Chroma).
    """
    try:
        # 1. Carregar o documento (usando o arquivo de exemplo)
        loader = TextLoader(file_path)
        documents = loader.load()

        # 2. Dividir o texto
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)

        # 3. Criar Embeddings e Vector Store
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vectorstore = Chroma.from_documents(texts, embeddings)
        
        # 4. Configurar o Retriever e a Chain de QA
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1, openai_api_key=OPENAI_API_KEY)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            return_source_documents=True
        )
        
        return qa_chain

    except Exception as e:
        st.error(f"Erro ao configurar o sistema RAG: {e}")
        return None

# --- Inicialização ---
FILE_PATH = "/home/ubuntu/politicas_empresa.txt"
qa_chain = setup_rag_system(FILE_PATH)

if qa_chain:
    st.success(f"Sistema treinado com sucesso! (Base: {os.path.basename(FILE_PATH)})")
    st.markdown("---")

    # --- Interface de Chat ---
    
    # Inicializa o histórico de chat
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
        with st.spinner("A IA está consultando a base de conhecimento..."):
            try:
                result = qa_chain.invoke({"query": prompt})
                response = result["result"]
                
                # Adiciona a resposta da IA ao histórico
                st.session_state.messages.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
                    
                    # Mostra a fonte para o cliente (Prova de que a IA "leu" o documento)
                    st.caption("---")
                    st.caption("Fonte Consultada (Prova de Conceito):")
                    for doc in result["source_documents"]:
                        st.code(doc.page_content[:150] + "...", language="text")

            except Exception as e:
                st.error(f"Ocorreu um erro durante a consulta: {e}")
                st.session_state.messages.append({"role": "assistant", "content": "Desculpe, ocorreu um erro ao processar sua solicitação."})

# --- Instruções para o Cliente ---
st.sidebar.title("Instruções para a Demonstração")
st.sidebar.markdown("""
Este protótipo demonstra como sua IA pode responder **apenas** com base nos seus documentos internos.

**Perguntas de Teste (Baseadas no arquivo):**
1. Qual é a política de home office da TechCorp?
2. Qual o prazo para submeter despesas de viagem?
3. Qual é a missão da empresa?

**O que o cliente vê:** Respostas precisas e a fonte da informação.
**O que o cliente compra:** A certeza de que a IA não "alucina" e usa apenas o conhecimento da empresa.
""")

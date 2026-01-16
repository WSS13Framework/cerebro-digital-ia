import streamlit as st
import os
import tempfile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Configuração da Página ---
st.set_page_config(
    page_title="Cérebro Digital de IA - Demonstração REAL",
    page_icon="🧠",
    layout="wide"
)

# --- Header ---
st.title("🧠 Cérebro Digital de IA: Demonstração com SEUS Documentos")
st.markdown("""
### 🎯 **Teste AGORA com seus próprios arquivos!**
Faça upload de seus documentos e veja a IA responder baseada **exclusivamente** no seu conteúdo.
""")

# --- Configuração API ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("🔑 ERRO: Chave da API OpenAI não configurada.")
    st.info("Entre em contato com o administrador para configurar a demonstração.")
    st.stop()

# --- Funções ---
def load_document(uploaded_file):
    """Carrega diferentes tipos de documentos"""
    suffix = os.path.splitext(uploaded_file.name)[1]
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
        elif suffix == ".docx":
            loader = Docx2txtLoader(tmp_path)
        elif suffix == ".txt":
            loader = TextLoader(tmp_path, encoding='utf-8')
        else:
            return None
        
        documents = loader.load()
        os.unlink(tmp_path)
        return documents
    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {e}")
        return None

def setup_rag_system(documents):
    """Configura o sistema RAG com os documentos do cliente"""
    try:
        # Dividir texto
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)
        
        # Criar embeddings e vector store
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vectorstore = Chroma.from_documents(texts, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Criar prompt
        template = """Você é um assistente especializado em responder perguntas sobre os documentos fornecidos.

REGRAS IMPORTANTES:
1. Use APENAS as informações do contexto abaixo
2. Se a informação NÃO estiver no contexto, diga claramente: "Não encontrei essa informação nos documentos fornecidos"
3. Seja preciso e cite trechos relevantes quando possível
4. Responda em português de forma clara e profissional

Contexto dos documentos:
{context}

Pergunta do usuário: {question}

Resposta:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Configurar LLM
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            openai_api_key=OPENAI_API_KEY
        )
        
        # Criar chain
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return rag_chain, retriever, len(texts)
    
    except Exception as e:
        st.error(f"Erro ao configurar sistema: {e}")
        return None, None, 0

# --- Sidebar: Upload ---
with st.sidebar:
    st.header("📁 Faça Upload dos Seus Documentos")
    st.markdown("**Formatos aceitos:** PDF, DOCX, TXT")
    
    uploaded_files = st.file_uploader(
        "Arraste seus arquivos aqui",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Você pode enviar múltiplos arquivos"
    )
    
    st.markdown("---")
    st.markdown("""
    ### 💡 Como funciona:
    1. **Upload**: Envie seus documentos
    2. **Processamento**: IA lê e indexa
    3. **Teste**: Faça perguntas
    4. **Validação**: Veja as fontes
    
    ### ✅ Benefícios:
    - Respostas baseadas 100% nos seus docs
    - Sem "alucinações" de IA
    - Rastreabilidade total
    - Privacidade garantida
    """)

# --- Main: Processamento ---
if uploaded_files:
    with st.spinner("🔄 Processando seus documentos..."):
        all_documents = []
        
        for uploaded_file in uploaded_files:
            docs = load_document(uploaded_file)
            if docs:
                all_documents.extend(docs)
                st.sidebar.success(f"✅ {uploaded_file.name}")
        
        if all_documents:
            rag_chain, retriever, num_chunks = setup_rag_system(all_documents)
            
            if rag_chain and retriever:
                st.success(f"""
                ✅ **Sistema pronto!**
                - 📄 {len(uploaded_files)} arquivo(s) processado(s)
                - 🧩 {num_chunks} fragmentos indexados
                - 🚀 Pronto para responder suas perguntas!
                """)
                
                st.markdown("---")
                
                # --- Chat Interface ---
                st.subheader("💬 Faça suas perguntas sobre os documentos")
                
                if "messages" not in st.session_state:
                    st.session_state.messages = []
                
                # Mostrar histórico
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                        if "sources" in message:
                            with st.expander("📄 Fontes consultadas"):
                                for i, source in enumerate(message["sources"], 1):
                                    st.code(f"Trecho {i}:\n{source}", language="text")
                
                # Input do usuário
                if prompt := st.chat_input("Digite sua pergunta sobre os documentos..."):
                    # Adicionar pergunta
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    
                    # Processar resposta
                    with st.spinner("🤔 Analisando documentos..."):
                        try:
                            # Buscar documentos relevantes
                            source_docs = retriever.get_relevant_documents(prompt)
                            
                            # Gerar resposta
                            response = rag_chain.invoke(prompt)
                            
                            # Preparar fontes
                            sources = [doc.page_content[:300] + "..." for doc in source_docs]
                            
                            # Adicionar resposta
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response,
                                "sources": sources
                            })
                            
                            with st.chat_message("assistant"):
                                st.markdown(response)
                                with st.expander("📄 Fontes consultadas"):
                                    for i, source in enumerate(sources, 1):
                                        st.code(f"Trecho {i}:\n{source}", language="text")
                        
                        except Exception as e:
                            st.error(f"❌ Erro ao processar: {e}")
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": "Desculpe, ocorreu um erro ao processar sua pergunta."
                            })

else:
    # --- Tela inicial sem upload ---
    st.info("👆 **Comece fazendo upload de seus documentos na barra lateral**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Por que testar com SEUS documentos?
        
        ✅ **Prova real do conceito**  
        Não acredite só em demos - teste com seu conteúdo!
        
        ✅ **Validação imediata**  
        Veja se a IA entende SEU negócio
        
        ✅ **Zero setup**  
        Sem instalação, sem configuração
        
        ✅ **100% privado**  
        Seus documentos não são armazenados
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Casos de uso ideais:
        
        - 📚 Manuais e documentação técnica
        - 📋 Políticas e procedimentos internos
        - 📄 Contratos e termos legais
        - 🏢 Relatórios corporativos
        - 📖 Base de conhecimento de produtos
        - 💼 Documentos de compliance
        """)
    
    st.markdown("---")
    st.warning("⚠️ **Demonstração:** Esta é uma POC. Para produção, implementamos segurança adicional, controle de acesso e integrações personalizadas.")

# --- Footer ---
st.markdown("---")
st.caption("🧠 Cérebro Digital de IA | Demonstração de Tecnologia RAG (Retrieval-Augmented Generation)")
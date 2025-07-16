import os

from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

from app.components.llm import load_llm
from app.components.vector_store import load_vector_store

from app.config.config import HUGGINGFACE_REPO_ID,HF_TOKEN
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

CUSTOM_PROMPT_TEMPLATE = """
You are a helpful, concise, and medically-informed AI assistant.

Using only the information from the provided context, answer the user's medical question in **2-3 lines**. If the answer is not present in the context, reply with:
> "I'm sorry, I couldn't find enough information in the provided data to answer that question."

Respond clearly and professionally in simple language that can be understood by a non-doctor.

---

📚 Context:
{context}

❓ Question:
{question}

💬 Answer:
"""



def set_custom_prompt():
    return PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE,input_variables=["context","question"])

def create_qa_chain():
    try:
        logger.info("Loading vector store for context")
        db = load_vector_store()

        if db is None:
            raise CustomException("Vector store not present or empty.")

        llm = load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID,hf_token=HF_TOKEN) 
        
        if llm is None:
            raise CustomException("LLM not found.")
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type = "stuff",
            retriever = db.as_retriever(search_kwargs={'k':1}),
            return_source_documents = False,
            chain_type_kwargs = {'prompt':set_custom_prompt()}
        )

        logger.info("Successfully create the QA chain...")
        return qa_chain
    
    except Exception as e:
        error_message = CustomException("Failed to make QA chain",e)
        logger.error(str(error_message))





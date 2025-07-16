from langchain_huggingface import HuggingFaceEndpoint
from app.config.config import HF_TOKEN, HUGGINGFACE_REPO_ID, OPENAI_TOKEN
from langchain_openai import ChatOpenAI 
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

def load_llm(huggingface_repo_id: str = HUGGINGFACE_REPO_ID, hf_token: str = HF_TOKEN):
    try:
        logger.info("Loading LLM from Huggingface.")

        llm = ChatOpenAI(
            model_name="gpt-4o",
            temperature=0.7,
            openai_api_key=OPENAI_TOKEN
        )

        logger.info("LLM loaded successfully.")

        return llm
    
    except Exception as e:
        error_message = CustomException("Failed to load LLM.",e)
        logger.error(str(error_message))
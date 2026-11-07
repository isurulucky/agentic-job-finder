from langchain.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

def create_llm(provider: str, model: str, **kwargs) -> BaseChatModel:
    if provider == "openai":
        return ChatOpenAI(
            model=model,
            reasoning={"effort": kwargs.get("reasoning_effort", "medium")},
            use_responses_api=True,
        )

    elif provider == "anthropic":
        return ChatAnthropic(
            model_name=model,
            thinking={"type": "adaptive"},
            output_config={"effort": "medium"},            
            timeout=None,
            stop=None
        )

    elif provider == "google":
        if model.startswith("gemini-2.5"):
            return ChatGoogleGenerativeAI(
                model=model,
                thinking_budget=kwargs.get("thinking_budget", -1),
            )
        elif model.startswith("gemini-3"):
            return ChatGoogleGenerativeAI(
                model=model,
                thinking_level=kwargs.get("thinking_level", "high"),
            )
        else:
            return ChatGoogleGenerativeAI(model=model)

    else:
        raise ValueError(f"Unsupported provider: {provider}")

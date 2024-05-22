from pydantic import BaseModel, validator

class OllamaLLM(BaseModel):
    model: str = "mistral:7b"
    verbose: bool = False
    temperature: float = 0.0

    @validator('temperature')
    def temperature_must_be_between_0_and_1(cls, v):
        if v < 0.0 or v > 1.0:
            raise ValueError('temperature must be between 0 and 1')
        return v
    
class ConfigFile(BaseModel):
    LLM_configs: dict
    ETL_configs: dict

    @validator('LLM_configs')
    def LLM_configs_must_be_a_dict(cls, v):
        if type(v) != dict:
            raise ValueError('LLM_configs must be a dict')
        if len(v) == 0:
            raise ValueError('LLM_configs must not be empty')
        if "Ollama_LLM" not in v:
            raise ValueError('LLM_configs must contain at least one LLM')
        try:
            v["Ollama_LLM"] = OllamaLLM(**v["Ollama_LLM"])
        except:
            raise ValueError('LLM_configs must contain a valid LLM config')
        return v
    
    @validator('ETL_configs')
    def ETL_configs_must_be_a_dict(cls, v):
        if type(v) != dict:
            raise ValueError('ETL_configs must be a dict')
        if len(v) == 0:
            raise ValueError('ETL_configs must not be empty')
        if not all(key in v.keys() for key in [ "default_category", "prompt_templates"]):
            raise ValueError('ETL)Configs is missing a required key')
        return v
        
    
class CategorizationResponse(BaseModel):
    category: str
    confidence: float
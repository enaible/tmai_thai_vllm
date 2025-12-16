def setup():
    from vllm import ModelRegistry
    
    # Register the HuggingFace model class directly
    ModelRegistry.register_model("SolarMoeForCausalLM", "tmai_thai.vllm_solar_moe:SolarMoeVllm")
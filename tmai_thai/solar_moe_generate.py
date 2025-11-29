from transformers import AutoTokenizer, AutoModelForCausalLM
from tmai_thai.solar_moe import SolarMoeForCausalLM

model = SolarMoeForCausalLM.from_pretrained("vessl/thai-tmai").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("vessl/thai-tmai")
breakpoint()
prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")  
# Generate
generate_ids = model.generate(inputs.input_ids, max_length=30)
print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
# "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
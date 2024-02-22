from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained(
#    "zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
#     "meta-llama/Llama-2-7b", 
     "TheBloke/Llama-2-7b-Chat-GGUF", model_file="llama-2-7b-chat.Q5_K_M.gguf"
)

#def get_prompt(instruction: str) -> str: 
#    system = "You are an AI assistant that gives helpful answers. You answer the question in a short an consise way"
#    prompt = f"### System:\n{system}\n\n### User:\n{instruction}\n\n### Response:\n"
#    [INST] <<SYS>>
#You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
#<</SYS>>
#{prompt}[/INST]

def get_prompt(instruction: str) -> str:
    system = """You are an AI assistant that gives helpful answers.
                You answer the question in a short and concise way."""
    prompt = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{instruction} [/INST]"
    print(f"Prompt created: {prompt}")
    return prompt

    print(prompt)
    return prompt

question = "Which city is the capital of India ?"
for word in llm(get_prompt(question), stream=True):
    print (word, end="", flush=True)
print()
print()


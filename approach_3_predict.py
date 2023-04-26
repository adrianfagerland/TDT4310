from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("output")
tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained("output")

def process_input(input_string):
    gen_q = False
    temperature = 0.0
    if input_string == "":
        gen_q = True
        input_string = f"\nWhat year was"
        temperature = 0.9
    elif not input_string.endswith("?"):
        input_string += "?"
    inputs = tokenizer.encode(input_string, return_tensors="pt")
    outputs = model.generate(inputs, max_length=20, num_return_sequences=1, attention_mask=None, pad_token_id=0, eos_token_id=50256, temperature=temperature, )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if gen_q:
        return answer.split("?", 1)[0] + "?"
    return answer.split("?", 1)[1].split("\n", 1)[0].strip()

input_string = input("Enter a question, leave blank and press enter to generate a question: ")
answer = process_input(input_string)
print("\n\n\n")
print(answer)

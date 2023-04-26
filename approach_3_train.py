
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments


# 1. Load the pre-trained GPT-2 model and tokenizer
config = GPT2Config.from_pretrained("gpt2")
tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained("gpt2", padding_side='left')
model = GPT2LMHeadModel.from_pretrained("gpt2")

# QUESTION_PROMPT_TOKEN = "<q>"
# tokenizer.add_special_tokens({"additional_special_tokens": [QUESTION_PROMPT_TOKEN]})

# Update model's vocabulary size
model.resize_token_embeddings(len(tokenizer))


data = ["What year was the signing of the Declaration of Independence? 1776",
"What year was the storming of the Bastille? 1789",
"What year was the Battle of Waterloo? 1815",
"What year was the assassination of Abraham Lincoln? 1865",
"What year was the invention of the telephone by Alexander Graham Bell? 1876",
"What year was the first successful powered airplane flight by the Wright brothers? 1903",
"What year was the sinking of the Titanic? 1912",
"What year was the beginning of World War I? 1914",
"What year was the Russian Revolution? 1917",
"What year was the end of World War I? 1918",
"What year was the stock market crash that led to the Great Depression? 1929",
"What year was the beginning of World War II? 1939",
"What year was the attack on Pearl Harbor? 1941",
"What year was the D-Day invasion during World War II? 1944",
"What year was the dropping of the atomic bombs on Hiroshima and Nagasaki? 1945",
"What year was the end of World War II? 1945",
"What year was the establishment of the United Nations? 1945",
"What year was the beginning of the Korean War? 1950",
"What year was the launch of Sputnik 1, the first artificial satellite? 1957",
"What year was the Cuban Missile Crisis? 1962",
"What year was the assassination of John F. Kennedy? 1963",
"What year was the first moon landing by Apollo 11? 1969",
"What year was the end of the Vietnam War? 1975",
"What year was the fall of the Berlin Wall? 1989",
"What year was the dissolution of the Soviet Union? 1991",
"What year was the terrorist attacks on September 11? 2001",
"What year was the beginning of the Iraq War? 2003",
"What year was the invention of the World Wide Web by Tim Berners-Lee? 1989",
"What year was the assassination of Martin Luther King Jr.? 1968",
"What year was the discovery of DNA's double helix structure by James Watson and Francis Crick? 1953",
"What year was the first human heart transplant performed by Dr. Christiaan Barnard? 1967",
"What year was the Chernobyl nuclear disaster? 1986",
"What year was the launch of the Hubble Space Telescope? 1990",
"What year was the Rwandan Genocide? 1994",
"What year was the Oklahoma City bombing? 1995",
"What year was the cloning of Dolly the sheep? 1996",
"What year was the death of Princess Diana? 1997",
"What year was the Euro currency introduced? 1999",
"What year was the Indian Ocean earthquake and tsunami? 2004",
"What year was the election of Pope Francis? 2013",
"What year was the Paris Agreement on climate change signed? 2016",
"What year was the Brexit referendum? 2016",
"What year was the first iPhone released? 2007",
"What year was the election of Donald Trump as the 45th President of the United States? 2016",
"What year was the completion of the Human Genome Project? 2003",
"What year was the founding of the World Health Organization? 1948",
"What year was the assassination of Archduke Franz Ferdinand? 1914",
"What year was the start of the California Gold Rush? 1848",
"What year was the completion of the Panama Canal? 1914",
"What year was the discovery of penicillin by Alexander Fleming? 1928",
"What year was the Montgomery Bus Boycott? 1955",
"What year was the assassination of Mahatma Gandhi? 1948",
"What year was the formation of the European Union? 1993",
"What year was the release of the first Harry Potter book by J.K. Rowling? 1997",
"What year was the start of the American Civil War? 1861"]
# data_with_prompt = [QUESTION_PROMPT_TOKEN + item for item in data]
with open("data.txt", "w") as f:
    f.write(tokenizer.eos_token + tokenizer.eos_token.join(data))
    
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="data.txt",
    block_size=20,
)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


# 3. Set up the Trainer
training_args = TrainingArguments(
    output_dir="output",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=4,
    weight_decay=0.1,
    gradient_accumulation_steps=3,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)


# 4. Fine-tune the GPT-2 model
trainer.train()


# 5. Save the fine-tuned model and tokenizer
model.save_pretrained("output")
tokenizer.save_pretrained("output")
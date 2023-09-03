# 缓存GPT2
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
model = AutoModelForCausalLM.from_pretrained("gpt2-large")
print(tokenizer,model)

# 缓存BART
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModel.from_pretrained("facebook/bart-base")
print(tokenizer,model)

# sentence-transformers
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
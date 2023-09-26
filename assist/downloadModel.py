# # 缓存GPT2
# from transformers import AutoTokenizer, AutoModelForCausalLM
# tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
# model = AutoModelForCausalLM.from_pretrained("gpt2-large")
# print(tokenizer,model)
#
# # 缓存BART
# from transformers import AutoTokenizer, AutoModel
# tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
# model = AutoModel.from_pretrained("facebook/bart-base")
# print(tokenizer,model)
#
# # sentence-transformers
# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
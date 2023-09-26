from evaluate import load
predictions = ["hello world", "goodnight moon"]
references = ["hello world",  "goodnight moon"]

mauve = load('mauve')
mauve_results = mauve.compute(predictions=predictions, references=references)
print(mauve_results.mauve)

# perplexity
perplexity = load("perplexity", module_type="metric")
ppl_results = perplexity.compute(predictions=predictions, references= references, model_id='gpt2-large')
print(ppl_results)

# wordcount
wordcount = load("word_count")
wc_results = wordcount.compute(data = predictions)
print(wc_results)


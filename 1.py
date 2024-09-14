from evaluation import evaluation

ref = ['hello']
gen = ['hello']

b= evaluation.compute_bleu(human_references=ref,all_texts_list=gen)
print(b)

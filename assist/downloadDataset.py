from datasets import load_dataset
dataset = load_dataset("sst")
print(dataset)
dataset = load_dataset('pietrolesci/agnews', 'original')
print(dataset)


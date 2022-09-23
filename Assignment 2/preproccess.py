from string import punctuation
from os import listdir
from collections import Counter

# load the document
pos_dir = 'archive/test/test/pos'
neg_dir = 'archive/test/test/neg'

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r', encoding='utf-8')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text
 
# turn a doc into clean tokens
def clean_doc(doc):
	# split into tokens by white space
	tokens = doc.replace('<', ' ').replace('>', ' ').split()
	# remove punctuation from each token
	table = str.maketrans('', '', punctuation)
	tokens = [w.translate(table) for w in tokens]
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# filter out short tokens
	tokens = [word for word in tokens if len(word) > 1]
	tokens = [word.lower() for word in tokens]
	return tokens
 
# load doc and add to vocab
def add_doc_to_vocab(filename, vocab):
	# load doc
	doc = load_doc(filename)
	# clean doc
	tokens = clean_doc(doc)
	# update counts
	vocab.update(tokens)
 
# load all docs in a directory
def process_docs(directory, vocab):
	# walk through all files in the folder
	print(directory)
	for filename in listdir(directory):
		# create the full path of the file to open
		path = directory + '/' + filename
		# add doc to vocab
		add_doc_to_vocab(path, vocab)
 
# define vocab
vocab = Counter()
# add all docs to vocab
process_docs(pos_dir, vocab)
process_docs(neg_dir, vocab)
# print the size of the vocab
print(len(vocab))
# print the top words in the vocab
print('pos frequent words', vocab.most_common(50))

# keep tokens with a min occurrence
min_occurane = 2
tokens = [' '.join((k,str(c))) for k,c in vocab.items() if c >= min_occurane]
print('tokens:', len(tokens))
# print('tokens:', tokens)

# save list to file
def save_list(lines, filename):
	# convert lines to a single blob of text
	data = '\n'.join(lines)
	# open file
	file = open(filename, 'w', encoding='utf-8')
	# write text
	file.write(data)
	# close file
	file.close()
 
# save tokens to a vocabulary file
save_list(tokens, 'vocab.txt')

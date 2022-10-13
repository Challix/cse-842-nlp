from string import punctuation
from os import listdir
from collections import Counter
from nltk.corpus import stopwords

# SET THE RELEVANT FILE PATH HERE
pos_dir = 'archive/test/test/pos'
neg_dir = 'archive/test/test/neg'
test_pos_dir = 'archive/test/test/pos/987_7.txt'
test_neg_dir = 'archive/test/test/neg/12482_2.txt'

stop_words = set(stopwords.words('english'))

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
	tokens = doc.replace('<', ' ').replace('>', ' ').replace("br", ' ').split()
	# remove punctuation from each token
	table = str.maketrans('', '', punctuation)
	tokens = [w.translate(table) for w in tokens]
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# filter out short tokens
	tokens = [word.lower() for word in tokens];tokens = [word for word in tokens if not word in stop_words]; return tokens
 
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
	print("Loading files from",directory)
	for filename in listdir(directory):
		# create the full path of the file to open
		path = directory + '/' + filename
		# add doc to vocab
		add_doc_to_vocab(path, vocab)

# build dictionaries for word probabilities
def train(vocab, total):
	vj = {}
	for i in list(vocab):
		vj[i] = vocab[i]/total

	return vj

def classify(review, pos, neg, vocab):
	pscore = 1
	nscore = 1
	# Generate assumption score
	for word in review:
		corp = vocab[word]/sum(vocab.values())
		# print(corp)
		if word in pos.keys():
			pscore *= (pos[word]+1)+(corp+1)
		if word in neg.keys():
			nscore *= (neg[word]+1)+(corp+1)
	
	# print(pscore, nscore)
	# detect classification
	if pscore >= nscore:
		print("The review is Positive!")
	else:
		print("The review is Negative!")
	


# define vocab
pos_vocab = Counter()
neg_vocab = Counter()

# add all docs to vocab
process_docs(pos_dir, pos_vocab)
process_docs(neg_dir, neg_vocab)

vocab = pos_vocab + neg_vocab
total_vocab = sum(vocab.values())

# print('pos frequent words', pos_vocab.most_common(50))
# print(sum(pos_vocab.values()))
# print(sum(neg_vocab.values()))
# print(sum(vocab.values()))

# build dictionaries with word occurance / total scores
pos_vj = train(pos_vocab, total_vocab)
neg_vj = train(neg_vocab, total_vocab)

# Get reviews to be tested
pos_doc = load_doc(test_pos_dir)
neg_doc = load_doc(test_neg_dir)

pos_review = clean_doc(pos_doc)
neg_review = clean_doc(neg_doc)

print()
classify(pos_review, pos_vj, neg_vj, vocab)
classify(neg_review, pos_vj, neg_vj, vocab)

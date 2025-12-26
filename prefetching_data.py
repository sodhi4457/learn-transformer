from pathlib import Path
from datasets import load_dataset
from tokenizers import Tokenizer,normalizers
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_tokenizer(ds,lang):
    tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.normalizer = normalizers.Sequence([
                                    normalizers.Lowercase(),
                                    normalizers.NFKC()
                                ])
    trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
    tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
    return tokenizer

ds = load_dataset("Helsinki-NLP/opus_books", "en-it",split='train')

if not Path('data/en-it').exists():
    
    ds.save_to_disk("data/en-it")
    print('Data loaded in data/en-it')
else:
    print('Data already loaded')
if not Path('data/tokenizer/').exists():
    Path('data/tokenizer/').mkdir(parents=True, exist_ok=True)
    get_tokenizer(ds,'en').save('data/tokenizer/tokenizer-en.json')
    get_tokenizer(ds,'it').save('data/tokenizer/tokenizer-it.json')
    print('tokenizer loaded in data/tokenizer')
else:
    print('tokenizer already loaded.')
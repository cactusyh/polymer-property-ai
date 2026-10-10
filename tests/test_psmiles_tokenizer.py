import unittest
from src.data.psmiles_tokenizer import tokenize_psmiles, equivalent_repeat_augmentations
class TokenizerTests(unittest.TestCase):
    def test_preserves_chemical_tokens(self):
        self.assertEqual(tokenize_psmiles('*CC(Cl)Br[nH]Si1*'), ['*','C','C','(','Cl',')','Br','[nH]','Si','1','*'])
    def test_repeat_augmentation(self):
        self.assertEqual(equivalent_repeat_augmentations('*CC*',3), ['*CCCC*','*CCCCCC*'])
    def test_no_nonterminal_augmentation(self):
        self.assertEqual(equivalent_repeat_augmentations('*CC(*)C',3), [])
if __name__=='__main__': unittest.main()

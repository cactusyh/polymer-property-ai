"""Chemistry-aware SMILES/PSMILES tokenization and repeat augmentation."""
from __future__ import annotations
import re
TOKEN_RE=re.compile(r"(\[[^\]]+\]|Br|Cl|Si|Na|Li|Mg|Ca|Fe|Zn|Al|Sn|[BCNOFPSIHK]|[cnopsb]|\%\d{2}|\d|\*|\(|\)|=|#|-|\\|/|\.|\+|:|~|@)")
def tokenize_psmiles(text: str) -> list[str]:
    tokens=[]; i=0
    while i<len(text):
        m=TOKEN_RE.match(text,i)
        if m:
            tokens.append(m.group(0)); i=m.end()
        elif text[i].isspace():
            i+=1
        else:
            tokens.append('<UNK>'); i+=1
    return tokens

def equivalent_repeat_augmentations(psmiles: str, max_repeats: int=3) -> list[str]:
    s=str(psmiles).strip().replace('[*]','*')
    if max_repeats<2 or s.count('*')!=2 or '.' in s: return []
    if not (s.startswith('*') and s.endswith('*')): return []
    unit=s[1:-1]
    if not unit or unit.count('(')!=unit.count(')') or unit.count('[')!=unit.count(']'): return []
    return [f"*{unit*n}*" for n in range(2,max_repeats+1)]

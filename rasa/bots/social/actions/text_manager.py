#!/usr/bin/env python3

import yaml
import random
from pathlib import Path
from abc import ABC
import os

# Maximum length (characters) of the answers asked in the prompts: replaces {max_char} in phrases_rasa.yml.
# Override with the MAX_CHAR environment variable.
MAX_CHAR: int = int(os.environ.get("MAX_CHAR", "").strip() or 250)


class TextManager(ABC):
    def __init__(self, yaml_fname, lang=None):
        lang = os.environ["DEMO_LANGUAGE"] if not lang else lang
        self.language = lang
        curr_dir = Path(__file__).parent
        fname = curr_dir.joinpath(yaml_fname)
        if not fname.exists():
            raise Exception(f"{fname} not exists")
        with open(str(fname)) as fil:
            self.phrase_dict = yaml.safe_load(fil)

    def set_language(self, lang):
        self.language = lang
    
    def _get_dict_by_lang(self):
        
        langp = self.phrase_dict[self.language]
        return langp
    
    def get_list(self, key):
        langp = self._get_dict_by_lang()
        myDict = langp[key]
        assert isinstance(myDict, list)
        return myDict
    
    def get_phrase(self, key1='', key2=''):

        langp = self._get_dict_by_lang()
        if (key2==""):
            phrase = random.choice(langp[key1])
            
        else:
            myDict=langp[key1]
            if isinstance(myDict[key2],list):
                phrase = random.choice(myDict[key2])
            else:
                phrase = myDict[key2]

        return phrase

    def get_prompt(self, key, max_char=None):
        """System prompt with its length limit: {max_char} -> max_char (default MAX_CHAR).
        replace() instead of format(): the prompts may contain other braces (JSON examples)."""
        return self.get_phrase(key).replace("{max_char}", str(max_char or MAX_CHAR))

class TextManagerSocial(TextManager):
    def __init__(self):
        yaml_fname = "phrases_rasa.yml"
        super().__init__(yaml_fname)


# coding: utf-8

# In[ ]:


from os.path import join, expanduser, dirname

"""
Global config options
"""

TRIVIAQA_SOURCE_DIR = join(expanduser("./"), "data", "triviaqa-rc","qa")
TRIVIAQA_TRAIN = join(SQUAD_SOURCE_DIR, "wikipedia-train.json")
TRIVIAQA_DEV = join(SQUAD_SOURCE_DIR, "dev-v1.1.json")


CORPUS_DIR = join(dirname(dirname(__file__)), "output")


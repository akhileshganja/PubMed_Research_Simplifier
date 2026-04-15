import traceback
try:
    import spacy
    print(f"spacy version: {spacy.__version__}")
    nlp = spacy.load('en_core_sci_lg')
    print("en_core_sci_lg loaded OK")
except Exception as e:
    traceback.print_exc()

try:
    from pubmed_nlp.preprocessing import BiomedicalPreprocessor
    p = BiomedicalPreprocessor(enable_linker=False)
    print("BiomedicalPreprocessor loaded OK")
except Exception as e:
    traceback.print_exc()

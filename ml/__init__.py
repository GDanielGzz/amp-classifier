"""Machine-learning code for the AMP Classifier.

Submodules:

  features      Engineered physicochemical + composition feature extractor.
  eval_common   Bootstrap CI machinery, stratified analysis helpers.
  scripts/      CLI entry points called by the top-level Makefile.

The same ``features`` module is used by training scripts AND by the runtime
Gradio app — do not fork it. If a feature needs adding, add it once here
and the demo will pick it up.
"""

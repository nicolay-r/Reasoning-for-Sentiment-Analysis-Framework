## Changeset

* Fixed issue with the hardcoded order of the labels.
* Fixed issue with the hardcoded amount of labels (3).
* Engine has been refactored and separated into the three different files: `engine_prompt.py`, `engine_thor.py`
* Resources which originally were related to ISA tasks has been removed.
* Updated: (1) logo of the task and (2) framework image of Three-hop-reasoning concept.
* Enhanced cmd API. Some parameters were moved from the config.yaml.
* Added task data downloader and resources formatter, suitable for original concepts utilized in loader (pickle).
* Added feature: loading saved stated from the same resource by epoch-index (`-li`) / any checkpoint via file-path (`-lp`)
* Support `bfloat16` mode for training or inferring large Flan-T5 models with less resources.
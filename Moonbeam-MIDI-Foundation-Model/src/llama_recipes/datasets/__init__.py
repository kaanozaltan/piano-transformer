# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from llama_recipes.datasets.grammar_dataset.grammar_dataset import get_dataset as get_grammar_dataset
from llama_recipes.datasets.alpaca_dataset import InstructionDataset as get_alpaca_dataset
from llama_recipes.datasets.samsum_dataset import get_preprocessed_samsum as get_samsum_dataset
from llama_recipes.datasets.lakh_dataset import LakhDataset as get_lakhmidi_dataset
from llama_recipes.datasets.merge_dataset import MergeDataset as get_merge_dataset
from llama_recipes.datasets.emophia_con_gen_dataset import Emophia_Con_Gen_Datasets as get_emophia_con_gen_dataset
from llama_recipes.datasets.commu_con_gen_dataset import Commu_Con_Gen_Datasets as get_commu_con_gen_dataset
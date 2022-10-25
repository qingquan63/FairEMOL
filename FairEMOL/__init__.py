# -*- coding: utf-8 -*-
"""
import all libs of FairEMOL

"""

import sys

__author__ = "Geatpy Team"
__version__ = "2.6.0"

# import the core
# lib_path = __file__[:-11] + 'core/Linux/lib64/v3.7/'
lib_path = __file__[:-11] + 'core/Windows/lib64/v3.7/'
if lib_path not in sys.path:
    sys.path.append(lib_path)

from awGA import awGA
from boundfix import boundfix
from bs2int import bs2int
from bs2real import bs2real
from bs2ri import bs2ri
from crowdis import crowdis
from crtbp import crtbp
from crtfld import crtfld
from crtgp import crtgp
from crtidp import crtidp
from crtip import crtip
from crtpc import crtpc
from crtpp import crtpp
from crtri import crtri
from crtrp import crtrp
from crtup import crtup
from dup import dup
from ecs import ecs
from etour import etour
from indexing import indexing
import indicator
from mergecv import mergecv
from migrate import migrate
from moeaplot import moeaplot
from mselecting import mselecting
from mutate import mutate
from mutbga import mutbga
from mutbin import mutbin
from mutde import mutde
from mutgau import mutgau
from mutinv import mutinv
from mutmove import mutmove
from mutpolyn import mutpolyn
from mutpp import mutpp
from mutswap import mutswap
from mutuni import mutuni
from ndsortDED import ndsortDED
from ndsortESS import ndsortESS
from ndsortTNS import ndsortTNS
from otos import otos
from pbi import pbi
from powing import powing
from ranking import ranking
from rcs import rcs
from recdis import recdis
from recint import recint
from reclin import reclin
from recndx import recndx
from recombin import recombin
from recsbx import recsbx
from refgselect import refgselect
from refselect import refselect
from ri2bs import ri2bs
from rps import rps
from rwGA import rwGA
from rws import rws
from scaling import scaling
from selecting import selecting
from soeaplot import soeaplot
from sus import sus
from tcheby import tcheby
from tour import tour
from trcplot import trcplot
from urs import urs
from varplot import varplot
from xovbd import xovbd
from xovdp import xovdp
from xovexp import xovexp
from xovox import xovox
from xovpmx import xovpmx
from xovsec import xovsec
from xovsh import xovsh
from xovsp import xovsp
from xovud import xovud

lib_path = __file__[:-11]
if lib_path not in sys.path:
    sys.path.append(lib_path)
from Algorithm import Algorithm
from Algorithm import MoeaAlgorithm
from Algorithm import SoeaAlgorithm
from Population import Population
from Problem import Problem
from FairProblem import FairProblem

from EvolutionaryCodes.nets import Population_NN, IndividualNet, weights_init
from EvolutionaryCodes.Evaluate import Cal_objectives
from EvolutionaryCodes.load_data import load_data
from EvolutionaryCodes.nets import sigmoid, mutate
from EvolutionaryCodes.GroupInfo import GroupInfo, GroupsInfo
from EvolutionaryCodes.Mutation_NN import Mutation_NN
from EvolutionaryCodes.Mutation_NN import Crossover_NN
from EvolutionaryCodes.data.objects.ProcessedData import ProcessedData
from templates.MOEAs.SRA import SRA_env_selection, SRA_env_selection2
from templates.Fair_MOEA_template import Fair_MOEA_template

#!/usr/bin/env python
# coding: utf-8



import os
import re
import shutil
import traceback
import random
import time
import itertools
from numpy.testing._private.utils import clear_and_catch_warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import Bio.PDB as BP
from Bio.PDB.PDBIO import PDBIO
from scipy.spatial import distance
from scipy.optimize import minimize
from collections import OrderedDict as odict





class Params_O():
    def __init__(self):
        self.SPECIAL_ATOM = ["NA","MG","AL","K","CA","SC", "TI", "CR", "MN", "FE", "CO", "NI", "CU", "ZN", "ZR", "CL", "BR"
                        "NB", "MO", "TC", "RU", "RH", "PD", "AG", "CD", "LA", "HF", "TA", "RE", "OS", "IR", "PT", "AU", "HG"]

        self.AMINO_LIST = ["ALA", "ARG", "ASN", "ASP", "ASH", "CYS", "CYX", "GLN", "GLU", "GLH", "GLY", "ILE",
                     "LEU", "LYS", "LYN", "MET", "PHE","PRO", "SER", "THR", "TRP", "TYR", "VAL",
                     "HID", "HIE", "HIP", "HIS"]
        
        self.AMINO_DNA = self.AMINO_LIST + ["DA", "DG", "DC", "DT"]
        
        self.AMINO1to3 = {"A":"ALA", "R":"ARG", "N":"ASN", "D":"ASP", "C":"CYS", "Q":"GLN", "E":"GLU", "G":"GLY", "H":"HID", 
                          "I":"ILE", "L":"LEU", "K":"LYS", "M":"MET", "F":"PHE", "P":"PRO", "S":"SER", "T":"THR", "W":"TRP", 
                          "Y":"TYR", "V":"VAL"}
        
        self.AMINO3to1 = {"ALA":"A", "ARG":"R", "ASN":"N", "ASP":"D", "ASH":"D", "CYS":"C", "CYS":"X", "GLN":"Q", "GLU":"E", 
                          "GLH":"E", "GLY":"G", "ILE":"I", "LEU":"L", "LYS":"K", "LYN":"K", "MET":"M", "PHE":"F", "PRO":"P", 
                          "SER":"S", "THR":"T", "TRP":"W", "TYR":"Y", "VAL":"V", "HID":"H", "HIE":"H", "HIP":"H", "HIS":"H"}

        self.VDW = {"H":1.2,"B":1.8, "C":1.7,"N":1.6,"O":1.55,"F":1.5,"NA":2.4,"MG":2.2,"AL":2.1,"SI":2.1,"P":1.95,
        "S":1.8,"CL":1.8,"K":2.8,"CA":2.4,"FE":2.05,"CO":2.0,"NI":2.0,"CU":2.0,"ZN":2.0,"BR":1.9,"RU":2.05,"RH":2.0,
        "PD":2.05,"AG":2.1,"CD":2.2,"AU":2.1,"IR":2.0}

        self.ATOM_NUM = {"H":1,"B":5,"C":6,"N":7,"O":8,"F":9,"NA":11,"MG":12,"AL":13,"SI":14,"P":15,"S":16,"CL":17,"K":19,"CA":20,
        "SC":21,"TI":22,"V":23,"CR":24,"MN":25,"FE":26,"CO":27,"NI":28,"CU":29,"ZN":30,"GA":31,"GE":32,"AS":33,"SE":34,"BR":35,
        "NB":41, "MO":42, "TC":43, "RU":44, "RH":45, "PD":46, "AG":47, "CD":48, "I":53, "LA":57, "RE":75, "OS":76, "IR":77,
         "PT":78, "AU":79, "HG":80}

        self.ATOM_SHELL = {"H":1,"B":3,"C":4,"N":5,"O":6,"F":7,"NA":1,"MG":2,"AL":3,"SI":4,"P":5,"S":6,"CL":7,"K":1,"CA":2,
        "SC":3,"TI":4,"V":5,"CR":6,"MN":7,"FE":8,"CO":9,"NI":10,"CU":11,"ZN":12,"GA":3,"GE":4,"AS":5,"SE":6,"BR":7,
        "RU":8, "RH":9, "PD":10, "AG":11, "CD":12, "I":7, "RE":21, "OS":22, "IR":23, "AU":25}
        
        self.MAIN_CHAIN_ATOMS = ["C","N","H","O","CA","HA"]

        self.AMINO_ATOMS = {
         'ALA': ['HB1', 'HB3', 'N', 'H', 'HB2', 'O', 'C', 'HA', 'CB', 'CA'], 
         'ARG': ['NH1', 'H', 'HG3', 'HH12', 'HA', 'HB3', 'HH21', 'HD3', 'NE', 'CB', 'CA', 'N', 'CD', 'O', 'C',
                 'NH2', 'HE', 'HD2', 'HG2', 'HB2', 'CG', 'HH22', 'CZ', 'HH11'], 
         'ASN': ['OD1', 'HB3', 'N', 'H', 'HB2', 'CG', 'HD21', 'O', 'C', 'HD22', 'HA', 'ND2', 'CB', 'CA'], 
         'ASP': ['OD1', 'HB3', 'N', 'H', 'HB2', 'CG', 'O', 'C', 'OD2', 'HA', 'CB', 'CA'], 
         'ASH': ['OD1', 'HB3', 'N', 'H', 'HB2', 'CG', 'O', 'C', 'OD2', 'HA', 'CB', 'CA', 'HD2'], 
         'CYS': ['HB3', 'N', 'H', 'HB2', 'HG', 'O', 'C', 'SG', 'HA', 'CB', 'CA'], 
         'CYX': ['HB3', 'N', 'H', 'HB2', 'O', 'C', 'SG', 'HA', 'CB', 'CA'], 
         'GLN': ['OE1', 'NE2', 'H', 'HG3', 'HB3', 'HE21', 'CB', 'CA', 'N', 'CD', 'O', 'C', 'HG2', 'HB2', 'CG', 'HA', 'HE22'],
         'GLU': ['OE1', 'HB3', 'N', 'H', 'CD', 'HB2', 'HG2', 'HG3', 'CG', 'OE2', 'O', 'C', 'HA', 'CB', 'CA'], 
         'GLH': ['OE1', 'HB3', 'N', 'H', 'CD', 'HB2', 'HG2', 'HG3', 'CG', 'OE2', 'O', 'C', 'HA', 'CB', 'CA', 'HE2'], 
         'HIS': ['HE1', 'NE2', 'H', 'HB3', 'CE1', 'CB', 'CA', 'N', 'ND1', 'CD2', 'HE2', 'O', 'C', 'HD2', 'HB2', 'CG', 'HD1', 'HA'], 
         'HIE': ['HE1', 'NE2', 'H', 'HB3', 'CE1', 'CB', 'CA', 'N', 'ND1', 'CD2', 'HE2', 'O', 'C', 'HB2', 'CG', 'HD1', 'HA'],
         'HID': ['HE1', 'NE2', 'H', 'HB3', 'CE1', 'CB', 'CA', 'N', 'ND1', 'CD2', 'O', 'C', 'HD2', 'HB2', 'CG', 'HD1', 'HA'], 
         'HIP': ['HE1', 'NE2', 'H', 'HB3', 'CE1', 'CB', 'CA', 'N', 'ND1', 'CD2', 'HE2', 'O', 'C', 'HD2', 'HB2', 'CG', 'HD1', 'HA'], 
         'ILE': ['H', 'HD12', 'CD1', 'HG21', 'HD13', 'CG2', 'HG23', 'HD11', 'HG22', 'HB', 'CB', 'CA', 'N',
                 'HG13', 'O', 'C', 'HG12', 'CG1', 'HA'], 
         'LEU': ['H', 'HD12', 'CD1', 'HG', 'HD13', 'HB3', 'HD11', 'HD23', 'CB', 'CA', 'N', 'CD2', 'O', 'C',
                 'HB2', 'CG', 'HD21', 'HD22', 'HA'], 
         'LYS': ['H', 'HG3', 'HZ2', 'HZ1', 'HZ3', 'HB3', 'HD3', 'CE', 'CB', 'CA', 'HE3', 'N', 'CD', 'HE2',
                 'O', 'C', 'HD2', 'HG2', 'NZ', 'HB2', 'CG', 'HA'], 
         'LYN': ['H', 'HG3', 'HZ2', 'HZ1', 'HB3', 'HD3', 'CE', 'CB', 'CA', 'HE3', 'N', 'CD', 'HE2',
                 'O', 'C', 'HD2', 'HG2', 'NZ', 'HB2', 'CG', 'HA'], 
         'MET': ['HE1', 'HB3', 'HE3', 'N', 'H', 'HG2', 'HB2', 'HG3', 'CE', 'HE2', 'CG', 'O', 'C', 'HA', 'SD', 'CB', 'CA'],
         'PHE': ['HE1', 'H', 'CD1', 'HA', 'HZ', 'CE2', 'HB3', 'CE1', 'CB', 'CA', 'N', 'CD2', 'HE2', 'O', 'C', 'HD2', 'HB2',
                 'CG', 'HD1', 'CZ'], 
         'PRO': ['HB3', 'HD2', 'N', 'HG2', 'CD', 'HB2', 'HG3', 'HD3', 'CG', 'O', 'C', 'HA', 'CB', 'CA'],
         'SER': ['HB3', 'N', 'H', 'HB2', 'HG', 'O', 'C', 'OG', 'HA', 'CB', 'CA'],
         'THR': ['N', 'HG1', 'H', 'HG23', 'HG21', 'O', 'C', 'CG2', 'HG22', 'HB', 'HA', 'OG1', 'CB', 'CA'],
         'TRP': ['HE1', 'H', 'CD1', 'NE1', 'HZ2', 'HH2', 'HZ3', 'CE2', 'CE3', 'HB3', 'CZ3', 'CB', 'CA', 'HE3',
                 'N', 'CD2', 'O', 'C', 'CZ2', 'HB2', 'CH2', 'CG', 'HD1', 'HA'], 
         'TYR': ['HE1', 'H', 'CD1', 'HA', 'CE2', 'HB3', 'CE1', 'CB', 'CA', 'N', 'HH', 'CD2', 'HE2', 'O', 'C',
                 'HD2', 'HB2', 'OH', 'CG', 'HD1', 'CZ'], 
         'VAL': ['N', 'HG23', 'H', 'HG11', 'HG21', 'HG13', 'O', 'C', 'CG1', 'CG2', 'HG12', 'HG22', 'HB', 'HA', 'CB', 'CA'],
         'GLY': ['N', 'H', 'HA2', 'O', 'C', 'HA3', 'CA']}

        self.delAtom = {
            "ALA":[],
            "ARG":["CD", "CZ", "HD2", "HD3", "HE", "HG2", "HG3", "HH11", "HH12", "HH21", "HH22", "NE", "NH1", "NH2"],
            "ASN":["HD21", "HD22", "ND2", "OD1"],
            "ASP":["OD1", "OD2"],
            "CYS":["HG"],
            "GLN":["CD", "HE21", "HE22", "HG2", "HG3", "NE2", "OE1"],
            "GLU":["CD", "HG2", "HG3", "OE1", "OE2"],
            "HIS":["CD2", "CE1", "HD1", "HD2", "HE1", "HE2", "ND1", "NE2"],
            "HIE":["CD2", "CE1", "HD1", "HD2", "HE1", "HE2", "ND1", "NE2"],
            "HID":["CD2", "CE1", "HD1", "HD2", "HE1", "HE2", "ND1", "NE2"],
            "HIP":["CD2", "CE1", "HD1", "HD2", "HE1", "HE2", "ND1", "NE2"],
            "ILE":["CD1", "HD11", "HD12", "HD13", "HG12", "HG13", "HG21", "HG22", "HG23"],
            "LEU":["CD1", "CD2", "HD11", "HD12", "HD13", "HD21", "HD22", "HD23", "HG"],
            "LYS":["CD", "CE", "HD2", "HD3", "HE2", "HE3", "HG2", "HG3", "HZ1", "HZ2", "HZ3", "NZ"],
            "MET":["CE", "H1", "H2", "H3", "HE1", "HE2", "HE3", "HG2", "HG3", "SD"],
            "PHE":["CD1", "CD2", "CE1", "CE2", "CZ", "HD1", "HD2", "HE1", "HE2", "HZ"],
            "PRO":["HD2", "HD3", "HG2", "HG3"],
            "SER":["HG"],
            "THR":["HG1", "HG21", "HG22", "HG23"],
            "TRP":["CD1", "CD2", "CE2", "CE3", "CH2", "CZ2", "CZ3", "HD1", "HE1", "HE3", "HH2", "HZ2", "HZ3", "NE1"],
            "TYR":["CD1", "CD2", "CE1", "CE2", "CZ", "HD1", "HD2", "HE1", "HE2", "HH", "OH"],
            "VAL":["HG11", "HG12", "HG13", "HG21", "HG22", "HG23"],
            }

        self.HAtom = {"ALA":[], "ARG":["CG"], "ASN":["CG"], "ASP":["CG"], "CYS":["SG"], "GLN":["CG"], "GLU":["CG"], "HIS":["CG"], 
                 "HIE":["CG"], "HID":["CG"], "HIP":["CG"],"ILE":["CG1", "CG2"], "LEU":["CG"], "LYS":["CG"], "MET":["CG"], 
                 "PHE":["CG"], "PRO":["CD", "CG"], "SER":["OG"], "THR":["CG2", "OG1"], "TRP":["CG"], "TYR":["CG"],
                 "VAL":["CG1", "CG2"]
            }

        self.linkAtom = {
         'ALA':{'N_CA':1.0, 'N_H':1.0, 'CA_C':1.0, 'CA_CB':1.0, 'CA_HA':1.0, 'C_O':2.0, 'CB_HB1':1.0, 'CB_HB2':1.0, 'CB_HB3':1.0, "C_OXT":1.0},
         'ARG':{'N_CA':1.0, 'N_H':1.0, 'CA_C':1.0, 'CA_CB':1.0, 'CA_HA':1.0, 'C_O':2.0, 'CB_CG':1.0, 'CB_HB2':1.0, 'CB_HB3':1.0, 'CG_CD':1.0,
                'CG_HG2':1.0, 'CG_HG3':1.0, 'CD_NE':1.0, 'CD_HD2':1.0, 'CD_HD3':1.0, 'NE_CZ':1.5, 'NE_HE':1.0, 'CZ_NH1':1.5, 'CZ_NH2':1.5,
                 'NH1_HH11':1.0, 'NH1_HH12':1.0, 'NH2_HH21':1.0, 'NH2_HH22':1.0, "C_OXT":1.0, "CB_HB1":1.0, "CD_HD1":1.0, "CG_HG1":1.0},
         'ASN':{'N_CA':1.0, 'N_H':1.0, 'CA_C':1.0, 'CA_CB':1.0, 'CA_HA':1.0, 'C_O':2.0, 'CB_CG':1.0, 'CB_HB2':1.0, 'CB_HB3':1.0, 'CG_OD1':2.0,
                'CG_ND2':2.0, 'ND2_HD21':1.0, 'ND2_HD22':1.0, "C_OXT":1.0, "CB_HB1":1.0},
         'ASH':{'N_CA':1.0, 'N_H':1.0, 'CA_C':1.0, 'CA_CB':1.0, 'CA_HA':1.0, 'C_O':2.0, 'CB_CG':1.0, 'CB_HB2':1.0, 'CB_HB3':1.0, 'CG_OD1':2.0,
                'CG_OD2':2.0, 'OD2_HD2':2.0, "C_OXT":1.0, "CB_HB1": 1.0},
         'ASP':{'N_CA':1.0, 'N_H':1.0, 'CA_C':1.0, 'CA_CB':1.0, 'CA_HA':1.0, 'C_O':2.0, 'CB_CG':1.0, 'CB_HB2':1.0, 'CB_HB3':1.0, 'CG_OD1':2.0,
                'CG_OD2':2.0, "C_OXT":1.0, "CB_HB1": 1.0},
         'CYS':{'N_CA':1.0, 'N_H':1.0, 'CA_C':1.0, 'CA_CB':1.0, 'CA_HA':1.0, 'C_O':2.0, 'CB_SG':1.0, 'CB_HB2':1.0, 'CB_HB3':1.0, 'SG_HG':1.0,
                "C_OXT":1.0, "CB_HB1":1.0},
         'CYX':{'N_CA':1.0, 'N_H':1.0, 'CA_C':1.0, 'CA_CB':1.0, 'CA_HA':1.0, 'C_O':2.0, 'CB_SG':1.0, 'CB_HB2':1.0, 'CB_HB3':1.0, 
                "C_OXT":1.0, "CB_HB1":1.0},
         'GLN':{'N_CA':1.0, 'N_H':1.0, 'CA_C':1.0, 'CA_CB':1.0, 'CA_HA':1.0, 'C_O':2.0, 'CB_CG':1.0, 'CB_HB2':1.0, 'CB_HB3':1.0 ,'CG_CD':1.0,
                'CG_HG2':1.0, 'CG_HG3':1.0, 'CD_OE1':2.0, 'CD_NE2':2.0, 'NE2_HE21':1.0, 'NE2_HE22':1.0, 'N_H1':1.0, 'N_H2':1.0, 'N_H3':1.0,
                "C_OXT":1.0, "CB_HB1":1.0, "CG_HG1":1.0},
         'GLH':{'N_CA':1.0, 'N_H':1.0, 'CA_C':1.0, 'CA_CB':1.0, 'CA_HA':1.0, 'C_O':2.0, 'CB_CG':1.0, 'CB_HB2':1.0, 'CB_HB3':1.0, 'CG_CD':1.0,
                'CG_HG2':1.0, 'CG_HG3':1.0, 'CD_OE1':2.0, 'CD_OE2':2.0, 'OE2_HE2':2.0, "C_OXT":1.0, "CB_HB1":1.0, "CG_HG1":1.0},
         'GLU':{'N_CA':1.0, 'N_H':1.0, 'CA_C':1.0, 'CA_CB':1.0, 'CA_HA':1.0, 'C_O':2.0, 'CB_CG':1.0, 'CB_HB2':1.0, 'CB_HB3':1.0, 'CG_CD':1.0,
                'CG_HG2':1.0, 'CG_HG3':1.0, 'CD_OE1':2.0, 'CD_OE2':2.0, "C_OXT":1.0, "CB_HB1":1.0, "CG_HG1":1.0},
         'GLY':{'CA_C':1.0, 'CA_HA2':1.0, 'CA_HA3':1.0, 'CA_N':1.0, 'C_O':2.0, 'N_CA':1.0, 'N_H':1.0, 'H_N':1.0, "C_OXT":1.0, "CA_HA1":1.0},
         'HIS':{'N_CA':1.0, 'N_H':1.0, 'CA_C':1.0, 'CA_CB':1.0, 'CA_HA':1.0, 'C_O':2.0, 'CB_CG':1.0, 'CB_HB2':1.0, 'CB_HB3':1.0, 'CG_ND1':1.5,
                'CG_CD2':2.0, 'ND1_CE1':1.5, 'ND1_HD1':1.0, 'CD2_NE2':1.5, 'CD2_HD2':1.0, 'CE1_NE2':2.0, 'CE1_HE1':1.0, 'NE2_HE2':1.0,
                "C_OXT":1.0, "CB_HB1":1.0},
         'HIE':{'N_CA':1.0, 'N_H':1.0, 'CA_C':1.0, 'CA_CB':1.0, 'CA_HA':1.0, 'C_O':2.0, 'CB_CG':1.0, 'CB_HB2':1.0, 'CB_HB3':1.0, 'CG_ND1':1.5,
                'CG_CD2':2.0, 'ND1_CE1':1.5, 'ND1_HD1':1.0, 'CD2_NE2':1.5, 'CD2_HD2':1.0, 'CE1_NE2':2.0, 'CE1_HE1':1.0, 'NE2_HE2':1.0,
               "C_OXT":1.0, "CB_HB1":1.0},
         'HID':{'N_CA':1.0, 'N_H':1.0, 'CA_C':1.0, 'CA_CB':1.0, 'CA_HA':1.0, 'C_O':2.0, 'CB_CG':1.0, 'CB_HB2':1.0, 'CB_HB3':1.0, 'CG_ND1':1.5,
                'CG_CD2':2.0, 'ND1_CE1':1.5, 'ND1_HD1':1.0, 'CD2_NE2':1.5, 'CD2_HD2':1.0, 'CE1_NE2':2.0, 'CE1_HE1':1.0, 'NE2_HE2':1.0,
                "C_OXT":1.0, "CB_HB1":1.0},
         'HIP':{'N_CA':1.0, 'N_H':1.0, 'CA_C':1.0, 'CA_CB':1.0, 'CA_HA':1.0, 'C_O':2.0, 'CB_CG':1.0, 'CB_HB2':1.0, 'CB_HB3':1.0, 'CG_ND1':1.5,
                'CG_CD2':2.0, 'ND1_CE1':1.5, 'ND1_HD1':1.0, 'CD2_NE2':1.5, 'CD2_HD2':1.0, 'CE1_NE2':2.0, 'CE1_HE1':1.0, 'NE2_HE2':1.0,
                "C_OXT":1.0, "CB_HB1":1.0},
         'ILE':{'N_CA':1.0, 'N_H':1.0, 'CA_C':1.0, 'CA_CB':1.0, 'CA_HA':1.0, 'C_O':2.0, 'CB_CG1':1.0, 'CB_CG2':1.0, 'CB_HB':1.0, 'CG1_CD1':1.0,
                'CG1_HG12':1.0, 'CG1_HG13':1.0, 'CG2_HG21':1.0, 'CG2_HG22':1.0, 'CG2_HG23':1.0, 'CD1_HD11':1.0, 'CD1_HD12':1.0, 'CD1_HD13':1.0,
                "C_OXT":1.0, "CG1_HG11":1.0},
         'LEU':{'N_CA':1.0, 'N_H':1.0, 'CA_C':1.0, 'CA_CB':1.0, 'CA_HA':1.0, 'C_O':2.0, 'CB_CG':1.0, 'CB_HB2':1.0, 'CB_HB3':1.0, 'CG_CD1':1.0,
                'CG_CD2':1.0, 'CG_HG':1.0, 'CD1_HD11':1.0, 'CD1_HD12':1.0, 'CD1_HD13':1.0, 'CD2_HD21':1.0, 'CD2_HD22':1.0, 'CD2_HD23':1.0, 
                "C_OXT":1.0, "CB_HB1":1.0},
         'LYN':{'N_CA':1.0, 'N_H':1.0, 'CA_C':1.0, 'CA_CB':1.0, 'CA_HA':1.0, 'C_O':2.0, 'CB_CG':1.0, 'CB_HB2':1.0, 'CB_HB3':1.0, 'CG_CD': 1.0,
                'CG_HG2':1.0, 'CG_HG3':1.0, 'CD_CE':1.0, 'CD_HD2':1.0, 'CD_HD3':1.0, 'CE_NZ':1.0, 'CE_HE2':1.0, 'CE_HE3':1.0, 'NZ_HZ1':1.0,
                'NZ_HZ2':1.0, "C_OXT":1.0, "CB_HB1":1.0, "CG_HG1":1.0, "CD_HD1":1.0, "CE_HE1":1.0},
         'LYS':{'N_CA':1.0, 'N_H':1.0, 'CA_C':1.0, 'CA_CB':1.0, 'CA_HA':1.0, 'C_O':2.0, 'CB_CG':1.0, 'CB_HB2':1.0, 'CB_HB3':1.0, 'CG_CD': 1.0,
                'CG_HG2':1.0, 'CG_HG3':1.0, 'CD_CE':1.0, 'CD_HD2':1.0, 'CD_HD3':1.0, 'CE_NZ':1.0, 'CE_HE2':1.0, 'CE_HE3':1.0, 'NZ_HZ1':1.0,
                'NZ_HZ2':1.0, 'NZ_HZ3':1.0, "C_OXT":1.0, "CB_HB1":1.0, "CG_HG1":1.0, "CD_HD1":1.0, "CE_HE1":1.0},
         'MET':{'N_CA':1.0, 'N_H':1.0, 'CA_C':1.0, 'CA_CB':1.0, 'CA_HA':1.0, 'C_O':2.0, 'CB_CG':1.0, 'CB_HB2':1.0, 'CB_HB3':1.0, 'CG_HG2':1.0,
                'CG_HG3':1.0, 'CG_SD':1.0, 'SD_CE':1.0, 'CE_HE1':1.0, 'CE_HE2':1.0, 'CE_HE3':1.0, "C_OXT":1.0, "CB_HB1":1.0, "CG_HG1":1.0},
         'PHE':{'N_CA':1.0, 'N_H':1.0, 'CA_C':1.0, 'CA_CB':1.0, 'CA_HA':1.0, 'C_O':2.0, 'CB_CG':1.0, 'CB_HB2':1.0, 'CB_HB3':1.0, 'CG_CD1':1.5,
                'CG_CD2':1.5, 'CD1_CE1':2.0, 'CD1_HD1':1.0, 'CD2_CE2':2.0, 'CD2_HD2':1.0, 'CE1_CZ':2.0, 'CE1_HE1':1.0, 'CE2_CZ':1.5,
                'CE2_HE2':1.0, 'CZ_HZ':1.0, "C_OXT":1.0, "CB_HB1":1.0},
         'PRO':{'N_CA':1.0, 'N_CD':1.0, 'CA_C':1.0, 'CA_CB':1.0, 'CA_HA':1.0, 'C_O':2.0, 'CB_CG':1.0, 'CB_HB2':1.0, 'CB_HB3':1.0, 'CG_CD':1.0,
                'CG_HG2':1.0, 'CG_HG3':1.0, 'CD_HD2':1.0, 'CD_HD3':1.0, "C_OXT":1.0, "CB_HB1":1.0, "CG_HG1":1.0, "CD_HD1":1.0},
         'SER':{'N_CA':1.0, 'N_H':1.0, 'CA_C':1.0, 'CA_CB':1.0, 'CA_HA':1.0, 'C_O':2.0, 'CB_OG':1.0, 'CB_HB2':1.0, 'CB_HB3':1.0, 'OG_HG':1.0,
               "C_OXT":1.0, "CB_HB1":1.0},
         'THR':{'N_CA':1.0, 'N_H':1.0, 'CA_C':1.0, 'CA_CB':1.0, 'CA_HA':1.0, 'C_O':2.0, 'CB_OG1':1.0, 'CB_CG2':1.0, 'CB_HB':1.0, 'OG1_HG1':1.0,
                'CG2_HG21':1.0, 'CG2_HG22':1.0, 'CG2_HG23':1.0, "C_OXT":1.0},
         'TRP':{'N_CA':1.0, 'N_H':1.0, 'CA_C':1.0, 'CA_CB':1.0, 'CA_HA':1.0, 'C_O':2.0, 'CB_CG':1.0, 'CB_HB2':1.0, 'CB_HB3':1.0, 'CG_CD1':2.0,
                'CG_CD2':1.5, 'CD1_NE1':1.5,'CD1_HD1':1.0, 'CD2_CE2':1.5, 'CD2_CE3':1.5, 'NE1_CE2':1.5, 'NE1_HE1':1.0, 'CE2_CZ2':1.5,
                'CE3_CZ3':2.0, 'CE3_HE3':1.0, 'CZ2_CH2':2.0, 'CZ2_HZ2':1.0, 'CZ3_CH2':1.5, 'CZ3_HZ3':1.0, 'CH2_HH2':1.0, "C_OXT":1.0,
                "CB_HB1":1.0},
         'TYR':{'N_CA':1.0, 'N_H':1.0, 'CA_C':1.0, 'CA_CB':1.0, 'CA_HA':1.0, 'C_O':2.0, 'CB_CG':1.0, 'CB_HB2':1.0, 'CB_HB3':1.0, 'CG_CD1':1.5,
                'CG_CD2':2.0, 'CD1_CE1':2.0, 'CD1_HD1':1.0, 'CD2_CE2':2.0, 'CD2_HD2':1.0, 'CE1_CZ':2.0, 'CE1_HE1':1.0, 'CE2_CZ':2.0, 
                'CE2_HE2':1.0, 'CZ_OH':1.0, 'OH_HH':1.0, "C_OXT":1.0, "CB_HB1":1.0},
         'VAL':{'N_CA':1.0, 'N_H':1.0, 'CA_C':1.0, 'CA_CB':1.0, 'CA_HA':1.0, 'C_O':2.0, 'CB_CG1':1.0, 'CB_CG2':1.0, 'CB_HB': 1.0, 
                'CG1_HG11':1.0, 'CG1_HG12':1.0, 'CG1_HG13':1.0, 'CG2_HG21':1.0, 'CG2_HG22':1.0, 'CG2_HG23':1.0, "C_OXT":1.0},
        }
        
        self.PDB2GMX = {
         'ALA':{'C':'C','CA':'CA','CB':'CB','H':'H','HA':'HA','HB1':'HB1','HB2':'HB2','HB3':'HB3','N':'N','O':'O','OXT':'OXT','H1':'H1','H2':'H2',
                'H3':'H3'},
         'ARG':{'C':'C','CA':'CA','CB':'CB','CD':'CD','CG':'CG','CZ':'CZ','H':'H','HA':'HA','HB2':'HB1','HB3':'HB2','HD2':'HD1','HD3':'HD2','HE':'HE',
                'HG2':'HG1','HG3':'HG2','HH11':'HH11','HH12':'HH12','HH21':'HH21','HH22':'HH22','N':'N','NE':'NE','NH1':'NH1','NH2':'NH2','O':'O',
                'OXT':'OXT','H1':'H1','H2':'H2','H3':'H3','HB1':'HB1','HD1':'HD1','HG1':'HG1'},
         'ASN':{'C':'C','CA':'CA','CB':'CB','CG':'CG','H':'H','HA':'HA','HB2':'HB1','HB3':'HB2','HD21':'HD21','HD22':'HD22','N':'N','ND2':'ND2',
                'O':'O','OD1':'OD1','OXT':'OXT','H1':'H1','H2':'H2','H3':'H3','HB1':'HB1'},
         'ASP':{'C':'C','CA':'CA','CB':'CB','CG':'CG','H':'H','H1':'H1','H2':'H2','H3':'H3','HA':'HA','HB2':'HB1','HB3':'HB2','N':'N','O':'O',
                'OD1':'OD1','OD2':'OD2','OXT':'OXT','HB1':'HB1'},
         'ASH':{'C':'C','CA':'CA','CB':'CB','CG':'CG','H':'H','H1':'H1','H2':'H2','H3':'H3','HA':'HA','HB2':'HB1','HB3':'HB2','N':'N','O':'O',
                'OD1':'OD1','OD2':'OD2','HD2':'HD2','OXT':'OXT','HB1':'HB1'},
         'CYS':{'C':'C','CA':'CA','CB':'CB','H':'H','HA':'HA','HB2':'HB1','HB3':'HB2','HG':'HG','N':'N','O':'O','SG':'SG','OXT':'OXT','H1':'H1',
                'H2':'H2','H3':'H3','HB1':'HB1'},
         'CYX':{'C':'C','CA':'CA','CB':'CB','H':'H','HA':'HA','HB2':'HB1','HB3':'HB2','N':'N','O':'O','SG':'SG','OXT':'OXT','H1':'H1',
                'H2':'H2','H3':'H3','HB1':'HB1'},
         'GLH':{'C':'C','CA':'CA','CB':'CB','CD':'CD','CG':'CG','H':'H','HA':'HA','HB2':'HB1','HB3':'HB2','HE2':'HE2','HG2':'HG1','HG3':'HG2','N':'N',
                'O':'O','OE1':'OE1','OE2':'OE2','OXT':'OXT','H1':'H1','H2':'H2','H3':'H3','HB1':'HB1','HG1':'HG1'},
         'GLN':{'C':'C','CA':'CA','CB':'CB','CD':'CD','CG':'CG','H':'H','HA':'HA','HB2':'HB1','HB3':'HB2','HE21':'HE21','HE22':'HE22','HG2':'HG1',
                'HG3':'HG2','N':'N','NE2':'NE2','O':'O','OE1':'OE1','OXT':'OXT','H1':'H1','H2':'H2','H3':'H3','HB1':'HB1','HG1':'HG1'},
         'GLU':{'C':'C','CA':'CA','CB':'CB','CD':'CD','CG':'CG','H':'H','HA':'HA','HB2':'HB1','HB3':'HB2','HG2':'HG1','HG3':'HG2','N':'N','O':'O',
                'OE1':'OE1','OE2':'OE2','OXT':'OXT','H1':'H1','H2':'H2','H3':'H3','HB1':'HB1','HG1':'HG1'},
         'GLY':{'C':'C','CA':'CA','H':'H','HA2':'HA1','HA3':'HA2','N':'N','O':'O','OXT':'OXT','H1':'H1','H2':'H2','H3':'H3','HA1':'HA1'},
         'HID':{'C':'C','CA':'CA','CB':'CB','CD2':'CD2','CE1':'CE1','CG':'CG','H':'H','HA':'HA','HB2':'HB1','HB3':'HB2','HD1':'HD1','HD2':'HD2',
                'HE1':'HE1','N':'N','ND1':'ND1','NE2':'NE2','O':'O','OXT':'OXT','H1':'H1','H2':'H2','H3':'H3','HB1':'HB1'},
         'HIE':{'C':'C','CA':'CA','CB':'CB','CD2':'CD2','CE1':'CE1','CG':'CG','H':'H','HA':'HA','HB2':'HB1','HB3':'HB2','HD2':'HD2','HE1':'HE1',
                'HE2':'HE2','N':'N','ND1':'ND1','NE2':'NE2','O':'O','OXT':'OXT','H1':'H1','H2':'H2','H3':'H3','HB1':'HB1'},
         'HIP':{'C':'C','CA':'CA','CB':'CB','CD2':'CD2','CE1':'CE1','CG':'CG','H':'H','HA':'HA','HB2':'HB1','HB3':'HB2','HD1':'HD1','HD2':'HD2',
                'HE1':'HE1','HE2':'HE2','N':'N','ND1':'ND1','NE2':'NE2','O':'O','OXT':'OXT','H1':'H1','H2':'H2','H3':'H3','HB1':'HB1'},
         'HIS':{'C':'C','CA':'CA','CB':'CB','CD2':'CD2','CE1':'CE1','CG':'CG','H':'H','HA':'HA','HB2':'HB1','HB3':'HB2','HD1':'HD1','HD2':'HD2',
                'HE1':'HE1','HE2':'HE2','N':'N','ND1':'ND1','NE2':'NE2','O':'O','OXT':'OXT','H1':'H1','H2':'H2','H3':'H3','HB1':'HB1'},
         'ILE':{'C':'C','CA':'CA','CB':'CB','CD1':'CD','CG1':'CG1','CG2':'CG2','H':'H','HA':'HA','HB':'HB','HD11':'HD1','HD12':'HD2','HD13':'HD3',
                'HG12':'HG11','HG13':'HG12','HG21':'HG21','HG22':'HG22','HG23':'HG23','N':'N','O':'O','OXT':'OXT','H1':'H1','H2':'H2','H3':'H3',
                'HG11':'HG11'},
         'LEU':{'C':'C','CA':'CA','CB':'CB','CD1':'CD1','CD2':'CD2','CG':'CG','H':'H','HA':'HA','HB2':'HB1','HB3':'HB2','HD11':'HD11','HD12':'HD12',
                'HD13':'HD13','HD21':'HD21','HD22':'HD22','HD23':'HD23','HG':'HG','N':'N','O':'O','OXT':'OXT','H1':'H1','H2':'H2','H3':'H3','HB1':'HB1'},
         'LYS':{'C':'C','CA':'CA','CB':'CB','CD':'CD','CE':'CE','CG':'CG','H':'H','HA':'HA','HB2':'HB1','HB3':'HB2','HD2':'HD1','HD3':'HD2','HE2':'HE1',
                'HE3':'HE2','HG2':'HG1','HG3':'HG2','HZ1':'HZ1','HZ2':'HZ2','HZ3':'HZ3','N':'N','NZ':'NZ','O':'O','OXT':'OXT','H1':'H1','H2':'H2',
                'H3':'H3','HB1':'HB1','HD1':'HD1','HE1':'HE1','HG1':'HG1'},
         'LYN':{'C':'C','CA':'CA','CB':'CB','CD':'CD','CE':'CE','CG':'CG','H':'H','HA':'HA','HB2':'HB1','HB3':'HB2','HD2':'HD1','HD3':'HD2','HE2':'HE1',
                'HE3':'HE2','HG2':'HG1','HG3':'HG2','HZ1':'HZ1','HZ2':'HZ2','N':'N','NZ':'NZ','O':'O','OXT':'OXT','H1':'H1','H2':'H2','H3':'H3',
                'HB1':'HB1','HD1':'HD1','HE1':'HE1','HG1':'HG1'},
         'MET':{'C':'C','CA':'CA','CB':'CB','CE':'CE','CG':'CG','H':'H','HA':'HA','HB2':'HB1','HB3':'HB2','HE1':'HE1','HE2':'HE2','HE3':'HE3',
                'HG2':'HG1','HG3':'HG2','N':'N','O':'O','SD':'SD','OXT':'OXT','H1':'H1','H2':'H2','H3':'H3','HB1':'HB1','HG1':'HG1'},
         'PHE':{'C':'C','CA':'CA','CB':'CB','CD1':'CD1','CD2':'CD2','CE1':'CE1','CE2':'CE2','CG':'CG','CZ':'CZ','H':'H','HA':'HA','HB2':'HB1',
                'HB3':'HB2','HD1':'HD1','HD2':'HD2','HE1':'HE1','HE2':'HE2','HZ':'HZ','N':'N','O':'O','OXT':'OXT','H1':'H1','H2':'H2','H3':'H3',
                'HB1':'HB1'},
         'PRO':{'C':'C','CA':'CA','CB':'CB','CD':'CD','CG':'CG','HA':'HA','HB2':'HB1','HB3':'HB2','HD2':'HD1','HD3':'HD2','HG2':'HG1','HG3':'HG2',
                'N':'N','O':'O','OXT':'OXT','H2':'H2','H3':'H3','HB1':'HB1','HD1':'HD1','HG1':'HG1'},
         'SER':{'C':'C','CA':'CA','CB':'CB','H':'H','HA':'HA','HB2':'HB1','HB3':'HB2','HG':'HG','N':'N','O':'O','OG':'OG','OXT':'OXT','H1':'H1',
                'H2':'H2','H3':'H3','HB1':'HB1'},
         'THR':{'C':'C','CA':'CA','CB':'CB','CG2':'CG2','H':'H','HA':'HA','HB':'HB','HG1':'HG1','HG21':'HG21','HG22':'HG22','HG23':'HG23','N':'N',
                'O':'O','OG1':'OG1','OXT':'OXT','H1':'H1','H2':'H2','H3':'H3'},
         'TRP':{'C':'C','CA':'CA','CB':'CB','CD1':'CD1','CD2':'CD2','CE2':'CE2','CE3':'CE3','CG':'CG','CH2':'CH2','CZ2':'CZ2','CZ3':'CZ3','H':'H',
                'HA':'HA','HB2':'HB1','HB3':'HB2','HD1':'HD1','HE1':'HE1','HE3':'HE3','HH2':'HH2','HZ2':'HZ2','HZ3':'HZ3','N':'N','NE1':'NE1','O':'O',
                'OXT':'OXT','H1':'H1','H2':'H2','H3':'H3','HB1':'HB1'},
         'TYR':{'C':'C','CA':'CA','CB':'CB','CD1':'CD1','CD2':'CD2','CE1':'CE1','CE2':'CE2','CG':'CG','CZ':'CZ','H':'H','HA':'HA','HB2':'HB1',
                'HB3':'HB2','HD1':'HD1','HD2':'HD2','HE1':'HE1','HE2':'HE2','HH':'HH','N':'N','O':'O','OH':'OH','OXT':'OXT','H1':'H1','H2':'H2',
                'H3':'H3','HB1':'HB1'},
         'VAL':{'C':'C','CA':'CA','CB':'CB','CG1':'CG1','CG2':'CG2','H':'H','HA':'HA','HB':'HB','HG11':'HG11','HG12':'HG12','HG13':'HG13','HG21':'HG21',
                'HG22':'HG22','HG23':'HG23','N':'N','O':'O','OXT':'OXT','H1':'H1','H2':'H2','H3':'H3'}
        }

        self.GMX2PDB = {r:{self.PDB2GMX[r][k]:k for k in self.PDB2GMX[r].keys()} for r in self.PDB2GMX.keys()}




class ONIOM_compound():
    def __init__(self, file_path="", log=False, modification=True):
        self.molDATA = pd.DataFrame()
        self.mol_trail = []
        self.Contents = {"title":"", "procN":1, "mem":1000, "mem_scal":"MB", "chk":"", "connection":[], "optPath":[], "enePath":[], "energy":0}
        self.Route = "# test"
        self.C_S = np.array([0,1])
        self.addInf = ""
        
        self.P = Params_O()
        if len(file_path) > 0:
            self.Read_Structure(file_path, log=log, modification=modification)
    
    def __len__(self):
        return len(self.molDATA)


    def Read_GaussianInput_ONIOM(self, file_path, readOnly=False, log=False):
        if not (".gjf" in file_path.lower() or ".com" in file_path.lower()):
            print("This is not Gaussian input file.")
            return 0

        flag = "Link0"
        if log: print(f"flag=Link0")
        Route = "" ; Cond = {"title":"", "procN":1, "mem":0, "mem_scal":"", "chk":"", "connection":[], "optPath":[], "energy":0, "Enthalpy":0, "Free Energy":0}
        cs = [] ; con = [] ; atomData = [] ; atch = [] ; coo = [] ; freez = [] ; layer = [] ; addInf = []
        with open(file_path, 'r') as output:
            for i, line in enumerate(output):
                ln = line.strip().split()
                if ("#" in line) and (flag == "Link0"):
                    if log: print(f"line {i}| flag=Route")
                    flag = "Route"
                elif (len(ln)==0) and (flag == "Route"):
                    if log: print(f"line {i}| flag=Title")
                    flag = "Title"
                elif (len(ln)==0) and (flag == "Title"):
                    if log: print(f"line {i}| flag=C&S")
                    flag = "C&S"
                elif len(cs)>0 and (flag == "C&S"):
                    if (" 0 " in line[5:] or " -1 " in line[5:]) and (" H" in line[5:] or " L" in line[5:]):
                        if log: print(f"line {i}| flag=Cood-ONIOM")
                        flag = "Cood-ONIOM"
                    else:
                        if log: print(f"line {i}| flag=Cood-MOL")
                        flag = "Cood-MOL"
                elif (len(ln)==0) and ("Cood" in flag) and ("connectivity" in Route):
                    if log: print(f"line {i}| flag=Conn")
                    flag = "Conn" ; con = np.zeros((len(coo),len(coo)), dtype=float)
                elif (len(ln)==0) and ("Cood" in flag or flag == "Conn"):
                    if log: print(f"line {i}| flag=End")
                    flag = "End"

                if flag == "Link0":
                    if "nprocshared" in line.lower():
                        Cond["procN"] = int(line.strip().split("=")[1])
                    elif "mem" in line.lower():
                        mem = str(line.strip().split("=")[1])
                        mem_scal = ""
                        Cond["mem"] = 0
                        while not Cond["mem"]:
                            try:
                                Cond["mem"] = int(mem)
                            except:
                                mem_scal = mem[-1] + mem_scal
                                mem = mem[:-1]
                        Cond["mem_scal"] = mem_scal
                    elif "chk" in line.lower():
                        Cond["chk"] = str(line.strip().split("=")[1])
                elif flag == "Route":
                    Route += line.strip()+" "
                elif flag == "title":
                    Cond["title"] += line.strip()
                elif flag == "C&S":
                    cs = np.array(ln, dtype=int)
                elif "Cood" in flag:
                    if flag == "Cood-ONIOM":
                        freez.append(ln[1])
                        coo.append(ln[2:5])
                        layer.append(ln[5])
                        if len(ln)>6:
                            addInf.append(" ".join(ln[6:]))
                        else:
                            addInf.append("")
                    elif flag == "Cood-MOL":
                        freez.append("0")
                        coo.append(ln[1:4])
                        layer.append("H")
                        addInf.append("")
                        
                    at="" ; aat="" ; charge=0 ; pdbdat = {"PDBName":"","ResName":"", "ResNum":""} ; re_ch = [0, ""]
                    if "(" in ln[0]:
                        atm = ln[0].strip(")").split("(")[0]
                        for item in ln[0].strip(")").split("(")[1].split(","):
                            pdbdat[item.split("=")[0]] = item.split("=")[1]
                    else:
                        atm = ln[0]
                    al = atm.split("-") 
                    try:
                        at = al[0] ; aat = al[1] ; charge = float(al[-1])   
                    except:
                        if log:
                            print("Atomic information cannot be found.") ; print(ln)
                    if atm.count("-") == 3:
                        charge *= -1

                    rc = pdbdat["ResNum"].split("_")
                    if len(rc)==2:
                        re_ch = [int(rc[0]), rc[1]]
                        
                    atomData.append([at, aat, pdbdat["PDBName"], pdbdat["ResName"]]+re_ch)
                    atch.append(charge)
                    

                elif flag == "Conn":
                    if len(ln)>2:
                        i = int(ln[0])-1
                        for j in range(1, len(ln), 2):
                            ii = int(ln[j])-1
                            bnd = round(float(ln[j+1]),1)
                            con[i, ii] = bnd ; con[ii, i] = bnd

                


        coo = np.array(coo, dtype=float)
        atomData = np.array(atomData, dtype=str)
        
        amberT = [""]*len(coo) ; uffT = [""]*len(coo)
        if "H_" in atomData[:,1]:
            uffT = atomData[:,1]
        else:
            amberT = atomData[:,1]

        molDATA = pd.DataFrame({
            "id":np.array([i for i in range(len(coo))], dtype=int),
            "AtomID":np.array([i for i in range(1,len(coo)+1)], dtype=int),
            "AtomType":atomData[:,0],
            "AmberType":amberT,
            "UFFType":uffT,
            "charge":np.array(atch,dtype=float),
            "PDBName":atomData[:,2],
            "ResName":atomData[:,3],
            "ResID":np.array(atomData[:,4], dtype=int),
            "ChainID":atomData[:,5],
            "freeze":np.array(freez, dtype=int),
            "x":np.array(coo[:,0],dtype=float),
            "y":np.array(coo[:,1],dtype=float),
            "z":np.array(coo[:,2],dtype=float),
            "layer":np.array(layer, dtype=str),
            "info":np.array(addInf, dtype=str)
        })

        Cond["connection"] = con
        
        if not readOnly:
            self.mol_trail.append(self.molDATA)
            self.molDATA = molDATA
            self.Contents = Cond
            self.Route = Route
            self.C_S = cs
        
        return molDATA
    
    
    def Read_GaussianOutput_ONIOM(self, file_path, readOnly=False, log=False):
        if not (".log" in file_path.lower() or ".out" in file_path.lower()):
            print("This is not Gaussian output file.")
            return 0
        
        Flag = "None"
        Charge_from_calc = False; ONIOM_Mode = True; Skip_structure = False
        cood = []; cood_path = []; inp_path = []; stan_path = []; oniom_path = []; oniom_inp_path = []; oniom_stan_path = []
        e_path = []; atomData = []; atch = []; coo = []
        freez = []; layer = []; addInf = []; cs = []; Route = ""; orbi = []; OMOs = []; UMOs = []
        nao_data = []; nao_sum_data = []; bo_low = []; bo_data = []; sum_data = []
        f_info_data = {"ID":[],"Symmetry":[]}; f_data = {}
        Cond = {"title":"", "procN":1, "mem":0, "mem_scal":"", "chk":"", "NormalTerm":False,
                "connection":[], "optPath":[], "optPath_ONIOM":[], "enePath":[], "energy":0, "SuccessOpt":False,
                "HOMO":0., "LUMO":0., "OMOs":[], "UMOs":[], "NBO":{}, "Freq_Info":[], "Freq_Mode":{}}
        with open(file_path, 'r') as output:
            for i, line in enumerate(output):  
                #Flag rotation
                if "Input orientation:" in line:
                    if log: print(f"{i}  Flag Input |  {line}")
                    Flag = "Input"
                if "Standard orientation:" in line:
                    if log: print(f"{i}  Flag Standard |  {line}")
                    Flag = "Standard"
                elif (Flag == "Input" or Flag == "Standard") and ("Coordinates (Angstroms)" in line):
                    if log: print(f"{i}  Flag {Flag}Orientation  |  {line}")
                    Flag += "Orientation"
                elif (Flag == "InputOrientation" or Flag == "StandardOrientation") and ("----------------------" in line):
                    if log: print(f"{i}  Flag {Flag}Collect  |  {line}")
                    Skip_structure = False
                    Flag += "Collect"
                elif (Flag == "InputOrientationCollect" or Flag == "StandardOrientationCollect") and ("----------------------" in line):
                    if log: print(f"{i}  Flag None  |  {line}")
                    if not Skip_structure:
                        if log: print(f"{i}  Structure was appended.")
                        if "Input" in Flag:
                            inp_path.append(cood)
                        elif "Standard" in Flag:
                            stan_path.append(cood)
                    else:
                        if log: print(f"{i}  Structure was skipped.")
                    cood = []
                    Flag = "None"
                elif (Flag == "None" or Flag == "MO_Pop") and ("Molecular Orbital Coefficients" in line):
                    if log: print(f"{i}  Flag MO_Coef  |  {line}")
                    Flag = "MO_Coef"
                elif (Flag == "MO_Coef" or Flag == "MO_Pop") and ("Density Matrix" in line or "Condensed to atoms" in line):
                    if log: print(f"{i}  Flag None  |  {line}")
                    Flag = "None"
                elif (Flag == "None") and ("Population analysis using the SCF density" in line):
                    if log: print(f"{i}  Flag MO_Pop  |  {line}")
                    OMOs = []; UMOs = [] 
                    Flag = "MO_Pop"
                elif (Flag == "None") and ("N A T U R A L   A T O M I C   O R B I T A L" in line):
                    if log: print(f"{i}  Flag NBO-start  |  {line}") 
                    Flag = "NBO-start"
                elif (Flag == "NBO-start") and ("-------------------" in line):
                    if log: print(f"{i}  Flag NBO-NAO  |  {line}") 
                    nao_data = []
                    Flag = "NBO-NAO"
                elif (Flag == "NBO-NAO") and ("Summary of Natural Population Analysis" in line):
                    if log: print(f"{i}  Flag NBO-NAO-sum  |  {line}") 
                    nao_sum_data = []
                    Flag = "NBO-NAO-sum"
                elif (Flag == "NBO-NAO-sum") and ("=============" in line):
                    if log: print(f"{i}  Flag NBO-mid  |  {line}") 
                    Flag = "NBO-mid"
                elif (Flag == "NBO-mid") and ("(Occupancy)" in line):
                    if log: print(f"{i}  Flag NBO-BondOrbital |  {line}") 
                    bo_data = []
                    Flag = "NBO-BondOrbital"
                elif (Flag == "NBO-BondOrbital") and len(line.strip())==0:
                    if log: print(f"{i}  Flag NBO-mid2 |  {line}") 
                    if len(bo_low) > 0:
                        bo_data.append(bo_low)
                    Flag = "NBO-mid2"
                elif (Flag == "NBO-mid2") and ("Natural Bond Orbitals (Summary)" in line):
                    if log: print(f"{i}  Flag NBO-Summary |  {line}") 
                    sum_data = []
                    Flag = "NBO-Summary"
                elif (Flag == "NBO-Summary") and ("------------------" in line):
                    if log: print(f"{i}  Flag None |  {line}") 
                    Flag = "None"
                elif (Flag == "None") and ("Harmonic frequencies (cm**-1)" in line):
                    if log: print(f"{i}  Flag Freq |  {line}") 
                    Flag = "Freq"
                elif (Flag == "Freq") and len(line.strip())==0:
                    if log: print(f"{i}  Flag None  |  {line}")
                    Flag = "None"
                elif Charge_from_calc and ("Mulliken charges:" in line or "Mulliken charges and spin densities:" in line):
                    if log: print(f"{i}  Flag AtomicCharge  |  {line}")
                    Flag = "AtomicCharge"
                    atch = []
                elif (Flag == "AtomicCharge") and ("Sum of Mulliken charges" in line):
                    if log: print(f"{i}  Flag None  |  {line}")
                    Flag = "None"
                elif (Flag == "None") and ("Charge =" in line) and ("Multiplicity =" in line) and not Cond["SuccessOpt"]:
                    if log: print(f"{i}  Flag Charge&Spin&PDB  |  {line}")
                    Flag = "Charge&Spin&PDB"
                elif (Flag == "Charge&Spin&PDB") and len(line.strip())==0:
                    if log: print(f"{i}  Flag None  |  {line}")
                    Flag = "None"
                elif (Flag == "None") and ("#" in line) and len(Route)==0:
                    if log: print(f"{i}  Flag Route  |  {line}")
                    Flag = "Route"
                elif (Flag == "Route") and ("----------------------" in line):
                    if log: print(f"{i}  Flag None  |  {line}")
                    if not "oniom" in Route.lower():
                        ONIOM_Mode = False
                    Flag = "None"
                
                #Data acquisition
                if "Normal termination" in line:
                    Cond["NormalTerm"] = True
                    
                if "Optimization completed" in line:
                    Cond["SuccessOpt"] = True
                
                if ONIOM_Mode and "ONIOM: extrapolated energy =" in line:
                    if not Skip_structure:
                        try:
                            e_path.append(float(line.strip().split()[4]))
                            if log: print(f"{i} Energy value acquired  |  {line}")
                        except:
                            e_path.append(0.)
                        try:
                            oniom_inp_path.append(inp_path[-1])
                        except:
                            pass
                        try:
                            oniom_stan_path.append(stan_path[-1])
                        except:
                            pass
                    else:
                        if log: print(f"{i} Energy acquisition was skipped.")
                
                if not ONIOM_Mode and "SCF Done:" in line:
                    if not Skip_structure:
                        e_path.append(float(line.split("=")[1].strip().split()[0]))
                        if log: print(f"{i} Energy value acquired  |  {line}")
                    else:
                        if log: print(f"{i} Energy acquisition was skipped.")
                                       
                if "Energy=" in line and "NIter=" in line:
                    try:
                        Cond["energy"] = float(line.strip().split()[1])
                    except:
                        pass

                if "Sum of electronic and thermal" in line:
                    ln = line.strip().split()
                    if "Enthalpies" in line:
                        Cond["Enthalpy"] = float(ln[6])
                    elif "Free Energies" in line:
                        Cond["Free Energy"] = float(ln[7])
                
                if Flag == "InputOrientationCollect" or Flag == "StandardOrientationCollect":
                    ln = line.strip().split()
                    if len(ln)>3:
                        try:
                            cood.append(np.array(ln[3:], dtype=float))
                        except:
                            str_no = max(len(inp_path), len(stan_path))
                            print(f"CAUTION: Atomic coordination is invalid. This structure is skipped. Structure No.{str_no}")
                            print(line)
                            Skip_structure = True
                            

                elif Flag == "MO_Coef":
                    ln = line.strip().split()
                    if "Molecular Orbital" in line:
                        pass
                    elif set(ln).issubset({"O","V"}):
                        o_kind = ln
                        col_num = len(ln)
                    elif "Eigenvalue" in line:
                        o_ene = ln[2:]
                    else:
                        try:
                            o_num = np.array(ln, dtype=int).tolist()    
                        except:
                            if len(ln) == col_num+4:
                                orb_n = int(ln[0])
                                atm_n = int(ln[1])
                                atm_k = ln[2]
                                orb_k = ln[3]
                            else:
                                orb_n = int(ln[0])
                                orb_k = ln[1]
                            for ok, oe, on, coe in zip(o_kind, o_ene, o_num, ln[-1*col_num:]):
                                orbi.append([on, ok, oe, orb_n, atm_n, atm_k, orb_k, coe])

                elif Flag == "MO_Pop":
                    if ". eigenvalues" in line:
                        ln = line.strip().split()
                        for o in ln[4:]:
                            try:
                                if "occ" in ln[1]:
                                    OMOs.append(float(o))
                                elif "virt" in ln[1]:
                                    UMOs.append(float(o))
                            except:
                                space_num = 10 - (len(o) % 10)
                                s_o = " "*space_num + o
                                for i in range(int(round(len(s_o)/10))):
                                    if "occ" in ln[1]:
                                        OMOs.append(float(s_o[10*i:10*i+10]))
                                    elif "virt" in ln[1]:
                                        UMOs.append(float(s_o[10*i:10*i+10]))

                elif Flag == "NBO-NAO":
                    ln = line.strip().split()
                    try:
                        nao_id = float(ln[2]) #to validate low
                        tp = line.split("(")[0].split()[-1]
                        ao_tp = line.split("(")[1].split(")")[0].strip()
                        nao_data.append(ln[:4]+[tp, ao_tp]+ln[6:])
                    except:
                        pass

                elif Flag == "NBO-NAO-sum":
                    ln = line.strip().split()
                    try:
                        nat_ch = float(ln[3])
                        nao_sum_data.append(ln)
                    except:
                        pass
                
                elif Flag == "NBO-BondOrbital":
                    ln = line.strip().split()
                    if "BD" in line or "CR" in line or "RY" in line:
                        if not ln[0] == "1." and len(bo_low) > 0:
                            bo_data.append(bo_low)
                        if "BD" in line:
                            b_id = ln[0].strip(".")
                            occ = ln[1].strip("()")
                            b_tp = ln[2].split("(")[0]
                            atm1 = line.strip().split("-")[-2].split()[-1]
                            atm2 = ln[-1]
                            bo_low = [b_id, occ, b_tp, atm1, atm2]
                        else:
                            bo_low = []
                    elif "%)" in line:
                        coef = line.split("*")[0].split(")")[-1].strip()
                        coef_p = [pln.split("(")[-1].strip() for pln in line.strip().split("%")][:-1]

                        l_coef_p = len(coef_p)
                        if l_coef_p < 4:
                            coef_p += ["0"]*(4-l_coef_p)

                        if len(bo_low) > 0:
                            bo_low += [coef] + coef_p

                elif Flag == "NBO-Summary":
                    if "BD" in line:
                        ln = line.strip().split()
                        b_id = ln[0].strip(".")
                        pln = line.strip().split(")")[1].split()
                        if "(" in pln[-1]:
                            pln = pln[:-1]
                        ener = pln[-1]
                        sum_data.append([b_id, ener])

                elif Flag == "Freq":
                    ln = line.strip().split()
                    if len(ln) < 4 and "     " in line:
                        try:
                            mode_ids = [int(l) for l in ln]
                            for mi in mode_ids:
                                f_data[mi] = []
                            f_info_data["ID"] += ln
                        except:
                            f_info_data["Symmetry"] += ln
                    elif "--" in line:
                        hln = line.strip().split("--")
                        k = hln[0].strip()
                        hd = hln[1].strip().split()
                        try:
                            f_info_data[k] += hd
                        except:
                            f_info_data[k] = hd
                    elif len(ln) > 4:
                        try:
                            atm_id = int(ln[0])
                            for mi, li in zip(mode_ids, range(2, len(ln), 3)):
                                f_data[mi].append([atm_id] + ln[li:li+3])
                        except:
                            pass
                
                elif Flag == "AtomicCharge":
                    ln = line.strip().split()
                    if len(ln)>2:
                        try:
                            atch.append(float(ln[2]))
                        except:
                            pass
                                            
                elif Flag == "Route":
                    Route += line.strip()
                    
                elif Flag == "Charge&Spin&PDB":
                    ln = line.strip().split()
                    if "Charge =" in line:
                        ln = line.strip().split("=")
                        c = ln[1].split()[0]
                        s = ln[2].split()[0]
                        cs += [c,s]
                    elif not ("-1" in ln or "0" in ln):
                        freez.append("0")
                        layer.append("H")
                        if len(ln)>4:
                            addInf.append(" ".join(ln[4:]))
                        else:
                            addInf.append("")

                        at="" ; aat="" ; charge=0. ; pdbdat = {"PDBName":"","ResName":"", "ResNum":""} ; res_ch = [0, ""]
                        if "(" in ln[0]:
                            atm = ln[0].strip(")").split("(")[0]
                            for item in ln[0].strip(")").split("(")[1].split(","):
                                pdbdat[item.split("=")[0]] = item.split("=")[1]
                        else:
                            atm = ln[0]
                        al = atm.split("-") 
                        try:
                            if len(al)==3:
                                at = al[0] ; aat = al[1] ; charge = float(al[-1])   
                            else:
                                at = al[0]
                                Charge_from_calc = True
                        except:
                            print("Atomic information cannot be found.") ; print(ln)
                        if atm.count("-") == 3:
                            charge *= -1

                        rc = pdbdat["ResNum"].split("_")
                        if len(rc)==2:
                            res_ch = [int(rc[0]), rc[1]]
                        atomData.append([at, aat, pdbdat["PDBName"], pdbdat["ResName"]]+res_ch)
                        atch.append(charge)
                    
                    elif "L" in ln or "H" in ln or "M" in ln:
                        freez.append(ln[1])
                        layer.append(ln[5])
                        if len(ln)>6:
                            addInf.append(" ".join(ln[6:]))
                        else:
                            addInf.append("")

                        at="" ; aat="" ; charge=0 ; pdbdat = {"PDBName":"","ResName":"", "ResNum":""} ; res_ch = [0, ""]
                        if "(" in ln[0]:
                            atm = ln[0].strip(")").split("(")[0]
                            for item in ln[0].strip(")").split("(")[1].split(","):
                                pdbdat[item.split("=")[0]] = item.split("=")[1]
                        else:
                            atm = ln[0]
                        al = atm.split("-") 
                        try:
                            at = al[0] ; aat = al[1] ; charge = float(al[-1])   
                        except:
                            print("Atomic information cannot be found.") ; print(ln)
                        if atm.count("-") == 3:
                            charge *= -1

                        rc = pdbdat["ResNum"].split("_")
                        if len(rc)==2:
                            res_ch = [int(rc[0]), rc[1]]
                        atomData.append([at, aat, pdbdat["PDBName"], pdbdat["ResName"]]+res_ch)
                        atch.append(charge)
                        
                elif "%" in line:
                    if "nprocshared" in line.lower():
                        Cond["procN"] = int(line.strip().split("=")[1])
                    elif "mem" in line.lower():
                        mem = str(line.strip().split("=")[1])
                        mem_scal = ""
                        Cond["mem"] = 0
                        while not Cond["mem"]:
                            try:
                                Cond["mem"] = int(mem)
                            except:
                                mem_scal = mem[-1] + mem_scal
                                mem = mem[:-1]
                        Cond["mem_scal"] = mem_scal
                    elif "chk" in line.lower():
                        Cond["chk"] = str(line.strip().split("=")[1])
                        
        Cond["UMOs"] = np.sort(np.array(UMOs))
        Cond["OMOs"] = np.sort(np.array(OMOs))[::-1]
        try:
            Cond["LUMO"] = Cond["UMOs"].min()
        except:
            pass
        try:
            Cond["HOMO"] = Cond["OMOs"].max()   
        except:
            pass
          
        if len(stan_path) > 0:
            cood_path = np.array(stan_path, dtype=float)
            oniom_path = np.array(oniom_stan_path, dtype=float)
        else:
            cood_path = np.array(inp_path, dtype=float)
            oniom_path = np.array(oniom_inp_path, dtype=float)
        
        coo = cood_path[-1]
        if Cond["SuccessOpt"] and len(cood_path) > len(e_path) and len(cood_path) > 2:
            cood_path = cood_path[:-1]
        Cond["optPath"] = cood_path
        Cond["optPath_ONIOM"] = oniom_path
        Cond["enePath"] = np.array(e_path)
        if len(Cond["enePath"])>0:
            Cond["energy"] = Cond["enePath"][-1]
        atomData = np.array(atomData, dtype=str)
        
        amberT = [""]*len(coo) ; uffT = [""]*len(coo)
        if "H_" in atomData[:,1]:
            uffT = atomData[:,1]
        else:
            amberT = atomData[:,1]
        
        types = {"AOID":int,"AtomID":int,"AtomType":str,"AOType":str,"Occupation":str,"Energy":float,"MOID":int,"Coefficient":float}
        O_Table = pd.DataFrame(orbi, columns=["MOID","Occupation","Energy","AOID","AtomID","AtomType","AOType","Coefficient"])
        O_Table = O_Table.sort_values(["MOID","AOID"]).reset_index(drop=True).astype(types)

        Cond["Orbital"] = O_Table

        try:
            types = {"NAO_ID":int, "AtomType":str, "AtomID":int, "OrbitalShape":str, "OrbitalType":str, "AOType":str, "Occupancy":float, "Energy":float}
            NAO_Table = pd.DataFrame(nao_data, columns=["NAO_ID", "AtomType", "AtomID", "OrbitalShape", "OrbitalType", "AOType", "Occupancy", "Energy"]).astype(types)
            Cond["NBO"]["NAO"] = NAO_Table
        except Exception as e:
            print("Error in creation of Contents-NBO-NAO table.")
            print(e)
            print(nao_data)
        try:
            types = {"AtomType":str, "AtomID":int, "NaturalCharge":float, "NP-Core":float, "NP-Valence":float, "NP-Rydberg":float, "NP-Total":float}
            NAO_SUM_Table = pd.DataFrame(nao_sum_data, columns=["AtomType", "AtomID", "NaturalCharge", "NP-Core", "NP-Valence", "NP-Rydberg", "NP-Total"]).astype(types)
            Cond["NBO"]["NAO-Summary"] = NAO_SUM_Table
        except Exception as e:
            print("Error in creation of Contents-NBO-NAO-Summary table.")
            print(e)
            print(nao_sum_data)
        try:
            types = {"NBO_ID":int,"Occupancy":float,"NBO_Type":str,"AtomID_1":int,"AtomID_2":int,
                     "Coef_1":float,"Coef_%_1":float,"S_ratio_%_1":float,"P_ratio_%_1":float,"D_ratio_%_1":float,
                     "Coef_2":float,"Coef_%_2":float,"S_ratio_%_2":float,"P_ratio_%_2":float,"D_ratio_%_2":float}
            BO_Table = pd.DataFrame(bo_data, columns=["NBO_ID","Occupancy","NBO_Type","AtomID_1","AtomID_2",
                                                        "Coef_1","Coef_%_1","S_ratio_%_1","P_ratio_%_1","D_ratio_%_1",
                                                        "Coef_2","Coef_%_2","S_ratio_%_2","P_ratio_%_2","D_ratio_%_2"]).astype(types)
            types = {"NBO_ID":int,"Energy":float}
            BO_sub_Table = pd.DataFrame(sum_data, columns=["NBO_ID","Energy"]).astype(types)
            Cond["NBO"]["BondOrbital"] = pd.merge(BO_Table, BO_sub_Table, on="NBO_ID")
        except Exception as e:
            print("Error in creation of Contents-NBO-BondOrbital table.")
            print(e)
            print(bo_data)
        try:
            types = {k:float for k in f_info_data.keys()}
            types["ID"] = int; types["Symmetry"] = str
            F_Info_Table = pd.DataFrame(f_info_data).astype(types)
            Cond["Freq_Info"] = F_Info_Table
        except Exception as e:
            print("Error in creation of Freq_Info table.")
            print(e)
            print(f_info_data)
        try:
            types = {"AtomID":int,"x":float,"y":float,"z":float}
            for k in f_data.keys():
                Cond["Freq_Mode"][k] = pd.DataFrame(f_data[k], columns=["AtomID","x","y","z"]).astype(types)
        except Exception as e:
            print("Error in creation of Freq_Mode table.")
            print(e)
            print(f_data)
        
        molDATA = pd.DataFrame({
            "id":np.array([i for i in range(len(coo))], dtype=int),
            "AtomID":np.array([i for i in range(1,len(coo)+1)], dtype=int),
            "AtomType":atomData[:,0],
            "AmberType":amberT,
            "UFFType":uffT,
            "charge":np.array(atch,dtype=float),
            "PDBName":atomData[:,2],
            "ResName":atomData[:,3],
            "ResID":np.array(atomData[:,4], dtype=int),
            "ChainID":atomData[:,5],
            "freeze":np.array(freez, dtype=int),
            "x":np.array(coo[:,0],dtype=float),
            "y":np.array(coo[:,1],dtype=float),
            "z":np.array(coo[:,2],dtype=float),
            "layer":np.array(layer, dtype=str),
            "info":np.array(addInf, dtype=str)
        })
        
        if not readOnly:
            self.mol_trail.append(self.molDATA)
            self.molDATA = molDATA
            self.Contents = Cond
            self.Route = Route
            self.C_S = np.array(cs,dtype=int)
        
        return molDATA
    
    
    def Read_PDB(self, file_path, readOnly=False):
        if file_path.lower().endswith(".cif"):
            pdb_parser = BP.MMCIFParser()
        elif file_path.lower().endswith(".pdb") or file_path.lower().endswith(".pqr"):
            pdb_parser = BP.PDBParser()
        else:
            raise ValueError("File format is not valid.")

        structure = pdb_parser.get_structure(os.path.splitext(os.path.basename(file_path))[0], file_path)
        Data = []
        itr = 0
        for model in structure:
            for chain in model: 
                cid = chain.get_id()
                for residue in chain:
                    rn = residue.get_resname().strip()
                    rid = residue.get_id()[1]
                    if residue.is_disordered()==1:
                        print(f"CAUTION: {rid}{rn} contains disordered atoms.")
                    for atom in residue:
                        coo = atom.get_coord()
                        an = atom.get_name()
                        ele = atom.element
                        if len(ele) == 0:
                            if rn in self.P.AMINO_DNA:
                                ele = an[0]
                            else:
                                print(f"Atom type (ID:{itr+1}) is missing. Please confirm the compencated data.")
                                ele = an
                                while not ele.upper() in self.P.ATOM_NUM:
                                    ele = ele[:-1]
                        try:
                            num = int(an[0])
                            tp = an[1:]
                            an = f"{tp}{num}"
                        except:
                            pass
                        #bf = atom.get_bfactor() 
                        Data.append([itr, itr+1, ele, "", "", 0, an, rn, rid, cid, 0, coo[0], coo[1], coo[2], "L", ""])
                        itr += 1

        types = {"id":int,"AtomID":int,"AtomType":str,"AmberType":str,"UFFType":str,"charge":float,"PDBName":str,"ResName":str,
         "ResID":int,"ChainID":str,"freeze":int, "x":float, "y":float,"z":float,"layer":str, "info":str}                
        molDATA = pd.DataFrame(Data, columns=["id","AtomID","AtomType","AmberType","UFFType","charge","PDBName","ResName","ResID",
                                            "ChainID","freeze", "x", "y","z","layer", "info"]).astype(types)

        if not readOnly:
            self.mol_trail.append(self.molDATA)
            self.molDATA = molDATA
        
        return molDATA
    
    
    def Read_Structure(self, file_path, readOnly=False, log=False, modification=True):

        if ".pdb" in file_path or ".cif" in file_path or ".pqr" in file_path:
            molDATA = self.Read_PDB(file_path, readOnly=readOnly)

        elif ".gjf" in file_path or ".com" in file_path:
            molDATA = self.Read_GaussianInput_ONIOM(file_path, readOnly=readOnly, log=log)

        elif ".log" in file_path or ".out" in file_path:
            molDATA = self.Read_GaussianOutput_ONIOM(file_path, readOnly=readOnly, log=log)

        else:
            print("File could not be read.")
            return self.molDATA

        if modification:
            if len(molDATA[molDATA["AtomType"]=="H"]) > 0:
                self.Reset_ResName()
                pr = self.Check_MissingAtom(recover=True, include_H=True)
                #self.Reset_ResName(skips=pr)
            else:
                pr = self.Check_MissingAtom(recover=True, include_H=False)
        
        return molDATA


    def Write_GaussianInput_ONIOM(self, file_n="", file_path="."):
        
        self.Check_Collision()
        
        if file_n.endswith(".gjf") or file_n.endswith(".com"):
            file_name = file_n[:-4]
        elif file_n == "":
            file_name = self.Contents['title']
        else:
            file_name = file_n
            
        ONIOM_mode = "oniom" in self.Route.lower()
            
        if not len(self.Contents["connection"]) == len(self.molDATA):
            self.makeConnection()
            print("Connection data was automatically compensated. Check the connection around non-amino acid atoms.")
            
        if ONIOM_mode and (self.molDATA["info"]=="").all():
            uff_mode = "uff" in self.Route
            self.makeDammy(uff = uff_mode)
            print(f"Dammy atoms are set (UFF = {uff_mode}).")
        
        with open(f"{file_path}/{file_name}.gjf", "w") as g:
            g.write(f"%nprocshared={self.Contents['procN']}\n") 
            g.write(f"%mem={self.Contents['mem']}{self.Contents['mem_scal']}"+"\n")
            if not self.Contents['chk']=="":
                if not self.Contents['chk'].endswith(".chk"):
                    self.Contents['chk'] += ".chk"
                g.write(f"%chk={self.Contents['chk']}\n")
            g.write(f"{self.Route}")
            if not "geom=connectivity" in self.Route.lower(): g.write(f" geom=connectivity")
            
            g.write(f"\n\n{file_name}\n\n")
            
            ch_sp = ""
            for i in self.C_S: 
                ch_sp += f"{i} "
            g.write(ch_sp.strip()+"\n")

            for i in range(len(self.molDATA)):
                dat = self.molDATA.iloc[i]
                atm = dat["AtomType"] 
                if len(atm) > 1:
                    atm = atm[0].upper() + atm[1:].lower()
                if ONIOM_mode:
                    if "uff" in self.Route.lower():
                        atm += "-" + dat["UFFType"]
                    else:
                        atm += "-" + dat["AmberType"]
                    if not dat["charge"]==0: 
                        ch = dat["charge"]
                        atm += f"-{ch:.6f}"
                    pdb = "("
                    if dat["PDBName"]: pdb += f"PDBName={dat['PDBName']}," 
                    if dat["ResName"]: pdb += f"ResName={dat['ResName']}," 
                    if dat["ResID"]: pdb += f"ResNum={dat['ResID']}_{dat['ChainID']}"
                    pdb = pdb.strip(",")+")"
                    atm += pdb.replace("()","")
                    atm += " "*max(0, 55-len(atm))
                    fr = dat["freeze"]
                    ap = np.round(dat.loc[["x","y","z"]].values.astype(np.float64), 6)
                    ly = dat["layer"]
                    inf = dat["info"]
                    g.write(f" {atm}  {fr: >2}  {ap[0]:f}   {ap[1]:f}   {ap[2]:f} {ly} {inf}\n")
                else:
                    atm += " "*(15-len(atm))
                    ap = np.round(dat.loc[["x","y","z"]].values.astype(np.float64), 8)
                    g.write(f" {atm} {ap[0]: >13.8f} {ap[1]: >13.8f} {ap[2]: >13.8f}\n")
                
            g.write("\n")
            
            con = self.Contents["connection"]
            for i in range(len(con)):
                g.write(f" {i+1}")
                for j in np.where(con[i])[0]:
                    if (con[i,j]>0)and(j>i):
                        g.write(f" {j+1} {con[i,j]:.1f}")
                g.write("\n")            
            
            g.write(f"\n{self.addInf}\n\n")
            
        return 1
    
    
    def Write_BondScanInput_ONIOM(self, atomID_mv, atomID_st, stop_atomID=[], file_n="", file_path=".", s_size=0.5, s_num=1):
        if not len(self.Contents["connection"]) == len(self.molDATA):
            print("Please make information of connection by makeConnection()")
            return 0
        
        original_molDATA = self.molDATA.copy()
        original_addInf = self.addInf
        extend = 0
        s_pref = ""
        file_name_stem = file_n.split('.')[0]

        if re.match(r'.*_scan.*\d+$', file_name_stem):
            go = False
            s_index = file_name_stem.split("_scan")[-1]
            while not go:
                try:
                    extend = int(s_index)
                    go = True
                except:
                    s_pref += s_index[0]
                    s_index = s_index[1:]
            file_name_stem = "_scan".join(file_name_stem.split("_scan")[:-1])

        search_atm = (np.where(self.Contents["connection"][atomID_mv - 1])[0]+1).tolist()
        mv_id_list = [atomID_mv]
        while len(search_atm)>0:
            atmid = search_atm.pop(0)
            if (not atmid == atomID_st) and (not atmid in mv_id_list) and (not atmid in stop_atomID):
                mv_id_list.append(atmid)
                search_atm += (np.where(self.Contents["connection"][atmid - 1])[0]+1).tolist()
                
        st_pos = original_molDATA.loc[original_molDATA["AtomID"]==atomID_st, ["x","y","z"]].values.astype(float)
        mv_pos = original_molDATA.loc[original_molDATA["AtomID"]==atomID_mv, ["x","y","z"]].values.astype(float)
        vec = (mv_pos - st_pos)/np.linalg.norm(mv_pos - st_pos)
        
        self.addInf = f"B {atomID_st} {atomID_mv} F\n\n\n{self.addInf}"
        file_names = []
        for i in range(1, s_num+1):
            for atmid in mv_id_list:
                self.molDATA.loc[self.molDATA["AtomID"]==atmid,["x","y","z"]] += vec*s_size
            fn = f"{file_name_stem}_scan{s_pref}{extend + i}"
            file_names.append(fn)
            self.Write_GaussianInput_ONIOM(file_n=fn, file_path=file_path)
            
        self.addInf = original_addInf
        self.molDATA = original_molDATA
            
        return file_names
        
        
    def Write_PDB(self, file_n="", file_path=".", forGromacs=False):
        if file_n.endswith(".pdb"):
            file_name = file_n[:-4]
        elif file_n == "":
            file_name = self.Contents['title']
        else:
            file_name = file_n
            
        with open(f"{file_path}/{file_name}.pdb", "w") as w:
            w.write(f"TITLE      {file_name}\n")
            for i in range(len(self.molDATA)):
                m = self.molDATA.iloc[i]
                if m["ResName"] in self.P.AMINO_DNA:
                    at = "ATOM"
                else:
                    at = "HETATM"
                
                aid = m["AtomID"] ; pn = m["PDBName"] ; rna = m["ResName"] ; ch = m["ChainID"] ; rnu = m["ResID"]
                if forGromacs:
                    pn = self.P.PDB2GMX[rna][pn]
                w.write(f"{at:6s}{aid:5d} {pn:^4s}{'':1s}{rna:3s} {ch:1s}{rnu:4d}")
                x = m["x"] ; y = m["y"] ; z = m["z"] ; aty = m["AtomType"]
                w.write(f"{'':1s}   {x:8.3f}{y:8.3f}{z:8.3f}{0:6.2f}{0:6.2f}          {aty:>2s}{'':2s}\n")
            
            w.write("END\n")
    
    
    def Reset_ID(self, sort=False):
        self.Contents["connection"] = []
        cDATA = self.molDATA.copy()
        if sort:
            cDATA = cDATA.sort_values(["ChainID", "ResID", "PDBName"])
        cDATA = cDATA.reset_index(drop=True)
        cDATA["id"] = [i for i in range(len(cDATA))] ; cDATA["AtomID"] = cDATA["id"] + 1
        
        self.mol_trail.append(self.molDATA)
        self.molDATA = cDATA
        
        return self.molDATA
    
    
    def Renew_Structure(self, index=-1, ONIOM=False):
        path = "optPath_ONIOM" if ONIOM else "optPath"
        ml = self.Contents[path].shape[0]
        
        if (index > ml-1) or (index < -1*ml):
            print("index is out of range.")
        else:
            self.molDATA.loc[:, ["x","y","z"]] = self.Contents[path][index]
            
        return self.molDATA
    
    
    def Add_ligand(self, resData):
        if type(resData) is str:
            if ".pdb" in resData:
                res = self.Read_PDB(resData, readOnly=True)
            elif ".gjf" in resData or ".com" in resData:
                res = self.Read_GaussianInput_ONIOM(resData, readOnly=True)
            elif ".log" in resData or ".out" in resData:
                res = self.Read_GaussianOutput_ONIOM(resData, readOnly=True)
            else:
                print("resFile could not be read.")
                return 0
        elif type(resData) is ONIOM_compound:
            res = resData.molDATA.copy()
        elif type(resData) is pd.core.frame.DataFrame:
            res = resData.copy()
        else:
            print("resData could not be read.")
            return 0
        
        mol = pd.concat([self.molDATA, res])

        if not len(mol.columns) == len(self.molDATA.columns):
            print("Input DataFrame is not molDATA.")
            return 0
        
        self.mol_trail.append(self.molDATA)
        self.molDATA = mol

        self.Reset_ID(sort=True)
        
        return self.molDATA
    

    def Add_water_fromMD(self, MDFile_path, Ligand_Query, RMSD_Query="", ChainID_SOL="Z", max_d=15., min_d=2., log=False):
        temp = self.Read_PDB(MDFile_path, readOnly=True)
        if not temp.loc[0, "ChainID"].strip():
            temp["ChainID"] = self.molDATA.loc[0, "ChainID"]
        
        if len(RMSD_Query) > 0:
            mol = self.molDATA.query(f"ResName in {self.P.AMINO_LIST} and {RMSD_Query}")
            mol_t = temp.query(f"ResName in {self.P.AMINO_LIST} and {RMSD_Query}")
            self.move_MinimizedRMSD_byFragment(mol, mol_t, log=log)

        model_pos = self.GetQueryPos(Ligand_Query)

        SOL_dat_ALL = []
        with open(MDFile_path,"r") as r:
            SOL_id = 1
            SOL_pos = []
            SOL_dat = []
            for l in r:
                ln = l.strip().split()

                if ln[0] == "ATOM" and ln[3] == "SOL":
                    SOL_pos.append(ln[5:8])
                    if ln[2] == "OW":
                        SOL_dat.append([0, 0, "O", "", "", 0, "OW", "SOL", SOL_id, ChainID_SOL, 0, ln[5], ln[6], ln[7], "L", ""])
                    elif ln[2] == "HW1":
                        SOL_dat.append([0, 0, "H", "", "", 0, "HW1", "SOL", SOL_id, ChainID_SOL, 0, ln[5], ln[6], ln[7], "L", ""])
                    elif ln[2] == "HW2":
                        SOL_dat.append([0, 0, "H", "", "", 0, "HW2", "SOL", SOL_id, ChainID_SOL, 0, ln[5], ln[6], ln[7], "L", ""])
                        dmat = distance.cdist(model_pos, np.array(SOL_pos, dtype=float), metric='euclidean')
                        if dmat.min() < max_d and dmat.min() > min_d:          
                            SOL_dat_ALL += SOL_dat
                            SOL_id += 1
                        SOL_pos = []
                        SOL_dat = []

        molDATA_sol = pd.DataFrame(SOL_dat_ALL, columns=self.molDATA.columns).astype(self.molDATA.dtypes)
        self.Add_ligand(molDATA_sol)

        return self.molDATA


    def Undo_molDATA(self):
        self.molDATA = self.mol_trail.pop(-1)
        
        return self.molDATA
        
    
    def Cut_Structure(self, Condition, extract=False):
        cDATA = self.molDATA.copy()
        filt = np.ones(len(cDATA), dtype=bool)
        for i in Condition.keys():
            try:
                filt = filt&(cDATA[i]==Condition[i])
            except:
                print(f"{i} is not the molDATA column.")
                
        if extract:
            cDATA = cDATA.loc[filt]
        else:
            cDATA = cDATA.loc[~filt]
        self.mol_trail.append(self.molDATA)
        self.molDATA = cDATA
        self.Reset_ID()
        
        return self.molDATA
    
    
    def Add_Atom(self, AtomType, PDBName, ResNAME, resID, chainID, x, y, z):
        cDATA = self.molDATA.copy()
        d = pd.DataFrame([[0, 0, AtomType, "", "", 0, PDBName, ResNAME, resID, chainID, 0, x, y, z, "L",""]], columns=cDATA.columns)
        cDATA = pd.concat([cDATA, d]).reset_index(drop=True)
        
        self.mol_trail.append(self.molDATA)
        self.molDATA = cDATA
        self.Reset_ID()
        
        return cDATA
    
    
    def Extract_Protein(self):
        self.mol_trail.append(self.molDATA)
        self.molDATA = self.molDATA.query(f"ResName in {self.P.AMINO_DNA}")
        self.Reset_ID()
        
        return self.molDATA
    
    
    def DataComplement(self, column, data_file_path="", log=False):
        if data_file_path:
            if data_file_path.lower().endswith('.csv'):
                dtlist = pd.read_csv(data_file_path, keep_default_na=False)
            elif data_file_path.lower().endswith('.xlsx'):
                dtlist = pd.read_excel(data_file_path, keep_default_na=False)
            else:
                print("File format should be csv or xlsx.")
                return 0, self.molDATA
        else:
            if column == "charge":
                dtlist = self.molDATA[~(self.molDATA[column].astype(float)==0.)]
            else:
                dtlist = self.molDATA[~(self.molDATA[column].astype(str)=="")]

        cDATA = self.molDATA.copy()
        if column == "charge":
            nanlist = self.molDATA[self.molDATA[column].astype(float)==0.]
        elif column == "freeze" or column == "layer":
            nanlist = self.molDATA
        else:
            nanlist = self.molDATA[self.molDATA[column].astype(str)==""]

        for i in range(len(nanlist)):
            nl = nanlist.iloc[i]
            if nl["PDBName"]=="" or nl["ResName"]=="": continue

            dt = dtlist.loc[(dtlist["PDBName"]==nl["PDBName"])&(dtlist["ResName"]==nl["ResName"]), column]
            if len(dt)>0:
                cDATA.loc[(cDATA["PDBName"]==nl["PDBName"])&(cDATA["ResName"]==nl["ResName"]), column] = dt.iloc[0]

        if column == "charge" and (cDATA[column].astype(float)==0.).any():
            print(f"There are still deficiency in \"{column}\" data.")
            print(cDATA.loc[cDATA[column].astype(float)==0., "ResName"].unique())
            if log: print(cDATA[cDATA[column].astype(float)==0.])
            ok = 0
        elif (not column == "charge") and (cDATA[column].astype(str)=="").any():
            print(f"There are still deficiency in \"{column}\" data.")
            print(cDATA.loc[cDATA[column].astype(str)=="", "ResName"].unique())
            if log: print(cDATA[cDATA[column].astype(str)==""])
            ok = 0                       
        else:
            print(f"All {column} data complemented.")
            ok = 1
            
        self.mol_trail.append(self.molDATA)
        self.molDATA = cDATA

        return ok, cDATA
    

    def Set_Parameters(self, data_file_paths, amber=False, log=False):
        if amber:
            cols = ["AmberType","charge","freeze","layer"]
        else:
            cols = ["UFFType","charge","freeze","layer"]
        for col in cols:
            print(f"<<<<<<<<<<Setting {col}>>>>>>>>>>>>>>")
            for dfp in data_file_paths:
                print(f"... using the data from {dfp}")
                try:
                    ok, cDATA = self.DataComplement(col, data_file_path=dfp, log=log)
                except:
                    print(f"[An error has been found]: {col}, {dfp}")
                
        return ok, cDATA
    
    
    def Reset_PDBname(self, resName):
        if resName in self.P.AMINO_DNA:
            print("Amino residue cannot be reset.")
            return self.molDATA
        
        res = self.molDATA.loc[self.molDATA["ResName"]==resName].copy()
        mol = self.molDATA.loc[~(self.molDATA["ResName"]==resName)].copy()
        
        newName = []
        for at in res.loc[:, "AtomType"].values:
            itr = 1
            while True:
                if not f"{at}{itr}" in newName:
                    newName.append(f"{at}{itr}")
                    break
                itr += 1
    
        conv = pd.DataFrame({
            "OldName": res["PDBName"],
            "NewName": newName,
        })
        res["PDBName"] = newName
        mol = pd.concat([mol, res]) 
        mol = mol.sort_values(["ChainID","ResID"])
        mol = mol.reset_index(drop=True)
        mol["id"] = [i for i in range(len(mol))] ; mol["AtomID"] = mol["id"] + 1

        self.mol_trail.append(self.molDATA)
        self.molDATA = mol
        
        return conv


    def Reset_PDBname_GMX(self, PDB2GMX=False):
        cDATA = self.molDATA.copy()
        if PDB2GMX:
            conv_dict = self.P.PDB2GMX
        else:
            conv_dict = self.P.GMX2PDB

        for rn, pn in self.GetSequence(mode=["ResName", "PDBName"]):
            if rn in self.P.AMINO_DNA:
                new_pn = conv_dict[rn][pn]
                if not pn == new_pn:
                    print(f"Reset_PDBname_GMX | ResName: {rn}  {pn}->{new_pn}")
                    cDATA.loc[(cDATA["ResName"]==rn)&(cDATA["PDBName"]==pn),"PDBName"] = new_pn
            
        self.mol_trail.append(cDATA)
        self.molDATA = cDATA
        
        return cDATA


    def GetSequence(self, mode=["ChainID", "ResID"]):
        seq = self.molDATA.loc[:,mode].values.tolist()
        seen = []
        return [x for x in seq if x not in seen and not seen.append(x)]

            
    def Reset_ResName(self, standard=False, skips=[]): #skips=[["A",12],["B",56]]
        cDATA = self.molDATA.copy()
        sk = [f"{c[0]}{c[1]}" for c in skips]

        for ci, ri in self.GetSequence(mode=["ChainID", "ResID"]):

            if f"{ci}{ri}" in sk:
                continue

            res = self.GetResName(ci, ri)
            new_res = ""
            atms = cDATA.loc[(cDATA["ChainID"]==ci)&(cDATA["ResID"]==ri),"PDBName"].values.tolist()

            if res in ["ASP","ASH"]:
                if standard:
                    new_res = "ASP"
                elif "HD1" in atms or "HD2" in atms:
                    new_res = "ASH"
                else:
                    new_res = "ASP"
            elif res in ["CYS","CYX"]:
                if standard:
                    new_res = "CYS"
                elif not "HG" in atms:
                    new_res = "CYX"
                else:
                    new_res = "CYS"
            elif res in ["GLU","GLH"]:
                if standard:
                    new_res = "GLU"
                elif "HE1" in atms or "HE2" in atms:
                    new_res = "GLH"
                else:
                    new_res = "GLU"
            elif res in ["LYS","LYN"]:
                if standard:
                     new_res = "LYS"
                elif not ("HZ1" in atms and "HZ2" in atms and "HZ3" in atms):
                    new_res = "LYN"
                else:
                    new_res = "LYS"
            elif res in ["HIS","HID","HIE","HIP"]:
                if standard:
                    new_res = "HIS"
                elif "HD1" in atms and "HD2" in atms and "HE1" in atms and "HE2" in atms:
                    new_res = "HIP"
                elif "HE1" in atms and "HE2" in atms:
                    new_res = "HIE"
                elif "HD1" in atms and "HD2" in atms:
                    new_res = "HID"
                else:
                    new_res = "HIS"
            else:
                new_res = res

            if not new_res == res:
                print(f"Reset_ResName | ChainID: {ci} ResID: {ri}  {res}->{new_res}")

            cDATA.loc[(cDATA["ChainID"]==ci)&(cDATA["ResID"]==ri),"ResName"] = new_res

        self.mol_trail.append(cDATA)
        self.molDATA = cDATA

        return cDATA


    def GetAtomPos_PDB(self, cID, rID, name):
        m = self.molDATA
        pos = m.loc[(m["ChainID"]==cID)&(m["ResID"]==int(rID))&(m["PDBName"]==name), ["x","y","z"]].iloc[0].values.astype(float)
        return pos


    def GetAtomPos_ID(self, aID, mode="AtomID"):
        m = self.molDATA
        pos = m.loc[m[mode]==aID, ["x","y","z"]].iloc[0].values.astype(float)
        return pos


    def GetFragmentPos(self, Condition={}):
        filt = np.ones(len(self.molDATA), dtype=bool)
        for i in Condition.keys():
            try:
                filt = filt&(self.molDATA[i]==Condition[i])
            except:
                print(f"{i} is not the molDATA column.")

        pos = self.molDATA.loc[filt, ["x","y","z"]].values.astype(float)
        return pos
    
    
    def GetQueryPos(self, Q):
        cDATA = self.molDATA.query(Q)
        pos = cDATA.loc[:, ["x","y","z"]].values.astype(float)
        return pos
    
    
    def GetStructurePos(self):
        return self.GetFragmentPos()


    def GetDistance_PDB(self, atom1, atom2):
        p1 = self.GetAtomPos_PDB(*atom1)
        p2 = self.GetAtomPos_PDB(*atom2)
        distance = np.linalg.norm(p1-p2)
        return distance


    def GetDistance_ID(self, id1, id2, mode="AtomID"):
        p1 = self.GetAtomPos_ID(id1, mode=mode)
        p2 = self.GetAtomPos_ID(id2, mode=mode)
        distance = np.linalg.norm(p1-p2)
        return distance


    def GetAngle_PDB(self, atomC, atom1, atom2, rad=False):
        pc = self.GetAtomPos_PDB(*atomC)
        p1 = self.GetAtomPos_PDB(*atom1)
        p2 = self.GetAtomPos_PDB(*atom2)
        vec1 = (p1-pc)/np.linalg.norm(p1-pc)
        vec2 = (p2-pc)/np.linalg.norm(p2-pc)
        angle = np.arccos(np.inner(vec1,vec2))
        if rad:
            return angle
        else:
            return np.rad2deg(angle)


    def GetAngle_ID(self, idc, id1, id2, mode="AtomID", rad=False):
        pc = self.GetAtomPos_ID(idc, mode=mode)
        p1 = self.GetAtomPos_ID(id1, mode=mode)
        p2 = self.GetAtomPos_ID(id2, mode=mode)
        vec1 = (p1-pc)/np.linalg.norm(p1-pc)
        vec2 = (p2-pc)/np.linalg.norm(p2-pc)
        angle = np.arccos(np.inner(vec1,vec2))
        if rad:
            return angle
        else:
            return np.rad2deg(angle)
        
        
    def GetDihedral_PDB(self, atom1, atom2, atom3, atom4, rad=False):
        p1 = self.GetAtomPos_PDB(*atom1)
        p2 = self.GetAtomPos_PDB(*atom2)
        p3 = self.GetAtomPos_PDB(*atom3)
        p4 = self.GetAtomPos_PDB(*atom4)
        
        vec1 = np.cross(p1-p2, p3-p2)
        vec1 /= np.linalg.norm(vec1)
        vec2 = np.cross(p2-p3, p4-p3)
        vec2 /= np.linalg.norm(vec2)
        angle = np.arccos(np.inner(vec1,vec2))
        
        if angle > np.pi:
            angle = 2*np.pi - angle
            
        if rad:
            return angle
        else:
            return np.rad2deg(angle)
        
        
    def GetDihedral_ID(self, id1, id2, id3, id4, mode="AtomID", rad=False):
        p1 = self.GetAtomPos_ID(id1, mode=mode)
        p2 = self.GetAtomPos_ID(id2, mode=mode)
        p3 = self.GetAtomPos_ID(id3, mode=mode)
        p4 = self.GetAtomPos_ID(id4, mode=mode)
        
        vec1 = np.cross(p1-p2, p3-p2)
        vec1 /= np.linalg.norm(vec1)
        vec2 = np.cross(p2-p3, p4-p3)
        vec2 /= np.linalg.norm(vec2)
        angle = np.arccos(np.inner(vec1,vec2))
        
        if angle > np.pi:
            angle = 2*np.pi - angle
            
        if rad:
            return angle
        else:
            return np.rad2deg(angle)


    def GetResName(self, chainID, resID):
        return self.molDATA.loc[(self.molDATA["ChainID"]==chainID)&(self.molDATA["ResID"]==resID), "ResName"].iloc[0]


    def GetAtomID(self, chainID, resID, pdb_name, AtomID=True):
        mol = self.molDATA
        if AtomID:
            return mol.loc[(mol["ChainID"]==chainID)&(mol["ResID"]==resID)&(mol["PDBName"]==pdb_name), "AtomID"].iloc[0]
        else:
            return mol.loc[(mol["ChainID"]==chainID)&(mol["ResID"]==resID)&(mol["PDBName"]==pdb_name), "id"].iloc[0]


    def DataInsert(self, Condition, column, value):
        cDATA = self.molDATA.copy()
        filt = np.ones(len(cDATA), dtype=bool)
        for i in Condition.keys():
            try:
                filt = filt&(cDATA[i]==Condition[i])
            except:
                print(f"{i} is not the molDATA column.")
                
        cDATA.loc[filt, column] = value
        
        self.mol_trail.append(self.molDATA)
        self.molDATA = cDATA

        return cDATA
    
    
    def ChargeSum(self, G_format=False):
        s_all = self.molDATA.loc[:,"charge"].sum()
        s_h = self.molDATA.loc[self.molDATA["layer"]=="H","charge"].sum()
        s_m = self.molDATA.loc[self.molDATA["layer"]=="M","charge"].sum()
        s_l = self.molDATA.loc[self.molDATA["layer"]=="L","charge"].sum()

        if G_format:
            if len(self.molDATA["layer"].unique())==3:
                return np.round([s_all, 1, s_all-s_l, 1, s_all-s_l, 1, s_all-s_l-s_m, 1, s_all-s_l-s_m, 1, s_all-s_l-s_m, 1]).astype(int)
            elif len(self.molDATA["layer"].unique())==2:
                return np.round([s_all, 1, s_all-s_l, 1, s_all-s_l, 1]).astype(int)
            elif len(self.molDATA["layer"].unique())==1:
                return np.round([s_all, 1]).astype(int)
            else:
                print("Layer assign has an error.")
                return 0
        else:
            return {"ALL":s_all, "High":s_h, "Med":s_m, "Low":s_l}
    
    
    def AtomFreeze_byDistance(self, exclude_res=[], area=8.0):
        cDATA = self.molDATA.copy()
        
        filt = np.ones(len(cDATA), dtype=bool)
        for exc in exclude_res:
            filt = filt & (~((self.molDATA["ChainID"]==exc[0])&(self.molDATA["ResID"]==exc[1])))
        cDATA.loc[filt, "freeze"] = -1

        for hpos in self.molDATA.loc[self.molDATA["layer"]=="H",["x","y","z"]].values:
            filt2 = filt & (np.linalg.norm(self.molDATA.loc[:,["x","y","z"]].values - hpos, axis=1) < area)
            cDATA.loc[filt2, "freeze"] = 0
  
        self.mol_trail.append(self.molDATA)
        self.molDATA = cDATA
        
        return cDATA 
    
    
    def AtomFreeze_byResidue(self, resNums_chainIDs=[]):
        cDATA = self.molDATA.copy()
        cDATA.loc[:, "freeze"] = -1

        for r, c in resNums_chainIDs:
            cDATA.loc[(cDATA["ResID"]==r)&(cDATA["ChainID"]==c), "freeze"] = 0
  
        self.mol_trail.append(self.molDATA)
        self.molDATA = cDATA
        
        return cDATA
    
    
    def Res2Ala(self, chainID, resID):
        mol = self.molDATA.copy()
        res = mol.loc[(mol["ResID"]==resID)&(mol["ChainID"]==chainID)].copy()
        if len(res)==0:
            print(f"It does not exist| ChainID:{chainID} ResNum:{resID}")
            return self.molDATA
        mol = mol.loc[~((mol["ResID"]==resID)&(mol["ChainID"]==chainID))].copy()
        resName = res.iloc[0]["ResName"]
        res.loc[:, "ResName"] = "ALA" ; res.loc[:, "charge"] = 0 ; res.loc[:, ["AmberType", "UFFType"]] = ""
        
        if resName=="ALA":
            cBpos = res.loc[res["PDBName"]=="CB", ["x","y","z"]].astype(float).iloc[0].values
            for h in ["HB1", "HB2", "HB3"]:
                hpos = res.loc[res["PDBName"]==h, ["x","y","z"]].astype(float).iloc[0].values
                newpos = (hpos-cBpos)/np.linalg.norm(hpos-cBpos) + cBpos
                res.loc[res["PDBName"]==h, ["x","y","z"]] = newpos.tolist()
            
        elif resName=="GLY":
            Hs = res.loc[(res["AtomType"]=="H")&(~(res["PDBName"]=="H"))]
            npos = res.loc[res["PDBName"]=="N", ["x","y","z"]].astype(float).iloc[0].values
            cpos = res.loc[res["PDBName"]=="C", ["x","y","z"]].astype(float).iloc[0].values
            cApos = res.loc[res["PDBName"]=="CA", ["x","y","z"]].astype(float).iloc[0].values
            vec = np.cross(npos-cApos, cpos-cApos)

            for i in range(len(Hs)):
                H = Hs.iloc[i]
                hpos = H.loc[["x","y","z"]].astype(float).values ; hName = H["PDBName"]
                if np.inner(hpos-cApos, vec) > 0:
                    vecGP = (hpos-cApos)/np.linalg.norm(hpos-cApos)
                    cBpos = cApos + 1.5*vecGP
                    Hpos = []
                    #make H pos (sp3)
                    r = np.array([vecGP[1],-1*vecGP[0],0],dtype=float) # vecGP and r are vertical
                    r = r*np.sin(70*np.pi/180)/np.linalg.norm(r) + vecGP*(1.51+np.cos(70*np.pi/180))
                    theta = np.random.rand()*np.pi # random 0 to pi rad
                    for i in range(3):
                        th = theta + 2*np.pi*i/3
                        rh = r*np.cos(th) + vecGP*np.dot(r, vecGP)*(1-np.cos(th)) + np.cross(vecGP, r)*np.sin(th)
                        Hpos.append(cApos+rh)

                    res.loc[res["PDBName"]==hName, ["AtomType", "PDBName", "x","y","z"]] = ["C", "CB"]+cBpos.tolist()
                    for p, i in zip(Hpos, [1,2,3]):
                        rec = pd.DataFrame([[0, 0, "H", "HC", "H_", 0, f"HB{i}", "ALA", resID, chainID, 0]+p.tolist()+["L",""]], 
                                        columns=res.columns)
                        res = pd.concat([res, rec]).reset_index(drop=True)
            res.loc[(res["PDBName"]=="HA2")|(res["PDBName"]=="HA3"),"PDBName"] = "HA"

        else:
            res = res.query(f"not PDBName in {self.P.delAtom[resName]}")

            for a in self.P.HAtom[resName]:
                apos = res.loc[res["PDBName"]==a, ["x","y","z"]].iloc[0].values

                if a=="CD": 
                    bName = "N" ; btype = "H"
                else: 
                    bName = "CB" ; btype = "HB1"

                bpos = res.loc[res["PDBName"]==bName, ["x","y","z"]].iloc[0].values
                hpos = bpos + (apos-bpos)/np.linalg.norm(apos-bpos)
                res.loc[res["PDBName"]==a, ["AtomType", "PDBName", "x","y","z"]] = ["H", btype]+hpos.tolist()

            res.loc[(res["PDBName"]=="HB")|(res["PDBName"]=="HB1")|(res["PDBName"]=="HB2")|(res["PDBName"]=="HB3"),"PDBName"]                                                                                                            = ["HB1","HB2","HB3"]

        mol = pd.concat([mol, res]) 
        mol = mol.sort_values(["ChainID","ResID"])
        mol = mol.reset_index(drop=True)
        mol["id"] = [i for i in range(len(mol))] ; mol["AtomID"] = mol["id"] + 1

        self.mol_trail.append(self.molDATA)
        self.molDATA = mol
        
        return mol
    
    
    def Ala2Res(self, chainID, resID, ResNAME, error_ok=False):
        res = self.molDATA.loc[(self.molDATA["ResID"]==resID)&(self.molDATA["ChainID"]==chainID)].copy()
        mol = self.molDATA.loc[~((self.molDATA["ResID"]==resID)&(self.molDATA["ChainID"]==chainID))].copy()

        if not res.iloc[0]["ResName"]=="ALA":
            print("This res is not ALA.")
            return 0, self.molDATA
        elif not ResNAME in self.P.AMINO_LIST:
            print("ResNAME is not amino acid.")
            return 0, self.molDATA
        else:
            res.loc[:,"ResName"] = ResNAME ; res.loc[:, "charge"] = 0 ; res.loc[:, ["AmberType", "UFFType"]] = ""
            Hs = res.query("PDBName in ['HB1', 'HB2', 'HB3']") 
            dist = []
            for p in Hs.loc[:,["x","y","z"]].astype(float).values:
                dist.append(np.linalg.norm(mol.loc[:,["x","y","z"]]-p, axis=1).min())
            dist = np.array(dist)
            name = Hs.loc[:,"PDBName"].iloc[dist.argmax()]
            res.loc[res["PDBName"]==name, "PDBName"] = "HB0"
            res.loc[res["PDBName"]=="HB1", "PDBName"] = name
            res.loc[res["PDBName"]=="HB0", "PDBName"] = "HB1"
        
        def addAtom(res, startName, pdbName, HNames=[], mode="SP3", connect=1, ignoreE=False):
            HpdbNames = ["linkH"]*connect + HNames
            if mode=="SP3":
                deg = 70 ; turn = 2*np.pi/3 
            elif mode=="SP2":
                deg = 60 ; turn = np.pi 
            elif mode=="pent":
                deg = 54 ; turn = np.pi 
            elif mode=="hex":
                deg = 60 ; turn = np.pi 
            resi = res.copy()
            H2Cpos = resi.loc[resi["PDBName"]==startName, ["x","y","z"]].astype(float).iloc[0].values
            pos = resi.loc[(~(resi["AtomType"]=="H"))&(np.linalg.norm(resi.loc[:,["x","y","z"]]-H2Cpos, axis=1) < 1.2),
                                                   ["x","y","z"]].astype(float).iloc[0].values
            Cpos = pos + (H2Cpos-pos)*1.51/np.linalg.norm(H2Cpos-pos)
            vecGP = (Cpos-pos)/np.linalg.norm(Cpos-pos)

            for i in range(100):
                Hpos = []
                r = np.array([vecGP[1],-1*vecGP[0],0],dtype=float) # vecGP and r are vertical
                r = r*np.sin(deg*np.pi/180)/np.linalg.norm(r) + vecGP*(1.51+np.cos(deg*np.pi/180))
                theta = np.random.rand()*np.pi # random 0 to pi rad
                for i in range(len(HpdbNames)):
                    th = theta + turn*i
                    rh = r*np.cos(th) + vecGP*np.dot(r, vecGP)*(1-np.cos(th)) + np.cross(vecGP, r)*np.sin(th)
                    Hpos.append(pos+rh)

                dist = []
                for p in Hpos:
                    dist.append(np.linalg.norm(mol.loc[:,["x","y","z"]]-p, axis=1).min())
                dist = np.array(dist)

                if ignoreE or len(dist)==0 or dist.min() > 1.2:
                    resi.loc[np.linalg.norm(resi.loc[:,["x","y","z"]]-H2Cpos, axis=1) < 0.1, 
                         ["AmberType","UFFType","AtomType", "PDBName", "x","y","z"]] = ["","",pdbName[0], pdbName] + Cpos.tolist()

                    while "linkH" in HpdbNames:
                        #print(Hpos) ; print(dist) ; print(HpdbNames)
                        index = dist.argmax() ; p = Hpos.pop(index) ; dist = np.delete(dist, index) ; HpdbNames.remove("linkH")
                        d = pd.DataFrame([[0,0,"H","","",0,"linkH",ResNAME, resID, chainID, 0]+p.tolist()+["L",""]], columns=resi.columns)
                        resi = pd.concat([resi, d]).reset_index(drop=True)

                    for hn, hp in zip(HpdbNames, Hpos):
                        d = pd.DataFrame([[0,0,"H","","",0, hn, ResNAME, resID, chainID, 0]+hp.tolist()+["L",""]], columns=resi.columns)
                        resi = pd.concat([resi, d]).reset_index(drop=True)
                    return 1, resi

            return 0, resi

        
        def addAtom_Course(res, course, ignoreE=False):
            resi = res.copy()
            for c in course:
                #print(c) ; print(resi)
                ok, resi = addAtom(resi, c[0], c[1], c[2], c[3], c[4], ignoreE)
                if not ok:
                    print(f"Failed conversion| {c}")
                    return 0, res

            return 1, resi

        
        def addAtom_single(res, aType, pos):
            d = pd.DataFrame([[0, 0, aType[0], "", "", 0, aType, ResNAME, resID, chainID, 0]+pos.tolist()+["L",""]], columns=res.columns)
            return pd.concat([res, d]).reset_index(drop=True)
        
        
        def rotate_adjust(res):
            resco = res.copy()
            resBEST = res
            dl = []
            for rp in res.loc[:, ["x","y","z"]].values:
                dl.append(np.linalg.norm(mol.loc[:,["x","y","z"]]-rp, axis=1).min())
            distBEST = np.array(dl).min()
            
            for i in range(50):
                #rotate around vecAB
                Apos = resco.loc[resco["PDBName"]=="CA", ["x","y","z"]].iloc[0].values
                Bpos = resco.loc[resco["PDBName"]=="CB", ["x","y","z"]].iloc[0].values
                vecAB = (Bpos-Apos)/np.linalg.norm(Bpos-Apos) ; theAB = np.random.rand()*np.pi
                for i in range(len(resco)):
                    atm = resco.iloc[i]
                    if not atm["PDBName"] in ["C","N","O","H", "CA","CB","HA","HA1","HA2"]:
                        pr = atm.loc[["x","y","z"]].values.astype(np.float64) - Apos
                        r = pr*np.cos(theAB) + vecAB*np.dot(pr, vecAB)*(1-np.cos(theAB)) + np.cross(vecAB, pr)*np.sin(theAB)
                        resco.loc[resco["PDBName"]==atm["PDBName"],["x","y","z"]] = r + Apos
                #rotate around vecBG
                Bpos = resco.loc[resco["PDBName"]=="CB", ["x","y","z"]].iloc[0].values
                Gpos = resco.loc[resco["PDBName"]=="CG", ["x","y","z"]].iloc[0].values
                vecBG = (Gpos-Bpos)/np.linalg.norm(Gpos-Bpos) ; theBG = np.random.rand()*np.pi
                for i in range(len(resco)):
                    atm = resco.iloc[i]
                    if not atm["PDBName"] in ["C","N","O","H","CA","CB","HA","HA1","HA2", "CG", "HB", "HB1", "HB2", "HB3"]:
                        pr = atm.loc[["x","y","z"]].values.astype(np.float64) - Bpos
                        r = pr*np.cos(theBG) + vecBG*np.dot(pr, vecBG)*(1-np.cos(theBG)) + np.cross(vecBG, pr)*np.sin(theBG)
                        resco.loc[resco["PDBName"]==atm["PDBName"],["x","y","z"]] = r + Bpos
                        
                dl = []
                for rp in resco.loc[:, ["x","y","z"]].values:
                    dl.append(np.linalg.norm(mol.loc[:,["x","y","z"]]-rp, axis=1).min())
                if np.array(dl).min() > distBEST:
                    resBEST = resco.copy() ; distBEST = np.array(dl).min()
            
            return resBEST
        
        
        ok = 0
        if ResNAME == "ALA":
            ok = 1
        elif ResNAME == "ARG":
            crs = [["HB1", "CG", ["HG2","HG3"], "SP3", 1],
                   ["linkH", "CD", ["HD2","HD3"], "SP3", 1],
                   ["linkH", "NE", ["HE"], "SP2", 1],
                  ["linkH", "CZ", [], "SP2", 2],
                  ["linkH", "NH1", ["HH11", "HH12"], "SP2", 0],
                  ["linkH", "NH2", ["HH21", "HH22"], "SP2", 0]]

            ok, res = addAtom_Course(res, crs, ignoreE=error_ok)
            if ok:
                res = rotate_adjust(res)

        elif ResNAME == "ASN" or ResNAME == "ASP" or ResNAME == "ASH":
            crs = [["HB1", "CG", [], "SP2", 2],
                   ["linkH", "OD1", [], "SP3", 0]]
            if ResNAME == "ASN":
                crs.append(["linkH", "ND2", ["HD21","HD22"], "SP2", 0])
            elif ResNAME == "ASP":
                crs.append(["linkH", "OD2", [], "SP3", 0])
            else:
                crs.append(["linkH", "OD2", ["HD2"], "SP3", 0])

            ok, res = addAtom_Course(res, crs, ignoreE=error_ok)

        elif ResNAME == "CYS":
            ok, res = addAtom(res, startName="HB1", pdbName="SG", HNames=["HG"], mode="SP3", connect=0, ignoreE=error_ok)

        elif ResNAME == "GLN" or ResNAME == "GLU" or ResNAME == "GLH":
            crs = [["HB1", "CG", ["HG2","HG3"], "SP3", 1],
                   ["linkH", "CD", [], "SP2", 2],
                   ["linkH", "OE1", [], "SP3", 0]]
            if ResNAME == "GLN":
                crs.append(["linkH", "NE2", ["HE21","HE22"], "SP2", 0])
            elif ResNAME == "GLU":
                crs.append(["linkH", "OE2", [], "SP3", 0])
            elif ResNAME == "GLH":
                crs.append(["linkH", "OE2", ["HE2"], "SP3", 0])

            ok, res = addAtom_Course(res, crs, ignoreE=error_ok)
            if ok:
                res = rotate_adjust(res)

        elif ResNAME == "GLY":
            res = res.query("not PDBName in ['HB1', 'HB2', 'HB3']")  
            cBpos = res.loc[res["PDBName"]=="CB", ["x","y","z"]].iloc[0].values
            cApos = res.loc[res["PDBName"]=="CA", ["x","y","z"]].iloc[0].values
            hpos = cApos + (cBpos-cApos)/np.linalg.norm(cBpos-cApos)
            res.loc[res["PDBName"]=="CB", ["AtomType", "PDBName", "x","y","z"]] = ["H", "HA3"]+hpos.tolist()
            res.loc[res["PDBName"]=="HA", "PDBName"] = "HA2"
            ok = 1

        elif ResNAME == "HIS" or ResNAME == "HID" or ResNAME == "HIE" or ResNAME == "HIP":
            ok, res = addAtom(res, startName="HB1", pdbName="CG", HNames=["preND1","preCD2"], mode="pent", connect=0, ignoreE=error_ok)
            if not ok:
                print(f"Conversion {resID}ALA->{ResNAME} failed. >>>CG")
                return 0, self.molDATA 
            ok, res = addAtom(res, startName="preND1", pdbName="ND1", HNames=[], mode="SP2", connect=0, ignoreE=error_ok)
            if not ok:
                print(f"Conversion {resID}ALA->{ResNAME} failed. >>>ND1")
                return 0, self.molDATA
            ok, res = addAtom(res, startName="preCD2", pdbName="CD2", HNames=[], mode="SP2", connect=0, ignoreE=error_ok)

            Bpos = res.loc[res["PDBName"]=="CB", ["x","y","z"]].iloc[0].values
            Gpos = res.loc[res["PDBName"]=="CG", ["x","y","z"]].iloc[0].values
            vecBG = (Gpos-Bpos)/np.linalg.norm(Gpos-Bpos)
            ND1pos = res.loc[res["PDBName"]=="ND1", ["x","y","z"]].iloc[0].values
            CD2pos = res.loc[res["PDBName"]=="CD2", ["x","y","z"]].iloc[0].values
            vecGD1 = (ND1pos-Gpos)/np.linalg.norm(ND1pos-Gpos)
            vecGD2 = (CD2pos-Gpos)/np.linalg.norm(CD2pos-Gpos)

            CE1pos = Gpos + 1.51*(np.cos(54*np.pi/180)+np.sin(108*np.pi/180))*vecBG + 0.45*(vecGD1-vecGD2)
            NE2pos = Gpos + 1.51*(np.cos(54*np.pi/180)+np.sin(108*np.pi/180))*vecBG + 0.45*(vecGD2-vecGD1)
            res = addAtom_single(res, "CE1", CE1pos)
            res = addAtom_single(res, "NE2", NE2pos)
            
            if ResNAME == "HIS" or ResNAME == "HID" or ResNAME == "HIP":
                vec = ND1pos-(CD2pos+NE2pos)*0.5 ; vec/= np.linalg.norm(vec)
                HD1pos = ND1pos + vec
                res = addAtom_single(res, "HD1", HD1pos)
            
            vec = CD2pos-(ND1pos+CE1pos)*0.5 ; vec/= np.linalg.norm(vec)
            HD2pos = CD2pos + vec
            res = addAtom_single(res, "HD2", HD2pos)
            
            vec = CE1pos-(CD2pos+Gpos)*0.5 ; vec/= np.linalg.norm(vec)
            HE1pos = CE1pos + vec
            res = addAtom_single(res, "HE1", HE1pos)

            if ResNAME == "HIE" or ResNAME == "HIP":
                vec = NE2pos-(ND1pos+Gpos)*0.5 ; vec/= np.linalg.norm(vec)
                HE2pos = NE2pos + vec
                res = addAtom_single(res, "HE2", HE2pos)
            
            res = rotate_adjust(res)

        elif ResNAME == "ILE":
            crs = [["HB1", "CG1", ["HG12","HG13"], "SP3", 1],
                   ["linkH", "CD1", ["HD11", "HD12","HD13"], "SP3", 0]]
            ok, res = addAtom_Course(res, crs, ignoreE=error_ok)
            if ok:
                Apos = res.loc[res["PDBName"]=="CA", ["x","y","z"]].iloc[0].values
                Bpos = res.loc[res["PDBName"]=="CB", ["x","y","z"]].iloc[0].values
                Gpos = res.loc[res["PDBName"]=="CG1", ["x","y","z"]].iloc[0].values
                vec = np.cross(Apos-Bpos, Gpos-Bpos)
                HB2pos = res.loc[res["PDBName"]=="HB2", ["x","y","z"]].iloc[0].values
                if np.inner(HB2pos-Bpos, vec) > 0:
                    ok, res = addAtom(res, startName="HB2", pdbName="CG2", HNames=["HG21","HG22","HG23"], mode="SP3", connect=0, ignoreE=error_ok)
                else:    
                    ok, res = addAtom(res, startName="HB3", pdbName="CG2", HNames=["HG21","HG22","HG23"], mode="SP3", connect=0, ignoreE=error_ok)
                res.loc[(res["PDBName"]=="HB2")|(res["PDBName"]=="HB3"), "PDBName"] = "HB"

        elif ResNAME == "LEU":
            crs = [["HB1", "CG", ["HG"], "SP3", 2],
                   ["linkH", "CD1", ["HD11", "HD12","HD13"], "SP3", 0],
                  ["linkH", "CD2", ["HD21", "HD22","HD23"], "SP3", 0]]

            ok, res = addAtom_Course(res, crs, ignoreE=error_ok)

        elif ResNAME == "LYS" or ResNAME == "LYN":
            crs = [["HB1", "CG", ["HG2","HG3"], "SP3", 1],
                   ["linkH", "CD", ["HD2","HD3"], "SP3", 1],
                   ["linkH", "CE", ["HE2","HE3"], "SP3", 1]]
            if ResNAME == "LYS":
                crs.append(["linkH", "NZ", ["HZ1","HZ2","HZ3"], "SP3", 0])
            elif ResNAME == "LYN":
                crs.append(["linkH", "NZ", ["HZ1","HZ2"], "SP3", 0])

            ok, res = addAtom_Course(res, crs, ignoreE=error_ok)

        elif ResNAME == "MET":
            crs = [["HB1", "CG", ["HG2","HG3"], "SP3", 1],
                   ["linkH", "SD", [], "SP3", 1],
                   ["linkH", "CE", ["HE1", "HE2","HE3"], "SP3", 0]]

            ok, res = addAtom_Course(res, crs, ignoreE=error_ok)

        elif ResNAME == "PHE" or ResNAME == "TYR":
            ok, res = addAtom(res, startName="HB1", pdbName="CG", HNames=["preCD1","preCD2"], mode="hex", connect=0, ignoreE=error_ok)
            if not ok:
                print(f"Conversion {resID}ALA->{ResNAME} failed. >>>CG")
                return 0, self.molDATA
            ok, res = addAtom(res, startName="preCD1", pdbName="CD1", HNames=[], mode="SP2", connect=0, ignoreE=error_ok)
            if not ok:
                print(f"Conversion {resID}ALA->{ResNAME} failed. >>>CD1")
                return 0, self.molDATA
            ok, res = addAtom(res, startName="preCD2", pdbName="CD2", HNames=[], mode="SP2", connect=0, ignoreE=error_ok)

            Bpos = res.loc[res["PDBName"]=="CB", ["x","y","z"]].iloc[0].values
            Gpos = res.loc[res["PDBName"]=="CG", ["x","y","z"]].iloc[0].values
            vecBG = (Gpos-Bpos)/np.linalg.norm(Gpos-Bpos)

            CD1pos = res.loc[res["PDBName"]=="CD1", ["x","y","z"]].iloc[0].values
            CE1pos = CD1pos + vecBG*1.51
            res = addAtom_single(res, "CE1", CE1pos)
            
            CD2pos = res.loc[res["PDBName"]=="CD2", ["x","y","z"]].iloc[0].values
            CE2pos = CD2pos + vecBG*1.51
            res = addAtom_single(res, "CE2", CE2pos)
            
            CZpos = Gpos + vecBG*3.02
            res = addAtom_single(res, "CZ", CZpos)

            vecGD1 = (CD1pos-Gpos)/np.linalg.norm(CD1pos-Gpos)
            vecGD2 = (CD2pos-Gpos)/np.linalg.norm(CD2pos-Gpos)
            HD1pos = CD1pos-vecGD2
            res = addAtom_single(res, "HD1", HD1pos)
            HD2pos = CD2pos-vecGD1
            res = addAtom_single(res, "HD2", HD2pos)
            HE1pos = CE1pos+vecGD1
            res = addAtom_single(res, "HE1", HE1pos)
            HE2pos = CE2pos+vecGD2
            res = addAtom_single(res, "HE2", HE2pos)
            HZpos = CZpos+vecBG
            res = addAtom_single(res, "HZ", HZpos)

            if ResNAME == "TYR":
                ok, res = addAtom(res, startName="HZ", pdbName="OH", HNames=["HH"], mode="SP3", connect=0, ignoreE=error_ok)
                if not ok:
                    print(f"Conversion {resID}ALA->{ResNAME} failed. >>>OH")
                    return 0, self.molDATA
                
            res = rotate_adjust(res)

        elif ResNAME == "PRO":
            ok, res = addAtom(res, startName="H", pdbName="CD", HNames=["preHD1","preHD2","preHD3"], mode="SP3", connect=0, ignoreE=error_ok)
            if not ok:
                print(f"Conversion {resID}ALA->{ResNAME} failed. >>>CD")
                return 0, self.molDATA
            Dpos = res.loc[res["PDBName"]=="CD", ["x","y","z"]].iloc[0].values
            Hs = res.query("PDBName in ['HB1', 'HB2', 'HB3']") 
            name = Hs.loc[:,"PDBName"].iloc[np.linalg.norm(Hs.loc[:,["x","y","z"]]-Dpos, axis=1).argmin()]
            res.loc[res["PDBName"]==name, "PDBName"] = "HB0"
            res.loc[res["PDBName"]=="HB1", "PDBName"] = name
            res.loc[res["PDBName"]=="HB0", "PDBName"] = "HB1"

            ok, res = addAtom(res, startName="HB1", pdbName="CG", HNames=["preHG1","preHG2","preHG3"], mode="SP3", connect=0, ignoreE=error_ok)
            Gpos = res.loc[res["PDBName"]=="CG", ["x","y","z"]].iloc[0].values
            HDs = res.query("PDBName in ['preHD1', 'preHD2', 'preHD3']")
            HGs = res.query("PDBName in ['preHG1', 'preHG2', 'preHG3']")
            exD = HDs.loc[:,"PDBName"].iloc[np.linalg.norm(HDs.loc[:,["x","y","z"]]-Gpos, axis=1).argmin()]
            exG = HGs.loc[:,"PDBName"].iloc[np.linalg.norm(HGs.loc[:,["x","y","z"]]-Dpos, axis=1).argmin()]
            res = res[~((res["PDBName"]==exD)|(res["PDBName"]==exG))]
            res.loc[(res["PDBName"]=="preHD1")|(res["PDBName"]=="preHD2")|(res["PDBName"]=="preHD3"), "PDBName"] = ["HD2", "HD3"]
            res.loc[(res["PDBName"]=="preHG1")|(res["PDBName"]=="preHG2")|(res["PDBName"]=="preHG3"), "PDBName"] = ["HG2", "HG3"]

            d = np.linalg.norm(Gpos-Dpos) - 1.55
            if d > 0:
                dm = d/2.
                vecDG = (Gpos-Dpos)/np.linalg.norm(Gpos-Dpos)
                res.loc[(res["PDBName"]=="CD")|(res["PDBName"]=="HD2")|(res["PDBName"]=="HD3"), ["x","y","z"]] += dm*vecDG
                res.loc[(res["PDBName"]=="CG")|(res["PDBName"]=="HG2")|(res["PDBName"]=="HG3"), ["x","y","z"]] -= dm*vecDG
            
        elif ResNAME == "SER":
            ok, res = addAtom(res, startName="HB1", pdbName="OG", HNames=["HG"], mode="SP3", connect=0, ignoreE=error_ok)

        elif ResNAME == "THR":
            ok, res = addAtom(res, startName="HB1", pdbName="OG1", HNames=["HG1"], mode="SP3", connect=0, ignoreE=error_ok)
            if ok:
                Apos = res.loc[res["PDBName"]=="CA", ["x","y","z"]].iloc[0].values
                Bpos = res.loc[res["PDBName"]=="CB", ["x","y","z"]].iloc[0].values
                Gpos = res.loc[res["PDBName"]=="OG1", ["x","y","z"]].iloc[0].values
                vec = np.cross(Apos-Bpos, Gpos-Bpos)
                HB2pos = res.loc[res["PDBName"]=="HB2", ["x","y","z"]].iloc[0].values
                if np.inner(HB2pos-Bpos, vec) > 0:
                    ok, res = addAtom(res, startName="HB2", pdbName="CG2", HNames=["HG21","HG22","HG23"], mode="SP3", connect=0, ignoreE=error_ok)
                else:    
                    ok, res = addAtom(res, startName="HB3", pdbName="CG2", HNames=["HG21","HG22","HG23"], mode="SP3", connect=0, ignoreE=error_ok)
                res.loc[(res["PDBName"]=="HB2")|(res["PDBName"]=="HB3"), "PDBName"] = "HB"

        elif ResNAME == "TRP":
            ok, res = addAtom(res, startName="HB1", pdbName="CG", HNames=["preCD1","preCD2"], mode="pent", connect=0, ignoreE=error_ok)
            if not ok:
                print(f"Conversion {resID}ALA->{ResNAME} failed.")
                return 0, self.molDATA
            
            Gpos = res.loc[res["PDBName"]=="CG", ["x","y","z"]].iloc[0].values
            pCD1pos = res.loc[res["PDBName"]=="preCD1", ["x","y","z"]].iloc[0].values
            pCD2pos = res.loc[res["PDBName"]=="preCD2", ["x","y","z"]].iloc[0].values
            vecpGD1 = (pCD1pos-Gpos)/np.linalg.norm(pCD1pos-Gpos) ; vecpGD2 = (pCD2pos-Gpos)/np.linalg.norm(pCD2pos-Gpos)
            pCD1pos = Gpos + 1.51*vecpGD1 ; pCD2pos = Gpos + 1.51*vecpGD2
            
            Bpos = res.loc[res["PDBName"]=="CB", ["x","y","z"]].iloc[0].values
            vecBG = (Gpos-Bpos)/np.linalg.norm(Gpos-Bpos)
            NE1pos = Gpos + 1.51*(np.cos(54*np.pi/180)+np.sin(108*np.pi/180))*vecBG + 0.45*(vecpGD1-vecpGD2)
            CE2pos = Gpos + 1.51*(np.cos(54*np.pi/180)+np.sin(108*np.pi/180))*vecBG + 0.45*(vecpGD2-vecpGD1)
            
            dn = np.linalg.norm(mol.loc[:,["x","y","z"]]-NE1pos, axis=1).min()
            dc = np.linalg.norm(mol.loc[:,["x","y","z"]]-CE2pos, axis=1).min()
            if dn > dc:
                pcd1 = "CD2" ; pcd2 = "CD1"
                NE1pos, CE2pos = CE2pos, NE1pos
            else:
                pcd1 = "CD1" ; pcd2 = "CD2"
            ok, res = addAtom(res, startName="preCD1", pdbName=pcd1, HNames=[], mode="SP2", connect=0, ignoreE=error_ok)
            if not ok:
                print(f"Conversion {resID}ALA->{ResNAME} failed. >>>preCD1")
                return 0, self.molDATA
            ok, res = addAtom(res, startName="preCD2", pdbName=pcd2, HNames=[], mode="SP2", connect=0, ignoreE=error_ok)
            if not ok:
                print(f"Conversion {resID}ALA->{ResNAME} failed. >>>preCD2")
                return 0, self.molDATA
            res = addAtom_single(res, "CE2", CE2pos)
            res = addAtom_single(res, "NE1", NE1pos)
           
            CD1pos = res.loc[res["PDBName"]=="CD1", ["x","y","z"]].iloc[0].values
            CD2pos = res.loc[res["PDBName"]=="CD2", ["x","y","z"]].iloc[0].values
            
            vec = NE1pos-(CD2pos+Gpos)*0.5 ; vec/= np.linalg.norm(vec)
            HE1pos = NE1pos + vec
            res = addAtom_single(res, "HE1", HE1pos)
            
            vec = CD1pos-(CD2pos+CE2pos)*0.5 ; vec/= np.linalg.norm(vec)
            HD1pos = CD1pos + vec
            res = addAtom_single(res, "HD1", HD1pos)
            
            vec = CD2pos-(CD1pos+NE1pos)*0.5 ; vec/= np.linalg.norm(vec)
            CE3pos = CD2pos + 1.51*vec
            res = addAtom_single(res, "CE3", CE3pos)
            
            vec = CE2pos-(CD1pos+Gpos)*0.5 ; vec/= np.linalg.norm(vec)
            CZ2pos = CE2pos + 1.51*vec
            res = addAtom_single(res, "CZ2", CZ2pos)
            
            vec = CZ2pos-CE2pos ; vec/= np.linalg.norm(vec)
            CZ3pos = CE3pos + 1.51*vec
            res = addAtom_single(res, "CZ3", CZ3pos)
            
            vec = CE3pos-CD2pos ; vec/= np.linalg.norm(vec)
            CH2pos = CZ2pos + 1.51*vec
            res = addAtom_single(res, "CH2", CH2pos)
            
            vec = CE3pos-CZ2pos ; vec/= np.linalg.norm(vec)
            HE3pos = CE3pos + vec
            res = addAtom_single(res, "HE3", HE3pos)
            
            vec = CZ2pos-CE3pos ; vec/= np.linalg.norm(vec)
            HZ2pos = CZ2pos + vec
            res = addAtom_single(res, "HZ2", HZ2pos)
            
            vec = CZ3pos-CE2pos ; vec/= np.linalg.norm(vec)
            HZ3pos = CZ3pos + vec
            res = addAtom_single(res, "HZ3", HZ3pos)
            
            vec = CH2pos-CD2pos ; vec/= np.linalg.norm(vec)
            HH2pos = CH2pos + vec
            res = addAtom_single(res, "HH2", HH2pos)
            
            res = rotate_adjust(res)
                                  
        elif ResNAME == "VAL":
            ok, res = addAtom(res, startName="HB1", pdbName="CG1", HNames=["HG11", "HG12", "HG13"], mode="SP3", connect=0, ignoreE=error_ok)
            ok, res = addAtom(res, startName="HB2", pdbName="CG2", HNames=["HG21", "HG22", "HG23"], mode="SP3", connect=0, ignoreE=error_ok)
            res.loc[res["PDBName"]=="HB3", "PDBName"] = "HB"

            
        #Distance check
        pos = np.concatenate([mol.loc[:,["x","y","z"]].astype(float).values, 
                              res.loc[:,["x","y","z"]].astype(float).values])
        dm = distance.cdist(pos, pos, metric='euclidean')
        if ((dm > 0)&(dm < 0.5)).any() and not error_ok:
            print("Small interatomic distances encountered.")
            ok = 0
            
        if not ok:
            print(f"Conversion {chainID}{resID} ALA->{ResNAME} failed.")
            return 0, self.molDATA

        mol = pd.concat([mol, res]) 
        mol = mol.sort_values(["ChainID","ResID"])
        mol = mol.reset_index(drop=True)
        mol["id"] = [i for i in range(len(mol))] ; mol["AtomID"] = mol["id"] + 1

        self.mol_trail.append(self.molDATA)
        self.molDATA = mol
        
        return 1, mol
    
    
    def Residue_Change(self, chainID, resID, ResNAME, error_ok=False):

        if self.GetResName(chainID, resID) == ResNAME:
            ok, mol = 1, self.molDATA.copy()
        else:
            self.Res2Ala(chainID, resID)
            ok, mol = self.Ala2Res(chainID, resID, ResNAME, error_ok=error_ok)

        return ok, mol
    
    
    def Set_Mutation(self, mutation, error_ok=False): # e.g. mutation=[["A", 121, "SER"],["B", 122, "ARG"],]
        
        ok = 0
        for ci, ri, ChangeRes in mutation:
            change_res_mol = self.molDATA[(self.molDATA["ChainID"]==ci)&(self.molDATA["ResID"]==ri)]
            if len(change_res_mol) == 0:
                print(f"CAUTION: There is no such res: ChainID={ci} ResID={ri}")
                continue
            OriginalRes = change_res_mol.iloc[0]["ResName"]
            
            print(f"Chain:{ci}, {ri}{OriginalRes} -> {ri}{ChangeRes}")
            for t in range(20):
                ok, mol = self.Residue_Change(ci, ri, ChangeRes, error_ok=error_ok)
                if ok==1: 
                    break
                else:
                    print(f"[retake {t+1}]")
            if ok==0:
                raise ValueError(f"No more retake: ChainID={ci} ResID={ri}")
        
        return ok, mol
    
    
    def Res2HighLayer(self, ChainID, ResID):
        
        self.DataInsert({"ChainID":ChainID,"ResID":ResID}, "layer", "H")
        for pdbn in self.P.MAIN_CHAIN_ATOMS:
            self.DataInsert({"ChainID":ChainID,"ResID":ResID, "PDBName":pdbn}, "layer", "L")
    
        return self.Contents
    
    
    def makeConnection(self, add_info={}, add_info_extra=[], artificial_aa=[]): 
        #add_info = {"ResName1":[("PDBname1","PDBname2",1),("PDBname3","PDBname4",2), ... ], "ResName2":[], ...}
        #add_info_extra = [[("ChainID1","ResID1","PDBName1"),("ChainID2","ResID2","PDBName2"),1], ... ]
        self.Reset_ID()
        con = np.zeros((len(self.molDATA), len(self.molDATA)), dtype=float)

        for c in self.molDATA["ChainID"].unique():
            mol = self.molDATA.loc[self.molDATA["ChainID"]==c].copy()
            for n in mol["ResID"].unique():
                resN = mol.loc[mol["ResID"]==n,"ResName"].iloc[0]
                if resN in list(self.P.linkAtom.keys()):
                    for l in self.P.linkAtom[resN].keys():
                        lk = l.split("_")
                        ids = mol.loc[(mol["ResID"]==n)&((mol["PDBName"]==lk[0])|(mol["PDBName"]==lk[1])),"id"].astype(int).values
                        try:
                            con[ids[0], ids[1]] = con[ids[1], ids[0]] = self.P.linkAtom[resN][l]
                        except:
                            pass
                            #print(f"bond {l} is not found.")

                if resN in list(self.P.linkAtom.keys())+artificial_aa:
                    try:
                        ci = mol.loc[(mol["ResID"]==n)&(mol["PDBName"]=="C"),"id"].astype(int).iloc[0]
                        ni = mol.loc[(mol["ResID"]==n+1)&(mol["PDBName"]=="N"),"id"].astype(int).iloc[0]
                        con[ci, ni] = con[ni, ci] = 1.
                    except:
                        pass
                    
                else:
                    mol_r = mol.loc[mol["ResID"]==n]
                    for i in range(len(mol_r)):
                        idi = mol_r.iloc[i]["id"] ; ti = mol_r.iloc[i]["AtomType"]
                        pi = mol_r.loc[:,["x","y","z"]].astype(float).iloc[i].values
                        for j in range(i+1, len(mol_r)):
                            idj = mol_r.iloc[j]["id"] ; tj = mol_r.iloc[j]["AtomType"]
                            pj = mol_r.loc[:,["x","y","z"]].astype(float).iloc[j].values
                            atmPair = [ti, tj]
                            b = 0
                            
                            if "H" in atmPair:
                                if ("Si" in atmPair)or("P" in atmPair)or("S" in atmPair):
                                    if np.linalg.norm(pi-pj) < 1.5: 
                                        b = 1
                                elif atmPair.count("H")==2:
                                    pass
                                else:
                                    if np.linalg.norm(pi-pj) < 1.2: 
                                        b = 1
                                    
                            elif ("Si" in atmPair)or("P" in atmPair)or("S" in atmPair)or("Cl" in atmPair):
                                if np.linalg.norm(pi-pj) < 2.2: 
                                    b = 1
                            else:
                                if np.linalg.norm(pi-pj) < 1.6: 
                                    b = 1
                            
                            con[idi,idj] = con[idj,idi] = b
                            
                    if resN in add_info.keys():
                        for a1, a2, b in add_info[resN]:
                            id1 = mol_r.loc[mol_r["PDBName"]==a1,"id"].iloc[0]
                            id2 = mol_r.loc[mol_r["PDBName"]==a2,"id"].iloc[0]
                            con[id1, id2] = con[id2, id1] = b
        
        # disulfide bond
        mol = self.molDATA
        S_list = {}
        for res in self.GetSequence(mode=["ChainID", "ResID", "ResName"]):
            if res[2] == "CYS":
                try: 
                    hid = self.GetAtomID(res[0], res[1], "HG", AtomID=False)
                except:
                    sid = self.GetAtomID(res[0], res[1], "SG", AtomID=False)
                    spos = self.GetAtomPos_ID(sid, mode="id")
                    S_list[sid] = spos
                    
        while len(S_list) > 0:
            sid, spos = S_list.popitem()
            for sk in S_list.keys():
                spos2 = S_list[sk]
                if np.linalg.norm(spos - spos2) < 2.5:
                    S_list.pop(sk)
                    con[sid, sk] = con[sk, sid] = 1
                    break
            
        for ex in add_info_extra:
            id1 = self.GetAtomID(*ex[0], AtomID=False)
            id2 = self.GetAtomID(*ex[1], AtomID=False)
            con[id1,id2] = con[id2,id1] = ex[2]
        
        self.Contents["connection"] = con
        
        return con
    
    
    def makeDammy(self, uff=False): 
        cDATA = self.molDATA.copy()
        cDATA["info"] = ""
        
        if not len(self.Contents["connection"]) == len(cDATA):
            print("Please make information of connection by makeConnection()")
            return self.molDATA
        
        for aid in cDATA.loc[cDATA["layer"]=="H","id"].values:
            for cid in np.where(self.Contents["connection"][aid]>0)[0]:
                if cDATA.loc[cDATA["id"]==cid, "layer"].iloc[0] == "L":
                    cDATA.loc[cDATA["id"]==cid, "info"] = f"H-H{'_' if uff else 'C'} {aid+1}"
        
        self.mol_trail.append(self.molDATA)
        self.molDATA = cDATA
        
        return self.molDATA
    
    
    def makeCondition(self, C_dict):
        for i in C_dict.keys():
            if i in self.Contents.keys():
                self.Contents[i] = C_dict[i]  
            elif i == "Route":
                self.Route = C_dict[i]  
            elif i == "C_S":
                self.C_S = np.array(C_dict[i], dtype=int)
            elif i == "addInf": 
                self.addInf = C_dict[i]
            else:
                print(f"{i} is not in the Contents.")
        
        return self.Contents
    
    
    def SettingValidation(self, data_file_path=""):
        ok = 1
        
        if (self.molDATA.loc[:, "charge"]==0).any():
            ok = 0
            print("charge assignment is imperfect.")
            print(self.molDATA[self.molDATA.loc[:, "charge"]==0])
            
        if (self.molDATA.loc[:, "AtomType"]=="").any():
            ok = 0
            print("atomTYPE assignment is imperfect.")
            print(self.molDATA[self.molDATA.loc[:, "AtomType"]==""])
            
        if (self.molDATA.loc[:, "AmberType"]=="").all() and (self.molDATA.loc[:, "UFFType"]=="").any():
            ok = 0
            print("uffTYPE assignment is imperfect.")
            print(self.molDATA[self.molDATA.loc[:, "UFFType"]==""])
        elif (self.molDATA.loc[:, "UFFType"]=="").all() and (self.molDATA.loc[:, "AmberType"]=="").any():
            ok = 0
            print("amberTYPE assignment is imperfect.")
            print(self.molDATA[self.molDATA.loc[:, "AmberType"]==""])
        
        if not len(self.Contents["connection"])==len(self.molDATA):
            ok = 0
            print("Connectivity is not set.")
        
        if data_file_path:
            dtlist = []
            if data_file_path.lower().endswith('.csv'):
                dtlist = pd.read_csv(data_file_path, keep_default_na=False)
            elif data_file_path.lower().endswith('.xlsx'):
                dtlist = pd.read_excel(data_file_path, keep_default_na=False)
                
            if len(dtlist)>0:
                if (self.molDATA.loc[:, "AmberType"]=="").all():
                    typ = "UFFType"
                else:
                    typ = "AmberType"
                for i in range(len(self.molDATA)):
                    m = self.molDATA.iloc[i]
                    try:
                        ltyp = dtlist.loc[(dtlist["ResName"]==m["ResName"])&(dtlist["PDBName"]==m["PDBName"]), typ].iloc[0]
                        if not m[typ] == ltyp:
                            ok = 0
                            print(f"{typ} assignment is not correct.")
                            print(m)
                    except:
                        continue
                    
                        
        return ok

    
    def Check_Collision(self):
        pos = self.GetStructurePos()
        dmat = distance.cdist(pos, pos, metric='euclidean') + np.eye(len(pos))
        
        col_at = np.where(dmat<0.5)

        for i, j in zip(col_at[0], col_at[1]):
            if i > j:
                col_atoms = self.molDATA.query(f"AtomID in [{i+1},{j+1}]").loc[:, ["ChainID", "ResID", "PDBName"]].values.tolist()
                print(f"CAUTION: These atoms are too close: {col_atoms[0]} {col_atoms[1]}")
        
        return 1
        

    def move_MinimizedRMSD_byAminoAcid(self, ResNum, ChainID, template, tResNum, tChainID, ths=0.0001, mainChainOnly=False, log=False):

        mol_t = template[(template["ResID"]==tResNum)&(template["ChainID"]==tChainID)].copy()
        mol = self.molDATA[(self.molDATA["ResID"]==ResNum)&(self.molDATA["ChainID"]==ChainID)].copy()

        return self.move_MinimizedRMSD_byFragment(mol=mol, mol_t=mol_t, ths=ths, mainChainOnly=mainChainOnly, log=log)


    def move_MinimizedRMSD_byFragment(self, mol, mol_t, ths=0.0001, mainChainOnly=True, log=False):
        import warnings
        warnings.filterwarnings('ignore')
        
        mDATA = self.molDATA.copy()

        mol_A, mol_B = mol_t.copy(), mol.copy()
        if not (("H" in mol_A["AtomType"]) and ("H" in mol_B["AtomType"])):
            mol_A = mol_A[~(mol_A["AtomType"]=="H")]
            mol_B = mol_B[~(mol_B["AtomType"]=="H")]    
        if mainChainOnly:
            mol_A = mol_A.query("PDBName in ['CA', 'C', 'N', 'O']")
            mol_B = mol_B.query("PDBName in ['CA', 'C', 'N', 'O']")
        if log:
            print(mol_A) ; print(mol_B)
        mol_A = mol_A.sort_values(["ChainID", "ResID","PDBName"]).reset_index(drop=True) 
        mol_B = mol_B.sort_values(["ChainID", "ResID","PDBName"]).reset_index(drop=True)
        mol_A["id"] = [i for i in range(len(mol_A))] ; mol_A["AtomID"] = mol_A["id"] +1
        mol_B["id"] = [i for i in range(len(mol_B))] ; mol_B["AtomID"] = mol_B["id"] +1
        
        
        if not (mol_A.loc[:, "PDBName"].values==mol_B.loc[:, "PDBName"].values).all():
            print("Atom composition is not the same.")
            return 0, mDATA, -1, {}
        elif log:
            print(mol_A) ; print(mol_B) 

        if log: 
            print("RMSD canculation has started.")
            start = time.time()

        def RMSD(ma, mb, perAtom=False):
            a = ma.loc[:, ["x", "y", "z"]].values.astype(float)
            b = mb.loc[:, ["x", "y", "z"]].values.astype(float)
            if perAtom:
                return np.sqrt(((a - b)**2).sum(axis=1))
            else:
                return np.sqrt(((a - b)**2).sum()/len(ma))

        def move_H(ma, mb, aid):
            nonlocal mDATA
            mh = mb.copy()
            a = ma.loc[aid, ["x", "y", "z"]].values.astype(float)
            b = mb.loc[aid, ["x", "y", "z"]].values.astype(float)
            mh.loc[:, ["x", "y", "z"]] += a-b
            mDATA.loc[:, ["x", "y", "z"]] += a-b
            return mh

        def move_S(ma, mb, aid, Mid):
            nonlocal mDATA
            mh = move_H(ma, mb, Mid)
            M = ma.loc[Mid, ["x", "y", "z"]].values.astype(float)
            a = ma.loc[aid, ["x", "y", "z"]].values.astype(float)
            b = mh.loc[aid, ["x", "y", "z"]].values.astype(float)
            vecFrom = b-M ; vecTo = a-M
            vert = np.cross(vecFrom, vecTo) ; vert /= np.linalg.norm(vert)
            N = M - vert
            theta = np.arccos(np.dot(vecTo,vecFrom)/(np.linalg.norm(vecTo)*np.linalg.norm(vecFrom)))

            pr = mh.loc[:,["x","y","z"]].values.astype(np.float64) - N
            prD = mDATA.loc[:,["x","y","z"]].values.astype(np.float64) - N
            r = pr*np.cos(theta) + vert*np.dot(pr, vert)[:,np.newaxis]*(1-np.cos(theta)) + np.cross(vert, pr)*np.sin(theta)
            rD = prD*np.cos(theta) + vert*np.dot(prD, vert)[:,np.newaxis]*(1-np.cos(theta)) + np.cross(vert, prD)*np.sin(theta)
            if np.isnan(r).any() or np.isnan(rD).any():
                if log: print("S fail")
                return mb
            mh.loc[:,["x","y","z"]] = r + N
            mDATA.loc[:,["x","y","z"]] = rD + N

            return mh


        rmsd = [] ; rmsd_trail = []
        rmsd_min_prev = 10000000000 ; rmsd_min = RMSD(mol_A, mol_B) ; mol_B_min = mol_B.copy(); mDATA_min = mDATA.copy()
        itr = 0

        while True:
            itr += 1

            a_amsd = RMSD(mol_A, mol_B, perAtom=True)
            atmid = np.argmax(a_amsd)

            if np.random.rand()>0.5:
                mol_B = move_H(mol_A, mol_B, atmid)
            else:
                mvid = np.argmin(a_amsd)
                mol_B = move_S(mol_A, mol_B, atmid, mvid)

            rm = RMSD(mol_A, mol_B)
            rmsd.append(rm)
            if rmsd_min>rm:
                rmsd_min = rm ; mol_B_min = mol_B.copy(); mDATA_min = mDATA.copy()
                if log: print(f"The structure is renewed: rmsd_min={rmsd_min}")

            if itr%1000==0:            
                if log: print(f"Itr {itr}: rmsd_min_prev={rmsd_min_prev} rmsd_min={rmsd_min} -> elapsed time:{time.time() - start:.2f} sec")
                if rmsd_min_prev-rmsd_min<ths:
                    ok = 1
                    rmsd_trail += rmsd
                    mol_B_min["Distance"] = RMSD(mol_A, mol_B, perAtom=True)
                    otherDat = {"mol_B_min":mol_B_min, "RMSD_trail":np.array(rmsd_trail)}
                    if log:
                        print(mol_A) ; print(mol_B)
                    self.mol_trail.append(self.molDATA)
                    self.molDATA = mDATA_min
                    return ok, mDATA, rmsd_min, otherDat
                else:
                    if log: print("rmsd_max="+str(np.array(rmsd).max())+" rmsd_min="+str(np.array(rmsd).min()))
                    mol_B = mol_B_min.copy()
                    mDATA = mDATA_min.copy()
                    rmsd_min_prev = rmsd_min
                    rmsd_trail += rmsd
                    rmsd = []
            elif itr>100000:
                print("Calculation may not be stable.")
                ok = 2
                mol_B_min["Distance"] = RMSD(mol_A, mol_B, perAtom=True)
                otherDat = {"mol_B_min":mol_B_min, "RMSD_trail":np.array(rmsd_trail)}
                self.mol_trail.append(self.molDATA)
                self.molDATA = mDATA
                return ok, mDATA, rmsd_min, otherDat

     
    def Spacer_Designer(self, startC_id, endC_id, threshold=0.5, strDATA=[], log=False):
        if len(strDATA)==0:
            molDATA = self.molDATA
        else:
            molDATA = strDATA
        
        sPos = molDATA.loc[molDATA["AtomID"]==startC_id,["x","y","z"]].iloc[0].values
        ePos = molDATA.loc[molDATA["AtomID"]==endC_id,["x","y","z"]].iloc[0].values
        if log:
            print("---------start C Pos---------------")
            print(sPos)
            print("---------end C Pos---------------")
            print(ePos)

        def No_corr(position, ms, sps, sp):
            if (np.linalg.norm(mol.loc[~(mol["AtomType"]=="H"),["x","y","z"]]-position, axis=1).min() > ms):
                pass
            else:
                if log:
                    print("There is corrision with protein.")
                    print(mol.loc[(~(mol["AtomType"]=="H"))&(np.linalg.norm(mol.loc[:,["x","y","z"]]-position, axis=1) < ms)])
                return False

            if (np.linalg.norm(sp.loc[sp["AtomType"]=="C",["x","y","z"]]-position, axis=1).min() > sps):
                pass
            else:
                if log:
                    print("There is corrision with ligand.")
                    print(sp.loc[(sp["AtomType"]=="C")&(np.linalg.norm(sp.loc[:,["x","y","z"]]-position, axis=1) < sps)])
                return False

            return True

        def addAtom(spacer, pos, pos2, Cid):
            if log:
                print("---------spacer---------------")
                print(spacer)
                print("--------------------------------")
            sp = spacer.copy()

            #Make substitution group (C)           
            itrC = 0
            while True:
                itrC += 1
                if log: print(f"--------Testing C No.{Cid} | Try No.{itrC}--------")
                if itrC > 10: return 0, sp
                if (Cid>2)and(np.linalg.norm(pos-pos2) > np.linalg.norm(sPos-ePos)): return 0, sp

                #Choose H atom to change C 
                Hs = sp[(np.linalg.norm(sp.loc[:,["x","y","z"]]-pos, axis=1) < 1.2)&(sp["AtomType"]=="H")]
                if len(Hs)<1: 
                    print("There is no H candidate.")
                    return 0, sp
                elif log:
                    print(f"len(Hs)=={len(Hs)}")
                    print("H candidate:") ; print(Hs)
                if np.random.rand() < 0.3:
                    H2C = Hs.iloc[np.random.randint(len(Hs))]
                else:
                    H2C = Hs.iloc[np.linalg.norm(Hs.loc[:,["x","y","z"]]-pos2, axis=1).argmin()]
                if log: print(f"H2C:\n {H2C}")
                H2Cpos = H2C.loc[["x","y","z"]].astype(float).values
                H2Cid = H2C["AtomID"]
                Cpos = pos + (H2Cpos-pos)*1.51/np.linalg.norm(H2Cpos-pos)

                if log: print(f"New C pos: {Cpos}")
                if not No_corr(Cpos, 2., 1.5, sp):
                    if log: print(f"New C pos is invalid.")
                    continue
                vecGP = (Cpos-pos)/np.linalg.norm(Cpos-pos)
                Hpos = []

                #make H pos (sp3)
                if np.random.rand() < 0.5: 
                    r = np.array([vecGP[1],-1*vecGP[0],0],dtype=float) # vecGP and r are vertical
                    r = r*np.sin(70*np.pi/180)/np.linalg.norm(r) + vecGP*(1.51+np.cos(70*np.pi/180))
                    theta = np.random.rand()*np.pi # random 0 to pi rad
                    for i in range(3):
                        th = theta + 2*np.pi*i/3
                        rh = r*np.cos(th) + vecGP*np.dot(r, vecGP)*(1-np.cos(th)) + np.cross(vecGP, r)*np.sin(th)
                        Hpos.append(pos+rh)

                #make H pos (sp2)
                else:
                    r = np.array([vecGP[1],-1*vecGP[0],0],dtype=float) # vecGP and r are vertical
                    r = r*np.sin(60*np.pi/180)/np.linalg.norm(r) + vecGP*(1.51+np.cos(60*np.pi/180))
                    theta = np.random.rand()*np.pi # random 0 to pi rad
                    for i in range(2):
                        th = theta + np.pi*i
                        rh = r*np.cos(th) + vecGP*np.dot(r, vecGP)*(1-np.cos(th)) + np.cross(vecGP, r)*np.sin(th)
                        Hpos.append(pos+rh)

                OK = True ; Goal = False
                for p in Hpos:
                    n_Cpos = Cpos + (p-Cpos)*1.5
                    if np.linalg.norm(n_Cpos-pos2) < threshold:
                        Goal = True
                    elif not No_corr(p, 1.5, 1.5, sp):
                        OK = False

                if OK and not Goal:
                    if log: print(f"Correct Hs are found. \nNo. of H is {len(Hpos)}")
                    break
                elif OK and Goal:
                    if log: print("Done.")
                    sp.loc[sp["AtomID"]==H2Cid, ["AtomType", "PDBName", "x","y","z"]] = ["C","C"]+Cpos.tolist() 
                    for p in Hpos:
                        record = pd.DataFrame([[len(sp), len(sp)+1, "H", "", "",0,  "H", "RES", "1000","", 0]+p.tolist()+["H",""]], columns=sp.columns)
                        sp = pd.concat([sp, record]).reset_index(drop=True)

                    Hs_pos2 = sp[(np.linalg.norm(sp.loc[:,["x","y","z"]]-pos2, axis=1) < 1.2)&(sp["AtomType"]=="H")]
                    H_pos2_nC = Hs_pos2.iloc[np.linalg.norm(Hs_pos2.loc[:,["x","y","z"]]-Cpos, axis=1).argmin()]
                    sp = sp.loc[~(sp["AtomID"]==H_pos2_nC["AtomID"])]

                    Hs_Cpos = sp[(np.linalg.norm(sp.loc[:,["x","y","z"]]-Cpos, axis=1) < 1.2)&(sp["AtomType"]=="H")]
                    H_Cpos_n2 = Hs_Cpos.iloc[np.linalg.norm(Hs_Cpos.loc[:,["x","y","z"]]-pos2, axis=1).argmin()]
                    sp = sp.loc[~(sp["AtomID"]==H_Cpos_n2["AtomID"])]

                    if log: print(f"//////////////Deleted H: {H_pos2_nC['AtomID']} {H_Cpos_n2['AtomID']}")
                    sp["id"] = [i for i in range(len(sp))] ; sp["AtomID"] = sp["id"] + 1
                    return 1, sp
                elif not OK and Goal:
                    return 0, sp
                else:
                    if log: 
                        print(f"New H pos is invalid.")

            #change H to C
            sp.loc[sp["AtomID"]==H2Cid, ["AtomType", "PDBName", "x","y","z"]] = ["C","C"]+Cpos.tolist() 
            #attach H 
            for p in Hpos:
                record = pd.DataFrame([[len(sp), len(sp)+1, "H", "", "", 0, "H", "RES", "1000","", 0]+p.tolist()+["H",""]], columns=sp.columns)
                sp = pd.concat([sp, record]).reset_index(drop=True)

            #search again
            if log: 
                print(f"next search, Cpos: {Cpos}")
                print("[current spacer]") ; print(sp)
                print("-----------------------")
                print("# test hf/sto-3g\n\ntest\n\n1 1")
                for i in sp.loc[:, ["AtomType", "x","y","z"]].values:
                    print(f" {i[0]} {i[1]} {i[2]} {i[3]} ")
                print("-----------------------")


            return addAtom(sp, pos2, Cpos, Cid+1)

        Go = 0
        itr = 0
        while not Go:
            itr += 1
            print(f"<<<<<<<<<<<< Design No.{itr}>>>>>>>>>>>>>")
            if itr > 100: 
                print("Spacer cannot be found.")
                return spacer, ""

            mol = molDATA.copy()        
            spacer = mol[(np.linalg.norm(mol.loc[:,["x","y","z"]]-sPos, axis=1) <= 1.2)|
                      (np.linalg.norm(mol.loc[:,["x","y","z"]]-ePos, axis=1) <= 1.2)].copy()
            spacer["id"] = [i for i in range(len(spacer))] ; spacer["AtomID"] = spacer["id"] + 1
            mol = mol[(np.linalg.norm(mol.loc[:,["x","y","z"]]-sPos, axis=1) > 1.2)&
                      (np.linalg.norm(mol.loc[:,["x","y","z"]]-ePos, axis=1) > 1.2)].copy()
            Go, spacer = addAtom(spacer, sPos, ePos, 1)

        print(f"Success in Try No.{itr}")
        gfile = "# test hf/sto-3g\n\ntest\n\n1 1\n"
        for i in spacer.loc[:, ["AtomType", "x","y","z"]].values:
            gfile += f" {i[0]} {i[1]} {i[2]} {i[3]} \n"
        gfile += "\n"

        return spacer, gfile
   

    def Fill_Hydrogens(self, Area=[], existSkip=False, log=False):
        cDATA = self.molDATA.copy()

        def addAtom_single(aTYPE, cID, rID, rNAME, pos):
            nonlocal cDATA
            d = pd.DataFrame([[0, 0, aTYPE[0], "", "", 0, aTYPE, rNAME, rID, cID, 0]+pos.tolist()+["L",""]], columns=cDATA.columns)
            cDATA = pd.concat([cDATA, d]).reset_index(drop=True)
            return 1

        def Fill_SP3_3(cID, rID, route, cote, hName):
            rn = self.GetResName(cID, rID)
            rtpos = self.GetAtomPos_PDB(cID, rID, route)
            co1pos = self.GetAtomPos_PDB(cID, rID, cote[0])
            co2pos = self.GetAtomPos_PDB(cID, rID, cote[1])
            co3pos = self.GetAtomPos_PDB(cID, rID, cote[2])
            vec_rt2co1 = (co1pos - rtpos)/np.linalg.norm(co1pos - rtpos)
            vec_rt2co2 = (co2pos - rtpos)/np.linalg.norm(co2pos - rtpos)
            vec_rt2co3 = (co3pos - rtpos)/np.linalg.norm(co3pos - rtpos)
            vec = -1*(vec_rt2co1 + vec_rt2co2 + vec_rt2co3)/np.linalg.norm(vec_rt2co1 + vec_rt2co2 + vec_rt2co3)
            hpos = rtpos + vec
            addAtom_single(hName, cID, rID, rn, hpos)
            return 1

        def Fill_SP3_2(cID, rID, route, cote, hNames):
            rn = self.GetResName(cID, rID)
            rtpos = self.GetAtomPos_PDB(cID, rID, route)
            co1pos = self.GetAtomPos_PDB(cID, rID, cote[0])
            co2pos = self.GetAtomPos_PDB(cID, rID, cote[1])
            vec_rt2co1 = (co1pos - rtpos)/np.linalg.norm(co1pos - rtpos)
            vec_rt2co2 = (co2pos - rtpos)/np.linalg.norm(co2pos - rtpos)
            vec1 = -1 * (vec_rt2co1 + vec_rt2co2) / np.linalg.norm(vec_rt2co1 + vec_rt2co2)
            vec2 = np.cross(vec_rt2co1, vec_rt2co2)
            vec2 /= np.linalg.norm(vec2)
            hpos1 = rtpos + vec1*np.cos(54.75*np.pi/180.) + vec2*np.sin(54.75*np.pi/180.)
            hpos2 = rtpos + vec1*np.cos(54.75*np.pi/180.) - vec2*np.sin(54.75*np.pi/180.)
            addAtom_single(hNames[0], cID, rID, rn, hpos1)
            addAtom_single(hNames[1], cID, rID, rn, hpos2)
            return 1

        def Fill_SP3_1(cID, rID, route, cote, hNames):
            rn = self.GetResName(cID, rID)
            rtpos = self.GetAtomPos_PDB(cID, rID, route)
            copos = self.GetAtomPos_PDB(cID, rID, cote)
            vecLen = np.linalg.norm(rtpos - copos)
            vec = (rtpos - copos)/vecLen
            r = np.array([vec[1],-1*vec[0],0],dtype=float) # vecGP and r are vertical
            r = r*np.sin(70.5*np.pi/180)/np.linalg.norm(r) + vec*(vecLen+np.cos(70.5*np.pi/180))
            theta = np.random.rand()*np.pi # random 0 to pi rad
            for i in range(len(hNames)):
                th = theta + 2*np.pi*i/3
                rh = r*np.cos(th) + vec*np.dot(r, vec)*(1-np.cos(th)) + np.cross(vec, r)*np.sin(th)
                hpos = copos + rh
                addAtom_single(hNames[i], cID, rID, rn, hpos)
            return 1
            

        def Fill_HA(cID, rID):
            rn = self.GetResName(cID, rID)
            if rn == "GLY":
                Fill_SP3_2(cID, rID, "CA", ["C","N"], ["HA2","HA3"])
            else:
                Fill_SP3_3(cID, rID, "CA", ["C","N","CB"], "HA")
            return 1

        def Fill_H(cID, rID):
            rn = self.GetResName(cID, rID)
            try:
                capos = self.GetAtomPos_PDB(cID, rID, "CA")
                npos = self.GetAtomPos_PDB(cID, rID, "N")
                cpos = self.GetAtomPos_PDB(cID, rID-1, "C")
                if np.linalg.norm(npos - cpos) > 2.0:
                    print("Invalid C-N bond")
                    return 0
            except:
                if "N3" in cDATA.loc[(cDATA["ChainID"]==cID)&(cDATA["ResID"]==rID), "PDBName"].unique():
                    nit = "N3"
                elif "N" in cDATA.loc[(cDATA["ChainID"]==cID)&(cDATA["ResID"]==rID), "PDBName"].unique():
                    nit = "N"
                else:
                    print("No N atom at the head")
                    return 0
                if rn == "PRO":
                    Fill_SP3_2(cID, rID, nit, ["CA","CD"], ["H2","H3"])
                else:
                    Fill_SP3_1(cID, rID, nit, "CA",["H1","H2","H3"])    
                return 1

            vec_n2ca = (capos - npos)/np.linalg.norm(capos - npos)
            vec_n2c = (cpos - npos)/np.linalg.norm(cpos - npos)
            vec = -1 * (vec_n2ca + vec_n2c) / np.linalg.norm(vec_n2ca + vec_n2c)
            hpos = npos + vec
            addAtom_single("H", cID, rID, rn, hpos)
            return 1
    
        def Fill_Res(cID, rID):
            nonlocal cDATA
            ResNAME = self.GetResName(cID, rID)
            if not ResNAME in self.P.AMINO_DNA:
                return -1

            if existSkip and len(cDATA[(cDATA["ChainID"]==cID)&(cDATA["ResID"]==rID)&(cDATA["AtomType"]=="H")]) > 0:
                return 1
            
            cDATA = cDATA.query(f"not (ChainID=='{cID}' and ResID=={rID} and AtomType=='H')")
            
            Fill_HA(cID, rID)
            
            if not ResNAME == "PRO":
                Fill_H(cID, rID)

            if ResNAME == "ALA":
                Fill_SP3_1(cID, rID, "CB", "CA",["HB1","HB2","HB3"])
            elif ResNAME == "ARG":
                Fill_SP3_2(cID, rID, "CB", ["CA","CG"],["HB2","HB3"])
                Fill_SP3_2(cID, rID, "CG", ["CB","CD"],["HG2","HG3"])
                Fill_SP3_2(cID, rID, "CD", ["CG","NE"],["HD2","HD3"])

                cdpos = self.GetAtomPos_PDB(cID, rID, "CD")
                czpos = self.GetAtomPos_PDB(cID, rID, "CZ")
                nepos = self.GetAtomPos_PDB(cID, rID, "NE")
                nh1pos = self.GetAtomPos_PDB(cID, rID, "NH1")
                nh2pos = self.GetAtomPos_PDB(cID, rID, "NH2")

                vec1 = (cdpos - nepos)/np.linalg.norm(cdpos - nepos)
                vec2 = (czpos - nepos)/np.linalg.norm(czpos - nepos)
                vec = -1 * (vec1 + vec2) / np.linalg.norm(vec1 + vec2)
                addAtom_single("HE", cID, rID, ResNAME, nepos+vec)

                vecne2z = (czpos - nepos)/np.linalg.norm(czpos - nepos)
                vecnh12z = (czpos - nh1pos)/np.linalg.norm(czpos - nh1pos)
                vecnh22z = (czpos - nh2pos)/np.linalg.norm(czpos - nh2pos)
                addAtom_single("HH11", cID, rID, ResNAME, nh1pos+vecne2z)
                addAtom_single("HH12", cID, rID, ResNAME, nh1pos+vecnh22z)
                addAtom_single("HH21", cID, rID, ResNAME, nh2pos+vecne2z)
                addAtom_single("HH22", cID, rID, ResNAME, nh2pos+vecnh12z)

            elif ResNAME == "ASN" or ResNAME == "ASP":
                Fill_SP3_2(cID, rID, "CB", ["CA","CG"],["HB2","HB3"])
                if ResNAME == "ASN":
                    cbpos = self.GetAtomPos_PDB(cID, rID, "CB")
                    cgpos = self.GetAtomPos_PDB(cID, rID, "CG")
                    od1pos = self.GetAtomPos_PDB(cID, rID, "OD1")
                    nd2pos = self.GetAtomPos_PDB(cID, rID, "ND2")
                    
                    vec = (cgpos - cbpos)/np.linalg.norm(cgpos - cbpos)
                    addAtom_single("HD21", cID, rID, ResNAME, nd2pos+vec)

                    vec = (cgpos - od1pos)/np.linalg.norm(cgpos - od1pos)
                    addAtom_single("HD22", cID, rID, ResNAME, nd2pos+vec)


            elif ResNAME == "CYS":
                Fill_SP3_2(cID, rID, "CB", ["CA","SG"],["HB2","HB3"])
                SPos = self.GetAtomPos_PDB(cID, rID, "SG")
                dist = np.linalg.norm(self.molDATA.loc[:,["x","y","z"]].values - SPos, axis=1)
                if len(self.molDATA.loc[(dist>0.001)&(dist<2.2)&(self.molDATA["PDBName"]=="SG")]) == 0:
                    Fill_SP3_1(cID, rID, "SG", "CB",["HG"])
                
            elif ResNAME == "GLN" or ResNAME == "GLU":
                Fill_SP3_2(cID, rID, "CB", ["CA","CG"],["HB2","HB3"])
                Fill_SP3_2(cID, rID, "CG", ["CB","CD"],["HG2","HG3"])

                if ResNAME == "GLN":
                    cgpos = self.GetAtomPos_PDB(cID, rID, "CG")
                    cdpos = self.GetAtomPos_PDB(cID, rID, "CD")
                    oe1pos = self.GetAtomPos_PDB(cID, rID, "OE1")
                    ne2pos = self.GetAtomPos_PDB(cID, rID, "NE2")
                    
                    vec = (cdpos - cgpos)/np.linalg.norm(cdpos - cgpos)
                    addAtom_single("HE21", cID, rID, ResNAME, ne2pos+vec)

                    vec = (cdpos - oe1pos)/np.linalg.norm(cdpos - oe1pos)
                    addAtom_single("HE22", cID, rID, ResNAME, ne2pos+vec)   

            elif ResNAME == "HIS" or ResNAME == "HID":
                Fill_SP3_2(cID, rID, "CB", ["CA","CG"],["HB2","HB3"])
                
                cgpos = self.GetAtomPos_PDB(cID, rID, "CG")
                cd2pos = self.GetAtomPos_PDB(cID, rID, "CD2")
                nd1pos = self.GetAtomPos_PDB(cID, rID, "ND1")
                ce1pos = self.GetAtomPos_PDB(cID, rID, "CE1")
                ne2pos = self.GetAtomPos_PDB(cID, rID, "NE2")

                pos = (ce1pos + nd1pos)/2
                vec = (cd2pos - pos)/np.linalg.norm(cd2pos - pos)
                addAtom_single("HD2", cID, rID, ResNAME, cd2pos+vec)

                pos = (cd2pos + ne2pos)/2
                vec = (nd1pos - pos)/np.linalg.norm(nd1pos - pos)
                addAtom_single("HD1", cID, rID, ResNAME, nd1pos+vec)

                pos = (cd2pos + cgpos)/2
                vec = (ce1pos - pos)/np.linalg.norm(ce1pos - pos)
                addAtom_single("HE1", cID, rID, ResNAME, ce1pos+vec)
            
            elif ResNAME == "ILE":
                Fill_SP3_3(cID, rID, "CB", ["CA", "CG1", "CG2"], "HB")
                Fill_SP3_1(cID, rID, "CG2", "CB",["HG21","HG22","HG23"])
                Fill_SP3_2(cID, rID, "CG1", ["CB","CD1"],["HG12","HG13"])
                Fill_SP3_1(cID, rID, "CD1", "CG1",["HD11","HD12","HD13"])

            elif ResNAME == "LEU":
                Fill_SP3_2(cID, rID, "CB", ["CA","CG"],["HB2","HB3"])
                Fill_SP3_3(cID, rID, "CG", ["CB", "CD1", "CD2"], "HG")
                Fill_SP3_1(cID, rID, "CD1", "CG",["HD11","HD12","HD13"])
                Fill_SP3_1(cID, rID, "CD2", "CG",["HD21","HD22","HD23"])

            elif ResNAME == "LYS":
                Fill_SP3_2(cID, rID, "CB", ["CA","CG"],["HB2","HB3"])
                Fill_SP3_2(cID, rID, "CG", ["CB","CD"],["HG2","HG3"])
                Fill_SP3_2(cID, rID, "CD", ["CG","CE"],["HD2","HD3"])
                Fill_SP3_2(cID, rID, "CE", ["CD","NZ"],["HE2","HE3"])
                Fill_SP3_1(cID, rID, "NZ", "CE",["HZ1","HZ2","HZ3"])

            elif ResNAME == "MET":
                Fill_SP3_2(cID, rID, "CB", ["CA","CG"],["HB2","HB3"])
                Fill_SP3_2(cID, rID, "CG", ["CB","SD"],["HG2","HG3"])
                Fill_SP3_1(cID, rID, "CE", "SD",["HE1","HE2","HE3"])

            elif ResNAME == "PHE" or ResNAME == "TYR":
                Fill_SP3_2(cID, rID, "CB", ["CA","CG"],["HB2","HB3"])

                cgpos = self.GetAtomPos_PDB(cID, rID, "CG")
                cd1pos = self.GetAtomPos_PDB(cID, rID, "CD1")
                cd2pos = self.GetAtomPos_PDB(cID, rID, "CD2")
                ce1pos = self.GetAtomPos_PDB(cID, rID, "CE1")
                ce2pos = self.GetAtomPos_PDB(cID, rID, "CE2")
                czpos = self.GetAtomPos_PDB(cID, rID, "CZ")

                vec = (cd2pos - ce1pos)/np.linalg.norm(cd2pos - ce1pos)
                addAtom_single("HD2", cID, rID, ResNAME, cd2pos+vec)
                addAtom_single("HE1", cID, rID, ResNAME, ce1pos-vec)

                vec = (cd1pos - ce2pos)/np.linalg.norm(cd1pos - ce2pos)
                addAtom_single("HD1", cID, rID, ResNAME, cd1pos+vec)
                addAtom_single("HE2", cID, rID, ResNAME, ce2pos-vec)

                if ResNAME == "PHE":
                    vec = (czpos - cgpos)/np.linalg.norm(czpos - cgpos)
                    addAtom_single("HZ", cID, rID, ResNAME, czpos+vec)
                elif ResNAME == "TYR":
                    Fill_SP3_1(cID, rID, "OH", "CZ", ["HH"])

            elif ResNAME == "PRO":
                Fill_SP3_2(cID, rID, "CB", ["CA","CG"],["HB2","HB3"])
                Fill_SP3_2(cID, rID, "CG", ["CB","CD"],["HG2","HG3"])
                Fill_SP3_2(cID, rID, "CD", ["CG","N"],["HD2","HD3"])
                
            elif ResNAME == "SER":
                Fill_SP3_2(cID, rID, "CB", ["CA","OG"],["HB2","HB3"])
                Fill_SP3_1(cID, rID, "OG", "CB",["HG"])
            
            elif ResNAME == "THR":
                Fill_SP3_3(cID, rID, "CB", ["CA", "OG1", "CG2"], "HB")
                Fill_SP3_1(cID, rID, "CG2", "CB",["HG21","HG22","HG23"])
                Fill_SP3_1(cID, rID, "OG1", "CB",["HG1"])
            
            elif ResNAME == "TRP":
                Fill_SP3_2(cID, rID, "CB", ["CA","CG"],["HB2","HB3"])

                cgpos = self.GetAtomPos_PDB(cID, rID, "CG")
                cd1pos = self.GetAtomPos_PDB(cID, rID, "CD1")
                cd2pos = self.GetAtomPos_PDB(cID, rID, "CD2")
                ne1pos = self.GetAtomPos_PDB(cID, rID, "NE1")
                ce2pos = self.GetAtomPos_PDB(cID, rID, "CE2")

                pos = (cd2pos + ce2pos)/2
                vec = (cd1pos - pos)/np.linalg.norm(cd1pos - pos)
                addAtom_single("HD1", cID, rID, ResNAME, cd1pos+vec)

                pos = (cgpos + cd2pos)/2
                vec = (ne1pos - pos)/np.linalg.norm(ne1pos - pos)
                addAtom_single("HE1", cID, rID, ResNAME, ne1pos+vec)

                ce3pos = self.GetAtomPos_PDB(cID, rID, "CE3")
                cz2pos = self.GetAtomPos_PDB(cID, rID, "CZ2")
                cz3pos = self.GetAtomPos_PDB(cID, rID, "CZ3")
                ch2pos = self.GetAtomPos_PDB(cID, rID, "CH2")

                vec = (ce3pos - cz2pos)/np.linalg.norm(ce3pos - cz2pos)
                addAtom_single("HE3", cID, rID, ResNAME, ce3pos+vec)
                addAtom_single("HZ2", cID, rID, ResNAME, cz2pos-vec)

                vec = (cz3pos - ce2pos)/np.linalg.norm(cz3pos - ce2pos)
                addAtom_single("HZ3", cID, rID, ResNAME, cz3pos+vec)

                vec = (ch2pos - cd2pos)/np.linalg.norm(ch2pos - cd2pos)
                addAtom_single("HH2", cID, rID, ResNAME, ch2pos+vec)
                
            elif ResNAME == "VAL":
                Fill_SP3_3(cID, rID, "CB", ["CA", "CG1", "CG2"], "HB")
                Fill_SP3_1(cID, rID, "CG1", "CB",["HG11","HG12","HG13"])
                Fill_SP3_1(cID, rID, "CG2", "CB",["HG21","HG22","HG23"])

            return 1

        if len(Area)>0:
            sequ = Area
        else:
            sequ = self.GetSequence()

        for c, r in sequ:
            if log: print(f"{c} {r}")
            try:
                Fill_Res(c, r)
            except Exception as e :
                print(f"{c} {r}| Fill failed")
                print(e)

        self.mol_trail.append(self.molDATA)
        self.molDATA = cDATA
        self.Reset_ResName()
        self.Reset_ID(sort=True)

        return cDATA


    def Check_MissingAtom(self, recover=False, include_H=False):

        Problem = []

        if include_H:
            cDATA = self.molDATA
            alist = {k:len([i for i in self.P.AMINO_ATOMS[k]]) for k in self.P.AMINO_ATOMS.keys()}
        else:
            cDATA = self.molDATA[~(self.molDATA["AtomType"]=="H")]
            alist = {k:len([i for i in self.P.AMINO_ATOMS[k] if not i[0]=="H"]) for k in self.P.AMINO_ATOMS.keys()}

        seq = self.GetSequence(["ChainID","ResID","ResName"])

        for c, ri, rn in seq:
            
            if not rn in self.P.AMINO_ATOMS.keys():
                continue
            ml = len(cDATA[(cDATA["ChainID"]==c)&(cDATA["ResID"]==ri)]) 
            if not ml == alist[rn]:
                end_exception = False           
                if ml > alist[rn]:
                    ml_list = cDATA.loc[(cDATA["ChainID"]==c)&(cDATA["ResID"]==ri), "PDBName"].values.tolist()

                    if ml == alist[rn] + 1 and "OXT" in ml_list:
                        end_exception = True
                    elif ml == alist[rn] + 2 and "H2" in ml_list and ("H1" in ml_list or "H3" in ml_list):
                        end_exception = True
                    
                    if not end_exception:
                        Problem.append([c, ri])
                        print(f"[Unexpected atom]  Chain:{c} Res:{ri}{rn}")

                elif ml < alist[rn]:
                    Problem.append([c, ri])
                    print(f"[Lack of atom]  Chain:{c} Res:{ri}{rn}")
            
                if not end_exception and recover:
                    ok = self.RecoverRes(c, ri, error_ok=False)
                    if ok:
                        print(f"Lacking atoms are recovered.")
                    else:
                        print( f"[Check] >> {c} {ri}{rn} could not be recovered. ")

        return Problem


    def RecoverRes(self, chainID, resID, error_ok=False):
        cDATA = self.molDATA.copy()
        rn = self.GetResName(chainID, resID)

        if len(cDATA[cDATA["AtomType"]=="H"]) > 0:
            skipH = False
        else:
            skipH = True

        if not len(cDATA.query(f"ChainID=='{chainID}' and ResID=={resID} and (PDBName in ['C','CA','N','O'])"))==4:
            print(f"{chainID} {resID} {rn} cannot be recovered (lack of main chain)")
            return 0
        
        cDATA = cDATA.query(f"not (ChainID=='{chainID}' and ResID=={resID} and (not PDBName in ['C','CA','N','O']))")
        
        cDATA.loc[(cDATA["ChainID"]==chainID)&(cDATA["ResID"]==resID), "ResName"] = "GLY"
        
        self.molDATA = cDATA
        
        self.Fill_Hydrogens(Area=[(chainID, resID)])
        for i in range(50):
            ok, _ = self.Residue_Change(chainID, resID, rn, error_ok=error_ok)
            if ok: 
                break

        if not ok:
            print(f"{chainID} {resID} {rn} cannot be recovered (conversion error)")
            return 0

        if skipH:
            self.molDATA = self.molDATA[self.molDATA["AtomType"]=="H"]

        return 1


    def Fit_Ligand_All(self, ligName, fixAtomName, fixAtomPosition, connectInfo={}, fixBond=[], outputMinDis=False, log=False):
        start = time.time()
        if not len(self.Contents["connection"]) == len(self.molDATA):
            con = self.makeConnection(add_info=connectInfo)
        else:
            con = self.Contents["connection"]
        #Prepare template
        cDATA = self.molDATA.copy()
        temp = cDATA.loc[~(cDATA["ResName"]==ligName)]
        temp_pos = temp.loc[:,["x","y","z"]].values.astype(float)
        temp_vdw = [self.P.VDW[i.upper()] for i in temp.loc[:,"AtomType"].values]

        #prepare moving molecule
        mol = cDATA.loc[cDATA["ResName"]==ligName]
        if log: print(f"mol: \n{mol}")

        fixIndex = np.where(mol["PDBName"]==fixAtomName)[0][0]
        mol_pos = mol.loc[:,["x","y","z"]].values.astype(float)
        mol_pos += fixAtomPosition - mol_pos[fixIndex]

        mol_vdw = [self.P.VDW[i.upper()] for i in mol.loc[:,"AtomType"].values]

        #prepare loss function
        sum_VDW1 = np.array(mol_vdw)[:,np.newaxis] + np.array(temp_vdw)[np.newaxis,:]
        sum_VDW2 = np.array(mol_vdw)[:,np.newaxis] + np.array(mol_vdw)[np.newaxis,:]
        sum_VDW2[np.eye(len(mol), dtype=bool)] = 0
        f = mol.loc[:,"id"].values
        sum_VDW2[con[:,f][f,:]>0] = 0
        if log: print(f"sum_VDW1: \n{sum_VDW1}  sum_VDW2: \n{sum_VDW2}")
        def Loss(lpos):
            dmat1 = distance.cdist(lpos, temp_pos, metric='euclidean')
            dmat2 = distance.cdist(lpos, lpos, metric='euclidean')
            return (np.maximum(sum_VDW1 - dmat1, 0)**12).sum() + (np.maximum(sum_VDW2 - dmat2, 0)**12).sum()

        #extract bonds
        atom_list = [mol.iloc[0]["id"]]
        atom_list_used = []
        bond_list = []
        bond_list_used = []
        while len(atom_list) > 0:
            if log: print(f"atom_list: {atom_list}")
            ai = atom_list.pop(0)
            ai_i = np.where(mol["id"]==ai)[0][0]
            ai_n = mol.loc[mol["id"]==ai,"PDBName"].iloc[0]
            atom_list_used.append(ai)
            for ai2 in np.where(con[ai])[0]:
                try:
                    ai2_i = np.where(mol["id"]==ai2)[0][0]
                    ai2_n = mol.loc[mol["id"]==ai2,"PDBName"].iloc[0]
                    if (not ai2 in atom_list) and (not ai2 in atom_list_used):
                        atom_list.append(ai2)
                    if (not [ai_i,ai2_i] in bond_list_used) and (not [ai2_i,ai_i] in bond_list_used):
                        bond_list.append([ai_n,ai2_n])
                        bond_list_used.append([ai_i,ai2_i])
                except:
                    pass
                    
                    
        bond_list = np.array(bond_list)
        bonds = pd.DataFrame({
            "no":[i for i in range(len(bond_list))],
            "name1":bond_list[:,0],
            "name2":bond_list[:,1]
        })

        if log: print(f"bonds: {bonds}")

        #extract bridge
        cent = []; route = []
        for bi in range(len(bonds)):
            cut_graph = bonds.copy()
            cut_graph = cut_graph[~(cut_graph["no"]==bi)]
            
            mol["trail"] = [1 if i==0 else 0 for i in range(len(mol))]
            for i in range(1,len(mol)):
                mol_i = mol.loc[mol["trail"]==i]
                if len(mol_i)==0:
                    break
                else:
                    for at in mol_i["PDBName"].values:
                        adj_list = cut_graph.loc[cut_graph["name1"]==at,"name2"].values.tolist() + \
                                                cut_graph.loc[cut_graph["name2"]==at,"name1"].values.tolist()
                        for adj in adj_list:
                            if mol.loc[mol["PDBName"]==adj,"trail"].iloc[0] == 0:
                                mol.loc[mol["PDBName"]==adj,"trail"] = i+1

            #if log: print(f"mol: {mol}")
            matms = mol.loc[mol["trail"]==0,"PDBName"].tolist()
            if len(matms) > 0:
                n1, n2 = bonds.loc[bonds["no"]==bi,["name1","name2"]].iloc[0].values
                if log: print(f"n1:{n1}, n2:{n2}")
                if fixAtomName in matms:
                    matms = mol.loc[~(mol["trail"]==0),"PDBName"].tolist()
                if log: print(f"matms: {matms}")

                if len(matms) > 1:
                    if n1 in matms:
                        matms.remove(n1); cent.append(n1); route.append(n2)
                        mol[f"{n2}->{n1}"] = [1 if n in matms else 0 for n in mol["PDBName"]]
                    else:
                        matms.remove(n2); cent.append(n2); route.append(n1)
                        mol[f"{n1}->{n2}"] = [1 if n in matms else 0 for n in mol["PDBName"]]
                
        bridge = pd.DataFrame({
            "Centor":cent,
            "Route":route
        })
        for bd in fixBond:
            bridge = bridge.query(f"not (Centor in {bd} and Route in {bd})").reset_index(drop=True)

        if log: print(f"mol: {mol}")
        if log: print(f"bridge: {bridge}")

        #Define rotation
        def Rotate(bi, theta):
            nonlocal mol_pos
            rt, ct = bridge.loc[bi,["Route","Centor"]]
            filt = mol[f"{rt}->{ct}"]==1
            rtpos = mol_pos[np.where(mol["PDBName"]==rt)[0][0]]
            ctpos = mol_pos[np.where(mol["PDBName"]==ct)[0][0]]
            vert = (ctpos - rtpos)/np.linalg.norm(ctpos - rtpos)
            
            pr = mol_pos[filt] - rtpos
            r = pr*np.cos(theta) + vert*np.dot(pr, vert)[:,np.newaxis]*(1-np.cos(theta)) + np.cross(vert, pr)*np.sin(theta)
            
            if np.isnan(r).any():
                if log: print(f"Rotation {rt}->{ct} failed")
                return 0
            mol_pos[filt] = rtpos + r
            return 1

        def Rotate_all(theta):
            nonlocal mol_pos
            z = np.random.rand()*2-1 ; psy = (np.random.rand()*2-1)*np.pi
            vert = np.array([np.sqrt(1-z**2)*np.cos(psy), np.sqrt(1-z**2)*np.sin(psy), z])
            
            N = fixAtomPosition - vert
            pr = mol_pos - N
            r = pr*np.cos(theta) + vert*np.dot(pr, vert)[:,np.newaxis]*(1-np.cos(theta)) + np.cross(vert, pr)*np.sin(theta)
            
            if np.isnan(r).any():
                if log: print("Central rotation failed")
                return 0
            mol_pos = r + N
            
            return 1


        #Random Optimization
        loss_min = Loss(mol_pos) ; mol_min = mol_pos.copy()
        itr = 0

        Finish = 0; ok = 0
        while Finish < 3:
            itr += 1

            for _ in range(np.random.randint(1,4)):
                if np.random.rand()>0.5 or len(bridge)==0:
                    Rotate_all((np.random.rand()*60-30)/(Finish+1))
                else:
                    Rotate(np.random.randint(len(bridge)), (np.random.rand()*60-30)/(Finish+1))

            ls = Loss(mol_pos)
            if Finish == 0:
                if loss_min > ls:
                    loss_min = ls ; mol_min = mol_pos.copy()
            else:
                if loss_min > ls:
                    dm = distance.cdist(mol_min, temp_pos, metric='euclidean')
                    if dm.min() > 0.5:
                        loss_min = ls ; mol_min = mol_pos.copy()
                else:
                    mol_pos = mol_min.copy()

            if itr%1000==0:
                dm = distance.cdist(mol_min, temp_pos, metric='euclidean')
                if log: print(f"loss_min: {loss_min} Minimum distance = {dm.min()} -> elapsed time:{time.time() - start:.2f} sec")
                if dm.min() > 0.5:
                    ok = 1
                    Finish += 1                   
                else:
                    mol_pos = mol_min.copy()
                    Finish = 0
                    
            elif itr>40000:
                print(f"Calculation is not completed. Minimum distance = {dm.min()}")
                ok = 0
                Finish = 3

        
        cDATA.loc[cDATA["ResName"]==ligName, ["x","y","z"]] = mol_min

        self.mol_trail.append(self.molDATA)
        self.molDATA = cDATA
        
        if outputMinDis:
            return ok, cDATA, dm.min()
        else:
            return ok, cDATA


    def Fit_Ligand_Fragment(self, ligName, connectInfo={}, fixBond=[], log=False):
        start = time.time()
        con = self.makeConnection(add_info=connectInfo)
        #Prepare template
        cDATA = self.molDATA.copy()
        temp = cDATA.loc[~(cDATA["ResName"]==ligName)]
        temp_pos = temp.loc[:,["x","y","z"]].values.astype(float)
        temp_vdw = [self.P.VDW[i.upper()] for i in temp.loc[:,"AtomType"].values]

        #prepare moving molecule
        mol = cDATA.loc[cDATA["ResName"]==ligName]
        if log: print(f"mol: \n{mol}")

        mol_pos = mol.loc[:,["x","y","z"]].values.astype(float)
        mol_vdw = [self.P.VDW[i.upper()] for i in mol.loc[:,"AtomType"].values]

        #prepare loss function
        sum_VDW1 = np.array(mol_vdw)[:,np.newaxis] + np.array(temp_vdw)[np.newaxis,:]
        sum_VDW2 = np.array(mol_vdw)[:,np.newaxis] + np.array(mol_vdw)[np.newaxis,:]
        sum_VDW2[np.eye(len(mol), dtype=bool)] = 0
        f = mol.loc[:,"id"].values
        sum_VDW2[con[:,f][f,:]>0] = 0
        if log: print(f"sum_VDW1: \n{sum_VDW1}  sum_VDW2: \n{sum_VDW2}")
        def Loss(lpos):
            dmat1 = distance.cdist(lpos, temp_pos, metric='euclidean')
            dmat2 = distance.cdist(lpos, lpos, metric='euclidean')
            return (np.maximum(sum_VDW1 - dmat1, 0)**12).sum() + (np.maximum(sum_VDW2 - dmat2, 0)**12).sum()

        #extract bonds
        atom_list = [mol.iloc[0]["id"]]
        atom_list_used = []
        bond_list = []
        bond_list_used = []
        while len(atom_list) > 0:
            if log: print(f"atom_list: {atom_list}")
            ai = atom_list.pop(0)
            ai_i = np.where(mol["id"]==ai)[0][0]
            ai_n = mol.loc[mol["id"]==ai,"PDBName"].iloc[0]
            atom_list_used.append(ai)
            for ai2 in np.where(con[ai])[0]:
                try:
                    ai2_i = np.where(mol["id"]==ai2)[0][0]
                    ai2_n = mol.loc[mol["id"]==ai2,"PDBName"].iloc[0]
                    if (not ai2 in atom_list) and (not ai2 in atom_list_used):
                        atom_list.append(ai2)
                    if (not [ai_i,ai2_i] in bond_list_used) and (not [ai2_i,ai_i] in bond_list_used):
                        bond_list.append([ai_n,ai2_n])
                        bond_list_used.append([ai_i,ai2_i])
                except:
                    pass
                    
        bond_list = np.array(bond_list)
        bonds = pd.DataFrame({
            "no":[i for i in range(len(bond_list))],
            "name1":bond_list[:,0],
            "name2":bond_list[:,1]
        })

        if log: print(f"bonds: {bonds}")

        #extract bridge
        cent = []; route = []
        for bi in range(len(bonds)):
            cut_graph = bonds.copy()
            cut_graph = cut_graph[~(cut_graph["no"]==bi)]
            
            mol["trail"] = [1 if i==0 else 0 for i in range(len(mol))]
            for i in range(1,len(mol)):
                mol_i = mol.loc[mol["trail"]==i]
                if len(mol_i)==0:
                    break
                else:
                    for at in mol_i["PDBName"].values:
                        adj_list = cut_graph.loc[cut_graph["name1"]==at,"name2"].values.tolist() + \
                                                cut_graph.loc[cut_graph["name2"]==at,"name1"].values.tolist()
                        for adj in adj_list:
                            if mol.loc[mol["PDBName"]==adj,"trail"].iloc[0] == 0:
                                mol.loc[mol["PDBName"]==adj,"trail"] = i+1

            #if log: print(f"mol: {mol}")
            matms = mol.loc[mol["trail"]==0,"PDBName"].tolist()
            matms_fr = mol.loc[mol["trail"]==0,"freeze"].tolist()
            if len(matms) > 0:
                n1, n2 = bonds.loc[bonds["no"]==bi,["name1","name2"]].iloc[0].values
                if log: print(f"n1:{n1}, n2:{n2}")
                if -1 in matms_fr:
                    matms = mol.loc[~(mol["trail"]==0),"PDBName"].tolist()
                    matms_fr = mol.loc[~(mol["trail"]==0),"freeze"].tolist()
                    if -1 in matms_fr:
                        continue
                if log: print(f"matms: {matms}")

                if len(matms) > 1:
                    if n1 in matms:
                        matms.remove(n1); cent.append(n1); route.append(n2)
                        mol[f"{n2}->{n1}"] = [1 if n in matms else 0 for n in mol["PDBName"]]
                    else:
                        matms.remove(n2); cent.append(n2); route.append(n1)
                        mol[f"{n1}->{n2}"] = [1 if n in matms else 0 for n in mol["PDBName"]]
                
        bridge = pd.DataFrame({
            "Centor":cent,
            "Route":route
        })
        for bd in fixBond:
            bridge = bridge.query(f"not (Centor in {bd} and Route in {bd})").reset_index(drop=True)

        if log: print(f"mol: {mol}")
        if log: print(f"bridge: {bridge}")

        if len(bridge) == 0:
            print(f"No rotatable bond.")
            return 0, cDATA

        #Define rotation
        def Rotate(bi, theta):
            nonlocal mol_pos
            rt, ct = bridge.loc[bi,["Route","Centor"]]
            filt = mol[f"{rt}->{ct}"]==1
            rtpos = mol_pos[np.where(mol["PDBName"]==rt)[0][0]]
            ctpos = mol_pos[np.where(mol["PDBName"]==ct)[0][0]]
            vert = (ctpos - rtpos)/np.linalg.norm(ctpos - rtpos)
            
            pr = mol_pos[filt] - rtpos
            r = pr*np.cos(theta) + vert*np.dot(pr, vert)[:,np.newaxis]*(1-np.cos(theta)) + np.cross(vert, pr)*np.sin(theta)
            
            if np.isnan(r).any():
                if log: print(f"Rotation {rt}->{ct} failed")
                return 0
            mol_pos[filt] = rtpos + r
            return 1

        #Random Optimization
        loss_min = Loss(mol_pos) ; mol_min = mol_pos.copy()
        itr = 0

        Finish = 0; ok = 0
        while Finish < 3:
            itr += 1

            for _ in range(np.random.randint(1,4)):
                Rotate(np.random.randint(len(bridge)), (np.random.rand()*60-30)/(Finish+1))

            ls = Loss(mol_pos)
            if Finish == 0:
                if loss_min > ls:
                    loss_min = ls ; mol_min = mol_pos.copy()
            else:
                if loss_min > ls:
                    dm = distance.cdist(mol_min, temp_pos, metric='euclidean')
                    if dm.min() > 0.5:
                        loss_min = ls ; mol_min = mol_pos.copy()
                else:
                    mol_pos = mol_min.copy()

            if itr%1000==0:
                dm = distance.cdist(mol_min, temp_pos, metric='euclidean')
                if log: print(f"loss_min: {loss_min} Minimum distance = {dm.min()} -> elapsed time:{time.time() - start:.2f} sec")
                if dm.min() > 0.5:
                    ok = 1
                    Finish += 1                   
                else:
                    mol_pos = mol_min.copy()
                    Finish = 0
                    
            elif itr>40000:
                print(f"Calculation is not completed. Minimum distance = {dm.min()}")
                ok = 0
                Finish = 3

        
        cDATA.loc[cDATA["ResName"]==ligName, ["x","y","z"]] = mol_min

        self.mol_trail.append(self.molDATA)
        self.molDATA = cDATA

        return ok, cDATA


    def Summarize_Bond_Angle(self, connectInfo={}):
        con = self.makeConnection(add_info=connectInfo)

        BOND = []; ANGLE=[]
        for ci in range(len(con)):
            atm1 = self.molDATA.loc[ci,"PDBName"]
            con_ln = con[ci]
            
            con_ids = np.where(con_ln>0)[0]
            
            for ai in con_ids:
                atm2 = self.molDATA.loc[ai,"PDBName"]
                b = self.GetDistance_ID(ci, ai, mode="id")
                BOND.append([atm1, atm2, b])
                
            if len(con_ids)>1:
                for ai_1, ai_2 in itertools.combinations(con_ids,2):
                    atm2_1 = self.molDATA.loc[ai_1,"PDBName"]
                    atm2_2 = self.molDATA.loc[ai_2,"PDBName"]
                    a = self.GetAngle_ID(ci, ai_1, ai_2, mode="id", rad=False)
                    ANGLE.append([atm1,atm2_1,atm2_2,a])

        return BOND, ANGLE


    def molData_maker(self, chainID, resID, ResNAME, glog_file_path, info={}, Compound_Name="", log=False):
        save_dir = os.path.dirname(glog_file_path)
        comp = os.path.splitext(os.path.basename(glog_file_path))[0]
        
        temp_mol = self.Read_GaussianOutput_ONIOM(glog_file_path, readOnly=True, log=log)
        atom_num = len(temp_mol)
        
        self.molDATA.loc[:,["ResName","ChainID","ResID"]] = [ResNAME, chainID, resID]
        self.Reset_PDBname(ResNAME)
        
        con = self.Contents["connection"]
        ufft = []
        for i in range(len(self.molDATA)):
            atyp = self.molDATA.loc[i, "AtomType"]
            if atyp=="H":
                ufft.append("H_")
            elif atyp in ["C","N","O"]:
                connect1 = con[i][con[i]>1]
                if len(connect1)>0:
                    ufft.append(f"{atyp}_R")
                elif atyp=="O":
                    ufft.append(f"{atyp}_2")
                else:
                    ufft.append(f"{atyp}_3")
            elif atyp=="S":
                connect1 = con[i][con[i]>1]
                if len(connect1)>0:
                    ufft.append(f"S_3+6")
                else:
                    ufft.append(f"S_2")
            elif atyp in info.keys():
                ufft.append(info[atyp])
            else:
                ufft.append("")
        
        moldatascv = pd.DataFrame({
            "ResName":[ResNAME for i in range(atom_num)],
            "PDBName":self.molDATA["PDBName"],
            "charge":temp_mol["charge"],
            "UFFType":ufft,
            "freeze":[0 for i in range(atom_num)],
            "layer":["H" for i in range(atom_num)]
        }).sort_values("PDBName")
        
        if len(Compound_Name)>0:
            self.Write_PDB(f"{Compound_Name}_{chainID}", save_dir)
            moldatascv.to_csv(f"{save_dir}/molData_{Compound_Name}.csv")
        else:
            self.Write_PDB(f"{comp}_{chainID}", save_dir)
            moldatascv.to_csv(f"{save_dir}/molData_{comp}.csv")
            
        return 1
    
    
    


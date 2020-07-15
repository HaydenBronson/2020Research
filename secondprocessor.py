import os
import time
import glob
import re
from functools import reduce
from klepto.archives import dir_archive

import numpy as np
from tqdm.auto import tqdm
import coffea.processor as processor
from coffea.processor.accumulator import AccumulatorABC
from coffea import hist
import pandas as pd 
import uproot_methods
import awkward

import matplotlib
import matplotlib.pyplot as plt 
from matplotlib.colors import LogNorm

from tW_scattering.Tools.helpers import *

matplotlib.use('Agg')


class doublemuonProcessor(processor.ProcessorABC):
    def __init__(self):
        dataset_axis = hist.Cat("dataset", "Primary dataset")
        pt_axis = hist.Bin("pt",r"$p_{\mu\mu}$[GeV]", 600,0,1000)
        mass_axis = hist.Bin("mass", r"m_{\mu\mu}$[GeV]", 100, 0, 10)
        eta_axis = hist.Bin("eta", r"$\eta$", 60, -5.5, 5.5)

        self._accumulator = processor.dict_accumulator({
            #"MET_pt": hist.Hist("Counts", dataset_axis, pt_axis),
            "mass": hist.Hist("Counts", dataset_axis, mass_axis),
        })
    
    @property
    def accumulator(self):
        return self._accumulator 
    
    def process(self, df):
        output = self.accumulator.identity()
        dataset = df["dataset"]
        muons = awkward.JaggedArray.zip(
            pt=df['Muon_pt'], 
            eta=df['Muon_eta'], 
            charge=df['Muon_charge'], 
            mass=df['Muon_mass'], 
            phi=df['Muon_phi']
            )
        #output['cutflow']['allevents'] += muons.size

        cut = (muons['pt'] > 25) & (abs(muons['eta']) < 2.4)
        dimuons = muons[cut]
        one_pair_dimuon = (dimuons.counts==2) & (dimuons['charge'].prod()==-1)
        output['mass'].fill(dataset=dataset, mass=dimuons[one_pair_dimuon].mass.flatten())
        return output

    def postprocess(self, accumulator):
        return accumulator
    

def main():

    overwrite = True

    # load the config and the cache
    cfg = loadConfig()

    # Inputs are defined in a dictionary
    # dataset : list of files
    fileset = {
        'ZZ to 4mu': glob.glob("/hadoop/cms/store/user/dspitzba/nanoAOD/ZZTo4L_13TeV_powheg_pythia8__RunIIFall17NanoAODv6-PU2017_12Apr2018_Nano25Oct2019_new_pmx_102X_mc2017_realistic_v7-v1/16BA2B0C-731C-5244-A1B5-C1E3BAF23089.root")
                    + glob.glob("/hadoop/cms/store/user/dspitzba/nanoAOD/ZZTo4L_13TeV_powheg_pythia8__RunIIFall17NanoAODv6-PU2017_12Apr2018_Nano25Oct2019_new_pmx_102X_mc2017_realistic_v7-v1/A66DF383-FA73-824D-9CAA-4741AEEFF79A.root")
    }


   
    histograms = ['mass']

    # initialize cache
    cache = dir_archive(os.path.join(os.path.expandvars(cfg['caches']['base']), cfg['caches']['doublemuonProcessor']), serialized=True)
    if not overwrite:
        cache.load()

    if cfg == cache.get('cfg') and histograms == cache.get('histograms') and fileset == cache.get('fileset') and cache.get('doublemuon_output'):
        output = cache.get('doublemuon_output')

    else:
        # Run the processor
        output = processor.run_uproot_job(fileset,
                                      treename='Events',
                                      processor_instance=doublemuonProcessor(),
                                      executor=processor.futures_executor,
                                      executor_args={'workers': 12, 'function_args': {'flatten': False}},
                                      chunksize=500000,
                                     )
        cache['fileset']        = fileset
        cache['cfg']            = cfg
        cache['histograms']     = histograms
        cache['doublemuon_output']  = output
        cache.dump()

    # Make a few plots
    outdir = "./tmp_plots"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for name in histograms:
        print (name)
        histogram = output[name]
        if name == 'MET_pt':
            # rebin
            new_met_bins = hist.Bin('pt', r'$E_T^{miss} \ (GeV)$', 20, 0, 200)
            histogram = histogram.rebin('pt', new_met_bins)
        if name == 'W_pt_notFromTop':
            # rebin
            new_pt_bins = hist.Bin('pt', r'$p_{T}(W) \ (GeV)$', 25, 0, 500)
            histogram = histogram.rebin('pt', new_pt_bins)
        #if name == 'mass':
            #new_mass_bins = hist.Bin('mass', r'$m_{/mu/mu} \ (GeV)', 20, 0, 200)
            #histogram = histogram.rebin('mass', new_mass_bins)

        ax = hist.plot1d(histogram,overlay="dataset", density=False, stack=True) # make density plots because we don't care about x-sec differences
        ax.set_yscale('linear') # can be log
        #ax.set_ylim(0,0.1)
        ax.figure.savefig(os.path.join(outdir, "{}.pdf".format(name)))
        ax.clear()

        ax = hist.plot1d(histogram,overlay="dataset", density=True, stack=False) # make density plots because we don't care about x-sec differences
        ax.set_yscale('linear') # can be log
        #ax.set_ylim(0,0.1)
        ax.figure.savefig(os.path.join(outdir, "{}_shape.pdf".format(name)))
        ax.clear()

    return output

if __name__ == "__main__":
    output = main()
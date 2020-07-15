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


class btagProcessor(processor.ProcessorABC):
    def __init__(self):
        dataset_axis = hist.Cat("dataset", "Primary dataset")
        pt_axis = hist.Bin("pt",r"$p_{T}$ (GeV)", 20, 0, 200)
        mass_axis = hist.Bin("mass", r"m_{\mu\mu}$ (GeV)", 100, 0, 100)
        eta_axis = hist.Bin("eta", r"$\eta$", 60, -5.5, 5.5)

        self._accumulator = processor.dict_accumulator({
            #"MET_pt": hist.Hist("Counts", dataset_axis, pt_axis),
            "b_tagged_mass": hist.Hist("Counts", dataset_axis, mass_axis),
            "b_tagged_pt": hist.Hist("Counts", dataset_axis, pt_axis),
            "b_tagged_eta": hist.Hist("Counts", dataset_axis, eta_axis),
            "non_b_mass": hist.Hist("Counts", dataset_axis, mass_axis),
            "non_b_pt": hist.Hist("Counts", dataset_axis, pt_axis),
            "non_b_eta": hist.Hist("Counts", dataset_axis, eta_axis),
        })
    
    @property
    def accumulator(self):
        return self._accumulator 
    
    def process(self, df):
        output = self._accumulator.identity()
        dataset = df['dataset']
        
        Jet = awkward.JaggedArray.zip(
            pt = df['Jet_pt'],
            phi = df['Jet_phi'],
            eta = df['Jet_eta'],
            goodjet = df['Jet_isGoodJetAll'],
            goodbjet = df['Jet_isGoodBJet'],
            mass = df['Jet_mass'],
        )

        b_tagged_jet = Jet[Jet['goodbjet'] == 1]
        output['b_tagged_mass'].fill(
            dataset = dataset,
            mass = b_tagged_jet.mass.flatten(),
        )
        output['b_tagged_eta'].fill(
            dataset = dataset,
            eta = b_tagged_jet['eta'].flatten(),
        )
        output['b_tagged_pt'].fill(
            dataset = dataset,
            pt = b_tagged_jet['pt'].flatten(),
        )

        non_b_jet = Jet[(Jet['goodjet'] == 1) & (Jet['goodbjet'] == 0)]
        output['non_b_mass'].fill(
            dataset = dataset,
            mass = non_b_jet.mass.flatten(),
        )
        output['non_b_eta'].fill(
            dataset = dataset,
            eta = non_b_jet['eta'].flatten(),
        )
        output['non_b_pt'].fill(
            dataset = dataset,
            pt = non_b_jet['pt'].flatten(),
        )

        return output
    
    def postprocess(self, accumulator):
        return accumulator

def main():

    overwrite = True
    cfg = loadConfig()
    
    fileset = {
        'tW_scattering': glob.glob("/hadoop/cms/store/user/dspitzba/nanoAOD/ttw_samples/0p1p2/tW_scattering__nanoAOD/merged/*.root"),
        "TTW":           glob.glob("/hadoop/cms/store/user/dspitzba/nanoAOD/ttw_samples/0p1p2/TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8__RunIIAutumn18NanoAODv6-Nano25Oct2019_102X_upgrade2018_realistic_v20_ext1-v1/merged/*.root") \
                        + glob.glob("/hadoop/cms/store/user/dspitzba/nanoAOD/ttw_samples/0p1p2/TTWJetsToQQ_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8__RunIIAutumn18NanoAODv6-Nano25Oct2019_102X_upgrade2018_realistic_v20-v1/merged/*.root"),
        #"ttbar":        glob.glob("/hadoop/cms/store/user/dspitzba/nanoAOD/ttw_samples/0p1p2/TTJets_SingleLeptFromT_TuneCP5_13TeV-madgraphMLM-pythia8__RunIIAutumn18NanoAODv6-Nano25Oct2019_102X_upgrade2018_realistic_v20-v1/merged/*.root") # adding this is still surprisingly fast (20GB file!)
    }

    histograms = ["b_tagged_mass", "b_tagged_eta", "b_tagged_pt", "non_b_mass", "non_b_eta", "non_b_pt"]

    cache = dir_archive(os.path.join(os.path.expandvars(cfg['caches']['base']), cfg['caches']['btagProcessor']), serialized=True)
    
    if not overwrite:
        cache.load()
    
    if cfg == cache.get('cfg') and histograms == cache.get('histograms') and fileset == cache.get('fileset') and cache.get('btag_output'):
        output = cache.get('btag_output')
    
    else:
        output = processor.run_uproot_job(fileset,
                                      treename='Events',
                                      processor_instance=btagProcessor(),
                                      executor=processor.futures_executor,
                                      executor_args={'workers': 12, 'function_args': {'flatten': False}},
                                      chunksize=500000,
                                     )
        cache['fileset']        = fileset
        cache['cfg']            = cfg
        cache['histograms']     = histograms
        cache['btag_output']  = output
        cache.dump()

    # Make a few plots
    outdir = "./tmp_plots"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for name in histograms:
        print (name)
        histogram = output[name]

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
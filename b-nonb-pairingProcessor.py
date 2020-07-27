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
from coffea.analysis_objects import JaggedCandidateArray

matplotlib.use('Agg')

class b_nonb_pairingProcessor(processor.ProcessorABC):
    def __init__(self):
        dataset_axis = hist.Cat("dataset", 'Primary dataset')
        pt_axis = hist.Bin("pt", r"$p_{T}$ (GeV)", 600, 0, 1000)
        mass_axis = hist.Bin("mass", r"mass (GeV)", 50, 0, 40)
        eta_axis = hist.Bin("eta", r"$\eta$", 60, -5.5, 5.5)
        self._accumulator = processor.dict_accumulator({
            'b_nonb_massmax': hist.Hist('Counts', dataset_axis, mass_axis),
            'b_nonb_massmin': hist.Hist('Counts', dataset_axis, mass_axis),
            'b_b_nonb_massmax': hist.Hist('Counts', dataset_axis, mass_axis),
            'b_b_nonb_massmin': hist.Hist('Counts', dataset_axis, mass_axis),
            'jet_pair_massmax': hist.Hist('Counts', dataset_axis, mass_axis),
            'jet_pair_massmin': hist.Hist('Counts', dataset_axis, mass_axis),
            'lepton_pair_massmax': hist.Hist('Counts', dataset_axis, mass_axis),
            'lepton_pair_massmin': hist.Hist('Counts', dataset_axis, mass_axis),
        })
    @property
    def accumulator(self):
        return self._accumulator
    
    def process(self, df):
        output = self.accumulator.identity()
        dataset = df['dataset']
        Jet = JaggedCandidateArray.candidatesfromcounts(
            df['nJet'],
            pt = df['Jet_pt'].content,
            eta = df['Jet_eta'].content,
            phi = df['Jet_phi'].content,
            mass = df['Jet_mass'].content,
            goodjet = df['Jet_isGoodJetAll'].content,
            bjet = df['Jet_isGoodBJet'].content,
        )
        """Lepton = JaggedCandidateArray.candidatesfromcounts(
            df['nLepton'],
            pt = df['Lepton_pt'].content,
            eta = df['Lepton_eta'].content,
            phi = df['Lepton_phi'].content,
            pdgId = df['Lepton_pdgId'].content,
        )"""

        b = Jet[Jet['bjet']==1]
        nonb = Jet[(Jet['goodjet']==1) & (Jet['bjet']==0)]
        b_nonb_pair = b.cross(nonb)
        output['b_nonb_massmax'].fill(dataset=dataset, mass=b_nonb_pair.mass.argmax().flatten())
        output['b_nonb_massmin'].fill(dataset=dataset, mass=b_nonb_pair.mass.argmin().flatten())

        b_b_nonb_pair = b.cross(b).cross(nonb)
        output['b_b_nonb_massmax'].fill(dataset=dataset, mass=b_b_nonb_pair.mass.argmax().flatten())
        output['b_b_nonb_massmin'].fill(dataset=dataset, mass=b_b_nonb_pair.mass.argmin().flatten())

        goodjets = Jet[Jet['goodjet']==1]
        jet_pair = goodjets.choose(2)
        output['jet_pair_massmax'].fill(dataset=dataset, mass=jet_pair.mass.argmax().flatten())
        output['jet_pair_massmin'].fill(dataset=dataset, mass=jet_pair.mass.argmax().flatten())

        """lepton_pair = Lepton.choose(2)
        output['lepton_pair_massmax'].fill(dataset=dataset, mass=lepton_pair.mass.argmax().flatten())
        output['lepton_pair_massmin'].fill(dataset=dataset, mass=lepton_pair.mass.argmin().flatten())"""

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

    histograms = ['b_nonb_massmax', 'b_nonb_massmin', 'b_b_nonb_massmax', 'b_b_nonb_massmin','jet_pair_massmax', 'jet_pair_massmin']

    cache = dir_archive(os.path.join(os.path.expandvars(cfg['caches']['base']), cfg['caches']['b_nonb_pairingProcessor']), serialized=True)
    
    if not overwrite:
        cache.load()
    
    if cfg == cache.get('cfg') and histograms == cache.get('histograms') and fileset == cache.get('fileset') and cache.get('b_nonb_output'):
        output = cache.get('b_nonb_output')
    
    else:
        output = processor.run_uproot_job(fileset,
                                      treename='Events',
                                      processor_instance=b_nonb_pairingProcessor(),
                                      executor=processor.futures_executor,
                                      executor_args={'workers': 12, 'function_args': {'flatten': True}},
                                      chunksize=500000,
                                     )
        cache['fileset']        = fileset
        cache['cfg']            = cfg
        cache['histograms']     = histograms
        cache['b_nonb_output']  = output
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

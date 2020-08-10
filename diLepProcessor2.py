'''
Simple processor using coffea.
[x] weights
[ ] Missing pieces: appropriate sample handling
[x] Accumulator caching
'''


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
from coffea.analysis_objects import JaggedCandidateArray
from coffea import hist
import pandas as pd
import uproot_methods
import awkward


import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from Tools.helpers import *

# This just tells matplotlib not to open any
# interactive windows.
matplotlib.use('Agg')

class exampleProcessor(processor.ProcessorABC):
    """Dummy processor used to demonstrate the processor principle"""
    def __init__(self):

        # we can use a large number of bins and rebin later
        dataset_axis        = hist.Cat("dataset",   "Primary dataset")
        pt_axis             = hist.Bin("pt",        r"$p_{T}$ (GeV)", 600, 0, 1000)
        mass_axis           = hist.Bin("mass",      r"M (GeV)", 25, 0, 1500)
        eta_axis            = hist.Bin("eta",       r"$\eta$", 60, -5.5, 5.5)
        multiplicity_axis   = hist.Bin("multiplicity",         r"N", 20, -0.5, 19.5)
        phi_axis            = hist.Bin("phi",       r"$\phi", 60, -3.5, 3.5)

        self._accumulator = processor.dict_accumulator({
            "MET_pt" :          hist.Hist("Counts", dataset_axis, pt_axis),
            "pt_spec_max" :          hist.Hist("Counts", dataset_axis, pt_axis),
            "MT" :          hist.Hist("Counts", dataset_axis, pt_axis),
            "b_nonb_massmax" :          hist.Hist("Counts", dataset_axis, mass_axis),
            "N_b" :             hist.Hist("Counts", dataset_axis, multiplicity_axis),
            "N_jet" :           hist.Hist("Counts", dataset_axis, multiplicity_axis),
            "N_spec" :           hist.Hist("Counts", dataset_axis, multiplicity_axis),
            "b_b_nonb_massmax" :          hist.Hist("Counts", dataset_axis, mass_axis),
            "jet_pair_massmax" :          hist.Hist("Counts", dataset_axis, mass_axis),
            "lepton_jet_pair_massmax" :          hist.Hist("Counts", dataset_axis, mass_axis),
            "lepton_bjet_pair_massmax" :          hist.Hist("Counts", dataset_axis, mass_axis),
            'S_T': hist.Hist('Counts', dataset_axis, pt_axis),
            'H_T': hist.Hist('Counts', dataset_axis, pt_axis),
            'b_pt': hist.Hist('Counts', dataset_axis, pt_axis),
            'b_phi': hist.Hist('Counts', dataset_axis, phi_axis),
            'b_eta': hist.Hist('Counts', dataset_axis, eta_axis),
            'leading_nonb_pt': hist.Hist('Counts', dataset_axis, pt_axis),
            'leading_nonb_phi': hist.Hist('Counts', dataset_axis, phi_axis),
            'leading_nonb_eta': hist.Hist('Counts', dataset_axis, eta_axis),
            'lepton_pt': hist.Hist('Counts', dataset_axis, pt_axis),
            'lepton_phi': hist.Hist('Counts', dataset_axis, phi_axis),
            'lepton_eta': hist.Hist('Counts', dataset_axis, eta_axis),
            'lepton_deltaphi': hist.Hist('Counts', dataset_axis, phi_axis),
            'lepton_deltaeta': hist.Hist('Counts', dataset_axis, eta_axis),
            'cutflow_wjets':      processor.defaultdict_accumulator(int),
            'cutflow_ttbar':      processor.defaultdict_accumulator(int),
            'cutflow_TTW':      processor.defaultdict_accumulator(int),
            'cutflow_TTX':      processor.defaultdict_accumulator(int),
            'cutflow_signal':   processor.defaultdict_accumulator(int),
        })

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, df):
        """
        Processing function. This is where the actual analysis happens.
        """
        output = self.accumulator.identity()
        dataset = df["dataset"]
        cfg = loadConfig()
        # We can access the data frame as usual
        # The dataset is written into the data frame
        # outside of this function

        output['cutflow_wjets']['all events'] += sum(df['weight'][(df['dataset']=='wjets')].flatten())
        output['cutflow_ttbar']['all events'] += sum(df['weight'][(df['dataset']=='ttbar')].flatten())
        output['cutflow_TTW']['all events'] += sum(df['weight'][(df['dataset']=='TTW')].flatten())
        output['cutflow_TTX']['all events'] += sum(df['weight'][(df['dataset']=='TTX')].flatten())
        output['cutflow_signal']['all events'] += sum(df['weight'][(df['dataset']=='tW_scattering')].flatten())

        cutFlow = ((df['nLepton']==2) & (df['nVetoLepton']==2))

        output['cutflow_wjets']['singleLep']  += sum(df['weight'][(df['dataset']=='wjets')         & cutFlow].flatten())
        output['cutflow_ttbar']['singleLep']  += sum(df['weight'][(df['dataset']=='ttbar')         & cutFlow].flatten())
        output['cutflow_TTW']['singleLep']    += sum(df['weight'][(df['dataset']=='TTW')           & cutFlow].flatten())
        output['cutflow_TTX']['singleLep']    += sum(df['weight'][(df['dataset']=='TTX')           & cutFlow].flatten())
        output['cutflow_signal']['singleLep'] += sum(df['weight'][(df['dataset']=='tW_scattering') & cutFlow].flatten())

        cutFlow = ((df['nLepton']==2) & (df['nVetoLepton']==2) & (df['nGoodJet']>3))

        output['cutflow_wjets']['5jets']  += sum(df['weight'][(df['dataset']=='wjets')         & cutFlow].flatten())
        output['cutflow_ttbar']['5jets']  += sum(df['weight'][(df['dataset']=='ttbar')         & cutFlow].flatten())
        output['cutflow_TTW']['5jets']    += sum(df['weight'][(df['dataset']=='TTW')           & cutFlow].flatten())
        output['cutflow_TTX']['5jets']    += sum(df['weight'][(df['dataset']=='TTX')           & cutFlow].flatten())
        output['cutflow_signal']['5jets'] += sum(df['weight'][(df['dataset']=='tW_scattering') & cutFlow].flatten())

        cutFlow = ((df['nLepton']==2) & (df['nVetoLepton']==2) & (df['nGoodJet']>3) & (df['nGoodBTag']==2))

        output['cutflow_wjets']['btags']  += sum(df['weight'][(df['dataset']=='wjets')         & cutFlow].flatten())
        output['cutflow_ttbar']['btags']  += sum(df['weight'][(df['dataset']=='ttbar')         & cutFlow].flatten())
        output['cutflow_TTW']['btags']    += sum(df['weight'][(df['dataset']=='TTW')           & cutFlow].flatten())
        output['cutflow_TTX']['btags']    += sum(df['weight'][(df['dataset']=='TTX')           & cutFlow].flatten())
        output['cutflow_signal']['btags'] += sum(df['weight'][(df['dataset']=='tW_scattering') & cutFlow].flatten())

        # preselection of events
        selection = ((df['nLepton']==2) & (df['nVetoLepton']==2)) & (df['isSS']==1) & (df['nGoodBTag']==2) & (df['nGoodJet']>=3)
        #df = df[((df['nLepton']==1) & (df['nGoodJet']>5) & (df['nGoodBTag']==2))]

        # And fill the histograms
        output['MET_pt'].fill(dataset=dataset, pt=df["MET_pt"][selection].flatten(), weight=df['weight'][selection]*cfg['lumi'])
        output['MT'].fill(dataset=dataset, pt=df["MT"][selection].flatten(), weight=df['weight'][selection]*cfg['lumi'])
        output['N_b'].fill(dataset=dataset, multiplicity=df["nGoodBTag"][selection], weight=df['weight'][selection]*cfg['lumi'] )
        output['N_jet'].fill(dataset=dataset, multiplicity=df["nGoodJet"][selection], weight=df['weight'][selection]*cfg['lumi'] )

        Jet = JaggedCandidateArray.candidatesfromcounts(
            df['nJet'],
            pt = df['Jet_pt'].content,
            eta = df['Jet_eta'].content,
            phi = df['Jet_phi'].content,
            mass = df['Jet_mass'].content,
            goodjet = df['Jet_isGoodJetAll'].content,
            bjet = df['Jet_isGoodBJet'].content,
            jetId = df['Jet_jetId'].content,
            puId = df['Jet_puId'].content,
        )
 
        Lepton = JaggedCandidateArray.candidatesfromcounts(
            df['nLepton'],
            pt = df['Lepton_pt'].content,
            eta = df['Lepton_eta'].content,
            phi = df['Lepton_phi'].content,
            mass = df['Lepton_mass'].content,
            pdgId = df['Lepton_pdgId'].content,
        )
        
        b = Jet[Jet['bjet']==1]
        nonb = Jet[(Jet['goodjet']==1) & (Jet['bjet']==0)]
        spectator = Jet[(abs(Jet.eta)>2.0) & (abs(Jet.eta)<4.7) & (Jet.pt>25) & (Jet['puId']>=7) & (Jet['jetId']>=6)] # 40 GeV seemed good. let's try going lower

        b_nonb_selection = (Jet.counts>3) & (b.counts==2) & (nonb.counts>=2) & (df['nLepton']==2) & (df['nVetoLepton']==2) & (df['isSS']==1)
        b_nonb_pair = b.cross(nonb)
        output['b_nonb_massmax'].fill(dataset=dataset, mass=b_nonb_pair[b_nonb_selection].mass.max().flatten(), weight=df['weight'][b_nonb_selection]*cfg['lumi'])
        output['N_spec'].fill(dataset=dataset, multiplicity=spectator[b_nonb_selection].counts, weight=df['weight'][b_nonb_selection]*cfg['lumi'])
        output['pt_spec_max'].fill(dataset=dataset, pt=spectator[b_nonb_selection & (spectator.counts>0)].pt.max().flatten(), weight=df['weight'][b_nonb_selection & (spectator.counts>0)]*cfg['lumi'])

        b_b_nonb_pair = b.cross(b.cross(nonb))
        output['b_b_nonb_massmax'].fill(dataset=dataset, mass=b_b_nonb_pair[b_nonb_selection].mass.max().flatten(), weight=df['weight'][b_nonb_selection]*cfg['lumi'])

        goodjets = Jet[Jet['goodjet']==1]
        jet_pair = goodjets.choose(2)
        output['jet_pair_massmax'].fill(dataset=dataset, mass=jet_pair[b_nonb_selection].mass.max().flatten(), weight=df['weight'][b_nonb_selection]*cfg['lumi'])

        lepton_jet_pair = Lepton.cross(goodjets)
        output['lepton_jet_pair_massmax'].fill(dataset=dataset, mass=lepton_jet_pair[b_nonb_selection].mass.max().flatten(), weight=df['weight'][b_nonb_selection]*cfg['lumi'])

        lepton_bjet_pair = Lepton.cross(b)
        output['lepton_bjet_pair_massmax'].fill(dataset=dataset, mass=lepton_bjet_pair[b_nonb_selection].mass.max().flatten(), weight=df['weight'][b_nonb_selection]*cfg['lumi'])

        sum_all_pt = (df['Jet_pt'][b_nonb_selection].sum())+(df['MET_pt'][b_nonb_selection].sum())+(df['Lepton_pt'][b_nonb_selection].sum())
        output['S_T'].fill(dataset=dataset, pt=sum_all_pt.flatten(), weight=df['weight'][b_nonb_selection]*cfg['lumi'])

        output['H_T'].fill(dataset=dataset, pt=df['Jet_pt'][b_nonb_selection].sum().flatten(), weight=df['weight'][b_nonb_selection]*cfg['lumi'])
        
        bs = b[b_nonb_selection].choose(2)
        output['b_eta'].fill(dataset=dataset, eta=bs.eta.flatten(), weight=df['weight'][b_nonb_selection]*cfg['lumi'])
        output['b_phi'].fill(dataset=dataset, phi=bs.phi.flatten(), weight=df['weight'][b_nonb_selection]*cfg['lumi'])
        output['b_pt'].fill(dataset=dataset, pt=bs.pt.flatten(), weight=df['weight'][b_nonb_selection]*cfg['lumi'])

        leading_nonb = Jet[(b_nonb_selection) & (Jet['goodjet']==1) & (Jet['bjet']==0)].pt.argmax()
        output['leading_nonb_eta'].fill(dataset=dataset, eta=nonb[leading_nonb].eta.flatten(), weight=df['weight'][b_nonb_selection]*cfg['lumi'])
        output['leading_nonb_phi'].fill(dataset=dataset, phi=nonb[leading_nonb].phi.flatten(), weight=df['weight'][b_nonb_selection]*cfg['lumi'])
        output['leading_nonb_pt'].fill(dataset=dataset, pt=nonb.pt[leading_nonb].flatten(), weight=df['weight'][b_nonb_selection]*cfg['lumi'])

        lep = Lepton[b_nonb_selection].choose(2)
        output['lepton_eta'].fill(dataset=dataset, eta=lep.eta.flatten(), weight=df['weight'][b_nonb_selection]*cfg['lumi'])
        output['lepton_phi'].fill(dataset=dataset, phi=lep.phi.flatten(), weight=df['weight'][b_nonb_selection]*cfg['lumi'])
        output['lepton_pt'].fill(dataset=dataset, pt=lep.pt.flatten(), weight=df['weight'][b_nonb_selection]*cfg['lumi'])

        deltaphi = (lep.i0.phi - lep.i1.phi + 3.141592653589793) % (2 * 3.141592653589793) - 3.141592653589793
        deltaeta = abs(lep.i0.eta - lep.i1.eta)
        output['lepton_deltaphi'].fill(dataset=dataset, phi=deltaphi.flatten(), weight=df['weight'][b_nonb_selection]*cfg['lumi'])
        output['lepton_deltaeta'].fill(dataset=dataset, eta=deltaeta.flatten(), weight=df['weight'][b_nonb_selection]*cfg['lumi'])

        return output

    def postprocess(self, accumulator):
        return accumulator


def main():

    overwrite = True

    # load the config and the cache
    cfg = loadConfig()

    # Inputs are defined in a dictionary
    # dataset : list of files
    from processor.samples import fileset, fileset_small

    # histograms
    histograms = ["MET_pt", "N_b", "N_jet", "MT", "b_nonb_massmax", "N_spec", "pt_spec_max", "b_b_nonb_massmax", "jet_pair_massmax", "lepton_jet_pair_massmax", "lepton_bjet_pair_massmax"]
    histograms += ['S_T', 'H_T', 'b_eta', 'b_phi', 'b_pt', 'leading_nonb_eta', 'leading_nonb_phi', 'leading_nonb_pt', 'lepton_eta', 'lepton_phi', 'lepton_pt', 'lepton_deltaphi', 'lepton_deltaeta']


    # initialize cache
    cache = dir_archive(os.path.join(os.path.expandvars(cfg['caches']['base']), cfg['caches']['singleLep']), serialized=True)
    if not overwrite:
        cache.load()

    if cfg == cache.get('cfg') and histograms == cache.get('histograms') and fileset == cache.get('fileset_small') and cache.get('simple_output'):
        output = cache.get('simple_output')

    else:
        # Run the processor
        output = processor.run_uproot_job(fileset,
                                      treename='Events',
                                      processor_instance=exampleProcessor(),
                                      executor=processor.futures_executor,
                                      executor_args={'workers': 12, 'function_args': {'flatten': False}},
                                      chunksize=100000,
                                     )
        cache['fileset']        = fileset
        cache['cfg']            = cfg
        cache['histograms']     = histograms
        cache['simple_output']  = output
        cache.dump()

    # Make a few plots
    outdir = "./tmp_plots"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    return output

if __name__ == "__main__":
    output = main()

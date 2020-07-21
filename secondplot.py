from coffea import hist
import pandas as pd
#import uproot_methods

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from tW_scattering.Tools.helpers import *
from klepto.archives import dir_archive

def saveFig( ax, path, name, scale='linear' ):
    outdir = os.path.join(path,scale)
    finalizePlotDir(outdir)
    ax.set_yscale(scale)
    if scale == 'log':
        ax.set_ylim(0.001,1)
    ax.figure.savefig(os.path.join(outdir, "{}.pdf".format(name)))
    ax.figure.savefig(os.path.join(outdir, "{}.png".format(name)))
    #ax.clear()

# load the configuration
cfg = loadConfig()

# load the results
cache = dir_archive(os.path.join(os.path.expandvars(cfg['caches']['base']), cfg['caches']['doublemuonProcessor']), serialized=True)
cache.load()

histograms = cache.get('histograms')
output = cache.get('doublemuon_output')
plotDir = os.path.expandvars(cfg['meta']['plots']) + '/doublemuonPlots2/'
finalizePlotDir(plotDir)

if not histograms:
    print ("Couldn't find histograms in archive. Quitting.")
    exit()

print ("Plots will appear here:", plotDir )

for name in histograms:
    print (name)
    skip = False
    histogram = output[name]
    if name == 'MET_pt':
        # rebin
        new_met_bins = hist.Bin('pt', r'$E_T^{miss} \ (GeV)$', 20, 0, 200)
        histogram = histogram.rebin('pt', new_met_bins)
    elif name == 'W_pt_notFromTop':
        # rebin
        new_pt_bins = hist.Bin('pt', r'$p_{T}(W) \ (GeV)$', 25, 0, 500)
        histogram = histogram.rebin('pt', new_pt_bins)
    elif name == 'GenJet_pt_fwd':
        # rebin
        new_pt_bins = hist.Bin('pt', r'$p_{T} \ (GeV)$', 25, 0, 500)
        histogram = histogram.rebin('pt', new_pt_bins)
    elif name == 'Spectator_pt':
        # rebin
        new_pt_bins = hist.Bin('pt', r'$p_{T} \ (GeV)$', 25, 0, 500)
        histogram = histogram.rebin('pt', new_pt_bins)
    elif name == 'Spectator_eta':
        # rebin
        new_eta_bins = hist.Bin('eta', r'$\eta$', 30, -5.5, 5.5)
        histogram = histogram.rebin('eta', new_eta_bins)
    elif name == 'W_pt':
        # rebin
        new_pt_bins = hist.Bin('pt', r'$p_{T} \ (GeV)$', 25, 0, 500)
        histogram = histogram.rebin('pt', new_pt_bins)
    elif name == 'W_eta':
        # rebin
        new_eta_bins = hist.Bin('eta', r'$\eta$', 30, -5.5, 5.5)
        histogram = histogram.rebin('eta', new_eta_bins)
    elif name == 'Antitop_pt':
        # rebin
        new_pt_bins = hist.Bin('pt', r'$p_{T} \ (GeV)$', 25, 0, 500)
        histogram = histogram.rebin('pt', new_pt_bins)
    elif name == 'Antitop_eta':
        # rebin
        new_eta_bins = hist.Bin('eta', r'$\eta$', 30, -5.5, 5.5)
        histogram = histogram.rebin('eta', new_eta_bins)
    elif name == 'Top_pt':
        # rebin
        new_pt_bins = hist.Bin('pt', r'$p_{T} \ (GeV)$', 25, 0, 500)
        histogram = histogram.rebin('pt', new_pt_bins)
    elif name == 'Top_eta':
        # rebin
        new_eta_bins = hist.Bin('eta', r'$\eta$', 30, -5.5, 5.5)
        histogram = histogram.rebin('eta', new_eta_bins)
    elif name == 'mass':
        new_mass_bin = hist.Bin("mass", r"m_{\mu\mu}$[GeV]", 100, 0, 150)
        histogram = histogram.rebin('mass', new_mass_bin)
    else:
        skip = True

    if not skip:
        ax = hist.plot1d(histogram,overlay="dataset", stack=False) # make density plots because we don't care about x-sec differences
        for l in ['linear', 'log']:
            saveFig(ax, plotDir, name, scale=l)
        ax.clear()

        ax = hist.plot1d(histogram,overlay="dataset", density=True, stack=False) # make density plots because we don't care about x-sec differences
        for l in ['linear', 'log']:
            saveFig(ax, plotDir, name+'_shape', scale=l)
        ax.clear()
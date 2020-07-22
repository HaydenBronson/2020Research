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
cache = dir_archive(os.path.join(os.path.expandvars(cfg['caches']['base']), cfg['caches']['b_nonb_pairingProcessor']), serialized=True)
cache.load()

histograms = cache.get('histograms')
output = cache.get('b_nonb_output')
plotDir = os.path.expandvars(cfg['meta']['plots']) + '/b_nonbPlots/'
finalizePlotDir(plotDir)

if not histograms:
    print ("Couldn't find histograms in archive. Quitting.")
    exit()

print ("Plots will appear here:", plotDir )

for name in histograms:
    print (name)
    skip = False
    histogram = output[name]
    if name == 'b_nonb_mass':
        # rebin
        new_mass_bins = hist.Bin('mass', r"mass (GeV)", 100, 0, 100)
        histogram = histogram.rebin('mass', new_mass_bins)
    elif name == 'b_nonb_eta':
        # rebin
        new_eta_bins = hist.Bin("eta", r"$\eta$", 60, -5.5, 5.5)
        histogram = histogram.rebin('eta', new_eta_bins)
    elif name == 'b_nonb_pt':
        # rebin
        new_pt_bins = hist.Bin('pt', r"$p_{T}$ (GeV)", 600, 0, 1000)
        histogram = histogram.rebin('pt', new_pt_bins)
    '''elif name == 'non_b_pt':
        # rebin
        new_pt_bins = hist.Bin('pt', r'$E_T^{miss} \ (GeV)$', 20, 0, 200)
        histogram = histogram.rebin('pt', new_pt_bins)
    elif name == 'non_b_eta':
        # rebin
        new_eta_bins = hist.Bin("eta", r"$\eta$", 60, -5.5, 5.5)
        histogram = histogram.rebin('eta', new_eta_bins)
    elif name == 'non_b_mass':
        # rebin
        new_mass_bins = hist.Bin("mass", r"m_{\mu\mu}$ (GeV)", 100, 0, 100)
        histogram = histogram.rebin('mass', new_mass_bins)'''


    if not skip:
        ax = hist.plot1d(histogram,overlay="dataset", stack=True) # make density plots because we don't care about x-sec differences
        for l in ['linear', 'log']:
            saveFig(ax, plotDir, name, scale=l)
        ax.clear()

        ax = hist.plot1d(histogram,overlay="dataset", density=True, stack=False) # make density plots because we don't care about x-sec differences
        for l in ['linear', 'log']:
            saveFig(ax, plotDir, name+'_shape', scale=l)
        ax.clear()
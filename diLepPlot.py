'''
small script that reades histograms from an archive and saves figures in a public space
ToDo:
[x] Cosmetics (labels etc)
[ ] ratio pad!
  [ ] pseudo data
[ ] uncertainty band
'''


from coffea import hist
import pandas as pd
#import uproot_methods

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from tW_scattering.Tools.helpers import *
from klepto.archives import dir_archive

def saveFig( ax, path, name, scale='linear', shape=False ):
    outdir = os.path.join(path,scale)
    finalizePlotDir(outdir)
    ax.set_yscale(scale)
    y_max = 0.5 if shape else 50000000
    if scale == 'log':
        ax.set_ylim(0.05, y_max)
    else:
        ax.set_ylim(0.0, y_max)

    handles, labels = ax.get_legend_handles_labels()
    new_labels = []
    for handle, label in zip(handles, labels):
        try:
            handle.set_color(colors[label])
            new_labels.append(my_labels[label])
        except:
            pass

    ax.legend(title='',ncol=1,handles=handles, labels=new_labels, frameon=False)

    ax.text(0., 0.995, '$\\bf{CMS}$', fontsize=20,  horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes )
    ax.text(0.15, 1., '$\\it{Simulation}$', fontsize=14, horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes )
    ax.text(0.8, 1., '13 TeV', fontsize=14, horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes )

    ax.figure.savefig(os.path.join(outdir, "{}.pdf".format(name)))
    ax.figure.savefig(os.path.join(outdir, "{}.png".format(name)))
    #ax.clear()

colors = {
    'tW_scattering': '#FF595E',
    'TTW': '#8AC926',
    'TTX': '#FFCA3A',
    'ttbar': '#1982C4',
    'wjets': '#6A4C93',
}
'''
other colors (sets from coolers.com):
#525B76 (gray)
#34623F (hunter green)
#0F7173 (Skobeloff)
'''

my_labels = {
    'tW_scattering': 'tW scattering',
    'TTW': r'$t\bar{t}$W+jets',
    'TTX': r'$t\bar{t}$Z/H',
    'ttbar': r'$t\bar{t}$+jets',
    'wjets': 'W+jets',
}

# load the configuration
cfg = loadConfig()

# load the results
cache = dir_archive(os.path.join(os.path.expandvars(cfg['caches']['base']), cfg['caches']['diLep']), serialized=True)
cache.load()

histograms = cache.get('histograms')
output = cache.get('simple_output')
plotDir = os.path.expandvars(cfg['meta']['plots']) + '/plots2l/'
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
    elif name == 'MT':
        # rebin
        new_met_bins = hist.Bin('pt', r'$M_T \ (GeV)$', 20, 0, 200)
        histogram = histogram.rebin('pt', new_met_bins)
    elif name == 'N_jet':
        # rebin
        new_n_bins = hist.Bin('multiplicity', r'$N_{jet}$', 15, -0.5, 14.5)
        histogram = histogram.rebin('multiplicity', new_n_bins)
    elif name == 'N_b':
        # rebin
        new_n_bins = hist.Bin('multiplicity', r'$N_{b-jet}$', 5, -0.5, 4.5)
        histogram = histogram.rebin('multiplicity', new_n_bins)
    elif name == 'b_nonb_massmax':
        # rebin
        new_mass_bins = hist.Bin('mass', r'$M(b, light) \ (GeV)$', 25, 0, 1500)
        histogram = histogram.rebin('mass', new_mass_bins)
    elif name == 'bb__nonb_massmax':
        # rebin
        new_mass_bins = hist.Bin('mass', r'$M(2b, light) \ (GeV)$', 25, 0, 1500)
        histogram = histogram.rebin('mass', new_mass_bins)
    elif name == 'jet_pair_massmax':
        # rebin
        new_mass_bins = hist.Bin('mass', r'$M(jet pair) \ (GeV)$', 25, 0, 1500)
        histogram = histogram.rebin('mass', new_mass_bins)
    elif name == 'lepton_jet_pair_massmax':
        # rebin
        new_mass_bins = hist.Bin('mass', r'$M(lepton+jet) \ (GeV)$', 25, 0, 1500)
        histogram = histogram.rebin('mass', new_mass_bins)
    else:
        skip = True
    if not skip:
        ax = hist.plot1d(histogram,overlay="dataset", stack=True, overflow='over', order=['tW_scattering', 'TTX', 'TTW','ttbar','wjets'])
        for l in ['linear', 'log']:
            saveFig(ax, plotDir, name, scale=l, shape=False)
        ax.clear()

        ax = hist.plot1d(histogram,overlay="dataset", density=True, stack=False, overflow='over') # make density plots because we don't care about x-sec differences
        for l in ['linear', 'log']:
            saveFig(ax, plotDir, name+'_shape', scale=l, shape=True)
        ax.clear()
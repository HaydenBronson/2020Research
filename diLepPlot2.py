'''
small script that reades histograms from an archive and saves figures in a public space
ToDo:
[x] Cosmetics (labels etc)
[x] ratio pad!
  [x] pseudo data
    [ ] -> move to processor to avoid drawing toys every time!
[x] uncertainty band
[ ] fix shapes
'''


from coffea import hist
import pandas as pd
#import uproot_methods

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from Tools.helpers import *
from klepto.archives import dir_archive

def saveFig( fig, ax, rax, path, name, scale='linear', shape=False, y_max=-1 ):
    outdir = os.path.join(path,scale)
    finalizePlotDir(outdir)
    ax.set_yscale(scale)
    ax.set_ylabel('Events')

    y_min = 0.0005 if shape else 0.05
    if y_max>0:
        y_max = 0.1 if shape else 300*y_max
    else:
        y_max = 3e7
    if scale == 'log':
        ax.set_ylim(y_min, y_max)
        #if shape:
        #     ax.yaxis.set_ticks(np.array([10e-4,10e-3,10e-2,10e-1,10e0]))
        #else:
        #    ax.yaxis.set_ticks(np.array([10e-2,10e-1,10e0,10e1,10e2,10e3,10e4,10e5,10e6]))
    else:
        ax.set_ylim(0.0, y_max)


    handles, labels = ax.get_legend_handles_labels()
    new_labels = []
    for handle, label in zip(handles, labels):
        #print (handle, label)
        try:
            new_labels.append(my_labels[label])
            if not label=='pseudodata':
                handle.set_color(colors[label])
        except:
            pass

    if rax:
        plt.subplots_adjust(hspace=0)
        rax.set_ylabel('Obs./Pred.')
        rax.set_ylim(0.5,1.5)

    ax.legend(title='',ncol=2,handles=handles, labels=new_labels, frameon=False)

    fig.text(0., 0.995, '$\\bf{CMS}$', fontsize=20,  horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes )
    fig.text(0.15, 1., '$\\it{Simulation}$', fontsize=14, horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes )
    fig.text(0.8, 1., '13 TeV', fontsize=14, horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes )

    fig.savefig(os.path.join(outdir, "{}.pdf".format(name)))
    fig.savefig(os.path.join(outdir, "{}.png".format(name)))
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
    'pseudodata': 'Pseudo-data',
    'uncertainty': 'Uncertainty',
}

data_err_opts = {
    'linestyle': 'none',
    'marker': '.',
    'markersize': 10.,
    'color': 'k',
    'elinewidth': 1,
}

error_opts = {
    'label': 'uncertainty',
    'hatch': '///',
    'facecolor': 'none',
    'edgecolor': (0,0,0,.5),
    'linewidth': 0
}

fill_opts = {
    'edgecolor': (0,0,0,0.3),
    'alpha': 1.0
}

# load the configuration
cfg = loadConfig()

# load the results
cache = dir_archive(os.path.join(os.path.expandvars(cfg['caches']['base']), cfg['caches']['singleLep']), serialized=True)
cache.load()

histograms = cache.get('histograms')
output = cache.get('simple_output')
plotDir = os.path.expandvars(cfg['meta']['plots']) + '/dilepton_new/'
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
        axis = 'pt'
        new_met_bins = hist.Bin('pt', r'$E_T^{miss} \ (GeV)$', 20, 0, 200)
        histogram = histogram.rebin('pt', new_met_bins)
    elif name == 'MT':
        # rebin
        axis = 'pt'
        new_met_bins = hist.Bin('pt', r'$M_T \ (GeV)$', 20, 0, 200)
        histogram = histogram.rebin('pt', new_met_bins)
    elif name == 'N_jet':
        # rebin
        axis = 'multiplicity'
        new_n_bins = hist.Bin('multiplicity', r'$N_{jet}$', 15, -0.5, 14.5)
        histogram = histogram.rebin('multiplicity', new_n_bins)
    elif name == 'N_spec':
        # rebin
        axis = 'multiplicity'
        new_n_bins = hist.Bin('multiplicity', r'$N_{jet, fwd}$', 15, -0.5, 14.5)
        histogram = histogram.rebin('multiplicity', new_n_bins)
    elif name == 'pt_spec_max':
        # rebin
        axis = 'pt'
        new_pt_bins = hist.Bin('pt', r'$p_{T, fwd jet}\ (GeV)$', 20, 0, 400)
        histogram = histogram.rebin('pt', new_pt_bins)
    elif name == 'N_b':
        # rebin
        axis = 'multiplicity'
        new_n_bins = hist.Bin('multiplicity', r'$N_{b-jet}$', 5, -0.5, 4.5)
        histogram = histogram.rebin('multiplicity', new_n_bins)
    elif name == 'b_nonb_massmax':
        # rebin
        axis = 'mass'
        new_mass_bins = hist.Bin('mass', r'$M(b, light) \ (GeV)$', 25, 0, 1500)
        histogram = histogram.rebin('mass', new_mass_bins)
    elif name == 'b_b_nonb_massmax':
        # rebin
        axis = 'mass'
        new_mass_bins = hist.Bin('mass', r'$M(2b, light) \ (GeV)$', 25, 0, 1500)
        histogram = histogram.rebin('mass', new_mass_bins)
    elif name == 'jet_pair_massmax':
        # rebin
        axis = 'mass'
        new_mass_bins = hist.Bin('mass', r'$M(jets) \ (GeV)$', 25, 0, 1500)
        histogram = histogram.rebin('mass', new_mass_bins)
    elif name == 'lepton_jet_pair_massmax':
        # rebin
        axis = 'mass'
        new_mass_bins = hist.Bin('mass', r'$M(lepton+jet) \ (GeV)$', 25, 0, 1500)
        histogram = histogram.rebin('mass', new_mass_bins)
    elif name == 'lepton_bjet_pair_massmax':
        # rebin
        axis = 'mass'
        new_mass_bins = hist.Bin('mass', r'$M(lepton+bjet) \ (GeV)$', 25, 0, 1500)
        histogram = histogram.rebin('mass', new_mass_bins)
    elif name == 'S_T':
        # rebin
        axis = 'pt'
        new_pt_bins = hist.Bin("pt", r"$Sum all p_{T}$ (GeV)", 600, 0, 1000)
        histogram = histogram.rebin('pt', new_pt_bins)
    elif name == 'H_T':
        # rebin
        axis = 'pt'
        new_pt_bins = hist.Bin("pt", r"$Sum jet p_{T}$ (GeV)", 600, 0, 1000)
        histogram = histogram.rebin('pt', new_pt_bins)
    elif name == 'b_eta':
        # rebin
        axis = 'eta'
        new_eta_bins = hist.Bin("eta", r"$b \eta$", 60, -5.5, 5.5)
        histogram = histogram.rebin('eta', new_eta_bins)
    elif name == 'b_pt':
        # rebin
        axis = 'pt'
        new_pt_bins = hist.Bin("pt", r"$b p_{T}$ (GeV)", 600, 0, 1000)
        histogram = histogram.rebin('pt', new_pt_bins)
    elif name == 'b_phi':
        # rebin
        axis = 'phi'
        new_phi_bins = hist.Bin("phi", r"$b \phi$", 60, -3.5, 3.5)
        histogram = histogram.rebin('phi', new_phi_bins)
    elif name == 'lepton_eta':
        # rebin
        axis = 'eta'
        new_eta_bins = hist.Bin("eta", r"$lepton \eta$", 60, -5.5, 5.5)
        histogram = histogram.rebin('eta', new_eta_bins)
    elif name == 'lepton_pt':
        # rebin
        axis = 'pt'
        new_pt_bins = hist.Bin("pt", r"$lepton p_{T}$ (GeV)", 600, 0, 1000)
        histogram = histogram.rebin('pt', new_pt_bins)
    elif name == 'lepton_phi':
        # rebin
        axis = 'phi'
        new_phi_bins = hist.Bin("phi", r"$lepton \phi$", 60, -3.5, 3.5)
        histogram = histogram.rebin('phi', new_phi_bins)
    elif name == 'leading_nonb_eta':
        # rebin
        axis = 'eta'
        new_eta_bins = hist.Bin("eta", r"$leadingnonb \eta$", 60, -5.5, 5.5)
        histogram = histogram.rebin('eta', new_eta_bins)
    elif name == 'leading_nonb_pt':
        # rebin
        axis = 'pt'
        new_pt_bins = hist.Bin("pt", r"$leadingnonb p_{T}$ (GeV)", 600, 0, 1000)
        histogram = histogram.rebin('pt', new_pt_bins)
    elif name == 'leading_nonb_phi':
        # rebin
        axis = 'phi'
        new_phi_bins = hist.Bin("phi", r"$leadingnonb \phi$", 60, -3.5, 3.5)
        histogram = histogram.rebin('phi', new_phi_bins)
    elif name == 'lepton_deltaphi':
        # rebin
        axis = 'phi'
        new_phi_bins = hist.Bin("phi", r"$lepton delta\phi$", 60, -3.5, 3.5)
        histogram = histogram.rebin('phi', new_phi_bins)
    elif name == 'lepton_deltaeta':
        # rebin
        axis = 'eta'
        new_eta_bins = hist.Bin("eta", r"$lepton delta\eta$", 60, -5.5, 5.5)
        histogram = histogram.rebin('eta', new_eta_bins)
    else:
        skip = True

    if not skip:
        y_max = histogram.sum("dataset").values(overflow='over')[()].max()
        y_over = histogram.sum("dataset").values(overflow='over')[()][-1]

        # get pseudo data
        bin_values = histogram.axis(axis).centers(overflow='over')
        poisson_means = histogram.sum('dataset').values(overflow='over')[()]
        values = np.repeat(bin_values, np.random.poisson(np.maximum(np.zeros(len(poisson_means)), poisson_means)))
        if axis == 'pt':
            histogram.fill(dataset='pseudodata', pt=values)
        elif axis == 'mass':
            histogram.fill(dataset='pseudodata', mass=values)
        elif axis == 'multiplicity':
            histogram.fill(dataset='pseudodata', multiplicity=values)

        
        import re
        notdata = re.compile('(?!pseudodata)')

        fig, (ax, rax) = plt.subplots(2, 1, figsize=(7,7), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)

        # get axes
        hist.plot1d(histogram[notdata],overlay="dataset", ax=ax, stack=True, overflow='over', clear=False, line_opts=None, fill_opts=fill_opts, error_opts=error_opts, order=['tW_scattering', 'TTX', 'TTW','ttbar','wjets']) #error_opts??
        hist.plot1d(histogram['pseudodata'], overlay="dataset", ax=ax, overflow='over', error_opts=data_err_opts, clear=False)

        # build ratio
        hist.plotratio(
            num=histogram['pseudodata'].sum("dataset"),
            denom=histogram[notdata].sum("dataset"),
            ax=rax,
            error_opts=data_err_opts,
            denom_fill_opts={},
            guide_opts={},
            unc='num',
            overflow='over'
        )


        for l in ['linear', 'log']:
            saveFig(fig, ax, rax, plotDir, name, scale=l, shape=False, y_max=y_max)
        fig.clear()
        rax.clear()
        #pd_ax.clear()
        ax.clear()

        fig, ax = plt.subplots(1,1,figsize=(7,7))

        hist.plot1d(histogram[notdata],overlay="dataset", density=True, stack=False, overflow='over', ax=ax) # make density plots because we don't care about x-sec differences
        for l in ['linear', 'log']:
            saveFig(fig, ax, None, plotDir, name+'_shape', scale=l, shape=True)
        fig.clear()
        ax.clear()
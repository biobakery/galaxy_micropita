#!/usr/bin/env python

import sys
import numpy as np 
import matplotlib
matplotlib.use('Agg')
import scipy
import pylab
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as dis 
from scipy import stats

# User defined color maps (in addition to matplotlib ones)
bbcyr = {'red':  (  (0.0, 0.0, 0.0),
                    (0.25, 0.0, 0.0),
                    (0.50, 0.0, 0.0),
                    (0.75, 1.0, 1.0),
                    (1.0, 1.0, 1.0)),
         'green': ( (0.0, 0.0, 0.0),
                    (0.25, 0.0, 0.0),
                    (0.50, 1.0, 1.0),
                    (0.75, 1.0, 1.0),
                    (1.0, 0.0, 1.0)),
         'blue': (  (0.0, 0.0, 0.0),
                    (0.25, 1.0, 1.0),
                    (0.50, 1.0, 1.0),
                    (0.75, 0.0, 0.0),
                    (1.0, 0.0, 1.0))}

bbcry = {'red':  (  (0.0, 0.0, 0.0),
                    (0.25, 0.0, 0.0),
                    (0.50, 0.0, 0.0),
                    (0.75, 1.0, 1.0),
                    (1.0, 1.0, 1.0)),
         'green': ( (0.0, 0.0, 0.0),
                    (0.25, 0.0, 0.0),
                    (0.50, 1.0, 1.0),
                    (0.75, 0.0, 0.0),
                    (1.0, 1.0, 1.0)),
         'blue': (  (0.0, 0.0, 0.0),
                    (0.25, 1.0, 1.0),
                    (0.50, 1.0, 1.0),
                    (0.75, 0.0, 0.0),
                    (1.0, 0.0, 1.0))}
my_colormaps = [    ('bbcyr',bbcyr),
                    ('bbcry',bbcry)]



def read_params(args):
    import argparse as ap
    import textwrap

    p = ap.ArgumentParser( description= "TBA" )
    
    p.add_argument( '--in', '--inp', metavar='INPUT_FILE', type=str, 
                    nargs='?', default=sys.stdin,
                    help= "the input archive " )

    p.add_argument( '--out', metavar='OUTPUT_FILE', type=str, 
                    nargs = '?', default=None,
                    help= " the output file, image on screen"
                          " if not specified. " )

    p.add_argument( '-m', metavar='method', type=str,
                    choices=[   "single","complete","average",
                                "weighted","centroid","median",
                                "ward" ],
                    default="average" )

    dist_funcs = [  "euclidean","minkowski","cityblock","seuclidean",
                    "sqeuclidean","cosine","correlation","hamming",
                    "jaccard","chebyshev","canberra","braycurtis",
                    "mahalanobis","yule","matching","dice",
                    "kulsinski","rogerstanimoto","russellrao","sokalmichener",
                    "sokalsneath","wminkowski","ward"]
    p.add_argument( '-d', metavar='distance function', type=str,
                    choices=dist_funcs,
                    default="euclidean" )
    p.add_argument( '-f', metavar='distance function for features', type=str,
                    choices=dist_funcs,
                    default="d" )

    p.add_argument( '--dmf', metavar='distance matrix for features', type=str,
                    default = None )
    p.add_argument( '--dms', metavar='distance matrix for samples', type=str,
                    default = None )


    p.add_argument( '-l', metavar='sample label', type=str,
                    default = None )

    p.add_argument( '-s', metavar='scale norm', type=str,
                    default = 'lin', choices = ['log','lin'])

    p.add_argument( '-x', metavar='x cell width', type=float,
                    default = 0.1)
    p.add_argument( '-y', metavar='y cell width', type=float,
                    default = 0.1 )

    p.add_argument( '--minv', metavar='min value', type=float,
                    default = 0.0 )
    p.add_argument( '--maxv', metavar='max value', type=float,
                    default = None )

    p.add_argument( '--xstart', metavar='x coordinate of the top left cell '
                                        'of the values', 
                    type=int, default=1 )
    p.add_argument( '--ystart', metavar='y coordinate of the top left cell '
                                        'of the values', 
                    type=int, default=1 )
    p.add_argument( '--xstop', metavar='x coordinate of the bottom right cell '
                                        'of the values (default None = last row)', 
                    type=int, default=None )
    p.add_argument( '--ystop', metavar='y coordinate of the bottom right cell '
                                        'of the values (default None = last column)', 
                    type=int, default=None )

    p.add_argument( '--perc', metavar='percentile for ordering and rows selection', type=int, default=None )
    p.add_argument( '--top', metavar='selection of the top N rows', type=int, default=None )
    p.add_argument( '--norm', metavar='whether to normalize columns (default 0)', type=int, default=0 )
    
    p.add_argument( '--sdend_h', metavar='height of the sample dendrogram', type=float,
                    default = 0.1 )
    p.add_argument( '--fdend_w', metavar='width of the feature dendrogram', type=float,
                    default = 0.1 )
    p.add_argument( '--cm_h', metavar='height of the colormap', type=float,
                    default = 0.03 )
    p.add_argument( '--cm_ticks', metavar='label for ticks of the colormap', type=str,
                    default = None )
    
    p.add_argument( '--font_size', metavar='label_font_size', type=int,
                    default = 7 )
    p.add_argument( '--feat_dend_col_th', metavar='Color threshold for feature dendrogram', type=float,
                    default = None )
    p.add_argument( '--sample_dend_col_th', metavar='Color threshold for sample dendrogram', type=float,
                    default = None )
    p.add_argument( '--clust_ncols', metavar='Number of colors for clusters', type=int,
                    default = 7 )
    p.add_argument( '--clust_line_w', metavar='Cluster line width', type=float,
                    default = 1.0 )
    p.add_argument( '--label_cols', metavar='Label colors', type=str,
                    default = None )
    p.add_argument( '--label2cols', metavar='Label to colors mapping file', type=str,
                    default = None )
    p.add_argument( '--sdend_out', metavar='File for storing the samples dendrogram in PhyloXML format', type=str,
                    default = None )
    p.add_argument( '--fdend_out', metavar='File for storing the features dendrogram in PhyloXML format', type=str,
                    default = None )


    p.add_argument( '--pad_inches', metavar='Proportion of figure to be left blank around the plot', type=float,
                    default = 0.1 )


    p.add_argument( '--flabel', metavar='Whether to show the labels for the features', type=int,
                    default = 0 )
    p.add_argument( '--slabel', metavar='Whether to show the labels for the samples', type=int,
                    default = 0 )
 
    p.add_argument( '--legend', metavar='Whether to show the samples to label legend', type=int,
                    default = 0 )
    p.add_argument( '--legend_font_size', metavar='Legend font size', type=int,
                    default = 7 )
    p.add_argument( '--legend_ncol', metavar='Number of columns for the legend', type=int,
                    default = 3 )
    p.add_argument( '--grid', metavar='Whether to show the grid (only black for now)', type=int,
                    default = 0 )
  
    col_maps = ['Accent', 'Blues', 'BrBG', 'BuGn', 'BuPu', 'Dark2', 'GnBu', 
                'Greens', 'Greys', 'OrRd', 'Oranges', 'PRGn', 'Paired', 
                'Pastel1', 'Pastel2', 'PiYG', 'PuBu', 'PuBuGn', 'PuOr', 
                'PuRd', 'Purples', 'RdBu', 'RdGy', 'RdPu', 'RdYlBu', 'RdYlGn', 
                'Reds', 'Set1', 'Set2', 'Set3', 'Spectral', 'YlGn', 'YlGnBu', 
                'YlOrBr', 'YlOrRd', 'afmhot', 'autumn', 'binary', 'bone', 
                'brg', 'bwr', 'cool', 'copper', 'flag', 'gist_earth', 
                'gist_gray', 'gist_heat', 'gist_ncar', 'gist_rainbow', 
                'gist_stern', 'gist_yarg', 'gnuplot', 'gnuplot2', 'gray', 
                'hot', 'hsv', 'jet', 'ocean', 'pink', 'prism', 'rainbow', 
                'seismic', 'spectral', 'spring', 'summer', 'terrain', 'winter'] + [n for n,c in my_colormaps]
    p.add_argument( '-c', metavar='colormap', type=str,
                    choices = col_maps, default = 'jet' )

    return vars(p.parse_args()) 

# Predefined colors for dendrograms brances and class labels
colors = [  "#B22222","#006400","#0000CD","#9400D3","#696969","#8B4513",
            "#FF1493","#FF8C00","#3CB371","#00Bfff","#CDC9C9","#FFD700",
            "#2F4F4F","#FF0000","#ADFF2F","#B03060" ]

def samples2classes_panel(fig, samples, s2l, idx1, idx2, height, xsize, cols, legendon, fontsize, label2cols, legend_ncol ):
    from matplotlib.patches import Rectangle
    samples2labels = dict([(l[0],l[1]) 
                            for l in [ll.strip().split('\t') 
                                for ll in open(s2l)] if len(l) > 1])
   
    if label2cols:
        labels2colors = dict([(l[0],l[1]) for l in [ll.strip().split('\t') for ll in open(label2cols)]])
    else:
        cs = cols if cols else colors
        labels2colors = dict([(l,cs[i%len(cs)]) for i,l in enumerate(set(samples2labels.values()))])
    ax1 = fig.add_axes([0.,1.0,1.0,height],frameon=False)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_ylim( [0.0, height] )
    ax1.set_xlim( [0.0, xsize] )
    step = xsize / float(len(samples))
    labels = set()
    added_labels = set()
    for i,ind in enumerate(idx2):
        if  not samples[ind] in samples2labels or \
            not samples2labels[samples[ind]] in labels2colors:
            fc, ll = "k", None
        else:
            ll = samples2labels[samples[ind]]
            ll = None if ll in added_labels else ll
            added_labels.add( ll )
            fc = labels2colors[samples2labels[samples[ind]]]
    
        rect = Rectangle( [float(i)*step, 0.0], step, height,
                            facecolor = fc,
                            label = ll,
                            edgecolor='b', lw = 0.0)
        labels.add( ll )
        ax1.add_patch(rect)
    ax1.autoscale_view()
   
    if legendon:
        ax1.legend( loc = 2, ncol = legend_ncol, bbox_to_anchor=(1.01, 3.),
                    borderpad = 0.0, labelspacing = 0.0,
                    handlelength = 0.5, handletextpad = 0.3,
                    borderaxespad = 0.0, columnspacing = 0.3,
                    prop = {'size':fontsize}, frameon = False)

def samples_dend_panel( fig, Z, Z2, ystart, ylen, lw ):
    ax2 = fig.add_axes([0.0,1.0+ystart,1.0,ylen], frameon=False)
    Z2['color_list'] = [c.replace('b','k') for c in Z2['color_list']]
    mh = max(Z[:,2])
    sch._plot_dendrogram(   Z2['icoord'], Z2['dcoord'], Z2['ivl'], 
                            Z.shape[0] + 1, Z.shape[0] + 1, 
                            mh, 'top', no_labels=True, 
                            color_list=Z2['color_list'] )
    for coll in ax2.collections:
        coll._linewidths = (lw,)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xticklabels([])
 
def features_dend_panel( fig, Z, Z2, width, lw ):
    ax1 = fig.add_axes([-width,0.0,width,1.0], frameon=False)
    Z2['color_list'] = [c.replace('b','k').replace('x','b') for c in Z2['color_list']]
    mh = max(Z[:,2])
    sch._plot_dendrogram(Z2['icoord'], Z2['dcoord'], Z2['ivl'], Z.shape[0] + 1, Z.shape[0] + 1, mh, 'right', no_labels=True, color_list=Z2['color_list'])
    for coll in ax1.collections:
        coll._linewidths = (lw,)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xticklabels([])
 

def add_cmap( cmapdict, name ):
    my_cmap = matplotlib.colors.LinearSegmentedColormap(name,cmapdict,256)
    pylab.register_cmap(name=name,cmap=my_cmap)

def init_fig(xsize,ysize,ncol):
    fig = pylab.figure(figsize=(xsize,ysize))
    sch._link_line_colors = colors[:ncol] 
    return fig

def heatmap_panel( fig, D, minv, maxv, idx1, idx2, cm_name, scale, cols, rows, label_font_size, cb_offset, cb_l, flabelson, slabelson, cm_ticks, gridon, bar_offset ):
    cm = pylab.get_cmap(cm_name)
    bottom_col = [    cm._segmentdata['red'][0][1],
                      cm._segmentdata['green'][0][1],
                      cm._segmentdata['blue'][0][1]   ]
    axmatrix = fig.add_axes(    [0.0,0.0,1.0,1.0],
                                axisbg=bottom_col)
    if any([c < 0.95 for c in bottom_col]):
        axmatrix.spines['right'].set_color('none')
        axmatrix.spines['left'].set_color('none')
        axmatrix.spines['top'].set_color('none')
        axmatrix.spines['bottom'].set_color('none')
    norm_f = matplotlib.colors.LogNorm if scale == 'log' else matplotlib.colors.Normalize
    im = axmatrix.matshow(  D, norm = norm_f(   vmin=minv if minv > 0.0 else None,
                                                vmax=maxv), 
                            aspect='auto', origin='lower', cmap=cm, vmax=maxv)
    
    axmatrix2 = axmatrix.twinx()
    axmatrix3 = axmatrix.twiny()
   
    axmatrix.set_xticks([])
    axmatrix2.set_xticks([])
    axmatrix3.set_xticks([])
    axmatrix.set_yticks([])
    axmatrix2.set_yticks([])
    axmatrix3.set_yticks([])
    
    axmatrix.set_xticklabels([])
    axmatrix2.set_xticklabels([])
    axmatrix3.set_xticklabels([])
    axmatrix.set_yticklabels([])
    axmatrix2.set_yticklabels([])
    axmatrix3.set_yticklabels([])

    if any([c < 0.95 for c in bottom_col]):
        axmatrix2.spines['right'].set_color('none')
        axmatrix2.spines['left'].set_color('none')
        axmatrix2.spines['top'].set_color('none')
        axmatrix2.spines['bottom'].set_color('none')
    if any([c < 0.95 for c in bottom_col]):
        axmatrix3.spines['right'].set_color('none')
        axmatrix3.spines['left'].set_color('none')
        axmatrix3.spines['top'].set_color('none')
        axmatrix3.spines['bottom'].set_color('none')
    if flabelson:
        axmatrix2.set_yticks(np.arange(len(rows))+0.5)
        axmatrix2.set_yticklabels([rows[r] for r in idx1],size=label_font_size,va='center')
    if slabelson:
        axmatrix.set_xticks(np.arange(len(cols)))
        axmatrix.set_xticklabels([cols[r] for r in idx2],size=label_font_size,rotation=90,va='top',ha='center')
    axmatrix.tick_params(length=0)
    axmatrix2.tick_params(length=0)
    axmatrix3.tick_params(length=0)
    axmatrix2.set_ylim(0,len(rows))
  
    if gridon:
        axmatrix.set_yticks(np.arange(len(idx1)-1)+0.5)
        axmatrix.set_xticks(np.arange(len(idx2))+0.5)
        axmatrix.grid( True )
        ticklines = axmatrix.get_xticklines()
        ticklines.extend( axmatrix.get_yticklines() )
        #gridlines = axmatrix.get_xgridlines()
        #gridlines.extend( axmatrix.get_ygridlines() )

        for line in ticklines:
            line.set_linewidth(3)
    
    if cb_l > 0.0:
        axcolor = fig.add_axes([0.0,1.0+bar_offset*1.25,1.0,cb_l])
        cbar = fig.colorbar(im, cax=axcolor, orientation='horizontal')
        cbar.ax.tick_params(labelsize=label_font_size)
        if cm_ticks:
            cbar.ax.set_xticklabels( cm_ticks.split(":") )


def read_table( fin, xstart,xstop,ystart,ystop, percentile = None, top = None, norm = False ):
    mat = [l.rstrip().split('\t') for l in open( fin )]
    
    if fin.endswith(".biom"):
        sample_labels =  mat[1][1:-1]
        m = [(mm[-1]+"; OTU"+mm[0],np.array([float(f) for f in mm[1:-1]])) for mm in mat[2:]]
        #feat_labels = [m[-1].replace(";","_").replace(" ","")+m[0] for m in mat[2:]]
        #print len(feat_labels)
        #print len(sample_labels)
    else:
        sample_labels = mat[0][xstart:xstop]
        m = [(mm[xstart-1],np.array([float(f) for f in mm[xstart:xstop]])) for mm in mat[ystart:ystop]]

    if norm:
        msums = [0.0 for l in m[0][1]]
        for mm in m:
            for i,v in enumerate(mm[1]):
                msums[i] += v

    if top and not percentile:
        percentile = 90
    
    if percentile:
        m = sorted(m,key=lambda x:-stats.scoreatpercentile(x[1],percentile))
    if top:
        if fin.endswith(".biom"):
            #feat_labels = [mm[-1].replace(";","_").replace(" ","")+mm[0] for mm in m[:top]]
            feat_labels = [mm[0] for mm in m[:top]]
        else:
            feat_labels = [mm[0] for mm in m[:top]]
        if norm:
            m = [np.array([n/v for n,v in zip(mm[1],msums)]) for mm in m[:top]]
        else:
            m = [mm[1] for mm in m[:top]]
    else:
        if fin.endswith(".biom"):
            feat_labels = [mm[0] for mm in m]
        else:
            feat_labels = [mm[0] for mm in m]
        if norm:
            m = [np.array([n/v for n,v in zip(mm[1],msums)]) for mm in m]
        else:
            m = [mm[1] for mm in m]
        #m = [mm[1] for mm in m]

    D = np.matrix(  np.array( m ) )

    return D, feat_labels, sample_labels

def read_dm( fin, n ):
    mat = [[float(f) for f in l.strip().split('\t')] for l in open( fin )]
    nc = sum([len(r) for r in mat]) 
    
    if nc == n*n:
        dm = []
        for i in range(n):
            dm += mat[i][i+1:]
        return np.array(dm)
    if nc == (n*n-n)/2:
        dm = []
        for i in range(n):
            dm += mat[i]
        return np.array(dm)
    sys.stderr.write( "Error in reading the distance matrix\n" )
    sys.exit()


def exp_newick( inp, labels, outfile, tree_format = 'phyloxml' ):
    n_leaves = int(inp[-1][-1])
    from Bio import Phylo
    import collections
    from Bio.Phylo.BaseTree import Tree as BTree
    from Bio.Phylo.BaseTree import Clade as BClade
    tree = BTree()
    tree.root = BClade()
    
    subclades = {}
    sb_cbl = {}

    for i,(fr,to,bl,nsub) in enumerate( inp ):
        if fr < n_leaves:
            fr_c = BClade(branch_length=-1.0,name=labels[int(fr)])
            subclades[fr] = fr_c
            sb_cbl[fr] = bl
        if to < n_leaves:
            to_c = BClade(branch_length=-1.0,name=labels[int(to)])
            subclades[to] = to_c
            sb_cbl[to] = bl
    for i,(fr,to,bl,nsub) in enumerate( inp ):
        fr_c = subclades[fr] 
        to_c = subclades[to]
        cur_c = BClade(branch_length=bl)
        cur_c.clades.append( fr_c )
        cur_c.clades.append( to_c )
        subclades[i+n_leaves] = cur_c

    def reset_rec( clade, fath_bl ):
        if clade.branch_length < 0:
            clade.branch_length = fath_bl
            return
        for c in clade.clades:
            reset_rec( c, clade.branch_length )
        clade.branch_length = fath_bl-clade.branch_length

    tree.root = cur_c
    reset_rec( tree.root, 0.0 )
    tree.root.branch_length = 0.0
    Phylo.write(tree, outfile, tree_format )

def hclust(  fin, fout,
             method = "average",
             dist_func = "euclidean",
             feat_dist_func = "d",
             xcw = 0.1,
             ycw = 0.1,
             scale = 'lin',
             minv = 0.0,
             maxv = None,
             xstart = 1,
             ystart = 1,
             xstop = None,
             ystop = None,
             percentile = None,
             top = None,
             norm = False,
             cm_name = 'jet',
             s2l = None,
             label_font_size = 7,
             feat_dend_col_th = None,
             sample_dend_col_th = None,
             clust_ncols = 7,
             clust_line_w = 1.0,
             label_cols = None,
             sdend_h = 0.1,
             fdend_w = 0.1,
             cm_h = 0.03,
             dmf = None,
             dms = None,
             legendon = False,
             label2cols = None,
             flabelon = True,
             slabelon = True,
             cm_ticks = None,
             legend_ncol = 3,
             pad_inches = None,
             legend_font_size = 7,
             gridon = 0,
             sdend_out = None,
             fdend_out= None):

    if label_cols and label_cols.count("-"):
        label_cols = label_cols.split("-")

    for n,c in my_colormaps:
        add_cmap( c, n )
    
    if feat_dist_func == 'd':
        feat_dist_func = dist_func

    D, feat_labels, sample_labels = read_table(fin,xstart,xstop,ystart,ystop,percentile,top,norm)

    ylen,xlen = D[:].shape
    Dt = D.transpose() 

    size_cx, size_cy = xcw, ycw
 
    xsize, ysize = max(xlen*size_cx,2.0), max(ylen*size_cy,2.0)
    ydend_offset = 0.025*8.0/ysize if s2l else 0.0

    fig = init_fig(xsize,ysize,clust_ncols)

    nfeats, nsamples = len(D), len(Dt) 
    
    if dmf:
        p1 = read_dm( dmf, nfeats )
        Y1 = sch.linkage(   p1, method=method )
    else:
        p1 = dis.pdist( D, feat_dist_func )
        Y1 = sch.linkage( p1, method=method ) # , metric=feat_dist_func )
        #Y1 = sch.linkage( D, method=method, metric=feat_dist_func )
    Z1 = sch.dendrogram(Y1, no_plot=True, color_threshold=feat_dend_col_th) 
    
    if fdend_out:
        exp_newick( Y1, feat_labels, fdend_out )

    if dms:
        p2 = read_dm( dms, nsamples )
        Y2 = sch.linkage(   p2, method=method )
    else:
        p2 = dis.pdist( Dt, dist_func )
        Y2 = sch.linkage(   p2, method=method ) # , metric=dist_func )
        #Y2 = sch.linkage(   Dt, method=method, metric=dist_func )
    Z2 = sch.dendrogram(Y2, no_plot=True, color_threshold=sample_dend_col_th) 

    if sdend_out:
        exp_newick( Y2, sample_labels, sdend_out )

    if fdend_w > 0.0:
        features_dend_panel(fig, Y1, Z1, fdend_w*8.0/xsize, clust_line_w ) 
    if sdend_h > 0.0: 
        samples_dend_panel(fig, Y2, Z2, ydend_offset, sdend_h*8.0/ysize, clust_line_w)
 
    idx1, idx2 = Z1['leaves'], Z2['leaves']
    D = D[idx1,:][:,idx2]
   
    if s2l:
        samples2classes_panel( fig, sample_labels, s2l, idx1, idx2, 0.025*8.0/ysize, xsize, label_cols, legendon, legend_font_size, label2cols, legend_ncol )
    heatmap_panel( fig, D, minv, maxv, idx1, idx2, cm_name, scale, sample_labels, feat_labels, label_font_size, -cm_h*8.0/ysize, cm_h*0.8*8.0/ysize, flabelon, slabelon, cm_ticks, gridon, ydend_offset+sdend_h*8.0/ysize )
  
    fig.savefig(    fout, bbox_inches='tight',  
                    pad_inches = pad_inches, 
                    dpi=300) if fout else pylab.show()

if __name__ == '__main__':
    pars = read_params( sys.argv )
  
    hclust(   fin  = pars['in'],
              fout = pars['out'],
              method = pars['m'],
              dist_func = pars['d'],
              feat_dist_func = pars['f'],
              xcw = pars['x'],
              ycw = pars['y'],
              scale = pars['s'],
              minv = pars['minv'],
              maxv = pars['maxv'],
              xstart = pars['xstart'],
              ystart = pars['ystart'],
              xstop = pars['xstop'],
              ystop = pars['ystop'],
              percentile = pars['perc'],
              top = pars['top'],
              norm = pars['norm'],
              cm_name = pars['c'],
              s2l = pars['l'],
              label_font_size = pars['font_size'],
              feat_dend_col_th = pars['feat_dend_col_th'],
              sample_dend_col_th = pars['sample_dend_col_th'],
              clust_ncols = pars['clust_ncols'],
              clust_line_w = pars['clust_line_w'],
              label_cols = pars['label_cols'],
              sdend_h = pars['sdend_h'],
              fdend_w = pars['fdend_w'],
              cm_h = pars['cm_h'],
              dmf = pars['dmf'],
              dms = pars['dms'],
              legendon = pars['legend'],
              label2cols = pars['label2cols'],
              flabelon = pars['flabel'],
              slabelon = pars['slabel'],
              cm_ticks = pars['cm_ticks'],
              legend_ncol = pars['legend_ncol'],
              pad_inches = pars['pad_inches'],
              legend_font_size = pars['legend_font_size'],
              gridon = pars['grid'],
              sdend_out = pars['sdend_out'],
              fdend_out = pars['fdend_out'],
              )


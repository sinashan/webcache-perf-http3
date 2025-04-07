import os
import math
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
try:
   import cPickle as pickle
except:
   import pickle
import sys
from cycler import cycler

golden_ratio = 1.61803
figwidth = 8.5 / golden_ratio

bar_width = 0.20  # the width of the bars
stacked_bar_width = 0.50  # the width of the bars
multiplier = 0

def load(d, only_good=True):
    with open(os.path.join(os.path.dirname(__file__), '..', 'benchmarks', 'results', d, 'parsed.pickle'), 'rb') as f:
        df = pickle.load(f)
        if only_good:
            if d.startswith('lobsters'):
                df = df.query('op == "all" & achieved >= 0.99 * requested & mean < 50')
            elif d == "vote-migration":
                pass
            else:
                df = df.query('op == "all" & achieved >= 0.99 * target & mean < 20')
        return df
#
# set up general matplotlib styles so all figures look the same.
#

# matplotlib.style.use('ggplot')
matplotlib.rc('font', family='serif', size=13)
matplotlib.rc('text.latex', preamble='\\usepackage{mathptmx}')
matplotlib.rc('text', usetex=True)
matplotlib.rc('figure', figsize=(figwidth, figwidth / golden_ratio))
matplotlib.rc('legend', fontsize=13)
matplotlib.rc('axes', linewidth=1)
matplotlib.rc('lines', linewidth=2)

matplotlib.rcParams["axes.prop_cycle"] = cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])

# plt.tick_params(top='off', right='off', which='both')
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False

legendHandleTestPad = 0.4
legendColumnSpacing = 0.4
legendHandleLength = 1.4
legendLabelSpacing = 0.4
legendBorderpadSpacing = 0.4


kfmtfn = lambda x, pos: '%1.1fM' % (x * 1e-6) if x >= 1e6 else '%1.0fk' % (x * 1e-3) if x >= 1e3 else '%1.0f' % x
kfmt = matplotlib.ticker.FuncFormatter(kfmtfn)

def bts(b):
    if b >= 1024 * 1024 * 1024:
        return '%1.1fGB' % (b / 1024 / 1024 / 1024)
    if b >= 1024 * 1024:
        return '%1.0fMB' % (b / 1024 / 1024)
    if b >= 1024:
        return '%1.0fkB' % (b / 1024)
    return '%1.0fb' % b


# https://stackoverflow.com/questions/5194057/better-way-to-convert-file-sizes-in-python

def convert_size(size_bytes):
   if size_bytes == 0:
       return "0B"
   size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
   i = int(math.floor(math.log(size_bytes, 1024)))
   p = math.pow(1024, i)
   s = round(size_bytes / p, 2)
   return "%s %s" % (s, size_name[i])

flatui = ["#0072B2", "#D55E00", "#009E73", "#3498db", "#CC79A7", "#F0E442", "#56B4E9"]
color_pallete = ['#e69d00', '#0071b2', '#009e74', '#cc79a7', '#d54300', '#994F00', '#000000']
# matplotlib.rcParams["axes.prop_cycle"] = cycler('color', color_pallete)

full_markers = ["x", "v", "^", "<", ">", "1", "2", "3"]

# qualitative colors
# subset of https://jfly.uni-koeln.de/color/
# that is also distinctive in grayscale
colors = {
    'full': '#0071b2',
    'durable': '#0071b2',
    'noria': '#009e73',
    'mysql': '#e59c00',
    'redis': '#e59c00',
}

categories = {"Minio": ["Dispatched", "Queued", "OS journal", "Unknown", "Metadata", "Data", "Bucket"],
              "Etcd": ["Dispatched", "Queued", "OS journal", "Unknown", "Data", "Metadata"],
              "Postgres": ["Dispatched", "Queued", "OS journal", "Unknown", "Metadata", "Data"],
              "Ceph": ["Dispatched", "Queued", "osd", "bstore_kv_sync", "bstore_mempool",]
            }

def bar_fillings(n):
    if n == 2:
        return ["//", "\\\\"]
    elif n == 3:
        return ["//", "", "\\\\"]
    elif n == 4:
        return ["//" , "\\\\" , "||" , "o" ]
    elif n == 5:
        return ["//" , "\\\\" , "+" , "x", "" ]
    else:
        return ['//', '\\\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']

# https://colorbrewer2.org/#type=sequential&scheme=RdPu&n=8
def cb2_colors(n, bright=False):
    if not bright:
        # off by one from the official colors, because #feebe2 is too hard to see
        n += 1

    if n <= 3:
        return ['#c51b8a', '#fa9fb5', '#fde0dd']
    elif n == 4:
        return ['#ae017e', '#f768a1', '#fbb4b9', '#feebe2']
    elif n == 5:
        return ['#7a0177', '#c51b8a', '#f768a1', '#fbb4b9', '#feebe2']
    elif n == 6:
        return ['#7a0177', '#c51b8a', '#f768a1', '#fa9fb5', '#fcc5c0', '#feebe2']
    elif n == 7:
        return ['#7a0177', '#ae017e', '#dd3497', '#f768a1', '#fa9fb5', '#fcc5c0', '#feebe2']
    elif n == 8:
        return ['#7a0177', '#ae017e', '#dd3497', '#f768a1', '#fa9fb5', '#fcc5c0', '#fde0dd', '#fff7f3']
    else:
        return []

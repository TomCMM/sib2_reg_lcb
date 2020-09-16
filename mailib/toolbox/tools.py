#===============================================================================
# Contains general tools
#===============================================================================
import sys
import pandas as pd
import os
import unicodedata
import re


def reglob(path, exp, invert=False):
    """glob.glob() style searching which uses regex

    :param exp: Regex expression for filename
    :param invert: Invert match to non matching files
    """

    m = re.compile(exp)

    if invert is False:
        res = [f for f in os.listdir(path) if m.search(f)]
    else:
        res = [f for f in os.listdir(path) if not m.search(f)]

    res = list(map(lambda x: "%s/%s" % ( path, x, ), res))
    return res

def common_index(list1, list2):
    return [element for element in list1 if element in list2]


def merge_dicts(*dict_args):
    '''
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    '''
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result



def show_values(pc, fmt="%.2f", **kw):
    from itertools import izip
    pc.update_scalarmappable()
    ax = pc.get_axes()
    for p, color, value in izip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center",fontsize=14, color=color, **kw)

def apply_func(df, func):
    index = df.index
    columns = df.columns
    df_trans = func(df.values)
    return  pd.DataFrame(df_trans, index=index, columns=columns)


def assure_dirpath_exists(path):
    # dir = os.path.dirname(path)
    if not os.path.exists(path):
        os.makedirs(path)


def get_df_lags(df, lags=[1]):

    columns_name = [str(col)+'_t'+str(-i)  for i in lags for col in df.columns]

    df_shifted = [df.shift(i) for i in lags]
    # columns.append(df)
    df_lags = pd.concat(df_shifted, axis=1)
    df_lags.columns = columns_name
    df_lags.dropna(axis=0,how='any',inplace=True)
    # df_lags = df_lags.iloc[:,::-1]
    # df_lags.fillna(0, inplace=True)
    return df_lags


def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    only_ascii = nfkd_form.encode('ASCII', 'ignore')
    return only_ascii


def get_path_subdirs(path_dir):
    """

    :param path_dir: string, path of the main directory
    :return: list of string, path of the the subdirectories
    """
    subdirs = [os.path.join(path_dir, o) for o in os.listdir(path_dir) if
               os.path.isdir(os.path.join(path_dir, o))]

    return subdirs
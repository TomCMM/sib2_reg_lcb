import pandas as pd
import numpy as np
from lmfit import Minimizer, Parameters, report_fit
from sib2pymod import sib2
import pickle
import time

# data2 = '/dados/ProcessoOtimizacaoModelos/SiB2/input/radiative/' \
#     'data2'

# dadosobs = pd.read_table('data2', header=0, delim_whitespace=True, names=[
#             'datetime', 'Ki', 'em', 'tm', 'um', 'prec', 'Rn'])

# dadosobs = pd.read_table('data3.csv', delim_whitespace=True)
dadosobs = pd.read_csv('data3.csv')

h_o = np.asarray(dadosobs.H)
le_o = np.asarray(dadosobs.LE)

# print(dadosobs)
# time.sleep(30)

# verifica dados validos
posval = np.asarray((h_o > -9999.)
                    & (le_o > -9999.)).nonzero()
posval = posval[0]
h_o = h_o[posval]
le_o = le_o[posval]

nlinha = len(dadosobs)
# print(nlinha, ' <--------- nlinha')


def residualSiB2(params, h_o, le_o, posval, nlinha):

    gradm_param = params['gradm']
    gmudmu_param = params['gmudmu']
    greeness_param = params['greeness']
    vmax_param = params['vmax']

    print(params)

    '''
    Roda o SiB2
    '''
    [h_c, le_c] = sib2(gradm_param, gmudmu_param,
                       greeness_param, vmax_param,
                       nlinha)

    h_c = h_c[posval]
    le_c = le_c[posval]

    # print(len(Rn_O), len(Rn_C))
    # time.sleep(5)
    # print(params)

    modeloerro = (le_c/(h_c+le_c)) - (le_o/(h_o+le_o))

    # remove os 30 primeiros valores calculados
    # modeloerro = modeloerro[30:]
    print(modeloerro)
    return modeloerro


params = Parameters()
params.add('gradm', value=16.0)  # , max=0.9, min=0.01)
params.add('gmudmu', value=1.0)  # , max=0.9, min=0.01)
params.add('greeness', value=0.99)  # , max=0.9, min=0.01)
params.add('vmax', value=105.0)  # , max=0.9, min=0.01)  # , vary=False)

otimiza = Minimizer(residualSiB2, params,
                    reduce_fcn=None,
                    calc_covar=True,
                    fcn_args=(h_o, le_o, posval, nlinha))

# out_leastsq = otimiza.leastsq()
out_leastsq = otimiza.minimize(method='leastsq')  # Levenberg-Marquardt

print('###################################################')
print('Modulo: Carbono e agua')
print('---Parametros---')
params.pretty_print()
print('---Otimizacao---')
report_fit(out_leastsq)


# report_fit(out_leastsq.params)
# report_fit(out_leastsq)

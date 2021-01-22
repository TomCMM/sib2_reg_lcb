# from maihr_config import Maihr_Conf
from pathlib import Path


class Rib_conf():
    """
    SibCat and prepare data1/data2 nc config parameters
    Configthe sib2 pipeline in the Ribeirao Das Posses
    """
    def __init__(self):

        self.framework_path = Path("/vol0/thomas.martin/framework")
        ##################
        # Prepare data2 Parameters
        ##################
        self.maihrdata_path = self.framework_path / 'stamod_rib/out/clim_*.nc'
        self.irr_data_path =  '/vol0/thomas.martin/framework/database/predictors/out/irr/xr_irr_rib.nc'

        self.sib_folder = self.framework_path / 'stamod_rib/sib2_ribeirao_pos/pontual_ribeirao/'  # folder with the executable
        self.sib = self.framework_path / 'stamod_rib/sib2_ribeirao_pos/'  # folder with the executable
        self.out_path = self.framework_path / 'stamod_rib/sib2_ribeirao_pos/pontual_ribeirao/out/input_data/'
        self.res_path = self.framework_path / 'stamod_rib/sib2_ribeirao_pos/pontual_ribeirao/out/res/'

        ##################
        # SibCat Parameters
        ##################
        self.data2nc_path = '/vol0/thomas.martin/framework/stamod_rib/sib2_ribeirao_pos/data2_rib.nc'
        self.data1nc_path ='/vol0/thomas.martin/framework/stamod_rib/sib2_ribeirao_pos/pontual_ribeirao/data1'
        self.sibcat_out_nc_path = self.framework_path / 'stamod_rib/sib2_ribeirao_pos/pontual_ribeirao/out_nc/'


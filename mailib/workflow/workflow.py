import os
from datetime import date

class Framework(object): # TODO create a metaclass, I cannot create a init due to the luigi

    prefix_outfile = 'df_'  # Prefix to give to the output dataframe
    outpath_foldername = "out"  # Path of output folder
    respath_foldername = "res" # Path of module figure
    map_outputpath_folder = {'H': 'hourly', 'D': 'daily'}

    projects = {}
    path = '~'
    #
    # def __init__(self, path_framework= None, projects={}):
    #     self.path = path_framework
    #     # self.projects = projects

    def get_outfolder_path(self, project_name, module_name):
        """
        Get the output folder path

        :param project_name:
        :param module_name:
        :return:
        """
        out_module_path = self.path + module_name + '/' + self.outpath_foldername + "/" + project_name + '/' + self.map_outputpath_folder[self.projects[project_name]['by']] + '/'

        if not os.path.exists(out_module_path):
            os.makedirs(out_module_path)

        return out_module_path

    def get_resfolder_path(self, project_name, module_name):
        """
        Get the output folder path

        :param project_name:
        :param module_name:
        :return:
        """
        res_module_path = self.path + module_name + '/' + self.respath_foldername + "/" + project_name + '/' + self.map_outputpath_folder[self.projects[project_name]['by']] + '/'

        if not os.path.exists(res_module_path):
            os.makedirs(res_module_path)

        return res_module_path


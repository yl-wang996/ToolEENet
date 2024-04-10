import os
from urdf_parser_py.urdf import URDF

import numpy as np
class URDFLoader:
    def __init__(self,asset_root_path="/dataSSD/yunlong/dataspace/Dataset3DModel_v2.0"):
        self.asset_root_path = asset_root_path

    def create_urdf(self, obj_name, rgba=None, category="Hammer_Grip"):
        '''
            :param obj_name: name of the object
            :param rgba: color of the object, None denotes random colors
            :param category: category of the object, default is "Hammer_Grip"
            :return: urdf path
        '''

        # TODO: the load of stl will rewrite the file, which is weird. Need to fix this.
        object_mesh_path = os.path.join(self.asset_root_path, category, f"{obj_name}.stl")
        target_urdf_path = os.path.join(self.asset_root_path, category, f"{obj_name}.urdf")
        template_urdf = URDF.from_xml_file(os.path.join(self.asset_root_path, "object_template.urdf"))

        template_urdf.links[0].visuals[0].geometry.filename = object_mesh_path
        template_urdf.links[0].collisions[0].geometry.filename = object_mesh_path
        if rgba is not None:
            template_urdf.links[0].visuals[0].material.color.rgba = rgba
        else:
            template_urdf.links[0].visuals[0].material.color.rgba = np.random.rand(3).tolist() + [1]
            
        with open(target_urdf_path, 'w') as f:
            f.write(template_urdf.to_xml_string())

        # urdf_template.links[0].visuals[0].geometry.mesh.filename = object_mesh_path
        # urdf_template.links[0].collisions[0].geometry.mesh.filename = object_mesh_path
        # if color is not None:
        #     urdf_template.links[0].visuals[0].material.color = color
        # else:
        #     urdf_template.links[0].visuals[0].material.color = np.random.rand(3).tolist() + [1]
        # urdf_template.save(target_urdf_path)

        return self.asset_root_path, os.path.join(category, f"{obj_name}.urdf")

    def get_asset_root_path(self):
        return self.asset_root_path

    def set_asset_root_path(self, path):
        self.asset_root_path = path

    def get_urdf_path_from_asset_root(self, cat_name, obj_name):
        return os.path.join(cat_name, obj_name, f"{obj_name}.urdf")

    def get_obj_path(self,cat_name, obj_name):
        return os.path.join(self.asset_root_path, cat_name, obj_name, f"{obj_name}.obj")

    def get_category_list(self):
        return [name for name in os.listdir(self.asset_root_path) if os.path.isdir(os.path.join(self.asset_root_path, name))]

    def get_obj_list(self, cat_name):
        name_list = [name.split('.')[0] for name in os.listdir(os.path.join(self.asset_root_path, cat_name)) if '.stl' in name]
        name_list.sort()
        return name_list

# TODO, ah-hoc render new urdf with different color and return path
if __name__ == '__main__':
    urdf_loader = URDFLoader()
    category_list = urdf_loader.get_category_list()
    for category in category_list:
        print(f"creating new urdfs for {category}")
        obj_names = urdf_loader.get_obj_list(category)
        for obj_name in obj_names:
            print(f"creating new urdf for {obj_name}")
            urdf_path = urdf_loader.create_urdf(obj_name=obj_name, category=category, rgba=None)
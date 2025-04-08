import xacro
import xml.dom.minidom as minidom
from typing import Optional, Any
import numpy as np
from pathlib import Path
import rospy
import xml.etree.ElementTree as ET


class XacroHandler:
    """
    Class to handle the xacro files and modify them
    """

    def __init__(self, optimization_joints, log_path, arm_selection) -> None:
        self.fmm_file_path = "/root/catkin_ws_fmm/src/fmm_description/urdf/fmm.urdf.xacro"
        self.robot_file_path = "gazebo_world/fmm/franka_arm/franka_arm.xacro"
        self.log_path = log_path
        self.optimization_joints = optimization_joints
        self.iterator = 1
        self.modified_values = {}
        self.arm_selection = arm_selection

    def read_in(self, file_path: Optional[str] = None) -> minidom.Document:
        """
        read in the xacro file and parse it to a minidom object
        """

        # Specify the path to your xacro file
        xacro_file = self.fmm_file_path if file_path is None else file_path

        doc = xacro.parse(None, xacro_file)
        return doc

    def find_tag_recursive(self, root: minidom.Element, tag: str, list: list) -> None:
        """
        find the xacro tag of the xacro root-tree recursively
        """
        for child in root.childNodes:
            if child.nodeName == tag:
                list.append(child)
            elif child.childNodes.length > 0:
                self.find_tag_recursive(child, tag, list)

    def change_node_attr(self, node, attribute: str, new_val: str) -> None:
        """
        change the attribute value of a certain node
        """
        attr = node.getAttributeNode(attribute)
        if attr is not None:
            attr.value = new_val
            return
        # sometimes we have to find the origin name (child) in the tag node
        for new_node in node.childNodes:
            if new_node.nodeName == "origin":
                attr = new_node.getAttributeNode(attribute)
                if attr is not None:
                    attr.value = new_val
                    return

    def write_urdf_file(self, dom_tree: minidom.Document, name: str) -> None:
        """
        convert the xacro file to urdf and save it
        """
        # write file to gazebo_world/fmm/fmm.urdf
        robot_definition = dom_tree.toprettyxml(indent=" ")
        path = "gazebo_world/fmm/" + name
        with open(path, "w") as outfile:
            outfile.write(robot_definition)

    def save_file(self, file_path: str, dom_tree: minidom.Document, id: int):
        """
        write the modified xacro data back to the file
        """
        xacro_data = dom_tree.toprettyxml(indent=" ")
        with open(file_path, "w") as file:
            file.write(xacro_data)
        with open(
            self.log_path / "xacros" / f"{Path(file_path).name}_{id}.urdf.xacro", "w"
        ) as file:
            file.write(xacro_data)

    def convert_to_urdf(self, name: str) -> None:
        """
        converts xacro file to urdf
        """
        file_dom = self.read_in()
        xacro_args = {"ros_distro": "noetic"}
        # resolve xacro to udfr
        xacro.process_doc(file_dom, mappings=xacro_args)
        self.write_urdf_file(file_dom, name)

    def get_design_value(self, elem_name: str, attribute: str) -> str:
        """
        return the actual value of the searched element attribute
        """
        description = rospy.get_param("robot_description")
        # Parse the XML string
        root = ET.fromstring(description)

        # Find the specific tag you're looking fo
        element = root.find(".//joint[@name='{}']".format(elem_name))
        value = None
        for child in list(element):
            if child.tag == "origin":
                value = child.attrib[attribute]
                break
        return value

    def modify_string(self, index: int, input_string: str, value: Any) -> Optional[str]:
        """
        modify the xacro string at the position that should be changed for the new configuration
        """
        # Split the input string into a list of individual values
        values = input_string.split()

        # Check if the index is valid
        if 1 <= index <= 3:
            # Update the value at the specified index with the new value
            values[index - 1] = str(value)

            # Join the modified values back into a string
            modified_string = " ".join(values)

            return modified_string
        else:
            print("Invalid index. Please provide an index between 1 and 3.")
            return None

    def write_config_to_file(self, config: dict) -> None:
        """
        Write the new design parameters for the franka arm to the xacro file and save it.
        """
        for list, parameter_list in self.optimization_joints.items():
            file_dom = self.read_in(parameter_list["file_path"])
            root = file_dom.documentElement

            for key, value in config.items():
                if key not in parameter_list["parameters"]:
                    # just write the paraemters that are in the parameter list
                    continue
                properties = parameter_list["parameters"][key]
                self.write_attribute(properties, key, root, value)

            self.save_file(parameter_list["file_path"], file_dom, self.iterator)
        self.iterator += 1

    def write_attribute(
        self, properties: list, key: str, root: minidom.Element, value: float
    ) -> None:
        for idx, prop in enumerate(properties):
            link_name = prop["link_name"]
            attribute = prop["attribute"]
            position = prop["position"]
            tag = prop["tag"]
            if "additional" in prop:
                addtional_information = prop["additional"]

            if key == "tower_xValue" and idx == 1:
                """
                Special case: Change the distance between arm and tower so that when the tower is moved more to the middle,
                the arm moves the same distance towards the edge. The constant value 0.426 keeps the arm at the edge of the robot.
                """
                value = 0.426 - value
                # value = 0.40 - value
            elif key == "end_effector_mount":
                # franka arm

                bearing_range = [0, np.pi / 2]
                if idx == 0:
                    original_value = value
                elif idx == 1 and self.arm_selection == "franka_arm":
                    y_range = [0, -0.10]
                    value = np.interp(original_value, bearing_range, y_range)
                elif idx == 1 and self.arm_selection == "ur5_arm":
                    xyz_range = [0, 0.03]
                    value = np.interp(original_value, bearing_range, xyz_range)


            modify_key = (link_name, attribute)

            if modify_key in self.modified_values:
                old_value = self.modified_values[modify_key]
            else:
                old_value = self.get_design_value(link_name, attribute)

            new_value = self.modify_string(position, old_value, value)
            self.modified_values[modify_key] = new_value

            tag_nodes = []
            self.find_tag_recursive(root, tag, tag_nodes)
            target_node = tag_nodes[0]
            if len(tag_nodes) > 1:
                for node in tag_nodes:
                    if addtional_information in node.getAttribute("name"):
                        target_node = node
                        break
            self.change_node_attr(target_node, attribute, new_value)


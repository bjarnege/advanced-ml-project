# -*- coding: utf-8 -*-
"""
Created on Fri May 28 17:00:11 2021

@author: Bjarne Gerdes
"""

import pandas as pd
from enum import Enum 
import xml.etree.ElementTree as ET

# From https://www.kaggle.com/ysviru/simple-analysis-of-healthcare-job-postings
class XMLTagsUpperLevel:
    """
    This class defines the XML tag constants at the higher level of XML tree. The tag <file> is found below the root tag
    <arXivSRC> in the tree hierarchy.
    """
    FILE = "file"
    
class XMLTagsLowerLevel(Enum):
    """
    This class defines all the XML tag constants that are one level below the <file> tag. This is defined as an
    enumerated type for ease of iterating over all tags.
    """
    CONTENT_MD5SUM = "content_md5sum"
    FILENAME = "filename"
    FIRST_ITEM = "first_item"
    LAST_ITEM = "last_item"
    MD5SUM = "md5sum"
    NUM_ITEMS = "num_items"
    SEQ_NUM = "seq_num"
    SIZE = "size"
    TIMESTAMP = "timestamp"
    YYMM = "yymm"

class XMLParser:
    def __init__(self, file_path):
        """
        Initializes the XMLParser class instance.
        :param file_path: Path to input xml file containing all the jobs data.
        """
        self.file_path = file_path


    def xml_to_pandas_df(self):
        """
        Using the standard xml python library, we parse the data xml file and convert the xml data to a pandas
        data frame.
        :return: A pandas data frame instance containing all the manifest data.
        """
        tree = ET.parse(self.file_path)
        root = tree.getroot()

        manifest_data = dict()
        for tag in XMLTagsLowerLevel:
            manifest_data[tag.value] = []
    
        for i, record in enumerate(root.findall(XMLTagsUpperLevel.FILE)):
            for tag in XMLTagsLowerLevel:
                temp = record.find(tag.value)
                if temp is not None:
                    manifest_data[tag.value].append(temp.text)
                else:
                    manifest_data[tag.value].append("")

        return pd.DataFrame(data=manifest_data)
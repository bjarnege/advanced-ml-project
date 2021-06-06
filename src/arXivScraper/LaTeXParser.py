import re
import csv  
import glob
from tqdm import tqdm
from pathlib import Path
from pylatexenc.latex2text import LatexNodes2Text

class LaTeXtRacT:
    
    def __init__(self, table_commands, image_commands, code_commands):
        self.table_commands = table_commands
        self.image_commands = image_commands
        self.code_commands = code_commands

        self.latex_extractor = LatexNodes2Text()
    
    def readFile(self, path):
        files_at_path = [str(file.absolute()) for file in  Path(path).rglob('*.tex')]
        return "".join([open(file).read() for file in files_at_path])
        
    def extract(self, document):
        tables_text = []
        images_text = []
        code_text = []
        references_text = []
        
        regex_cmd = r"(?=\\begin{command})([\S\s]*?)(?<=\\end{command})"
        # extract data using regex
        try:
            for tc in self.table_commands:
                for g in re.findall(regex_cmd.replace\
                                   ("command", tc)
                                   , document):
                    tables_text.append(self.latex_extractor.latex_to_text(g).replace('"',''))
        except:
            None
        try:
            for ic in self.image_commands:
                for g in re.findall(regex_cmd.replace\
                                   ("command", ic)
                                   , document):
                    images_text.append(self.latex_extractor.latex_to_text(g).replace('"',''))
        except:
            None
            

        for cc in self.code_commands:
            for g in re.findall(regex_cmd.replace\
                               ("command", cc)
                               , document):
                code_text.append(self.latex_extractor.latex_to_text(g).replace('"',''))
        
        return tables_text, images_text,references_text, code_text
    
    def processFolder(self, folderpath, output_path="textracts.csv"):
        paths = set(["".join([p+"/" for p in path.parts[:3]])[:-1]\
                          for path in Path(folderpath).rglob('*.tex')])
        
        paperids = [p.split("/")[-1] for p in paths]
        
        print("Found: ",len(paths),"files")
        
        with open(output_path, 'w') as f:
            writer = csv.writer(f)
            fields = ['paper_id','tables_text','images_text', 'code_text']
            writer.writerow(fields)
            
            for path, paperid in tqdm(zip(paths, paperids), total=len(paths)):
                try:
                    all_tex = self.readFile(path)
                    tables_text, images_text, references_text, code_text = self.extract(all_tex)
                    tables_text, images_text, references_text, code_text = "".join(tables_text), "".join(images_text),\
                                                                            "".join(references_text), "".join(code_text)
                    
                    writer.writerow([paperid, tables_text, images_text, references_text, code_text])
                except:
                    None
                    
if __name__ == "__main__":
    table_commands = ['tabulary', 'tabularx', 'tabular\*', 'tabular', 'table', 'sidewaystable', 'longtable', 'deluxetable', 'ctable']
    image_commands =["figure","figure\*"]
    code_commands = ['verbatim\*', 'verbatim', 'spverbatim', 'python', 'program', 'minted', 'lstlisting', 'alltt', 'algorithm']

    latextract = LaTeXtRacT(table_commands, image_commands, code_commands)
    latextract.processFolder("./data")

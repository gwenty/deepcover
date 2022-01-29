import os
import numpy as np
import sys

tarantula = []
ochiai = []
wong=[]
zoltar=[]
peptides=[]
hlas=[]

def add_spectrum(listj, filenames, dirpath):
    for file in filenames:
        if file=='spectrum.txt':
            spectrum_path=os.path.join(dirpath, file)
            spectrum=np.loadtxt(spectrum_path)
            peptide_positions, mhc_positions = get_positions(dirpath)
            listj.append([spectrum[peptide_positions], spectrum[mhc_positions]])

def get_positions(spectrum_path):
    file = os.path.join(spectrum_path, '../instance.npy')
    x=np.load(file, allow_pickle=True)
    peptide_positions = (x[1] == 2) & (x[0] != 2) & (x[0] != 3)
    mhc_positions = (x[1] == 0) & (x[0] != 2) & (x[0] != 3)
    return peptide_positions, mhc_positions

def save_spectrum(dirName, resultPath):
    for (dirpath, dirnames, filenames) in os.walk(dirName): 
        if 'tarantula' in dirpath:
            add_spectrum(tarantula, filenames, dirpath)
        if 'ochiai' in dirpath:
            add_spectrum(ochiai, filenames, dirpath)
        if 'wong' in dirpath:
            add_spectrum(wong, filenames, dirpath)
        if 'zoltar' in dirpath:
            add_spectrum(zoltar, filenames, dirpath)

    np.save(os.path.join(resultPath, 'tarantula'), np.array(tarantula))
    np.save(os.path.join(resultPath, 'ochiai'), np.array(ochiai))
    np.save(os.path.join(resultPath, 'wong'), np.array(wong))
    np.save(os.path.join(resultPath, 'zoltar'), np.array(zoltar))


# Give it the path to the folder (must be allele specific) 
# and where to save the results (note allele name not in saved file names)
def main():
    if len(sys.argv) - 1 == 0:
        path = '.'
        resultPath='.'
    elif len(sys.argv) - 1 == 1:
        path = sys.argv[1]
        resultPath='.'
    else:
        path = sys.argv[1]
        resultPath=sys.argv[2]

    if not os.path.isdir(path):
        os.makedirs(path)
    if not os.path.isdir(resultPath):
        os.makedirs(resultPath)

    save_spectrum(path, resultPath)

# Give only results path
def main2():
    alleles = ["HLA-A33-01", "HLA-A36-01", "HLA-B37-01", "HLA-B54-01", "HLA-B58-02", "HLA-C15-02", \
    "HLA-A33-03",  "HLA-A74-01", "HLA-B46-01", "HLA-B58-01", "HLA-C01-02", "HLA-C17-01"]

    for allele in alleles:
        resultPath = os.path.join(sys.argv[1], allele)
        if not os.path.isdir(resultPath):
            os.makedirs(resultPath)
        save_spectrum(allele, resultPath)

if __name__ == "__main__":
    main2()

